package com.lonelyme.bandbuddy.engine

import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** Runs one original 7.8 second HTDemucs window through native DSP, LiteRT, and native DSP. */
class DemucsWindowSeparator(modelRuntime: ModelRuntime) : Closeable {
    companion object {
        const val SAMPLES = 343_980
        const val SPECTRUM_FRAMES = 336
        const val SOURCES = 6
        const val OUTPUT_FLOATS = SOURCES * 2 * SAMPLES
        private const val MIX_FLOATS = 2 * SAMPLES
        private const val SPEC_FLOATS = 4 * 2_048 * SPECTRUM_FRAMES
        private const val FREQUENCY_FLOATS = SOURCES * 4 * 2_048 * SPECTRUM_FRAMES
    }

    private val session = modelRuntime.openSession()
    private val mix = direct(MIX_FLOATS)
    private val spec = direct(SPEC_FLOATS)
    private val frequency = direct(FREQUENCY_FLOATS)
    private val time = direct(OUTPUT_FLOATS)
    private val output = direct(OUTPUT_FLOATS)
    private val inputs = arrayOf<Any>(mix, spec)
    private val outputs = mutableMapOf<Int, Any>(0 to frequency, 1 to time)
    private var preprocessNanos = 0L
    private var neuralCoreWallNanos = 0L
    private var postprocessNanos = 0L

    val backend: ModelRuntime.Backend
        get() = session.backend

    /** Input and output are planar: channel/source first, then samples. */
    fun separate(stereoPlanar: FloatArray): FloatArray =
        FloatArray(OUTPUT_FLOATS).also { separateInto(stereoPlanar, it) }

    /** Reuses all native tensor buffers and writes into a caller-owned array. */
    fun separateInto(stereoPlanar: FloatArray, destination: FloatArray) {
        require(stereoPlanar.size == MIX_FLOATS) { "窗口必须是 $SAMPLES 样本的双声道平面 PCM" }
        require(destination.size == OUTPUT_FLOATS) { "输出必须恰好容纳六个双声道分轨" }

        mix.clear()
        mix.put(stereoPlanar)
        mix.rewind()
        spec.clear()
        val preprocessStarted = System.nanoTime()
        check(NativeDemucsBridge.preprocess(mix, spec)) { "原生 STFT 预处理失败" }
        preprocessNanos = System.nanoTime() - preprocessStarted

        mix.rewind()
        spec.rewind()
        frequency.clear()
        time.clear()
        val neuralCoreStarted = System.nanoTime()
        session.run(inputs, outputs)
        neuralCoreWallNanos = System.nanoTime() - neuralCoreStarted

        frequency.rewind()
        time.rewind()
        output.clear()
        val postprocessStarted = System.nanoTime()
        check(NativeDemucsBridge.postprocess(frequency, time, output)) { "原生 iSTFT 后处理失败" }
        postprocessNanos = System.nanoTime() - postprocessStarted
        output.rewind()
        output.get(destination)
    }

    fun lastNeuralCoreInferenceNanos(): Long? = session.lastInferenceNanos()

    fun lastPreprocessNanos(): Long = preprocessNanos

    fun lastNeuralCoreWallNanos(): Long = neuralCoreWallNanos

    fun lastPostprocessNanos(): Long = postprocessNanos

    fun qnnProfilingResult(): ByteArray = session.profilingResult()

    override fun close() = session.close()

    private fun direct(floats: Int): FloatBuffer = ByteBuffer
        .allocateDirect(floats * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
}
