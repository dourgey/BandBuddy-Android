package com.lonelyme.bandbuddy.engine

import android.content.Context
import com.lonelyme.bandbuddy.model.ModelSpec
import com.lonelyme.bandbuddy.model.ModelStore
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Opens the fixed-shape six-stem neural core as a verified mixed-precision graph.
 *
 * The compute-heavy convolutions run on Qualcomm HTP FP16/HMX. Normalization
 * and attention remain on the FP32 CPU path because running those reductions
 * in HTP FP16 measurably corrupts Demucs output. HTP is mandatory: if the
 * delegate cannot be created, separation fails instead of silently presenting
 * a CPU-only run as NPU acceleration.
 */
class ModelRuntime(private val context: Context) {
    companion object {
        private const val MODEL_OPERATOR_COUNT = 3_504
        private val HTP_NODE_IDS = setOf(
            586, 867,
            2639, 2645, 2777, 2784, 2923, 2929,
            3157, 3164, 3303,
        )
        private val CPU_NODE_IDS = (0 until MODEL_OPERATOR_COUNT)
            .filterNot(HTP_NODE_IDS::contains)
            .toIntArray()
        private val MIX_SHAPE = intArrayOf(1, 2, DemucsWindowSeparator.SAMPLES)
        private val SPEC_SHAPE = intArrayOf(1, 4, 2_048, DemucsWindowSeparator.SPECTRUM_FRAMES)
        private val FREQUENCY_SHAPE = intArrayOf(
            1,
            DemucsWindowSeparator.SOURCES,
            4,
            2_048,
            DemucsWindowSeparator.SPECTRUM_FRAMES
        )
        private val TIME_SHAPE = intArrayOf(1, DemucsWindowSeparator.SOURCES, 2, DemucsWindowSeparator.SAMPLES)

        fun clearCompiledCache(context: Context) {
            val cacheDirectory = File(context.codeCacheDir, "qnn-htp")
            cacheDirectory.listFiles().orEmpty()
                .filter { file -> file.isFile }
                .forEach { file -> runCatching { file.delete() } }
        }
    }

    fun isInstalled(): Boolean = ModelStore(context).isCurrentReady()

    fun hasCompiledCache(): Boolean = qnnCacheDirectory().listFiles().orEmpty().any {
        it.isFile && it.length() > 0L && it.name.startsWith(ModelSpec.CACHE_TOKEN)
    }

    /** A light-weight capability check suitable for the settings screen. */
    fun statusSummary(): String {
        cleanupLegacyOrtArtifacts()
        requireHtpFp16()
        return "Qualcomm HTP NPU · QNN ${QnnDelegate.getVersion().joinToString(".")} · 混合精度"
    }

    fun openSession(): QnnSession {
        cleanupLegacyOrtArtifacts()
        requireHtpFp16()
        val model = mapModel()
        val qnnCache = qnnCacheDirectory().apply { mkdirs() }
        val qnnOptions = QnnDelegate.Options().apply {
            setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
            setCacheDir(qnnCache.absolutePath)
            setModelToken(ModelSpec.CACHE_TOKEN)
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
            setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            // Delegate only the 11 high-MAC convolutions whose FP16 output
            // passed the Torch/LiteRT/device quality gate. This becomes seven
            // real HTP partitions; all normalization and attention stay FP32.
            setSkipNodeIds(CPU_NODE_IDS)
            // The two longest time-branch convolutions stay on FP32 because
            // their 7.8-second HTP graphs exceed this phone's prepare limit.
            setHtpOptimizationStrategy(QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_PREPARE)
            setProfiling(QnnDelegate.Options.ProfilingOptions.BASIC_PROFILING)
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
        }

        val delegate = QnnDelegate(qnnOptions)
        try {
            check(delegate.isAvailable) { "Qualcomm HTP NPU delegate is unavailable" }
            val interpreterOptions = Interpreter.Options().apply {
                // XNNPACK executes the planned FP32 portion efficiently after
                // QNN has claimed the verified HTP convolution partitions.
                setUseXNNPACK(true)
                setNumThreads(4)
                addDelegate(delegate)
            }
            val interpreter = Interpreter(model, interpreterOptions)
            val session = QnnSession(interpreter, delegate, model)
            session.verifyContract()
            return session
        } catch (error: Throwable) {
            delegate.close()
            throw error
        }
    }

    /** Compiles the graph through QNN and validates its exact tensor ABI. */
    fun verifyRuntime() = openSession().use { Unit }

    private fun requireHtpFp16() {
        check(NativeDemucsBridge.supportsCurrentAbi()) { "Native DSP ABI does not match the six-stem model" }
        check(isInstalled()) { "分轨模型尚未下载或版本不匹配" }
        check(QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)) {
            "This device does not expose the Qualcomm HTP FP16 NPU runtime"
        }
    }

    private fun qnnCacheDirectory(): File = File(context.codeCacheDir, "qnn-htp")

    /** Removes only the two exact model files left by pre-QNN app versions. */
    private fun cleanupLegacyOrtArtifacts() {
        val legacyDirectory = File(context.filesDir, "models")
        listOf("htdemucs_6s.core.onnx", "htdemucs_6s.core.onnx.data").forEach { name ->
            runCatching { File(legacyDirectory, name).delete() }
        }
        if (legacyDirectory.listFiles().isNullOrEmpty()) runCatching { legacyDirectory.delete() }
    }

    private fun mapModel(): MappedByteBuffer {
        val file = ModelStore(context).currentModelFile()
        return FileInputStream(file).channel.use { channel ->
            channel.map(FileChannel.MapMode.READ_ONLY, 0L, file.length())
        }
    }

    class QnnSession internal constructor(
        private val interpreter: Interpreter,
        private val delegate: QnnDelegate,
        @Suppress("unused") private val mappedModel: MappedByteBuffer
    ) : Closeable {
        internal fun verifyContract() {
            check(interpreter.inputTensorCount == 2) { "Unexpected LiteRT input count" }
            check(interpreter.outputTensorCount == 2) { "Unexpected LiteRT output count" }
            check(interpreter.getInputTensor(0).shape().contentEquals(MIX_SHAPE)) { "Unexpected mix input shape" }
            check(interpreter.getInputTensor(1).shape().contentEquals(SPEC_SHAPE)) { "Unexpected spectrum input shape" }
            check(interpreter.getOutputTensor(0).shape().contentEquals(FREQUENCY_SHAPE)) { "Unexpected frequency output shape" }
            check(interpreter.getOutputTensor(1).shape().contentEquals(TIME_SHAPE)) { "Unexpected waveform output shape" }
        }

        fun run(inputs: Array<Any>, outputs: MutableMap<Int, Any>) {
            delegate.performanceVote()
            try {
                interpreter.runForMultipleInputsOutputs(inputs, outputs)
            } finally {
                delegate.performanceRelease()
            }
        }

        fun lastInferenceNanos(): Long? = interpreter.lastNativeInferenceDurationNanoseconds

        fun profilingResult(): ByteArray = delegate.profilingResult

        override fun close() {
            interpreter.close()
            delegate.close()
        }
    }
}
