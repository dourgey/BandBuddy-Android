package cn.lonelyme.bandbuddy

import android.content.Context
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import cn.lonelyme.bandbuddy.engine.DemucsWindowSeparator
import cn.lonelyme.bandbuddy.engine.NativeDemucsBridge
import cn.lonelyme.bandbuddy.model.ModelStore
import com.qualcomm.qti.QnnDelegate
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Device-only quality/performance gate for experimental QNN backend layouts.
 *
 * Run one candidate per instrumentation process:
 *
 *   -e backendCandidate htp-all-fp16
 *   -e backendCandidate htp-broad-fp16
 *   -e backendCandidate htp-convs-fp16
 *   -e backendCandidate gpu-fp16
 *   -e backendCandidate gpu-fp32
 *   -e backendCandidate gpu-broad-fp16
 *   -e backendCandidate gpu-broad-fp32
 *   -e backendCandidate gpu-convs-fp16
 *   -e backendCandidate gpu-convs-fp32
 *   -e backendCandidate production
 *
 * This class deliberately does not change [cn.lonelyme.bandbuddy.engine.ModelRuntime].
 * A candidate must first pass host-side Torch comparison before it can be
 * considered for the production runtime.
 */
@RunWith(AndroidJUnit4::class)
class InferenceBackendCandidateInstrumentedTest {
    companion object {
        private const val TAG = "BandBuddyBackend"
        private const val MODEL_OPERATOR_COUNT = 3_504
        private val PRODUCTION_HTP_NODES = setOf(
            586, 867,
            2639, 2645, 2777, 2784, 2923, 2929,
            3157, 3164, 3303,
        )
        // Every Conv2D/TransposeConv node in the pinned 3,504-op graph except
        // the two known 7.8-second tail convolutions (3307 and 3487).
        private val CONVOLUTION_ACCELERATOR_NODES = setOf(
            20, 27, 79, 112, 164, 197, 206, 212, 242, 277, 307, 342,
            353, 360, 436, 469, 545, 578, 586, 592, 622, 657, 687, 722,
            730, 737, 765, 798, 826, 859, 867, 873, 903, 938, 968, 1003,
            1012, 1019, 1047, 1080, 1108, 1141, 1150, 1156, 1186, 1221,
            1251, 1286, 2207, 2216, 2246, 2281, 2311, 2349, 2356, 2366,
            2394, 2427, 2455, 2492, 2500, 2509, 2539, 2574, 2604, 2639,
            2645, 2655, 2683, 2716, 2744, 2777, 2784, 2793, 2823, 2858,
            2888, 2923, 2929, 2939, 3015, 3048, 3124, 3157, 3164, 3173,
            3203, 3238, 3268, 3303, 3317, 3369, 3402, 3454,
        )
    }

    @Test
    fun exportsTwoRealWindowsForTorchGate() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val candidate = Candidate.from(
            requireNotNull(
                InstrumentationRegistry.getArguments().getString("backendCandidate")
            ) { "Pass -e backendCandidate <candidate>" }
        )
        val modelStore = ModelStore(context)
        check(modelStore.isCurrentReady()) {
            "Download the verified production model before running backend candidates"
        }
        // Keep adb-provided calibration data in app-internal storage. On Android 14,
        // files pushed directly into the app-specific external directory are visible
        // to adb but can still be rejected by the app-facing FUSE layer.
        val inputRoot = File(context.filesDir, "backend-matrix-input")
        val outputRoot = File(
            requireNotNull(context.getExternalFilesDir(null)),
            "backend-matrix-output/${candidate.id}",
        ).apply { check(mkdirs() || isDirectory) }
        val cache = File(context.codeCacheDir, "backend-matrix-${candidate.id}-v1")
            .apply { check(mkdirs() || isDirectory) }

        val delegate = QnnDelegate(candidate.options(context, cache))
        check(delegate.isAvailable) { "QNN delegate unavailable for ${candidate.id}" }
        val modelFile = modelStore.currentModelFile()
        val mappedModel = FileInputStream(modelFile).channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0L, channel.size())
        }

        val mix = direct(2 * DemucsWindowSeparator.SAMPLES)
        val spec = direct(4 * 2_048 * DemucsWindowSeparator.SPECTRUM_FRAMES)
        val frequency = direct(
            DemucsWindowSeparator.SOURCES * 4 * 2_048 * DemucsWindowSeparator.SPECTRUM_FRAMES
        )
        val time = direct(DemucsWindowSeparator.OUTPUT_FLOATS)
        val output = direct(DemucsWindowSeparator.OUTPUT_FLOATS)
        val reports = JSONArray()

        try {
            val prepareStarted = SystemClock.elapsedRealtime()
            Interpreter(mappedModel, Interpreter.Options().apply {
                setUseXNNPACK(true)
                setNumThreads(4)
                addDelegate(delegate)
            }).use { interpreter ->
                val prepareMillis = SystemClock.elapsedRealtime() - prepareStarted
                Log.i(TAG, "candidate=${candidate.id} prepareMs=$prepareMillis")

                listOf("f1", "wuyun").forEach { caseName ->
                    val inputFile = File(inputRoot, "$caseName.interleaved.f32le")
                    check(
                        inputFile.isFile &&
                            inputFile.length() ==
                            DemucsWindowSeparator.SAMPLES * 2L * Float.SIZE_BYTES
                    ) { "Missing exact calibration input: $inputFile" }
                    loadInterleavedInput(inputFile, mix)

                    spec.clear()
                    mix.rewind()
                    val preprocessStarted = System.nanoTime()
                    check(NativeDemucsBridge.preprocess(mix, spec))
                    val preprocessNanos = System.nanoTime() - preprocessStarted

                    mix.rewind()
                    spec.rewind()
                    frequency.clear()
                    time.clear()
                    val inferenceStarted = System.nanoTime()
                    if (candidate.performanceVote) delegate.performanceVote()
                    try {
                        interpreter.runForMultipleInputsOutputs(
                            arrayOf(mix, spec),
                            mutableMapOf<Int, Any>(0 to frequency, 1 to time),
                        )
                    } finally {
                        if (candidate.performanceVote) delegate.performanceRelease()
                    }
                    val inferenceWallNanos = System.nanoTime() - inferenceStarted

                    frequency.rewind()
                    time.rewind()
                    output.clear()
                    val postprocessStarted = System.nanoTime()
                    check(NativeDemucsBridge.postprocess(frequency, time, output))
                    val postprocessNanos = System.nanoTime() - postprocessStarted

                    val stats = outputStats(output)
                    val caseRoot = File(outputRoot, caseName)
                        .apply { check(mkdirs() || isDirectory) }
                    writeFloatBuffer(File(caseRoot, "mix.f32le"), mix)
                    writeFloatBuffer(File(caseRoot, "final.f32le"), output)

                    val memory = Debug.MemoryInfo().also(Debug::getMemoryInfo)
                    val report = JSONObject()
                        .put("name", caseName)
                        .put("prepareMillis", prepareMillis)
                        .put("preprocessNanos", preprocessNanos)
                        .put("inferenceWallNanos", inferenceWallNanos)
                        .put(
                            "liteRtNativeNanos",
                            interpreter.lastNativeInferenceDurationNanoseconds ?: JSONObject.NULL,
                        )
                        .put("postprocessNanos", postprocessNanos)
                        .put("finite", stats.finite)
                        .put("peak", stats.peak)
                        .put("rms", stats.rms)
                        .put("totalPssKb", memory.totalPss)
                    reports.put(report)
                    Log.i(
                        TAG,
                        "candidate=${candidate.id} case=$caseName " +
                            "inferenceMs=${inferenceWallNanos / 1_000_000.0} " +
                            "nativeMs=${interpreter.lastNativeInferenceDurationNanoseconds?.div(1_000_000.0)} " +
                            "peak=${stats.peak} rms=${stats.rms} pssKb=${memory.totalPss}",
                    )
                    assertTrue("$caseName output contains NaN/Infinity", stats.finite)
                    assertTrue("$caseName output has implausible peak ${stats.peak}", stats.peak < 10f)
                }
            }

            val summary = JSONObject()
                .put("candidate", candidate.id)
                .put("cases", reports)
            File(outputRoot, "device-summary.json")
                .writeText(summary.toString(2), Charsets.UTF_8)
            File(outputRoot, "qnn-profile.bin").writeBytes(delegate.profilingResult)
        } finally {
            delegate.close()
        }
    }

    private enum class Candidate(
        val id: String,
        val backend: QnnDelegate.Options.BackendType,
        val gpuPrecision: QnnDelegate.Options.GpuPrecision? = null,
        val skippedNodes: IntArray? = null,
        val performanceVote: Boolean = false,
    ) {
        HTP_ALL_FP16(
            id = "htp-all-fp16",
            backend = QnnDelegate.Options.BackendType.HTP_BACKEND,
            performanceVote = true,
        ),
        HTP_BROAD_FP16(
            id = "htp-broad-fp16",
            backend = QnnDelegate.Options.BackendType.HTP_BACKEND,
            // Keep the numerically sensitive transformer and the two known
            // 7.8-second VTCM-heavy tail convolutions on XNNPACK FP32.
            skippedNodes = ((1_291..2_203) + listOf(3_307, 3_487)).toIntArray(),
            performanceVote = true,
        ),
        HTP_CONVS_FP16(
            id = "htp-convs-fp16",
            backend = QnnDelegate.Options.BackendType.HTP_BACKEND,
            skippedNodes = (0 until MODEL_OPERATOR_COUNT)
                .filterNot(CONVOLUTION_ACCELERATOR_NODES::contains)
                .toIntArray(),
            performanceVote = true,
        ),
        GPU_FP16(
            id = "gpu-fp16",
            backend = QnnDelegate.Options.BackendType.GPU_BACKEND,
            gpuPrecision = QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16,
        ),
        GPU_FP32(
            id = "gpu-fp32",
            backend = QnnDelegate.Options.BackendType.GPU_BACKEND,
            gpuPrecision = QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP32,
        ),
        GPU_BROAD_FP16(
            id = "gpu-broad-fp16",
            backend = QnnDelegate.Options.BackendType.GPU_BACKEND,
            gpuPrecision = QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16,
            // Avoid the Transformer softmax program-build OOM and the two
            // longest tail convolutions while retaining broad GPU coverage.
            skippedNodes = ((1_291..2_203) + listOf(3_307, 3_487)).toIntArray(),
        ),
        GPU_BROAD_FP32(
            id = "gpu-broad-fp32",
            backend = QnnDelegate.Options.BackendType.GPU_BACKEND,
            gpuPrecision = QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP32,
            skippedNodes = ((1_291..2_203) + listOf(3_307, 3_487)).toIntArray(),
        ),
        GPU_CONVS_FP16(
            id = "gpu-convs-fp16",
            backend = QnnDelegate.Options.BackendType.GPU_BACKEND,
            gpuPrecision = QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16,
            skippedNodes = (0 until MODEL_OPERATOR_COUNT)
                .filterNot(CONVOLUTION_ACCELERATOR_NODES::contains)
                .toIntArray(),
        ),
        GPU_CONVS_FP32(
            id = "gpu-convs-fp32",
            backend = QnnDelegate.Options.BackendType.GPU_BACKEND,
            gpuPrecision = QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP32,
            skippedNodes = (0 until MODEL_OPERATOR_COUNT)
                .filterNot(CONVOLUTION_ACCELERATOR_NODES::contains)
                .toIntArray(),
        ),
        PRODUCTION(
            id = "production",
            backend = QnnDelegate.Options.BackendType.HTP_BACKEND,
            skippedNodes = (0 until MODEL_OPERATOR_COUNT)
                .filterNot(PRODUCTION_HTP_NODES::contains)
                .toIntArray(),
            performanceVote = true,
        );

        fun options(context: Context, cache: File): QnnDelegate.Options =
            QnnDelegate.Options().apply {
                setCacheDir(cache.absolutePath)
                setModelToken("bandbuddy-backend-matrix-$id-a9fcc89e-v1")
                setBackendType(backend)
                setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
                setProfiling(QnnDelegate.Options.ProfilingOptions.BASIC_PROFILING)
                skippedNodes?.let(::setSkipNodeIds)
                if (backend == QnnDelegate.Options.BackendType.HTP_BACKEND) {
                    setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
                    setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
                    setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
                    setHtpPerformanceMode(
                        QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST
                    )
                    setHtpOptimizationStrategy(
                        QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_PREPARE
                    )
                } else {
                    setGpuPrecision(requireNotNull(this@Candidate.gpuPrecision))
                    setGpuPerformanceMode(
                        QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_HIGH
                    )
                }
            }

        companion object {
            fun from(value: String): Candidate = entries.firstOrNull { it.id == value }
                ?: error("Unknown backendCandidate=$value; expected ${entries.joinToString { it.id }}")
        }
    }

    private data class OutputStats(val finite: Boolean, val peak: Float, val rms: Double)

    private fun loadInterleavedInput(file: File, destination: FloatBuffer) {
        val interleaved = ByteBuffer.wrap(file.readBytes())
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
        destination.clear()
        repeat(DemucsWindowSeparator.SAMPLES) { frame ->
            destination.put(frame, interleaved.get(frame * 2))
            destination.put(
                DemucsWindowSeparator.SAMPLES + frame,
                interleaved.get(frame * 2 + 1),
            )
        }
        destination.rewind()
    }

    private fun outputStats(source: FloatBuffer): OutputStats {
        val values = source.duplicate().apply { rewind() }
        var finite = true
        var peak = 0f
        var squareSum = 0.0
        while (values.hasRemaining()) {
            val value = values.get()
            finite = finite && value.isFinite()
            peak = maxOf(peak, abs(value))
            squareSum += value.toDouble() * value
        }
        return OutputStats(
            finite = finite,
            peak = peak,
            rms = sqrt(squareSum / source.capacity()),
        )
    }

    private fun writeFloatBuffer(file: File, source: FloatBuffer) {
        source.rewind()
        FileOutputStream(file).channel.use { channel ->
            val bytes = ByteBuffer.allocate(256 * 1_024).order(ByteOrder.LITTLE_ENDIAN)
            while (source.hasRemaining()) {
                bytes.clear()
                repeat(minOf(source.remaining(), bytes.capacity() / Float.SIZE_BYTES)) {
                    bytes.putFloat(source.get())
                }
                bytes.flip()
                while (bytes.hasRemaining()) channel.write(bytes)
            }
        }
        source.rewind()
    }

    private fun direct(floats: Int): FloatBuffer = ByteBuffer
        .allocateDirect(floats * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
}
