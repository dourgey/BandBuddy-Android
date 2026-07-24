package com.lonelyme.bandbuddy

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.media.MediaMetadataRetriever
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import com.lonelyme.bandbuddy.engine.DemucsWindowSeparator
import com.lonelyme.bandbuddy.engine.AacM4aWriter
import com.lonelyme.bandbuddy.engine.ModelRuntime
import com.lonelyme.bandbuddy.engine.LongSongSeparator
import com.lonelyme.bandbuddy.data.SongRecord
import com.lonelyme.bandbuddy.data.SongStatus
import com.lonelyme.bandbuddy.data.StemRecord
import com.lonelyme.bandbuddy.data.StemType
import com.lonelyme.bandbuddy.data.TrackState
import com.lonelyme.bandbuddy.engine.MultiStemPlayer
import com.lonelyme.bandbuddy.engine.NativeDemucsBridge
import com.lonelyme.bandbuddy.model.ModelStore
import com.lonelyme.bandbuddy.playback.PlaybackService
import com.lonelyme.bandbuddy.worker.SeparationWorker
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Interpreter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.io.FileInputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.sin

@RunWith(AndroidJUnit4::class)
class DemucsInferenceInstrumentedTest {
    @Test
    fun originalWindowCandidateRunsThroughVerifiedMixedNpuPartition() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile = File(context.filesDir, "diagnostic-8s/htdemucs_6s.core.8s.fp16-safe.tflite")
        check(modelFile.isFile && modelFile.length() == 117_784_760L) {
            "Missing verified original-window candidate: $modelFile"
        }
        val samples = 343_980
        val spectrumFrames = 336
        val mix = direct(2 * samples)
        val spec = direct(4 * 2_048 * spectrumFrames)
        val frequency = direct(6 * 4 * 2_048 * spectrumFrames)
        val time = direct(6 * 2 * samples)
        val cache = File(context.codeCacheDir, "qnn-diagnostic-8s-safe-convs-v1").apply { mkdirs() }
        val options = QnnDelegate.Options().apply {
            setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
            setCacheDir(cache.absolutePath)
            setModelToken("bandbuddy-diagnostic-8s-a9fcc89e-v1")
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
            setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            setHtpOptimizationStrategy(QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_PREPARE)
            // The two longest final time-branch convolutions exceed this phone's
            // practical HTP graph-prepare limit at the original 7.8 s context.
            // The remaining heavy convolutions still execute as real NPU graphs.
            val htpNodes = setOf(
                586, 867,
                2639, 2645, 2777, 2784, 2923, 2929,
                3157, 3164, 3303,
            )
            setSkipNodeIds((0 until 3504).filterNot(htpNodes::contains).toIntArray())
            setProfiling(QnnDelegate.Options.ProfilingOptions.BASIC_PROFILING)
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
        }
        val delegate = QnnDelegate(options)
        val mappedModel = FileInputStream(modelFile).channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
        try {
            val started = SystemClock.elapsedRealtime()
            Interpreter(mappedModel, Interpreter.Options().apply {
                setUseXNNPACK(true)
                setNumThreads(4)
                addDelegate(delegate)
            }).use { interpreter ->
                assertTrue(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1, 2, samples)))
                assertTrue(interpreter.getInputTensor(1).shape().contentEquals(intArrayOf(1, 4, 2_048, spectrumFrames)))
                delegate.performanceVote()
                try {
                    interpreter.runForMultipleInputsOutputs(
                        arrayOf(mix, spec),
                        mutableMapOf<Int, Any>(0 to frequency, 1 to time),
                    )
                } finally {
                    delegate.performanceRelease()
                }
                Log.i(
                    "BandBuddy8s",
                    "prepare+inference=${SystemClock.elapsedRealtime() - started}ms, " +
                        "nativeInference=${interpreter.lastNativeInferenceDurationNanoseconds}ns",
                )
            }
            frequency.rewind()
            time.rewind()
            assertTrue(frequency.get().isFinite())
            assertTrue(time.get().isFinite())
        } finally {
            delegate.close()
        }
    }

    @Test
    fun exportOriginalWindowRealAudioForTorchComparison() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val samples = 343_980
        val spectrumFrames = 336
        val modelFile = File(context.filesDir, "diagnostic-8s/htdemucs_6s.core.8s.fp16-safe.tflite")
        check(modelFile.isFile && modelFile.length() == 117_784_760L)
        val inputRoot = File(context.filesDir, "diagnostic-8s-input")
        val exportRoot = File(
            requireNotNull(context.getExternalFilesDir(null)),
            "diagnostic-8s-real",
        ).apply {
            if (exists()) deleteRecursively()
            check(mkdirs())
        }
        val cache = File(context.codeCacheDir, "qnn-diagnostic-8s-safe-convs-v1").apply { mkdirs() }
        val options = QnnDelegate.Options().apply {
            setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
            setCacheDir(cache.absolutePath)
            setModelToken("bandbuddy-diagnostic-8s-a9fcc89e-v1")
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
            setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            setHtpOptimizationStrategy(QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_PREPARE)
            val htpNodes = setOf(
                586, 867,
                2639, 2645, 2777, 2784, 2923, 2929,
                3157, 3164, 3303,
            )
            setSkipNodeIds((0 until 3504).filterNot(htpNodes::contains).toIntArray())
            setProfiling(QnnDelegate.Options.ProfilingOptions.BASIC_PROFILING)
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
        }
        val delegate = QnnDelegate(options)
        val mappedModel = FileInputStream(modelFile).channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
        try {
            Interpreter(mappedModel, Interpreter.Options().apply {
                setUseXNNPACK(true)
                setNumThreads(4)
                addDelegate(delegate)
            }).use { interpreter ->
                listOf("f1", "wuyun").forEach { name ->
                    val input = File(inputRoot, "$name.interleaved.f32le")
                    check(input.isFile && input.length() == samples * 2L * Float.SIZE_BYTES) {
                        "Missing exact original-window calibration input: $input"
                    }
                    val interleaved = ByteBuffer.wrap(input.readBytes())
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .asFloatBuffer()
                    val mix = direct(2 * samples)
                    repeat(samples) { frame ->
                        mix.put(frame, interleaved.get(frame * 2))
                        mix.put(samples + frame, interleaved.get(frame * 2 + 1))
                    }
                    val spec = direct(4 * 2_048 * spectrumFrames)
                    val frequency = direct(6 * 4 * 2_048 * spectrumFrames)
                    val time = direct(6 * 2 * samples)
                    val output = direct(6 * 2 * samples)
                    mix.rewind()
                    assertTrue(NativeDemucsBridge.preprocess(mix, spec))
                    mix.rewind(); spec.rewind(); frequency.clear(); time.clear()
                    delegate.performanceVote()
                    try {
                        interpreter.runForMultipleInputsOutputs(
                            arrayOf(mix, spec),
                            mutableMapOf<Int, Any>(0 to frequency, 1 to time),
                        )
                    } finally {
                        delegate.performanceRelease()
                    }
                    frequency.rewind(); time.rewind(); output.clear()
                    assertTrue(NativeDemucsBridge.postprocess(frequency, time, output))
                    output.rewind()
                    var peak = 0f
                    while (output.hasRemaining()) {
                        val value = output.get()
                        assertTrue("$name output contains a non-finite sample", value.isFinite())
                        peak = maxOf(peak, kotlin.math.abs(value))
                    }
                    assertTrue("$name output has an implausible peak: $peak", peak < 10f)

                    val export = File(exportRoot, name).apply { check(mkdirs()) }
                    writeFloatBuffer(File(export, "mix.f32le"), mix)
                    writeFloatBuffer(File(export, "spec.f32le"), spec)
                    writeFloatBuffer(File(export, "frequency.f32le"), frequency)
                    writeFloatBuffer(File(export, "time.f32le"), time)
                    writeFloatBuffer(File(export, "final.f32le"), output)
                    Log.i("BandBuddy8s", "$name real-audio peak=$peak")
                }
            }
        } finally {
            delegate.close()
        }
    }

    @Test
    fun fourSecondCandidateRunsThroughVerifiedMixedNpuPartition() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile = File(context.filesDir, "diagnostic-4s/htdemucs_6s.core.4s.fp16-safe.tflite")
        check(modelFile.isFile && modelFile.length() == 114_651_784L) {
            "Missing verified four-second candidate: $modelFile"
        }
        val samples = 176_400
        val spectrumFrames = 173
        val mix = direct(2 * samples)
        val spec = direct(4 * 2_048 * spectrumFrames)
        val frequency = direct(6 * 4 * 2_048 * spectrumFrames)
        val time = direct(6 * 2 * samples)
        val cache = File(context.codeCacheDir, "qnn-diagnostic-4s-heavy-convs").apply { mkdirs() }
        val options = QnnDelegate.Options().apply {
            setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
            setCacheDir(cache.absolutePath)
            setModelToken("bandbuddy-diagnostic-4s-672001ed-v1")
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
            setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            setHtpOptimizationStrategy(QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_PREPARE)
            val htpNodes = setOf(
                585, 866,
                2638, 2644, 2776, 2783, 2922, 2928,
                3156, 3163, 3302, 3306, 3486,
            )
            setSkipNodeIds((0 until 3503).filterNot(htpNodes::contains).toIntArray())
            setProfiling(QnnDelegate.Options.ProfilingOptions.BASIC_PROFILING)
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
        }
        val delegate = QnnDelegate(options)
        val mappedModel = FileInputStream(modelFile).channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
        try {
            val started = SystemClock.elapsedRealtime()
            Interpreter(mappedModel, Interpreter.Options().apply {
                setUseXNNPACK(true)
                setNumThreads(4)
                addDelegate(delegate)
            }).use { interpreter ->
                assertTrue(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1, 2, samples)))
                assertTrue(interpreter.getInputTensor(1).shape().contentEquals(intArrayOf(1, 4, 2_048, spectrumFrames)))
                delegate.performanceVote()
                try {
                    interpreter.runForMultipleInputsOutputs(
                        arrayOf(mix, spec),
                        mutableMapOf<Int, Any>(0 to frequency, 1 to time),
                    )
                } finally {
                    delegate.performanceRelease()
                }
                Log.i(
                    "BandBuddy4s",
                    "prepare+inference=${SystemClock.elapsedRealtime() - started}ms, " +
                        "nativeInference=${interpreter.lastNativeInferenceDurationNanoseconds}ns",
                )
            }
            frequency.rewind()
            time.rewind()
            assertTrue(frequency.get().isFinite())
            assertTrue(time.get().isFinite())
        } finally {
            delegate.close()
        }
    }

    @Test
    fun sixSecondCandidateRunsThroughVerifiedMixedNpuPartition() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile = File(context.filesDir, "diagnostic-6s/htdemucs_6s.core.6s.fp16-safe.tflite")
        check(modelFile.isFile && modelFile.length() == 116_303_560L) {
            "Missing verified six-second candidate: $modelFile"
        }
        val samples = 264_600
        val spectrumFrames = 259
        val mix = direct(2 * samples)
        val spec = direct(4 * 2_048 * spectrumFrames)
        val frequency = direct(6 * 4 * 2_048 * spectrumFrames)
        val time = direct(6 * 2 * samples)
        val cache = File(context.codeCacheDir, "qnn-diagnostic-6s-safe-convs").apply { mkdirs() }
        val options = QnnDelegate.Options().apply {
            setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
            setCacheDir(cache.absolutePath)
            setModelToken("bandbuddy-diagnostic-6s-2bd39fbd-v2")
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
            setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            setHtpOptimizationStrategy(QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_PREPARE)
            val htpNodes = setOf(
                586, 867,
                2639, 2645, 2777, 2784, 2923, 2929,
                3157, 3164, 3303, 3307,
            )
            setSkipNodeIds((0 until 3504).filterNot(htpNodes::contains).toIntArray())
            setProfiling(QnnDelegate.Options.ProfilingOptions.BASIC_PROFILING)
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
        }
        val delegate = QnnDelegate(options)
        val mappedModel = FileInputStream(modelFile).channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
        try {
            val started = SystemClock.elapsedRealtime()
            Interpreter(mappedModel, Interpreter.Options().apply {
                setUseXNNPACK(true)
                setNumThreads(4)
                addDelegate(delegate)
            }).use { interpreter ->
                assertTrue(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1, 2, samples)))
                assertTrue(interpreter.getInputTensor(1).shape().contentEquals(intArrayOf(1, 4, 2_048, spectrumFrames)))
                delegate.performanceVote()
                try {
                    interpreter.runForMultipleInputsOutputs(
                        arrayOf(mix, spec),
                        mutableMapOf<Int, Any>(0 to frequency, 1 to time),
                    )
                } finally {
                    delegate.performanceRelease()
                }
                Log.i(
                    "BandBuddy6s",
                    "prepare+inference=${SystemClock.elapsedRealtime() - started}ms, " +
                        "nativeInference=${interpreter.lastNativeInferenceDurationNanoseconds}ns",
                )
            }
            frequency.rewind()
            time.rewind()
            assertTrue(frequency.get().isFinite())
            assertTrue(time.get().isFinite())
        } finally {
            delegate.close()
        }
    }

    @Test
    fun exportDeterministicQnnMixedPrecisionForTorchComparison() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val samples = DemucsWindowSeparator.SAMPLES
        val mix = direct(2 * samples)
        repeat(samples) { index ->
            val position = index.toDouble() / 44_100.0
            mix.put(index, (.37 * sin(2.0 * Math.PI * 220.0 * position) +
                .19 * sin(2.0 * Math.PI * 880.0 * position)).toFloat())
            mix.put(samples + index, (.33 * sin(2.0 * Math.PI * 329.63 * position) +
                .17 * sin(2.0 * Math.PI * 1760.0 * position)).toFloat())
        }
        val spec = direct(4 * 2_048 * DemucsWindowSeparator.SPECTRUM_FRAMES)
        val frequency = direct(6 * 4 * 2_048 * DemucsWindowSeparator.SPECTRUM_FRAMES)
        val time = direct(6 * 2 * samples)
        val output = direct(6 * 2 * samples)
        mix.rewind()
        assertTrue(NativeDemucsBridge.preprocess(mix, spec))

        val cache = File(context.codeCacheDir, "qnn-diagnostic-heavy-convs-htp").apply { mkdirs() }
        val options = QnnDelegate.Options().apply {
            setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
            setCacheDir(cache.absolutePath)
            setModelToken("bandbuddy-diagnostic-heavy-convs-htp-a9fcc89e-v1")
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
            setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            setHtpOptimizationStrategy(QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_PREPARE)
            // Preserve all numerically sensitive normalization/attention work
            // in LiteRT FP32. The selected 11 convolutions account for most of
            // the model's MACs and run as real QNN HTP partitions.
            val htpNodes = setOf(
                586, 867,
                2639, 2645, 2777, 2784, 2923, 2929,
                3157, 3164, 3303,
            )
            setSkipNodeIds((0 until 3504).filterNot(htpNodes::contains).toIntArray())
            setProfiling(QnnDelegate.Options.ProfilingOptions.BASIC_PROFILING)
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
        }
        val delegate = QnnDelegate(options)
        val modelFile = ModelStore(context).currentModelFile()
        check(ModelStore(context).isCurrentReady()) { "Download the model in BandBuddy settings before this test" }
        val mappedModel = FileInputStream(modelFile).channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0L, modelFile.length())
        }
        try {
            Interpreter(mappedModel, Interpreter.Options().apply {
                setUseXNNPACK(true)
                setNumThreads(4)
                addDelegate(delegate)
            }).use { interpreter ->
                mix.rewind(); spec.rewind(); frequency.clear(); time.clear()
                delegate.performanceVote()
                try {
                    interpreter.runForMultipleInputsOutputs(
                        arrayOf(mix, spec),
                        mutableMapOf<Int, Any>(0 to frequency, 1 to time)
                    )
                } finally {
                    delegate.performanceRelease()
                }
            }
        } finally {
            delegate.close()
        }
        frequency.rewind(); time.rewind(); output.clear()
        assertTrue(NativeDemucsBridge.postprocess(frequency, time, output))

        val export = File(requireNotNull(context.getExternalFilesDir(null)), "diagnostic-qnn-mixed-precision").apply {
            if (exists()) deleteRecursively()
            check(mkdirs())
        }
        writeFloatBuffer(File(export, "mix.f32le"), mix)
        writeFloatBuffer(File(export, "spec.f32le"), spec)
        writeFloatBuffer(File(export, "frequency.f32le"), frequency)
        writeFloatBuffer(File(export, "time.f32le"), time)
        writeFloatBuffer(File(export, "final.f32le"), output)
    }

    @Test
    fun exportDeterministicQnnWindowForTorchComparison() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val samples = DemucsWindowSeparator.SAMPLES
        val mix = direct(2 * samples)
        repeat(samples) { index ->
            val position = index.toDouble() / 44_100.0
            mix.put(index, (.37 * sin(2.0 * Math.PI * 220.0 * position) +
                .19 * sin(2.0 * Math.PI * 880.0 * position)).toFloat())
            mix.put(samples + index, (.33 * sin(2.0 * Math.PI * 329.63 * position) +
                .17 * sin(2.0 * Math.PI * 1760.0 * position)).toFloat())
        }
        val spec = direct(4 * 2_048 * DemucsWindowSeparator.SPECTRUM_FRAMES)
        val frequency = direct(6 * 4 * 2_048 * DemucsWindowSeparator.SPECTRUM_FRAMES)
        val time = direct(6 * 2 * samples)
        val output = direct(6 * 2 * samples)
        mix.rewind()
        assertTrue(NativeDemucsBridge.preprocess(mix, spec))
        mix.rewind(); spec.rewind(); frequency.clear(); time.clear()
        ModelRuntime(context).openSession().use { session ->
            session.run(arrayOf(mix, spec), mutableMapOf(0 to frequency, 1 to time))
        }
        frequency.rewind(); time.rewind(); output.clear()
        assertTrue(NativeDemucsBridge.postprocess(frequency, time, output))

        val export = File(requireNotNull(context.getExternalFilesDir(null)), "diagnostic-qnn").apply {
            if (exists()) deleteRecursively()
            check(mkdirs())
        }
        writeFloatBuffer(File(export, "mix.f32le"), mix)
        writeFloatBuffer(File(export, "spec.f32le"), spec)
        writeFloatBuffer(File(export, "frequency.f32le"), frequency)
        writeFloatBuffer(File(export, "time.f32le"), time)
        writeFloatBuffer(File(export, "final.f32le"), output)
    }

    @Test
    fun exportRealCalibrationWindowsForTorchComparison() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val externalRoot = requireNotNull(context.getExternalFilesDir(null))
        val inputRoot = File(externalRoot, "diagnostic-input")
        val exportRoot = File(externalRoot, "diagnostic-qnn-real").apply {
            if (exists()) deleteRecursively()
            check(mkdirs())
        }
        val samples = DemucsWindowSeparator.SAMPLES

        ModelRuntime(context).openSession().use { session ->
            listOf("f1", "wuyun").forEach { name ->
                val input = File(inputRoot, "$name.interleaved.f32le")
                check(input.isFile && input.length() == samples * 2L * Float.SIZE_BYTES) {
                    "Missing exact stereo calibration window: $input"
                }
                val interleaved = ByteBuffer.wrap(input.readBytes())
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                val mix = direct(2 * samples)
                repeat(samples) { frame ->
                    mix.put(frame, interleaved.get(frame * 2))
                    mix.put(samples + frame, interleaved.get(frame * 2 + 1))
                }
                val spec = direct(4 * 2_048 * DemucsWindowSeparator.SPECTRUM_FRAMES)
                val frequency = direct(6 * 4 * 2_048 * DemucsWindowSeparator.SPECTRUM_FRAMES)
                val time = direct(6 * 2 * samples)
                val output = direct(6 * 2 * samples)
                mix.rewind()
                assertTrue(NativeDemucsBridge.preprocess(mix, spec))
                mix.rewind(); spec.rewind(); frequency.clear(); time.clear()
                session.run(arrayOf(mix, spec), mutableMapOf(0 to frequency, 1 to time))
                frequency.rewind(); time.rewind(); output.clear()
                assertTrue(NativeDemucsBridge.postprocess(frequency, time, output))

                val export = File(exportRoot, name).apply { check(mkdirs()) }
                writeFloatBuffer(File(export, "mix.f32le"), mix)
                writeFloatBuffer(File(export, "spec.f32le"), spec)
                writeFloatBuffer(File(export, "frequency.f32le"), frequency)
                writeFloatBuffer(File(export, "time.f32le"), time)
                writeFloatBuffer(File(export, "final.f32le"), output)
            }
        }
    }

    @Test
    fun separatesRealF1ClipEndToEnd() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val source = File(context.filesDir, "diagnostic-e2e/song/source/f1-8s.wav")
        check(source.isFile) { "Missing F1 end-to-end calibration clip: $source" }
        val song = SongRecord(
            id = "diagnostic-f1-8s",
            title = "F1 · 8s validation",
            artist = "local calibration",
            durationMs = 8_000,
            sourcePath = source.absolutePath,
            stems = emptyList(),
            status = SongStatus.QUEUED,
            jobStage = "质量校验",
            progress = 0,
        )
        val stems = LongSongSeparator(context).separate(
            song,
            onProgress = { android.util.Log.i("BandBuddyE2E", "${it.percent}% ${it.stage}") },
        ).stems
        assertEquals(6, stems.size)
        val export = File(
            requireNotNull(context.getExternalFilesDir(null)),
            "diagnostic-e2e-output",
        ).apply {
            if (exists()) deleteRecursively()
            check(mkdirs())
        }
        stems.forEach { stem ->
            val file = File(stem.path)
            assertTrue("Missing ${stem.type} end-to-end output", file.isFile && file.length() > 1_024L)
            assertEquals("m4a", file.extension)
            file.copyTo(File(export, "${stem.type.name.lowercase()}.m4a"), overwrite = true)
        }
    }

    @Test
    fun stopCalibrationAndExportNoisyF1() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val repository = com.lonelyme.bandbuddy.data.LocalSongRepository(context)
        val songs = repository.load().filter { it.title == "F1" || it.title == "乌云典当记" }
        songs.forEach { song ->
            SeparationWorker.cancel(context, song.id).result.get(10, TimeUnit.SECONDS)
        }

        val f1 = songs.firstOrNull { it.title == "F1" } ?: error("F1 is missing")
        val export = File(requireNotNull(context.getExternalFilesDir(null)), "diagnostic-f1").apply {
            if (exists()) deleteRecursively()
            check(mkdirs())
        }
        f1.stems.forEach { stem ->
            File(stem.path).copyTo(File(export, "${stem.type.name.lowercase()}.m4a"), overwrite = true)
        }

        songs.forEach { song ->
            repository.updateSong(song.id) {
                it.copy(
                    stems = emptyList(),
                    status = SongStatus.FAILED,
                    jobStage = "质量校验未通过",
                    error = "已停止：分轨输出异常，等待修复"
                )
            }
        }
        repository.cleanupOrphans()
        assertEquals(6, export.listFiles().orEmpty().size)
    }

    @Test
    fun productionWorkerSeparatesF1AndWuyunOnHtp() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val repository = com.lonelyme.bandbuddy.data.LocalSongRepository(context)
        listOf("F1", "乌云典当记").forEach { title ->
            val original = repository.load().firstOrNull { it.title == title }
                ?: error("Calibration song is missing: $title")
            repository.updateSong(original.id) {
                it.copy(
                    stems = emptyList(),
                    status = SongStatus.QUEUED,
                    jobStage = "等待本地分轨",
                    progress = 0,
                    error = null
                )
            }
            SeparationWorker.enqueue(context, original.id, replace = true)

            val deadline = SystemClock.elapsedRealtime() + 20 * 60_000L
            var lastSnapshot = ""
            var completed: SongRecord? = null
            while (SystemClock.elapsedRealtime() < deadline) {
                val current = repository.load().first { it.id == original.id }
                val snapshot = "${current.status} ${current.progress}% ${current.jobStage}"
                if (snapshot != lastSnapshot) {
                    lastSnapshot = snapshot
                    Log.i("BandBuddyLongRun", "$title · $snapshot")
                    println("BAND_BUDDY_LONG_RUN=$title · $snapshot")
                }
                when (current.status) {
                    SongStatus.READY -> {
                        completed = current
                        break
                    }
                    SongStatus.FAILED -> throw AssertionError(
                        "$title separation failed at ${current.progress}%: ${current.error}"
                    )
                    else -> SystemClock.sleep(1_000L)
                }
            }

            val song = completed ?: throw AssertionError("$title separation timed out")
            assertEquals("$title did not produce exactly six stems", 6, song.stems.size)
            assertEquals(StemType.entries.toSet(), song.stems.map { it.type }.toSet())
            song.stems.forEach { stem ->
                val file = File(stem.path)
                assertTrue("$title ${stem.type} is not a non-empty M4A", file.isFile && file.extension == "m4a" && file.length() > 1_024L)
                MediaMetadataRetriever().use { retriever ->
                    retriever.setDataSource(file.absolutePath)
                    val duration = retriever
                        .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLongOrNull() ?: 0L
                    assertTrue(
                        "$title ${stem.type} duration $duration differs from ${song.durationMs}",
                        kotlin.math.abs(duration - song.durationMs) <= 1_500L
                    )
                }
            }
            MultiStemPlayer().use { player ->
                player.load(song.stems, positionMs = 0L, speed = 1f)
                player.setMix(StemType.entries.associateWith { TrackState() }, masterVolume = 0f)
                player.play()
                SystemClock.sleep(350L)
                assertTrue("$title six-stem transport did not start", player.isPlaying && player.positionMs > 0L)
            }
        }
    }

    @Test
    fun sixContinuousAacStemsEncodeAndLoadTogether() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = File(context.cacheDir, "aac-six-e2e").apply {
            deleteRecursively()
            mkdirs()
        }
        val writers = StemType.entries.map { type ->
            AacM4aWriter(File(root, "${type.name.lowercase()}.m4a"))
        }
        try {
            repeat(4) { block ->
                writers.forEachIndexed { stem, writer ->
                    writer.write(11_025) { frame, channel ->
                        val absoluteFrame = block * 11_025 + frame
                        val frequency = 110.0 + stem * 55.0 + channel * 3.0
                        (.12 * sin(2.0 * Math.PI * frequency * absoluteFrame / 44_100.0)).toFloat()
                    }
                }
            }
            writers.forEach(AacM4aWriter::close)
            val stems = StemType.entries.map { type ->
                val file = File(root, "${type.name.lowercase()}.m4a")
                assertTrue("Missing AAC stem ${type.label}", file.length() > 1_024L)
                MediaMetadataRetriever().use { retriever ->
                    retriever.setDataSource(file.absolutePath)
                    val duration = retriever
                        .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLongOrNull() ?: 0L
                    assertTrue("Unexpected AAC duration $duration ms", duration in 950L..1_100L)
                }
                StemRecord(type, file.absolutePath)
            }
            MultiStemPlayer().use { player ->
                player.load(stems, positionMs = 0, speed = 1f)
                player.setMix(StemType.entries.associateWith { TrackState() }, masterVolume = 0f)
                player.play()
                Thread.sleep(350)
                assertTrue(player.isPlaying)
                assertTrue(player.positionMs > 0L)
            }
        } finally {
            writers.forEach { runCatching { it.close() } }
            root.deleteRecursively()
        }
    }

    @Test
    fun nativeStftMatchesPyTorchAndReconstructsTheWindow() {
        val samples = DemucsWindowSeparator.SAMPLES
        val mix = direct(2 * samples)
        repeat(samples) { index ->
            val position = index.toDouble() / 44_100.0
            mix.put(index, (.37 * sin(2.0 * Math.PI * 220.0 * position) +
                .19 * sin(2.0 * Math.PI * 880.0 * position)).toFloat())
            mix.put(samples + index, (.33 * sin(2.0 * Math.PI * 329.63 * position) +
                .17 * sin(2.0 * Math.PI * 1760.0 * position)).toFloat())
        }
        val spec = direct(4 * 2_048 * 336)
        assertTrue("Native STFT rejected valid buffers", NativeDemucsBridge.preprocess(mix, spec))

        // Values generated by the pinned PyTorch HTDemucs._spec implementation.
        val reference = listOf(
            SpecValue(0, 0, 0, .35543293f),
            SpecValue(0, 20, 0, -2.7119725f), SpecValue(1, 20, 0, 2.0120382f),
            SpecValue(0, 20, 50, -.86138773f), SpecValue(1, 20, 50, -5.164964f),
            SpecValue(0, 82, 50, -1.7828593f), SpecValue(1, 82, 50, -2.292351f),
            SpecValue(2, 0, 0, .20544231f),
            SpecValue(2, 31, 0, 2.8050494f), SpecValue(3, 31, 0, -.06472957f),
            SpecValue(2, 31, 50, .80080664f), SpecValue(3, 31, 50, -4.72925f),
            SpecValue(2, 163, 86, 1.7259982f), SpecValue(3, 163, 86, -.4627036f)
        )
        reference.forEach { expected ->
            val offset = (expected.component * 2_048 + expected.bin) *
                DemucsWindowSeparator.SPECTRUM_FRAMES + expected.frame
            assertEquals("STFT mismatch at $expected", expected.value, spec.get(offset), .002f)
        }

        // Route the mixture spectrum through the inverse boundary as source 0.
        val frequency = direct(6 * 4 * 2_048 * 336)
        frequency.put(spec.duplicate().apply { rewind() }).rewind()
        val time = direct(6 * 2 * samples)
        val output = direct(6 * 2 * samples)
        assertTrue("Native iSTFT rejected valid buffers", NativeDemucsBridge.postprocess(frequency, time, output))

        var innerMaxError = 0f
        var totalError = 0.0
        repeat(2) { channel ->
            repeat(samples) { index ->
                val error = kotlin.math.abs(output.get(channel * samples + index) - mix.get(channel * samples + index))
                totalError += error
                if (index in 1_536 until samples - 1_536) innerMaxError = maxOf(innerMaxError, error)
            }
        }
        assertTrue("iSTFT inner max error was $innerMaxError", innerMaxError <= .00005f)
        // The strict inner max assertion guards every stable overlap-add sample.
        assertTrue("iSTFT mean error was ${totalError / (2 * samples)}", totalError / (2 * samples) <= .0012)
    }

    @Test
    fun zeroWindowRunsThroughNativeDspAndQnnHtpCore() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val input = FloatArray(2 * DemucsWindowSeparator.SAMPLES)
        val output = FloatArray(DemucsWindowSeparator.OUTPUT_FLOATS)
        DemucsWindowSeparator(ModelRuntime(context)).use { separator ->
            repeat(2) { run ->
                val started = System.nanoTime()
                separator.separateInto(input, output)
                val message = buildString {
                    append("BAND_BUDDY_HTP_RUN=").append(run + 1)
                    append(" totalMs=").append((System.nanoTime() - started) / 1_000_000.0)
                    append(" preprocessMs=").append(separator.lastPreprocessNanos() / 1_000_000.0)
                    append(" qnnWallMs=").append(separator.lastNeuralCoreWallNanos() / 1_000_000.0)
                    append(" liteRtNativeMs=").append(separator.lastNeuralCoreInferenceNanos()?.div(1_000_000.0))
                    append(" postprocessMs=").append(separator.lastPostprocessNanos() / 1_000_000.0)
                    append(" outputMin=").append(output.minOrNull())
                    append(" outputMax=").append(output.maxOrNull())
                    append(" outputRms=").append(kotlin.math.sqrt(output.sumOf { it.toDouble() * it } / output.size))
                }
                println(message)
                android.util.Log.i("BandBuddyPerf", message)
            }
        }

        assertEquals(6 * 2 * DemucsWindowSeparator.SAMPLES, output.size)
        assertTrue("Model output contains NaN/Infinity", output.all(Float::isFinite))
    }

    @Test
    fun short48kWavDecodesResamplesSeparatesAndWritesSixStems() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val root = File(context.cacheDir, "separation-e2e/song/source").apply { mkdirs() }.parentFile!!.parentFile!!
        val source = File(root, "source/test.wav")
        writeTestWav(source, sampleRate = 48_000, frames = 12_000)
        val song = SongRecord(
            id = "e2e", title = "E2E", artist = "test", durationMs = 250,
            sourcePath = source.absolutePath, stems = emptyList(), status = SongStatus.QUEUED,
            jobStage = "等待", progress = 0
        )
        try {
            val stems = LongSongSeparator(context).separate(song, onProgress = {}).stems
            assertEquals(6, stems.size)
            stems.forEach { stem ->
                val file = File(stem.path)
                assertTrue("Missing stem ${stem.type}", file.isFile)
                assertTrue("Empty M4A stem ${stem.type}", file.length() > 1_024L)
                assertEquals("m4a", file.extension)
                MediaMetadataRetriever().use { retriever ->
                    retriever.setDataSource(file.absolutePath)
                    val duration = retriever
                        .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLongOrNull() ?: 0L
                    assertTrue("Unexpected M4A duration $duration ms for ${stem.type}", duration in 200L..400L)
                }
            }
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun sixStemPlayerLoadsAndRunsRealTransport() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val root = File(context.cacheDir, "player-e2e").apply { mkdirs() }
        val stems = StemType.entries.map { type ->
            val file = File(root, type.fileName)
            writeTestWav(file, sampleRate = 44_100, frames = 44_100)
            StemRecord(type, file.absolutePath)
        }
        try {
            MultiStemPlayer().use { player ->
                player.load(stems, positionMs = 0, speed = 1f)
                player.setMix(StemType.entries.associateWith { TrackState() }, masterVolume = 0f)
                player.setSpeed(.8f)
                player.play()
                Thread.sleep(450)
                assertTrue(player.isPlaying)
                assertTrue(player.positionMs > 0)
                player.pause()
                assertTrue(!player.isPlaying)
            }
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun playbackServiceKeepsTheRealTransportOutsideCompose() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = File(context.cacheDir, "service-player-e2e").apply { mkdirs() }
        val stems = StemType.entries.map { type ->
            val file = File(root, type.fileName)
            writeTestWav(file, sampleRate = 44_100, frames = 44_100)
            StemRecord(type, file.absolutePath)
        }
        val song = SongRecord(
            id = "service-e2e", title = "Service E2E", artist = "test", durationMs = 1_000,
            sourcePath = null, stems = stems, status = SongStatus.READY, jobStage = "可练习", progress = 100
        )
        val repository = com.lonelyme.bandbuddy.data.LocalSongRepository(context)
        repository.addSong(song)
        val connected = CountDownLatch(1)
        var service: PlaybackService? = null
        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                service = (binder as PlaybackService.PlaybackBinder).service
                connected.countDown()
            }
            override fun onServiceDisconnected(name: ComponentName?) = Unit
        }
        try {
            assertTrue(context.bindService(Intent(context, PlaybackService::class.java), connection, Context.BIND_AUTO_CREATE))
            assertTrue("Playback service did not bind", connected.await(5, TimeUnit.SECONDS))
            val playback = requireNotNull(service)
            playback.load(song)
            playback.play()
            Thread.sleep(500)
            assertTrue(playback.isPlaying)
            assertTrue(playback.positionMs > 0)
            playback.pause()
            assertTrue(!playback.isPlaying)
        } finally {
            runCatching { context.unbindService(connection) }
            repository.delete(song)
            root.deleteRecursively()
        }
    }

    private fun writeTestWav(file: File, sampleRate: Int, frames: Int) {
        file.parentFile?.mkdirs()
        RandomAccessFile(file, "rw").use { output ->
            val dataBytes = frames * 4
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray()); header.putInt(36 + dataBytes); header.put("WAVE".toByteArray())
            header.put("fmt ".toByteArray()); header.putInt(16); header.putShort(1); header.putShort(2)
            header.putInt(sampleRate); header.putInt(sampleRate * 4); header.putShort(4); header.putShort(16)
            header.put("data".toByteArray()); header.putInt(dataBytes); output.write(header.array())
            val samples = ByteBuffer.allocate(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
            repeat(frames) { index ->
                val value = (sin(2.0 * Math.PI * 440.0 * index / sampleRate) * 8_000).toInt().toShort()
                samples.putShort(value); samples.putShort(value)
            }
            output.write(samples.array())
        }
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

    private data class SpecValue(
        val component: Int,
        val bin: Int,
        val frame: Int,
        val value: Float
    )

    private fun direct(floats: Int): FloatBuffer = ByteBuffer.allocateDirect(floats * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
}
