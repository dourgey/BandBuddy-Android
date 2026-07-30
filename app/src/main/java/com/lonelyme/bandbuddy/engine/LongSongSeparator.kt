package com.lonelyme.bandbuddy.engine

import android.content.Context
import com.lonelyme.bandbuddy.data.SongRecord
import com.lonelyme.bandbuddy.data.StemRecord
import com.lonelyme.bandbuddy.data.StemType
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

class LongSongSeparator(private val context: Context) {
    companion object {
        private const val WINDOW = DemucsWindowSeparator.SAMPLES
        private const val STRIDE = WINDOW * 3 / 4
        private val sourceTypes = listOf(StemType.DRUMS, StemType.BASS, StemType.OTHER, StemType.VOCALS, StemType.GUITAR, StemType.PIANO)
    }

    data class Progress(val stage: String, val percent: Int)
    data class SeparationOutput(
        val stems: List<StemRecord>,
        val beatGrid: BeatGridAnalysis?,
        val beatGridAnalysisVersion: Int
    )

    fun separate(
        song: SongRecord,
        onProgress: (Progress) -> Unit,
        isCancelled: () -> Boolean = { false }
    ): SeparationOutput {
        val source = song.sourcePath?.let(::File) ?: error("歌曲没有原始音频")
        val songDirectory = source.parentFile?.parentFile ?: error("歌曲目录无效")
        val normalized = File(songDirectory, "normalized/mix.f32le")
        onProgress(Progress("准备音频", 2))
        val decoded = AudioDecoder().decode(source, normalized)
        check(decoded.frames > 0) { "解码结果为空" }
        onProgress(Progress("生成波形", 4))
        WaveformPeaks.generateFromFloatPcm(decoded.file, decoded.frames, File(songDirectory, "peaks/mix.f32le"))
        if (isCancelled()) error("CANCELLED")
        onProgress(Progress("分析歌曲节拍", 5))
        val mixBeatGrid = try {
            BeatGridDetector.detectFromStereoFloatPcm(decoded.file, decoded.frames, isCancelled = isCancelled)
        } catch (error: Throwable) {
            if (error.message == "CANCELLED") throw error
            null
        }
        if (isCancelled()) error("CANCELLED")

        val temporaryDirectory = File(songDirectory, "stems.part").apply {
            if (exists()) deleteRecursively()
            check(mkdirs()) { "无法创建六轨临时目录" }
        }
        val writers = mutableListOf<AacM4aWriter>()
        try {
            sourceTypes.forEach { type ->
                writers += AacM4aWriter(File(temporaryDirectory, outputFileName(type)))
            }
        } catch (error: Throwable) {
            writers.forEach { runCatching { it.close() } }
            temporaryDirectory.deleteRecursively()
            throw error
        }
        val accumulator = FloatArray(12 * WINDOW)
        val accumulatedWeight = FloatArray(WINDOW)
        val weights = triangularWeights()
        val mix = FloatArray(2 * WINDOW)
        val separated = FloatArray(DemucsWindowSeparator.OUTPUT_FLOATS)
        val encodedWindow = ByteArray(2 * WINDOW * Float.SIZE_BYTES)
        val windowCount = ((decoded.frames + STRIDE - 1) / STRIDE).toInt()
        var offset = 0L
        var windowIndex = 0
        val runtime = ModelRuntime(context)

        try {
            RandomAccessFile(decoded.file, "r").use { pcm ->
                val preferredBackend = runtime.preferredBackend()
                onProgress(Progress(
                    when {
                        preferredBackend == ModelRuntime.Backend.CPU ->
                            "准备 CPU FP32 兼容模式 · 速度较慢"
                        runtime.hasCompiledCache() ->
                            "加载 NPU / CPU 混合模型"
                        else ->
                            "首次准备 NPU · 约 1 分钟"
                    },
                    6
                ))
                DemucsWindowSeparator(runtime).use { separator ->
                    val separationStage = when (separator.backend) {
                        ModelRuntime.Backend.QUALCOMM_HTP -> "正在本地分轨"
                        ModelRuntime.Backend.CPU -> "CPU 兼容模式分轨 · 速度较慢"
                    }
                    while (offset < decoded.frames) {
                        if (isCancelled()) error("CANCELLED")
                        val actual = min(WINDOW.toLong(), decoded.frames - offset).toInt()
                        val padLeft = (WINDOW - actual) / 2
                        readWindowInto(pcm, offset, actual, padLeft, encodedWindow, mix)
                        onProgress(Progress(
                            separationStage,
                            8 + (windowIndex * 85 / windowCount)
                        ))
                        separator.separateInto(mix, separated)
                        for (sample in 0 until actual) {
                            val modelSample = padLeft + sample
                            val weight = weights[modelSample]
                            accumulatedWeight[sample] += weight
                            for (channel in 0 until 12) {
                                accumulator[channel * WINDOW + sample] += separated[channel * WINDOW + modelSample] * weight
                            }
                        }
                        val hasNext = offset + STRIDE < decoded.frames
                        val flush = if (hasNext) min(STRIDE.toLong(), decoded.frames - offset).toInt() else actual
                        writeAndShift(writers, accumulator, accumulatedWeight, flush)
                        offset += flush
                        windowIndex++
                    }
                }
            }
            onProgress(Progress("封装六轨音频", 95))
            writers.forEach(AacM4aWriter::close)
            val finalDirectory = File(songDirectory, "stems")
            if (finalDirectory.exists()) finalDirectory.deleteRecursively()
            check(temporaryDirectory.renameTo(finalDirectory)) { "无法提交六轨结果" }
            val stems = sourceTypes.map { StemRecord(it, File(finalDirectory, outputFileName(it)).absolutePath) }
            normalized.delete()
            onProgress(Progress("综合鼓轨与贝斯检测节拍", 97))
            val jointBeatGrid = try {
                detectRhythmStems(stems, songDirectory, isCancelled)
            } catch (error: Throwable) {
                if (error.message == "CANCELLED") throw error
                null
            }
            onProgress(Progress("完成", 100))
            return SeparationOutput(
                stems = stems,
                beatGrid = jointBeatGrid ?: mixBeatGrid,
                beatGridAnalysisVersion = when {
                    jointBeatGrid != null -> BeatGridDetector.ANALYSIS_VERSION
                    mixBeatGrid != null -> 1
                    else -> 0
                }
            )
        } catch (error: Throwable) {
            writers.forEach { runCatching { it.close() } }
            temporaryDirectory.deleteRecursively()
            throw error
        }
    }

    private fun detectRhythmStems(
        stems: List<StemRecord>,
        songDirectory: File,
        isCancelled: () -> Boolean
    ): BeatGridAnalysis? {
        val drumsSource = stems.firstOrNull { it.type == StemType.DRUMS }?.path?.let(::File) ?: return null
        val bassSource = stems.firstOrNull { it.type == StemType.BASS }?.path?.let(::File) ?: return null
        val analysisDirectory = File(songDirectory, "normalized").apply { mkdirs() }
        val drumsFile = File(analysisDirectory, "beat-drums.f32le")
        val bassFile = File(analysisDirectory, "beat-bass.f32le")
        fun clean(file: File) {
            file.delete()
            File(file.parentFile, "${file.name}.decoded").delete()
            File(file.parentFile, "${file.name}.part").delete()
        }
        return try {
            val decoder = AudioDecoder()
            val drums = decoder.decode(drumsSource, drumsFile, maxDurationSeconds = 180)
            if (isCancelled()) error("CANCELLED")
            val bass = decoder.decode(bassSource, bassFile, maxDurationSeconds = 180)
            if (isCancelled()) error("CANCELLED")
            BeatGridDetector.detectFromDrumsAndBassPcm(
                drumsFile = drums.file,
                drumsFrames = drums.frames,
                bassFile = bass.file,
                bassFrames = bass.frames,
                isCancelled = isCancelled
            )
        } finally {
            clean(drumsFile)
            clean(bassFile)
        }
    }

    private fun readWindowInto(
        file: RandomAccessFile,
        frameOffset: Long,
        frames: Int,
        padLeft: Int,
        bytes: ByteArray,
        output: FloatArray
    ) {
        java.util.Arrays.fill(output, 0f)
        val byteCount = frames * 2 * Float.SIZE_BYTES
        file.seek(frameOffset * 2 * 4)
        file.readFully(bytes, 0, byteCount)
        val source = ByteBuffer.wrap(bytes, 0, byteCount).order(ByteOrder.LITTLE_ENDIAN)
        repeat(frames) { index ->
            output[padLeft + index] = source.float
            output[WINDOW + padLeft + index] = source.float
        }
    }

    private fun writeAndShift(writers: List<AacM4aWriter>, accumulator: FloatArray, weights: FloatArray, frames: Int) {
        writers.forEachIndexed { source, writer ->
            writer.write(frames) { frame, channel ->
                val plane = source * 2 + channel
                val weight = weights[frame]
                if (weight > 1e-8f) accumulator[plane * WINDOW + frame] / weight else 0f
            }
        }
        val remaining = WINDOW - frames
        for (plane in 0 until 12) {
            val base = plane * WINDOW
            accumulator.copyInto(accumulator, base, base + frames, base + WINDOW)
            java.util.Arrays.fill(accumulator, base + remaining, base + WINDOW, 0f)
        }
        weights.copyInto(weights, 0, frames, WINDOW)
        java.util.Arrays.fill(weights, remaining, WINDOW, 0f)
    }

    private fun triangularWeights() = FloatArray(WINDOW) { index ->
        val half = WINDOW / 2
        (if (index < half) index + 1 else WINDOW - index).toFloat() / half
    }

    private fun outputFileName(type: StemType): String = "${type.name.lowercase()}.m4a"
}
