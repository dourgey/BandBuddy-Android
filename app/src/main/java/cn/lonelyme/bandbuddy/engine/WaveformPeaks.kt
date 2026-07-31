package cn.lonelyme.bandbuddy.engine

import cn.lonelyme.bandbuddy.data.SongRecord
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

object WaveformPeaks {
    private const val COUNT = 720

    fun generateFromFloatPcm(source: File, frames: Long, destination: File) {
        if (frames <= 0) return
        write(calculate(source, frames), destination)
    }

    fun generateFromAudioFiles(sources: List<File>, destination: File) {
        require(sources.size == 6) { "需要完整的六个分轨" }
        val aggregate = FloatArray(COUNT)
        var expectedFrames = -1L
        sources.forEachIndexed { index, source ->
            val temporary = File(destination.parentFile, "decode-$index.f32le")
            val decoded = AudioDecoder().decode(source, temporary)
            try {
                if (expectedFrames < 0) expectedFrames = decoded.frames
                require(kotlin.math.abs(decoded.frames - expectedFrames) <= 4_410) { "六个分轨的时长不一致" }
                val peaks = calculate(decoded.file, decoded.frames)
                peaks.indices.forEach { bucket -> aggregate[bucket] = maxOf(aggregate[bucket], peaks[bucket]) }
            } finally {
                temporary.delete()
                File(temporary.parentFile, "${temporary.name}.decoded").delete()
            }
        }
        write(aggregate, destination)
    }

    private fun calculate(source: File, frames: Long): FloatArray {
        val peaks = FloatArray(COUNT)
        val bytes = ByteArray(8 * 8_192)
        var frame = 0L
        RandomAccessFile(source, "r").use { input ->
            while (frame < frames) {
                val wanted = minOf(8_192L, frames - frame).toInt()
                input.readFully(bytes, 0, wanted * 8)
                val values = ByteBuffer.wrap(bytes, 0, wanted * 8).order(ByteOrder.LITTLE_ENDIAN)
                repeat(wanted) {
                    val amplitude = maxOf(abs(values.float), abs(values.float))
                    val bucket = (((frame + it) * COUNT) / frames).toInt().coerceIn(0, COUNT - 1)
                    peaks[bucket] = maxOf(peaks[bucket], amplitude)
                }
                frame += wanted
            }
        }
        return peaks
    }

    private fun write(peaks: FloatArray, destination: File) {
        destination.parentFile?.mkdirs()
        val temporary = File(destination.parentFile, "${destination.name}.part")
        val output = ByteBuffer.allocate(COUNT * 4).order(ByteOrder.LITTLE_ENDIAN)
        peaks.forEach { output.putFloat(it.coerceIn(0f, 1f)) }
        temporary.writeBytes(output.array())
        if (destination.exists()) destination.delete()
        check(temporary.renameTo(destination)) { "无法提交波形索引" }
    }

    fun load(song: SongRecord): FloatArray {
        val songRoot = song.sourcePath?.let { File(it).parentFile?.parentFile }
            ?: song.stems.firstOrNull()?.let { File(it.path).parentFile?.parentFile }
            ?: return FloatArray(0)
        val file = File(songRoot, "peaks/mix.f32le")
        if (!file.isFile || file.length() % 4L != 0L) return FloatArray(0)
        val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(buffer.remaining() / 4) { buffer.float }
    }
}
