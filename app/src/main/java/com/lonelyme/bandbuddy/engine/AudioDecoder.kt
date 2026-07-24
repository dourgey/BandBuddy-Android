package com.lonelyme.bandbuddy.engine

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class DecodedPcm(val file: File, val frames: Long)

/** Decodes a managed source into little-endian float32, 44.1 kHz stereo PCM. */
class AudioDecoder {
    fun decode(
        source: File,
        destination: File,
        maxDurationSeconds: Int? = null
    ): DecodedPcm {
        destination.parentFile?.mkdirs()
        val extractor = MediaExtractor()
        extractor.setDataSource(source.absolutePath)
        val trackIndex = (0 until extractor.trackCount).firstOrNull {
            extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: error("文件中没有可解码的音频轨道")
        extractor.selectTrack(trackIndex)
        val inputFormat = extractor.getTrackFormat(trackIndex)
        val mime = inputFormat.getString(MediaFormat.KEY_MIME) ?: error("未知音频格式")
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(inputFormat, null, null, 0)
        codec.start()

        var sampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var encoding = AudioFormat.ENCODING_PCM_16BIT
        require(channels in 1..2) { "当前仅支持单声道或双声道音频" }
        var inputEnded = false
        var outputEnded = false
        var totalFrames = 0L
        val info = MediaCodec.BufferInfo()
        val decodedTemporary = File(destination.parentFile, "${destination.name}.decoded")

        try {
            FileOutputStream(decodedTemporary).use { output ->
                while (!outputEnded) {
                    if (!inputEnded) {
                        val inputIndex = codec.dequeueInputBuffer(10_000)
                        if (inputIndex >= 0) {
                            val buffer = codec.getInputBuffer(inputIndex) ?: error("解码输入缓冲区不可用")
                            val size = extractor.readSampleData(buffer, 0)
                            if (size < 0) {
                                codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputEnded = true
                            } else {
                                codec.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }
                    when (val outputIndex = codec.dequeueOutputBuffer(info, 10_000)) {
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val format = codec.outputFormat
                            sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            encoding = runCatching { format.getInteger(MediaFormat.KEY_PCM_ENCODING) }.getOrDefault(AudioFormat.ENCODING_PCM_16BIT)
                            require(channels in 1..2) { "解码输出声道数不受支持" }
                        }
                        else -> if (outputIndex >= 0) {
                            val buffer = codec.getOutputBuffer(outputIndex) ?: error("解码输出缓冲区不可用")
                            buffer.order(ByteOrder.LITTLE_ENDIAN)
                            buffer.position(info.offset); buffer.limit(info.offset + info.size)
                            val bytesPerSample = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) 4 else 2
                            val frameCount = info.size / (bytesPerSample * channels)
                            val frameLimit = maxDurationSeconds
                                ?.coerceAtLeast(1)
                                ?.toLong()
                                ?.times(sampleRate)
                                ?: Long.MAX_VALUE
                            val framesToWrite = minOf(frameCount.toLong(), frameLimit - totalFrames)
                                .coerceAtLeast(0)
                                .toInt()
                            val converted = ByteBuffer.allocate(framesToWrite * 2 * 4).order(ByteOrder.LITTLE_ENDIAN)
                            repeat(framesToWrite) {
                                val left = readSample(buffer, encoding)
                                val right = if (channels == 1) left else readSample(buffer, encoding)
                                converted.putFloat(left); converted.putFloat(right)
                            }
                            output.write(converted.array(), 0, converted.position())
                            totalFrames += framesToWrite
                            outputEnded = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0 ||
                                totalFrames >= frameLimit
                            codec.releaseOutputBuffer(outputIndex, false)
                        }
                    }
                }
            }
            if (sampleRate == 44_100) {
                if (destination.exists()) destination.delete()
                check(decodedTemporary.renameTo(destination)) { "无法提交解码后的音频" }
                return DecodedPcm(destination, totalFrames)
            }
            val resampledFrames = resample(decodedTemporary, destination, totalFrames, sampleRate)
            decodedTemporary.delete()
            return DecodedPcm(destination, resampledFrames)
        } finally {
            runCatching { codec.stop() }; codec.release(); extractor.release()
        }
    }

    private fun readSample(buffer: ByteBuffer, encoding: Int): Float = when (encoding) {
        AudioFormat.ENCODING_PCM_FLOAT -> buffer.float.coerceIn(-1f, 1f)
        AudioFormat.ENCODING_PCM_16BIT -> buffer.short / 32768f
        else -> error("不支持的 PCM 编码：$encoding")
    }

    private fun resample(source: File, destination: File, sourceFrames: Long, sourceRate: Int): Long {
        val outputFrames = (sourceFrames * 44_100L / sourceRate).coerceAtLeast(1)
        val temporary = File(destination.parentFile, "${destination.name}.part")
        RandomAccessFile(source, "r").use { input ->
            FileOutputStream(temporary).use { output ->
                val reader = StereoFrameReader(input, sourceFrames)
                val batch = ByteBuffer.allocate(8_192 * 8).order(ByteOrder.LITTLE_ENDIAN)
                var outputFrame = 0L
                while (outputFrame < outputFrames) {
                    batch.clear()
                    val count = minOf(8_192L, outputFrames - outputFrame).toInt()
                    repeat(count) { local ->
                        val sourcePosition = (outputFrame + local) * sourceRate.toDouble() / 44_100.0
                        val lower = sourcePosition.toLong().coerceAtMost(sourceFrames - 1)
                        val upper = (lower + 1).coerceAtMost(sourceFrames - 1)
                        val fraction = (sourcePosition - lower).toFloat()
                        val left0 = reader.sample(lower, 0); val left1 = reader.sample(upper, 0)
                        val right0 = reader.sample(lower, 1); val right1 = reader.sample(upper, 1)
                        batch.putFloat(left0 + (left1 - left0) * fraction)
                        batch.putFloat(right0 + (right1 - right0) * fraction)
                    }
                    output.write(batch.array(), 0, batch.position())
                    outputFrame += count
                }
            }
        }
        if (destination.exists()) destination.delete()
        check(temporary.renameTo(destination)) { "无法提交重采样音频" }
        return outputFrames
    }
}

private class StereoFrameReader(private val file: RandomAccessFile, private val frames: Long) {
    private val blockFrames = 8_192
    private var blockStart = -1L
    private var blockCount = 0
    private val buffer = ByteArray(blockFrames * 8)
    private var floats = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN)

    fun sample(index: Long, channel: Int): Float {
        val safe = index.coerceIn(0, frames - 1)
        if (blockStart < 0 || safe < blockStart || safe >= blockStart + blockCount) load(safe)
        val offset = ((safe - blockStart) * 8 + channel * 4).toInt()
        return floats.getFloat(offset)
    }

    private fun load(index: Long) {
        blockStart = (index / blockFrames) * blockFrames
        blockCount = minOf(blockFrames.toLong(), frames - blockStart).toInt()
        file.seek(blockStart * 8)
        file.readFully(buffer, 0, blockCount * 8)
        floats = ByteBuffer.wrap(buffer, 0, blockCount * 8).order(ByteOrder.LITTLE_ENDIAN)
    }
}
