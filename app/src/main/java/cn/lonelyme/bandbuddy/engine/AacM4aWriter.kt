package cn.lonelyme.bandbuddy.engine

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.Closeable
import java.io.File
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Streams stereo float PCM into one gapless AAC-LC track inside an M4A file.
 *
 * A single encoder is kept alive for the entire stem. This avoids the encoder
 * priming gaps and timestamp drift that would be introduced by concatenating
 * one compressed file per inference window.
 */
internal class AacM4aWriter(
    private val file: File,
    private val sampleRate: Int = 44_100,
    private val channelCount: Int = 2,
    bitRate: Int = 160_000
) : Closeable {
    companion object {
        private const val CODEC_TIMEOUT_US = 10_000L
        private const val MAX_END_OF_STREAM_POLLS = 1_000

        // The platform AAC-LC encoder used on the target device emits two
        // priming frames. Keep those packets (AAC transform overlap depends on
        // them), but place them before t=0 so the MP4 edit timeline starts at
        // the first submitted PCM sample.
        private const val AAC_ENCODER_PRIMING_FRAMES = 2_048L
    }

    private val codec: MediaCodec
    private val muxer: MediaMuxer
    private val bufferInfo = MediaCodec.BufferInfo()
    private var trackIndex = -1
    private var muxerStarted = false
    private var submittedFrames = 0L
    private var closed = false

    init {
        require(channelCount == 2) { "分轨编码器需要双声道 PCM" }
        file.parentFile?.mkdirs()
        if (file.exists()) check(file.delete()) { "无法替换旧的分轨文件" }

        val format = MediaFormat.createAudioFormat(
            MediaFormat.MIMETYPE_AUDIO_AAC,
            sampleRate,
            channelCount
        ).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
            setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 32 * 1_024)
        }

        codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            muxer = MediaMuxer(file.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            codec.start()
        } catch (error: Throwable) {
            runCatching { codec.release() }
            file.delete()
            throw error
        }
    }

    fun write(count: Int, sample: (frame: Int, channel: Int) -> Float) {
        check(!closed) { "AAC 编码器已经关闭" }
        require(count >= 0) { "写入帧数不能为负数" }
        var sourceFrame = 0
        var stalledPolls = 0
        while (sourceFrame < count) {
            val inputIndex = codec.dequeueInputBuffer(CODEC_TIMEOUT_US)
            if (inputIndex < 0) {
                drain(endOfStream = false)
                check(++stalledPolls < MAX_END_OF_STREAM_POLLS) { "AAC 编码器长时间没有可用输入缓冲区" }
                continue
            }
            stalledPolls = 0
            val input = codec.getInputBuffer(inputIndex) ?: error("AAC 编码输入缓冲区不可用")
            input.clear()
            input.order(ByteOrder.LITTLE_ENDIAN)
            val chunkFrames = minOf(count - sourceFrame, input.remaining() / (channelCount * Short.SIZE_BYTES))
            check(chunkFrames > 0) { "AAC 编码输入缓冲区过小" }
            repeat(chunkFrames) { localFrame ->
                repeat(channelCount) { channel ->
                    val pcm = (sample(sourceFrame + localFrame, channel)
                        .coerceIn(-1f, 1f) * Short.MAX_VALUE)
                        .roundToInt()
                        .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                    input.putShort(pcm.toShort())
                }
            }
            val byteCount = chunkFrames * channelCount * Short.SIZE_BYTES
            codec.queueInputBuffer(
                inputIndex,
                0,
                byteCount,
                framesToMicroseconds(submittedFrames),
                0
            )
            submittedFrames += chunkFrames
            sourceFrame += chunkFrames
            drain(endOfStream = false)
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        var failure: Throwable? = null
        try {
            queueEndOfStream()
            drain(endOfStream = true)
        } catch (error: Throwable) {
            failure = error
        }
        runCatching { codec.stop() }.onFailure { if (failure == null) failure = it }
        runCatching { codec.release() }.onFailure { if (failure == null) failure = it }
        if (muxerStarted) {
            runCatching { muxer.stop() }.onFailure { if (failure == null) failure = it }
        }
        runCatching { muxer.release() }.onFailure { if (failure == null) failure = it }
        failure?.let { throw it }
    }

    private fun queueEndOfStream() {
        var polls = 0
        while (true) {
            val inputIndex = codec.dequeueInputBuffer(CODEC_TIMEOUT_US)
            if (inputIndex >= 0) {
                codec.queueInputBuffer(
                    inputIndex,
                    0,
                    0,
                    framesToMicroseconds(submittedFrames),
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM
                )
                return
            }
            drain(endOfStream = false)
            check(++polls < MAX_END_OF_STREAM_POLLS) { "AAC 编码器无法结束输入" }
        }
    }

    private fun drain(endOfStream: Boolean) {
        var emptyPolls = 0
        while (true) {
            when (val outputIndex = codec.dequeueOutputBuffer(
                bufferInfo,
                if (endOfStream) CODEC_TIMEOUT_US else 0L
            )) {
                MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) return
                    check(++emptyPolls < MAX_END_OF_STREAM_POLLS) { "AAC 编码器没有返回结束标记" }
                }
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    check(!muxerStarted) { "AAC 输出格式重复变化" }
                    trackIndex = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                    muxerStarted = true
                }
                else -> if (outputIndex >= 0) {
                    emptyPolls = 0
                    val encoded = codec.getOutputBuffer(outputIndex)
                        ?: error("AAC 编码输出缓冲区不可用")
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                        bufferInfo.size = 0
                    }
                    if (bufferInfo.size > 0) {
                        check(muxerStarted && trackIndex >= 0) { "AAC 封装器尚未就绪" }
                        encoded.position(bufferInfo.offset)
                        encoded.limit(bufferInfo.offset + bufferInfo.size)
                        val codecPresentationTimeUs = bufferInfo.presentationTimeUs
                        bufferInfo.presentationTimeUs = codecPresentationTimeUs -
                            framesToMicroseconds(AAC_ENCODER_PRIMING_FRAMES)
                        try {
                            muxer.writeSampleData(trackIndex, encoded, bufferInfo)
                        } finally {
                            bufferInfo.presentationTimeUs = codecPresentationTimeUs
                        }
                    }
                    val ended = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    codec.releaseOutputBuffer(outputIndex, false)
                    if (ended) return
                }
            }
        }
    }

    private fun framesToMicroseconds(frames: Long): Long = frames * 1_000_000L / sampleRate
}
