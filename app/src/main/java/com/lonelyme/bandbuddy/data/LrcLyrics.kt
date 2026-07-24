package com.lonelyme.bandbuddy.data

import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.charset.Charset

data class LyricLine(
    val timestampMs: Long,
    val text: String
)

object LrcLyrics {
    private val timestampPattern = Regex("""\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?]""")
    private val offsetPattern = Regex("""^\s*\[offset:([+-]?\d+)]\s*$""", RegexOption.IGNORE_CASE)

    fun parse(file: File): List<LyricLine> = parse(decode(file.readBytes()))

    fun parse(content: String): List<LyricLine> {
        var offsetMs = 0L
        content.lineSequence().forEach { rawLine ->
            offsetPattern.matchEntire(rawLine)?.groupValues?.get(1)?.toLongOrNull()?.let { offsetMs = it }
        }

        return buildList {
            content.lineSequence().forEach { rawLine ->
                val matches = timestampPattern.findAll(rawLine).toList()
                if (matches.isEmpty()) return@forEach
                val text = timestampPattern.replace(rawLine, "").trim()
                if (text.isBlank()) return@forEach
                matches.forEach { match ->
                    val minutes = match.groupValues[1].toLong()
                    val seconds = match.groupValues[2].toLong()
                    val fraction = match.groupValues[3]
                    val fractionMs = when (fraction.length) {
                        0 -> 0L
                        1 -> fraction.toLong() * 100
                        2 -> fraction.toLong() * 10
                        else -> fraction.take(3).toLong()
                    }
                    add(
                        LyricLine(
                            timestampMs = (minutes * 60_000 + seconds * 1_000 + fractionMs + offsetMs)
                                .coerceAtLeast(0),
                            text = text
                        )
                    )
                }
            }
        }.sortedBy(LyricLine::timestampMs).distinct()
    }

    private fun decode(bytes: ByteArray): String {
        if (bytes.size >= 3 &&
            bytes[0] == 0xEF.toByte() &&
            bytes[1] == 0xBB.toByte() &&
            bytes[2] == 0xBF.toByte()
        ) {
            return bytes.copyOfRange(3, bytes.size).toString(StandardCharsets.UTF_8)
        }
        if (bytes.size >= 2 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xFE.toByte()) {
            return bytes.copyOfRange(2, bytes.size).toString(StandardCharsets.UTF_16LE)
        }
        if (bytes.size >= 2 && bytes[0] == 0xFE.toByte() && bytes[1] == 0xFF.toByte()) {
            return bytes.copyOfRange(2, bytes.size).toString(StandardCharsets.UTF_16BE)
        }

        val utf8 = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
        return runCatching { utf8.decode(ByteBuffer.wrap(bytes)).toString() }
            .getOrElse { bytes.toString(Charset.forName("GB18030")) }
    }
}
