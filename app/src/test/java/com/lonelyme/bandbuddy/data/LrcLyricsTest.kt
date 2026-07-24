package com.lonelyme.bandbuddy.data

import org.junit.Assert.assertEquals
import org.junit.Test

class LrcLyricsTest {
    @Test
    fun parsesMultipleTimestampsAndOffset() {
        val lyrics = LrcLyrics.parse(
            """
            [ar:测试歌手]
            [offset:120]
            [00:01.50][00:03.250]第一句
            [01:02]第二句
            """.trimIndent()
        )

        assertEquals(
            listOf(
                LyricLine(1_620, "第一句"),
                LyricLine(3_370, "第一句"),
                LyricLine(62_120, "第二句")
            ),
            lyrics
        )
    }
}
