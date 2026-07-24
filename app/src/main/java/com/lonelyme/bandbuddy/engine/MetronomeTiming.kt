package com.lonelyme.bandbuddy.engine

import kotlin.math.ceil

data class NextMetronomeBeat(
    val beatIndex: Long,
    val delayMs: Long,
    val intervalMs: Long
)

object MetronomeTiming {
    fun nextBeat(
        songPositionMs: Long,
        bpm: Float,
        beatOffsetMs: Long,
        playbackRate: Float,
        minimumLeadMs: Long = 20
    ): NextMetronomeBeat? {
        if (!bpm.isFinite() || bpm <= 0f || !playbackRate.isFinite() || playbackRate <= 0f) return null
        val beatDurationMs = 60_000.0 / bpm
        var beatIndex = ceil((songPositionMs - beatOffsetMs) / beatDurationMs - 1e-9).toLong()
        var delayMs = (beatOffsetMs + beatIndex * beatDurationMs - songPositionMs) / playbackRate
        val intervalMs = beatDurationMs / playbackRate
        while (delayMs < minimumLeadMs) {
            beatIndex += 1
            delayMs += intervalMs
        }
        return NextMetronomeBeat(
            beatIndex = beatIndex,
            delayMs = delayMs.toLong().coerceAtLeast(1),
            intervalMs = intervalMs.toLong().coerceAtLeast(1)
        )
    }
}
