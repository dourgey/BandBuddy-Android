package com.lonelyme.bandbuddy.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class MetronomeTimingTest {
    @Test
    fun schedulesFromSongGridAndScalesWithPlaybackSpeed() {
        assertEquals(
            NextMetronomeBeat(2, 250, 500),
            MetronomeTiming.nextBeat(875, 120f, 125, 1f)
        )
        assertEquals(
            NextMetronomeBeat(2, 500, 1_000),
            MetronomeTiming.nextBeat(875, 120f, 125, .5f)
        )
    }

    @Test
    fun skipsBeatWithoutEnoughSchedulingLead() {
        assertEquals(
            NextMetronomeBeat(3, 500, 500),
            MetronomeTiming.nextBeat(1_125, 120f, 125, 1f)
        )
    }
}
