package com.lonelyme.bandbuddy.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PracticeStateTest {
    @Test
    fun selectingSoloClearsEveryOtherSolo() {
        val initial = PracticeState(
            tracks = StemType.entries.associateWith { TrackState(solo = it == StemType.VOCALS) }
        )

        val updated = initial.withTrackState(StemType.DRUMS, TrackState(solo = true))

        assertTrue(updated.tracks.getValue(StemType.DRUMS).solo)
        assertFalse(updated.tracks.getValue(StemType.VOCALS).solo)
        assertEquals(1, updated.tracks.values.count(TrackState::solo))
    }

    @Test
    fun normalizationRepairsLegacyMultipleSolosAndUnsupportedValues() {
        val legacy = PracticeState(
            speed = 2f,
            countInBeats = 6,
            tracks = StemType.entries.associateWith {
                TrackState(solo = it == StemType.VOCALS || it == StemType.DRUMS)
            }
        ).normalized()

        assertEquals(1.5f, legacy.speed)
        assertEquals(0, legacy.countInBeats)
        assertEquals(1, legacy.tracks.values.count(TrackState::solo))
    }

    @Test
    fun muteAndSoloCannotRemainActiveOnTheSameTrack() {
        val initial = PracticeState()

        val soloWinsLegacyConflict = initial.withTrackState(
            StemType.GUITAR,
            TrackState(muted = true, solo = true)
        )
        assertTrue(soloWinsLegacyConflict.tracks.getValue(StemType.GUITAR).solo)
        assertFalse(soloWinsLegacyConflict.tracks.getValue(StemType.GUITAR).muted)

        val muted = soloWinsLegacyConflict.withTrackState(
            StemType.GUITAR,
            TrackState(muted = true, solo = false)
        )
        assertTrue(muted.tracks.getValue(StemType.GUITAR).muted)
        assertFalse(muted.tracks.getValue(StemType.GUITAR).solo)
    }
}
