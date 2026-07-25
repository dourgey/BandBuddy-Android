package com.lonelyme.bandbuddy.engine

import kotlin.math.abs
import kotlin.math.roundToLong
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TapTempoEstimatorTest {
    @Test
    fun stableJitteredTapsProduceAccurateTempoAndPhase() {
        val estimator = TapTempoEstimator()
        val jitter = listOf(24, -18, 11, -27, 17, 0, -13, 20, -8, 5)
        var result: TapTempoResult? = null

        jitter.forEachIndexed { index, error ->
            val position = 137L + index * 500L + error
            result = estimator.tap(
                songPositionMs = position,
                realtimeMs = 10_000L + position,
                playbackSpeed = 1f,
            ) ?: result
        }

        assertNotNull(result)
        val measured = requireNotNull(result)
        assertEquals(120f, measured.bpm, .45f)
        assertTrue(circularError(measured.beatOffsetMs, 137L, measured.bpm) <= 18L)
        assertTrue(measured.jitterMs < 30f)
        assertTrue(measured.phaseConsistency > .95f)
    }

    @Test
    fun playbackSpeedDoesNotChangeSongTempoEstimate() {
        val estimator = TapTempoEstimator()
        var result: TapTempoResult? = null
        val speed = .8f

        repeat(8) { index ->
            val position = 180L + index * 600L
            result = estimator.tap(
                songPositionMs = position,
                realtimeMs = 20_000L + (index * 600L / speed).toLong(),
                playbackSpeed = speed,
            ) ?: result
        }

        assertNotNull(result)
        val measured = requireNotNull(result)
        assertEquals(100f, measured.bpm, .1f)
        assertTrue(circularError(measured.beatOffsetMs, 180L, measured.bpm) <= 1L)
    }

    @Test
    fun oneInaccurateTapDoesNotPullTheGrid() {
        val estimator = TapTempoEstimator()
        val errors = listOf(4, -7, 9, 148, -8, 6, -4, 3, 0)
        var result: TapTempoResult? = null

        errors.forEachIndexed { index, error ->
            val position = 95L + index * 500L + error
            result = estimator.tap(position, 30_000L + position, 1f) ?: result
        }

        assertNotNull(result)
        val measured = requireNotNull(result)
        assertEquals(120f, measured.bpm, .35f)
        assertTrue(circularError(measured.beatOffsetMs, 95L, measured.bpm) <= 12L)
    }

    @Test
    fun missingBeatAndDuplicateTapAreHandledAfterLock() {
        val estimator = TapTempoEstimator()
        var realtime = 40_000L
        var result: TapTempoResult? = null

        listOf(100L, 600L, 1_100L, 1_600L, 2_100L, 3_100L).forEach { position ->
            realtime = 40_000L + position
            result = estimator.tap(position, realtime, 1f) ?: result
        }
        val countBeforeDuplicate = estimator.tapCount
        assertNull(estimator.tap(3_240L, realtime + 140L, 1f))
        assertEquals(countBeforeDuplicate, estimator.tapCount)

        result = estimator.tap(3_600L, 43_600L, 1f) ?: result
        assertNotNull(result)
        val measured = requireNotNull(result)
        assertEquals(120f, measured.bpm, .1f)
        assertTrue(circularError(measured.beatOffsetMs, 100L, measured.bpm) <= 1L)
    }

    @Test
    fun transportJumpStartsANewTapSession() {
        val estimator = TapTempoEstimator()
        repeat(4) { index ->
            estimator.tap(120L + index * 500L, 50_120L + index * 500L, 1f)
        }
        assertEquals(4, estimator.tapCount)

        assertNull(estimator.tap(300L, 52_300L, 1f))
        assertEquals(1, estimator.tapCount)
    }

    @Test
    fun irregularTapsDoNotReplaceTheBeatGrid() {
        val estimator = TapTempoEstimator()
        var result: TapTempoResult? = null

        listOf(100L, 400L, 1_150L, 1_430L, 2_240L, 2_510L, 3_390L, 3_640L)
            .forEach { position ->
                result = estimator.tap(position, 60_000L + position, 1f) ?: result
            }

        assertNull("Unexpected grid: $result", result)
    }

    @Test
    fun tempoRemainsAccurateAcrossPracticeSpeedsAndCommonBpms() {
        val jitter = listOf(-21, 13, -9, 25, -16, 7, 0, 18, -12, 5)
        val bpms = listOf(60f, 90f, 123.4f, 180f, 240f)
        val speeds = listOf(.5f, .8f, 1f, 1.2f, 1.5f)

        bpms.forEach { expectedBpm ->
            speeds.forEach { speed ->
                val estimator = TapTempoEstimator()
                val period = 60_000.0 / expectedBpm
                val offset = 83L
                var result: TapTempoResult? = null
                jitter.forEachIndexed { index, error ->
                    val position = (offset + index * period + error).roundToLong()
                    val realtime = 80_000L + ((position - offset) / speed).roundToLong()
                    result = estimator.tap(position, realtime, speed) ?: result
                }

                val measured = requireNotNull(result) { "$expectedBpm BPM at ${speed}x did not lock" }
                assertEquals(expectedBpm, measured.bpm, 1f)
                assertTrue(
                    "$expectedBpm BPM at ${speed}x offset=${measured.beatOffsetMs}",
                    circularError(measured.beatOffsetMs, offset, measured.bpm) <= 22L,
                )
            }
        }
    }

    private fun circularError(actual: Long, expected: Long, bpm: Float): Long {
        val period = 60_000.0 / bpm
        val error = ((actual - expected + period / 2.0) % period + period) % period - period / 2.0
        return abs(error).toLong()
    }
}
