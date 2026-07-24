package com.lonelyme.bandbuddy.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sin

class BeatGridDetectorTest {
    @Test
    fun detectsTempoAndBeatOffset() {
        val result = BeatGridDetector.detectFromMonoSamples(
            clickTrack(bpm = 123.4, seconds = 45, sampleRate = 8_000, offsetMs = 137),
            sampleRate = 8_000
        )

        assertNotNull(result)
        assertEquals(123.4f, result!!.bpm, .2f)
        val beatDurationMs = 60_000.0 / result.bpm
        val phaseError = ((result.beatOffsetMs - 137 + beatDurationMs / 2) % beatDurationMs +
            beatDurationMs) % beatDurationMs - beatDurationMs / 2
        assertTrue(abs(phaseError) <= 15)
    }

    @Test
    fun detectsAlternatingStrongAndWeakBeats() {
        val result = BeatGridDetector.detectFromMonoSamples(
            clickTrack(bpm = 96.0, seconds = 30, sampleRate = 4_000, alternate = true),
            sampleRate = 4_000
        )

        assertNotNull(result)
        assertTrue(abs(result!!.bpm - 96f) <= 2f)
    }

    @Test
    fun rejectsSilence() {
        assertNull(BeatGridDetector.detectFromMonoSamples(FloatArray(40_000), 4_000))
    }

    @Test
    fun combinesBassPulseWithDrumSubdivisionsWithoutDoublingTempo() {
        val sampleRate = 8_000
        val tracks = drumAndBassTracks(
            bpm = 72.0,
            seconds = 50,
            sampleRate = sampleRate,
            offsetMs = 143,
            subdivisionStrength = .72
        )

        val result = BeatGridDetector.detectFromDrumsAndBassSamples(
            drums = tracks.first,
            bass = tracks.second,
            sampleRate = sampleRate
        )

        assertNotNull(result)
        assertEquals(72f, result!!.bpm, .7f)
        assertPhaseNear(result, expectedOffsetMs = 143, toleranceMs = 24)
    }

    @Test
    fun keepsGenuinelyFastTempoWhenBothSourcesSupportEveryBeat() {
        val sampleRate = 8_000
        val tracks = drumAndBassTracks(
            bpm = 158.0,
            seconds = 40,
            sampleRate = sampleRate,
            offsetMs = 91,
            subdivisionStrength = .28
        )

        val result = BeatGridDetector.detectFromDrumsAndBassSamples(
            drums = tracks.first,
            bass = tracks.second,
            sampleRate = sampleRate
        )

        assertNotNull(result)
        assertEquals(158f, result!!.bpm, 1.2f)
        assertPhaseNear(result, expectedOffsetMs = 91, toleranceMs = 24)
    }

    @Test
    fun retainsSlowMusicalPulseWhenDrumsHaveStrongEighthNotes() {
        val sampleRate = 8_000
        val tracks = drumAndBassTracks(
            bpm = 60.0,
            seconds = 55,
            sampleRate = sampleRate,
            offsetMs = 176,
            subdivisionStrength = .86
        )

        val result = BeatGridDetector.detectFromDrumsAndBassSamples(
            drums = tracks.first,
            bass = tracks.second,
            sampleRate = sampleRate
        )

        assertNotNull(result)
        assertEquals(60f, result!!.bpm, .7f)
        assertPhaseNear(result, expectedOffsetMs = 176, toleranceMs = 25)
    }

    private fun assertPhaseNear(result: BeatGridAnalysis, expectedOffsetMs: Int, toleranceMs: Int) {
        val beatDurationMs = 60_000.0 / result.bpm
        val error = ((result.beatOffsetMs - expectedOffsetMs + beatDurationMs / 2) % beatDurationMs +
            beatDurationMs) % beatDurationMs - beatDurationMs / 2
        assertTrue("phase error was $error ms", abs(error) <= toleranceMs)
    }

    private fun drumAndBassTracks(
        bpm: Double,
        seconds: Int,
        sampleRate: Int,
        offsetMs: Int,
        subdivisionStrength: Double
    ): Pair<FloatArray, FloatArray> {
        val drums = FloatArray(seconds * sampleRate)
        val bass = FloatArray(seconds * sampleRate)
        val beatSamples = sampleRate * 60.0 / bpm
        val offsetSamples = offsetMs / 1_000.0 * sampleRate
        var subdivision = 0
        while (offsetSamples + subdivision * beatSamples / 2 < drums.size) {
            val start = (offsetSamples + subdivision * beatSamples / 2).toInt()
            val amplitude = if (subdivision % 2 == 0) 1.0 else subdivisionStrength
            for (sample in 0 until minOf(130, drums.size - start)) {
                val transient = amplitude * exp(-sample / 20.0) *
                    (sin(sample * 1.63) + .35 * sin(sample * .47))
                drums[start + sample] += transient.toFloat()
            }
            subdivision += 1
        }

        var beat = 0
        while (offsetSamples + beat * beatSamples < bass.size) {
            val start = (offsetSamples + beat * beatSamples + sampleRate * .012).toInt()
            for (sample in 0 until minOf(sampleRate / 5, bass.size - start)) {
                val secondsFromAttack = sample.toDouble() / sampleRate
                val attack = (sample / (sampleRate * .006)).coerceIn(0.0, 1.0)
                val note = .88 * attack * exp(-secondsFromAttack * 7.0) *
                    sin(2.0 * Math.PI * 82.0 * secondsFromAttack)
                bass[start + sample] += note.toFloat()
            }
            beat += 1
        }
        return drums to bass
    }

    private fun clickTrack(
        bpm: Double,
        seconds: Int,
        sampleRate: Int,
        alternate: Boolean = false,
        offsetMs: Int = 0
    ): FloatArray {
        val samples = FloatArray(seconds * sampleRate)
        val beatSamples = sampleRate * 60.0 / bpm
        val offsetSamples = offsetMs / 1_000.0 * sampleRate
        var beat = 0
        while (offsetSamples + beat * beatSamples < samples.size) {
            val start = (offsetSamples + beat * beatSamples).toInt()
            val amplitude = if (alternate && beat % 2 == 1) .42 else 1.0
            var offset = 0
            while (offset < 100 && start + offset < samples.size) {
                samples[start + offset] = (
                    amplitude * exp(-offset / 18.0) * sin(offset * 1.7)
                    ).toFloat()
                offset += 1
            }
            beat += 1
        }
        return samples
    }
}
