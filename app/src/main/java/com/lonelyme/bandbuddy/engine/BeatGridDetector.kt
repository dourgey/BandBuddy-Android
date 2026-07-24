package com.lonelyme.bandbuddy.engine

import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.round
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

data class BeatGridAnalysis(
    val bpm: Float,
    val confidence: Float,
    val beatOffsetMs: Long
)

/**
 * Detects tempo and the phase of the beat grid from decoded stereo float PCM.
 *
 * This is a Kotlin port of the desktop BandBuddy detector. It uses a
 * multi-band onset envelope, autocorrelation, peak intervals and a robust
 * least-squares grid refinement. At most the first three minutes are analyzed.
 */
object BeatGridDetector {
    const val ANALYSIS_VERSION = 2

    private const val MIN_BPM = 45
    private const val MAX_BPM = 240
    private const val TARGET_ENVELOPE_RATE = 100
    private const val MAX_ANALYSIS_SECONDS = 180L

    private enum class OnsetProfile(val bandWeights: DoubleArray) {
        FULL_MIX(doubleArrayOf(.85, 1.0, 1.05, 1.15)),
        DRUMS(doubleArrayOf(1.12, .88, 1.02, 1.24)),
        BASS(doubleArrayOf(1.58, 1.12, .34, .12))
    }

    private data class Envelope(val values: DoubleArray, val framesPerSecond: Double)
    private data class Peak(val frame: Int, val strength: Double)
    private data class GridFit(
        val phaseFrames: Double,
        val coverage: Double,
        val occupancy: Double,
        val score: Double
    )
    private data class TempoCandidate(
        val lag: Int,
        val bpm: Double,
        val autocorrelation: Double,
        var intervalScore: Double,
        val grid: GridFit,
        var score: Double = 0.0
    )
    private data class JointTempoCandidate(
        val lag: Int,
        val bpm: Double,
        val drumAutocorrelation: Double,
        val bassAutocorrelation: Double,
        val fusedAutocorrelation: Double,
        var drumInterval: Double,
        var bassInterval: Double,
        var fusedInterval: Double,
        val drumGrid: GridFit,
        val bassGrid: GridFit,
        val fusedGrid: GridFit,
        var autocorrelation: Double = 0.0,
        var intervalScore: Double = 0.0,
        var gridScore: Double = 0.0,
        var score: Double = 0.0
    )

    fun detectFromStereoFloatPcm(
        file: File,
        frames: Long,
        sampleRate: Int = 44_100,
        isCancelled: () -> Boolean = { false }
    ): BeatGridAnalysis? {
        if (!file.isFile || frames <= 0 || sampleRate <= 0) return null
        val availableFrames = min(frames, file.length() / (2 * Float.SIZE_BYTES))
        val analyzedFrames = min(availableFrames, sampleRate * MAX_ANALYSIS_SECONDS)
        if (analyzedFrames < sampleRate * 4L) return null
        val envelope = StereoFloatPcmReader(file).use { reader ->
            buildOnsetEnvelope(
                analyzedFrames,
                sampleRate,
                isCancelled,
                OnsetProfile.FULL_MIX.bandWeights
            ) { reader.nextMono() }
        } ?: return null
        return analyzeEnvelope(envelope)
    }

    /**
     * Jointly estimates the musical pulse from separated drums and bass.
     *
     * Drum transients provide timing precision while bass attacks confirm the
     * slower musical pulse. Scoring both envelopes independently prevents
     * hi-hat subdivisions from being mistaken for a doubled BPM, and the fused
     * envelope lets coincident kick/bass attacks anchor the beat phase.
     */
    fun detectFromDrumsAndBassPcm(
        drumsFile: File,
        drumsFrames: Long,
        bassFile: File,
        bassFrames: Long,
        sampleRate: Int = 44_100,
        isCancelled: () -> Boolean = { false }
    ): BeatGridAnalysis? {
        if (!drumsFile.isFile || !bassFile.isFile || sampleRate <= 0) return null
        val availableDrums = min(drumsFrames, drumsFile.length() / (2 * Float.SIZE_BYTES))
        val availableBass = min(bassFrames, bassFile.length() / (2 * Float.SIZE_BYTES))
        val analyzedFrames = min(min(availableDrums, availableBass), sampleRate * MAX_ANALYSIS_SECONDS)
        if (analyzedFrames < sampleRate * 4L) return null

        val drums = StereoFloatPcmReader(drumsFile).use { reader ->
            buildOnsetEnvelope(
                analyzedFrames,
                sampleRate,
                isCancelled,
                OnsetProfile.DRUMS.bandWeights
            ) { reader.nextMono() }
        } ?: return null
        val bass = StereoFloatPcmReader(bassFile).use { reader ->
            buildOnsetEnvelope(
                analyzedFrames,
                sampleRate,
                isCancelled,
                OnsetProfile.BASS.bandWeights
            ) { reader.nextMono() }
        } ?: return analyzeEnvelope(drums)
        return analyzeJointEnvelopes(drums, bass) ?: analyzeEnvelope(drums)
    }

    internal fun detectFromMonoSamples(samples: FloatArray, sampleRate: Int): BeatGridAnalysis? {
        if (sampleRate <= 0 || samples.size < sampleRate * 4) return null
        var index = 0
        val envelope = buildOnsetEnvelope(
            samples.size.toLong(),
            sampleRate,
            { false },
            OnsetProfile.FULL_MIX.bandWeights
        ) {
            samples[index++].toDouble()
        } ?: return null
        return analyzeEnvelope(envelope)
    }

    internal fun detectFromDrumsAndBassSamples(
        drums: FloatArray,
        bass: FloatArray,
        sampleRate: Int
    ): BeatGridAnalysis? {
        if (sampleRate <= 0 || min(drums.size, bass.size) < sampleRate * 4) return null
        val sampleCount = min(drums.size, bass.size)
        var drumIndex = 0
        val drumEnvelope = buildOnsetEnvelope(
            sampleCount.toLong(),
            sampleRate,
            { false },
            OnsetProfile.DRUMS.bandWeights
        ) { drums[drumIndex++].toDouble() } ?: return null
        var bassIndex = 0
        val bassEnvelope = buildOnsetEnvelope(
            sampleCount.toLong(),
            sampleRate,
            { false },
            OnsetProfile.BASS.bandWeights
        ) { bass[bassIndex++].toDouble() } ?: return analyzeEnvelope(drumEnvelope)
        return analyzeJointEnvelopes(drumEnvelope, bassEnvelope) ?: analyzeEnvelope(drumEnvelope)
    }

    private inline fun buildOnsetEnvelope(
        sampleCount: Long,
        sampleRate: Int,
        isCancelled: () -> Boolean,
        bandWeights: DoubleArray,
        nextSample: () -> Double
    ): Envelope? {
        val hopSize = max(16, (sampleRate.toDouble() / TARGET_ENVELOPE_RATE).roundToInt())
        val frameCount = (sampleCount / hopSize).toInt()
        if (frameCount < TARGET_ENVELOPE_RATE * 4) return null

        val bandEnergy = Array(4) { DoubleArray(frameCount) }
        val lowCoefficient = filterCoefficient(160.0, sampleRate)
        val middleCoefficient = filterCoefficient(700.0, sampleRate)
        val highCoefficient = filterCoefficient(2_500.0, sampleRate)
        val dcCoefficient = exp(-2.0 * PI * 25.0 / sampleRate)
        var previousSample = 0.0
        var previousHighPass = 0.0
        var lowPass = 0.0
        var middlePass = 0.0
        var highPass = 0.0
        var peakEnergy = 0.0

        for (frame in 0 until frameCount) {
            if (frame % TARGET_ENVELOPE_RATE == 0 && isCancelled()) error("CANCELLED")
            var lowSum = 0.0
            var lowerMiddleSum = 0.0
            var upperMiddleSum = 0.0
            var highSum = 0.0
            repeat(hopSize) {
                val input = nextSample()
                val dcBlocked = input - previousSample + dcCoefficient * previousHighPass
                previousSample = input
                previousHighPass = dcBlocked
                lowPass += lowCoefficient * (dcBlocked - lowPass)
                middlePass += middleCoefficient * (dcBlocked - middlePass)
                highPass += highCoefficient * (dcBlocked - highPass)
                val lowerMiddle = middlePass - lowPass
                val upperMiddle = highPass - middlePass
                val high = dcBlocked - highPass
                lowSum += lowPass * lowPass
                lowerMiddleSum += lowerMiddle * lowerMiddle
                upperMiddleSum += upperMiddle * upperMiddle
                highSum += high * high
            }
            val sums = doubleArrayOf(lowSum, lowerMiddleSum, upperMiddleSum, highSum)
            for (band in sums.indices) {
                val energy = sums[band] / hopSize
                bandEnergy[band][frame] = energy
                if (energy > peakEnergy) peakEnergy = energy
            }
        }
        if (peakEnergy < 1e-10) return null

        val rawFlux = DoubleArray(frameCount)
        for (frame in 1 until frameCount) {
            var flux = 0.0
            for (band in bandEnergy.indices) {
                val current = kotlin.math.ln(bandEnergy[band][frame] + peakEnergy * 1e-7 + 1e-14)
                val previous = kotlin.math.ln(bandEnergy[band][frame - 1] + peakEnergy * 1e-7 + 1e-14)
                flux += bandWeights[band] * max(0.0, current - previous)
            }
            rawFlux[frame] = flux
        }

        val prefix = DoubleArray(frameCount + 1)
        val prefixSquares = DoubleArray(frameCount + 1)
        for (frame in 0 until frameCount) {
            val value = rawFlux[frame]
            prefix[frame + 1] = prefix[frame] + value
            prefixSquares[frame + 1] = prefixSquares[frame] + value * value
        }

        val adaptive = DoubleArray(frameCount)
        val radius = max(8, (sampleRate.toDouble() / hopSize * .45).roundToInt())
        for (frame in 0 until frameCount) {
            val start = max(0, frame - radius)
            val end = min(frameCount, frame + radius + 1)
            val count = end - start
            val mean = (prefix[end] - prefix[start]) / count
            val meanSquare = (prefixSquares[end] - prefixSquares[start]) / count
            val deviation = sqrt(max(0.0, meanSquare - mean * mean))
            adaptive[frame] = max(0.0, rawFlux[frame] - mean * .72 - deviation * .08)
        }

        val ceiling = percentile(adaptive, .975)
        if (ceiling <= 1e-8) return null
        var energy = 0.0
        for (frame in 0 until frameCount) {
            var value = min(adaptive[frame], ceiling)
            val left = adaptive[max(0, frame - 1)]
            val right = adaptive[min(frameCount - 1, frame + 1)]
            if (value < left || value < right) value *= .35
            adaptive[frame] = value
            energy += value * value
        }
        if (energy < 1e-8) return null

        val scale = sqrt(energy / frameCount)
        for (frame in adaptive.indices) adaptive[frame] /= scale
        return Envelope(adaptive, sampleRate.toDouble() / hopSize)
    }

    private fun analyzeJointEnvelopes(drums: Envelope, bass: Envelope): BeatGridAnalysis? {
        if (abs(drums.framesPerSecond - bass.framesPerSecond) > .01) return null
        val framesPerSecond = drums.framesPerSecond
        val fused = fuseEnvelopes(drums, bass)
        val drumPeaks = pickPeaks(drums.values, framesPerSecond)
        val bassPeaks = pickPeaks(bass.values, framesPerSecond)
        val fusedPeaks = pickPeaks(fused.values, framesPerSecond)
        if (drumPeaks.size < 8) return analyzeEnvelope(bass)
        if (bassPeaks.size < 6 || fusedPeaks.size < 8) return analyzeEnvelope(drums)

        val minimumLag = max(2, floor(framesPerSecond * 60 / MAX_BPM).toInt())
        val maximumLag = min(fused.values.size - 2, ceil(framesPerSecond * 60 / MIN_BPM).toInt())
        val drumCorrelation = correlationCache(drums.values, maximumLag)
        val bassCorrelation = correlationCache(bass.values, maximumLag)
        val fusedCorrelation = correlationCache(fused.values, maximumLag)

        val candidates = mutableListOf<JointTempoCandidate>()
        var maximumDrumInterval = 0.0
        var maximumBassInterval = 0.0
        var maximumFusedInterval = 0.0
        for (lag in minimumLag..maximumLag) {
            val drumInterval = intervalEvidence(drumPeaks, lag)
            val bassInterval = intervalEvidence(bassPeaks, lag)
            val fusedInterval = intervalEvidence(fusedPeaks, lag)
            maximumDrumInterval = max(maximumDrumInterval, drumInterval)
            maximumBassInterval = max(maximumBassInterval, bassInterval)
            maximumFusedInterval = max(maximumFusedInterval, fusedInterval)
            candidates += JointTempoCandidate(
                lag = lag,
                bpm = 60.0 * framesPerSecond / lag,
                drumAutocorrelation = harmonicCorrelation(drumCorrelation, lag),
                bassAutocorrelation = harmonicCorrelation(bassCorrelation, lag),
                fusedAutocorrelation = harmonicCorrelation(fusedCorrelation, lag),
                drumInterval = drumInterval,
                bassInterval = bassInterval,
                fusedInterval = fusedInterval,
                drumGrid = evaluateGrid(drums.values, drumPeaks, lag, framesPerSecond),
                bassGrid = evaluateGrid(bass.values, bassPeaks, lag, framesPerSecond),
                fusedGrid = evaluateGrid(fused.values, fusedPeaks, lag, framesPerSecond)
            )
        }
        if (maximumDrumInterval <= 1e-9 || maximumFusedInterval <= 1e-9) return null

        candidates.forEach { candidate ->
            candidate.drumInterval = clamp(candidate.drumInterval / maximumDrumInterval, 0.0, 1.0)
            candidate.bassInterval = if (maximumBassInterval > 1e-9) {
                clamp(candidate.bassInterval / maximumBassInterval, 0.0, 1.0)
            } else {
                0.0
            }
            candidate.fusedInterval = clamp(candidate.fusedInterval / maximumFusedInterval, 0.0, 1.0)

            val correlationAgreement = sqrt(
                candidate.drumAutocorrelation * candidate.bassAutocorrelation
            )
            candidate.autocorrelation = (
                candidate.drumAutocorrelation * .36 +
                    candidate.bassAutocorrelation * .24 +
                    candidate.fusedAutocorrelation * .25 +
                    correlationAgreement * .15
                )
            val intervalAgreement = sqrt(candidate.drumInterval * candidate.bassInterval)
            candidate.intervalScore = (
                candidate.drumInterval * .42 +
                    candidate.bassInterval * .27 +
                    candidate.fusedInterval * .22 +
                    intervalAgreement * .09
                )
            val phaseAgreement = gridPhaseAgreement(
                candidate.drumGrid.phaseFrames,
                candidate.bassGrid.phaseFrames,
                candidate.lag.toDouble()
            )
            candidate.gridScore = (
                candidate.fusedGrid.score * .47 +
                    candidate.drumGrid.score * .30 +
                    candidate.bassGrid.score * .16 +
                    phaseAgreement * .07
                )
            candidate.score = (
                candidate.autocorrelation * .42 +
                    candidate.intervalScore * .28 +
                    candidate.gridScore * .30
                ) * tempoPrior(candidate.bpm)
        }
        candidates.sortByDescending(JointTempoCandidate::score)
        val best = resolveJointOctave(candidates) ?: return null
        if (best.score < .15 || best.autocorrelation < .035 || best.gridScore < .15) return null

        val refined = refineBeatGrid(fusedPeaks, best.lag.toDouble(), best.fusedGrid.phaseFrames)
        val rawBpm = 60.0 * framesPerSecond / refined.first
        if (!rawBpm.isFinite() || rawBpm < MIN_BPM || rawBpm > MAX_BPM) return null
        val bpm = round(rawBpm * 10.0) / 10.0
        val beatDurationMs = 60_000.0 / bpm
        val beatOffsetMs = round(
            positiveModulo(refined.second / framesPerSecond * 1_000.0 + beatDurationMs / 2, beatDurationMs) -
                beatDurationMs / 2
        ).toLong()

        val runnerUp = candidates.firstOrNull {
            it !== best && abs(it.bpm - best.bpm) / best.bpm > .025
        }
        val margin = runnerUp?.let {
            clamp((best.score - it.score) / max(best.score, 1e-9), 0.0, 1.0)
        } ?: 1.0
        val sourceAgreement = sqrt(bassPulse(best) * drumPulse(best))
        val quality = (
            best.autocorrelation * .36 +
                best.gridScore * .29 +
                best.intervalScore * .23 +
                sourceAgreement * .12
            )
        val confidence = clamp((quality - .05) / .72 * .86 + margin * .14, 0.0, 1.0)
        return BeatGridAnalysis(bpm.toFloat(), confidence.toFloat(), beatOffsetMs)
    }

    private fun fuseEnvelopes(drums: Envelope, bass: Envelope): Envelope {
        val size = min(drums.values.size, bass.values.size)
        val values = DoubleArray(size)
        for (frame in 0 until size) {
            var nearbyBass = 0.0
            for (offset in -3..3) {
                val index = frame + offset
                if (index in 0 until size) nearbyBass = max(nearbyBass, bass.values[index])
            }
            val drum = max(0.0, drums.values[frame])
            val bassAtFrame = max(0.0, bass.values[frame])
            val agreement = sqrt(drum * nearbyBass)
            values[frame] = drum * .52 + bassAtFrame * .28 + agreement * .42
        }
        var energy = 0.0
        for (value in values) energy += value * value
        val scale = sqrt(energy / max(1, size))
        if (scale > 1e-9) {
            for (frame in values.indices) values[frame] /= scale
        }
        return Envelope(values, drums.framesPerSecond)
    }

    private fun correlationCache(values: DoubleArray, maximumLag: Int): DoubleArray {
        val cache = DoubleArray(min(values.size - 2, maximumLag * 3) + 1)
        for (lag in 1 until cache.size) cache[lag] = centeredCorrelation(values, lag)
        return cache
    }

    private fun harmonicCorrelation(cache: DoubleArray, lag: Int): Double {
        val primary = max(0.0, cache.getOrElse(lag) { 0.0 })
        val double = max(0.0, cache.getOrElse(lag * 2) { 0.0 })
        val triple = max(0.0, cache.getOrElse(lag * 3) { 0.0 })
        val half = max(0.0, cache.getOrElse((lag / 2.0).roundToInt()) { 0.0 })
        return clamp((primary + double * .5 + triple * .2 - half * .08) / 1.7, 0.0, 1.0)
    }

    private fun gridPhaseAgreement(left: Double, right: Double, period: Double): Double {
        val distance = abs(positiveModulo(left - right + period / 2, period) - period / 2)
        val tolerance = max(2.0, period * .11)
        return exp(-.5 * (distance / tolerance).pow(2))
    }

    private fun drumPulse(candidate: JointTempoCandidate): Double =
        candidate.drumAutocorrelation * .48 +
            candidate.drumInterval * .32 +
            candidate.drumGrid.score * .20

    private fun bassPulse(candidate: JointTempoCandidate): Double =
        candidate.bassAutocorrelation * .48 +
            candidate.bassInterval * .34 +
            candidate.bassGrid.score * .18

    /**
     * Resolves the common half/double-tempo ambiguity using bass as the musical
     * pulse authority. The slower candidate is accepted only when it retains
     * strong joint evidence; genuinely fast songs therefore remain fast.
     */
    private fun resolveJointOctave(candidates: List<JointTempoCandidate>): JointTempoCandidate? {
        var best = candidates.firstOrNull() ?: return null
        if (best.bpm >= 105.0) {
            val half = candidates
                .filter { abs(it.bpm / best.bpm - .5) < .045 }
                .maxByOrNull(JointTempoCandidate::score)
            if (half != null &&
                half.bpm >= 55.0 &&
                half.score >= best.score * .76 &&
                half.gridScore >= best.gridScore * .80 &&
                bassPulse(half) >= bassPulse(best) * .92 &&
                (
                    bassPulse(half) >= bassPulse(best) * 1.04 ||
                        half.fusedGrid.occupancy >= best.fusedGrid.occupancy * .96
                    )
            ) {
                best = half
            }
        }
        if (best.bpm < 65.0) {
            val doubled = candidates
                .filter { abs(it.bpm / best.bpm - 2.0) < .045 }
                .maxByOrNull(JointTempoCandidate::score)
            if (doubled != null &&
                doubled.score >= best.score * .88 &&
                bassPulse(doubled) >= bassPulse(best) * .90 &&
                drumPulse(doubled) >= drumPulse(best) * 1.04
            ) {
                best = doubled
            }
        }
        return best
    }

    private fun analyzeEnvelope(envelope: Envelope): BeatGridAnalysis? {
        val values = envelope.values
        val framesPerSecond = envelope.framesPerSecond
        val peaks = pickPeaks(values, framesPerSecond)
        if (peaks.size < 8) return null

        val minimumLag = max(2, floor(framesPerSecond * 60 / MAX_BPM).toInt())
        val maximumLag = min(values.size - 2, ceil(framesPerSecond * 60 / MIN_BPM).toInt())
        val correlationCache = DoubleArray(min(values.size - 2, maximumLag * 3) + 1)
        for (lag in 1 until correlationCache.size) {
            correlationCache[lag] = centeredCorrelation(values, lag)
        }

        val candidates = mutableListOf<TempoCandidate>()
        var maximumIntervalScore = 0.0
        for (lag in minimumLag..maximumLag) {
            val primary = max(0.0, correlationCache.getOrElse(lag) { 0.0 })
            val double = max(0.0, correlationCache.getOrElse(lag * 2) { 0.0 })
            val triple = max(0.0, correlationCache.getOrElse(lag * 3) { 0.0 })
            val half = max(0.0, correlationCache.getOrElse((lag / 2.0).roundToInt()) { 0.0 })
            val autocorrelation = clamp((primary + double * .5 + triple * .2 - half * .08) / 1.7, 0.0, 1.0)
            val intervalScore = intervalEvidence(peaks, lag)
            maximumIntervalScore = max(maximumIntervalScore, intervalScore)
            candidates += TempoCandidate(
                lag = lag,
                bpm = 60.0 * framesPerSecond / lag,
                autocorrelation = autocorrelation,
                intervalScore = intervalScore,
                grid = evaluateGrid(values, peaks, lag, framesPerSecond)
            )
        }
        if (maximumIntervalScore <= 1e-9) return null

        candidates.forEach { candidate ->
            candidate.intervalScore = clamp(candidate.intervalScore / maximumIntervalScore, 0.0, 1.0)
            candidate.score = (
                candidate.autocorrelation * .4 +
                    candidate.intervalScore * .32 +
                    candidate.grid.score * .28
                ) * tempoPrior(candidate.bpm)
        }
        candidates.sortByDescending(TempoCandidate::score)
        var best = candidates.firstOrNull() ?: return null
        if (best.bpm < 70) {
            val doubled = candidates
                .filter { abs(it.bpm / best.bpm - 2.0) < .045 }
                .maxByOrNull(TempoCandidate::score)
            if (doubled != null &&
                doubled.score >= best.score * .64 &&
                doubled.intervalScore >= best.intervalScore * .97
            ) {
                best = doubled
            }
        }
        if (best.score < .16 || best.autocorrelation < .035 || best.grid.score < .16) return null

        val refined = refineBeatGrid(peaks, best.lag.toDouble(), best.grid.phaseFrames)
        val rawBpm = 60.0 * framesPerSecond / refined.first
        if (!rawBpm.isFinite() || rawBpm < MIN_BPM || rawBpm > MAX_BPM) return null
        val bpm = round(rawBpm * 10.0) / 10.0
        val beatDurationMs = 60_000.0 / bpm
        val beatOffsetMs = round(
            positiveModulo(refined.second / framesPerSecond * 1_000.0 + beatDurationMs / 2, beatDurationMs) -
                beatDurationMs / 2
        ).toLong()

        val runnerUp = candidates.firstOrNull { abs(it.bpm - best.bpm) / best.bpm > .025 }
        val margin = runnerUp?.let {
            clamp((best.score - it.score) / max(best.score, 1e-9), 0.0, 1.0)
        } ?: 1.0
        val quality = best.autocorrelation * .46 + best.grid.score * .34 + best.intervalScore * .2
        val confidence = clamp((quality - .05) / .72 * .88 + margin * .12, 0.0, 1.0)
        return BeatGridAnalysis(bpm.toFloat(), confidence.toFloat(), beatOffsetMs)
    }

    private fun centeredCorrelation(values: DoubleArray, lag: Int): Double {
        if (lag <= 0 || lag >= values.size - 2) return 0.0
        val count = values.size - lag
        var leftMean = 0.0
        var rightMean = 0.0
        for (index in lag until values.size) {
            leftMean += values[index]
            rightMean += values[index - lag]
        }
        leftMean /= count
        rightMean /= count

        var product = 0.0
        var leftEnergy = 0.0
        var rightEnergy = 0.0
        for (index in lag until values.size) {
            val left = values[index] - leftMean
            val right = values[index - lag] - rightMean
            product += left * right
            leftEnergy += left * left
            rightEnergy += right * right
        }
        val scale = sqrt(leftEnergy * rightEnergy)
        return if (scale > 1e-9) product / scale else 0.0
    }

    private fun pickPeaks(values: DoubleArray, framesPerSecond: Double): List<Peak> {
        var mean = 0.0
        var meanSquare = 0.0
        for (value in values) {
            mean += value
            meanSquare += value * value
        }
        mean /= values.size
        val deviation = sqrt(max(0.0, meanSquare / values.size - mean * mean))
        val threshold = mean + deviation * .18
        val radius = max(1, (framesPerSecond * .025).roundToInt())
        val minimumDistance = max(1, (framesPerSecond * .035).roundToInt())
        val peaks = mutableListOf<Peak>()

        for (frame in radius until values.size - radius) {
            val strength = values[frame]
            if (strength < threshold) continue
            var maximum = true
            for (offset in 1..radius) {
                if (values[frame - offset] > strength || values[frame + offset] >= strength) {
                    maximum = false
                    break
                }
            }
            if (!maximum) continue
            val previous = peaks.lastOrNull()
            if (previous != null && frame - previous.frame < minimumDistance) {
                if (strength > previous.strength) peaks[peaks.lastIndex] = Peak(frame, strength)
            } else {
                peaks += Peak(frame, strength)
            }
        }
        return peaks
    }

    private fun intervalEvidence(peaks: List<Peak>, lag: Int): Double {
        var score = 0.0
        for (left in peaks.indices) {
            val first = peaks[left]
            for (right in left + 1 until min(peaks.size, left + 9)) {
                val second = peaks[right]
                val distance = second.frame - first.frame
                val multiple = (distance.toDouble() / lag).roundToInt()
                if (multiple < 1) continue
                if (multiple > 4) break
                val tolerance = max(1.25, lag * .045 * sqrt(multiple.toDouble()))
                val deviation = abs(distance - multiple * lag).toDouble()
                if (deviation > tolerance * 3) continue
                val match = exp(-.5 * (deviation / tolerance).pow(2))
                score += sqrt(first.strength * second.strength) * match / multiple
            }
        }
        return score
    }

    private fun evaluateGrid(
        values: DoubleArray,
        peaks: List<Peak>,
        lag: Int,
        framesPerSecond: Double
    ): GridFit {
        val bins = max(16, lag * 4)
        val folded = DoubleArray(bins)
        var totalStrength = 0.0
        for (peak in peaks) {
            val position = positiveModulo(peak.frame + .5, lag.toDouble()) / lag * bins
            val lowerFloor = floor(position)
            val lower = lowerFloor.toInt() % bins
            val fraction = position - lowerFloor
            folded[lower] += peak.strength * (1 - fraction)
            folded[(lower + 1) % bins] += peak.strength * fraction
            totalStrength += peak.strength
        }

        val smoothingRadius = max(2, (framesPerSecond * .03 / lag * bins).roundToInt())
        var bestBin = 0
        var bestStrength = 0.0
        for (bin in 0 until bins) {
            var strength = 0.0
            var weightTotal = 0.0
            for (offset in -smoothingRadius..smoothingRadius) {
                val weight = exp(-.5 * (offset / max(1.0, smoothingRadius * .48)).pow(2))
                strength += folded[positiveModulo(bin + offset, bins)] * weight
                weightTotal += weight
            }
            strength /= weightTotal
            if (strength > bestStrength) {
                bestStrength = strength
                bestBin = bin
            }
        }

        val phaseFrames = bestBin.toDouble() / bins * lag
        val phaseTolerance = max(1, (framesPerSecond * .035).roundToInt())
        var alignedStrength = 0.0
        for (peak in peaks) {
            val phaseDistance = abs(
                positiveModulo(peak.frame + .5 - phaseFrames + lag / 2.0, lag.toDouble()) - lag / 2.0
            )
            if (phaseDistance <= phaseTolerance) {
                alignedStrength += peak.strength *
                    exp(-.5 * (phaseDistance / max(1.0, phaseTolerance * .6)).pow(2))
            }
        }
        val coverage = if (totalStrength > 1e-9) clamp(alignedStrength / totalStrength, 0.0, 1.0) else 0.0

        var hitSum = 0.0
        var hitSquares = 0.0
        var beatCount = 0
        var beat = phaseFrames
        while (beat - lag >= 0) beat -= lag
        while (beat < 0) beat += lag
        while (beat < values.size) {
            val center = round(beat - .5).toInt()
            var hit = 0.0
            for (offset in -phaseTolerance..phaseTolerance) {
                val frame = center + offset
                if (frame in values.indices) hit = max(hit, values[frame])
            }
            hitSum += hit
            hitSquares += hit * hit
            beatCount += 1
            beat += lag
        }
        val occupancy = if (beatCount > 0 && hitSquares > 1e-9) {
            clamp(hitSum * hitSum / (beatCount * hitSquares), 0.0, 1.0)
        } else {
            0.0
        }
        return GridFit(phaseFrames, coverage, occupancy, sqrt(coverage * occupancy))
    }

    private fun refineBeatGrid(
        peaks: List<Peak>,
        initialPeriod: Double,
        initialPhase: Double
    ): Pair<Double, Double> {
        var period = initialPeriod
        var phase = initialPhase
        repeat(4) {
            val tolerance = max(2.0, min(7.0, period * .12))
            var weightSum = 0.0
            var beatSum = 0.0
            var timeSum = 0.0
            var beatSquareSum = 0.0
            var beatTimeSum = 0.0
            for (peak in peaks) {
                val time = peak.frame + .5
                val beat = round((time - phase) / period)
                val residual = time - (phase + beat * period)
                if (abs(residual) > tolerance) continue
                val robust = (1 - (residual / tolerance).pow(2)).pow(2)
                val weight = sqrt(peak.strength) * robust
                weightSum += weight
                beatSum += weight * beat
                timeSum += weight * time
                beatSquareSum += weight * beat * beat
                beatTimeSum += weight * beat * time
            }
            val denominator = beatSquareSum - beatSum * beatSum / max(weightSum, 1e-9)
            if (weightSum < 4 || denominator < 1e-6) return@repeat
            val nextPeriod = (beatTimeSum - beatSum * timeSum / weightSum) / denominator
            if (!nextPeriod.isFinite() || abs(nextPeriod / initialPeriod - 1) > .06) return@repeat
            period = nextPeriod
            phase = (timeSum - period * beatSum) / weightSum
        }
        return period to phase
    }

    private fun tempoPrior(bpm: Double): Double = when {
        bpm in 65.0..190.0 -> 1.0
        bpm < 65 -> .82 + .18 * clamp((bpm - MIN_BPM) / (65 - MIN_BPM), 0.0, 1.0)
        else -> 1.0 - .14 * clamp((bpm - 190) / (MAX_BPM - 190), 0.0, 1.0)
    }

    private fun filterCoefficient(frequency: Double, sampleRate: Int): Double =
        1.0 - exp(-2.0 * PI * min(frequency, sampleRate * .42) / sampleRate)

    private fun percentile(values: DoubleArray, fraction: Double): Double {
        val sorted = values.filter { it > 0 }.toDoubleArray()
        if (sorted.isEmpty()) return 0.0
        sorted.sort()
        return sorted[min(sorted.lastIndex, floor(sorted.lastIndex * fraction).toInt())]
    }

    private fun positiveModulo(value: Double, modulus: Double): Double =
        ((value % modulus) + modulus) % modulus

    private fun positiveModulo(value: Int, modulus: Int): Int =
        ((value % modulus) + modulus) % modulus

    private fun clamp(value: Double, minimum: Double, maximum: Double): Double =
        max(minimum, min(maximum, value))

    private class StereoFloatPcmReader(file: File) : Closeable {
        private val channel = FileInputStream(file).channel
        private val buffer = ByteBuffer.allocateDirect(256 * 1024)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply { limit(0) }
        private var ended = false

        fun nextMono(): Double {
            while (buffer.remaining() < 2 * Float.SIZE_BYTES && !ended) {
                buffer.compact()
                if (channel.read(buffer) < 0) ended = true
                buffer.flip()
            }
            if (buffer.remaining() < 2 * Float.SIZE_BYTES) return 0.0
            return (buffer.float + buffer.float) * .5
        }

        override fun close() {
            channel.close()
        }
    }
}
