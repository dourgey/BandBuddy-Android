package com.lonelyme.bandbuddy.engine

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.roundToLong
import kotlin.math.sin

data class TapTempoResult(
    val bpm: Float,
    val beatOffsetMs: Long,
    val tapCount: Int,
    val jitterMs: Float,
    val phaseConsistency: Float,
)

/**
 * Robustly fits an isochronous beat grid to taps captured on the playback timeline.
 *
 * A repeated-median slope keeps one inaccurate tap from pulling the tempo, residual
 * filtering removes gross timing errors, and a circular phase mean finds the beat
 * offset without being confused by the period boundary.
 */
class TapTempoEstimator(
    private val minimumBpm: Float = 20f,
    private val maximumBpm: Float = 400f,
    private val minimumTaps: Int = 4,
    private val windowSize: Int = 12,
) {
    private data class TapPoint(
        val songPositionMs: Double,
        val realtimeMs: Long,
        val beatIndex: Long,
    )

    private val taps = ArrayDeque<TapPoint>()
    private var periodHintMs: Double? = null

    val tapCount: Int
        get() = taps.size

    init {
        require(minimumBpm > 0f && maximumBpm > minimumBpm)
        require(minimumTaps >= 3)
        require(windowSize >= minimumTaps)
    }

    fun reset() {
        taps.clear()
        periodHintMs = null
    }

    /**
     * Adds one user tap.
     *
     * [songPositionMs] must be read directly from the playing transport. Wall-clock
     * time and [playbackSpeed] are used only to detect pauses, seeks, and loop jumps;
     * tempo is fitted in song time so it remains correct at non-1x playback speeds.
     */
    fun tap(
        songPositionMs: Long,
        realtimeMs: Long,
        playbackSpeed: Float,
    ): TapTempoResult? {
        if (songPositionMs < 0L || realtimeMs < 0L || !playbackSpeed.isFinite() || playbackSpeed <= 0f) {
            reset()
            return null
        }

        val last = taps.lastOrNull()
        if (last != null) {
            val wallDelta = realtimeMs - last.realtimeMs
            val songDelta = songPositionMs - last.songPositionMs
            if (wallDelta <= TAP_DEBOUNCE_MS) return null

            val maximumPeriodMs = 60_000.0 / minimumBpm
            val sessionGapMs = max(
                DEFAULT_SESSION_GAP_MS.toDouble(),
                maximumPeriodMs / playbackSpeed * SESSION_GAP_PERIODS,
            ).roundToLong().coerceAtMost(MAXIMUM_SESSION_GAP_MS)
            val expectedSongDelta = wallDelta * playbackSpeed
            val discontinuityTolerance = max(
                MINIMUM_TRANSPORT_TOLERANCE_MS,
                expectedSongDelta * TRANSPORT_TOLERANCE_RATIO,
            )
            if (
                wallDelta > sessionGapMs ||
                songDelta <= 0.0 ||
                abs(songDelta - expectedSongDelta) > discontinuityTolerance
            ) {
                reset()
                return addFirstTap(songPositionMs, realtimeMs)
            }

            periodHintMs?.let { period ->
                if (songDelta < period * DUPLICATE_TAP_RATIO) return null
                if (songDelta > period * MAXIMUM_INFERRED_BEAT_STEP + period * .45) {
                    reset()
                    return addFirstTap(songPositionMs, realtimeMs)
                }
            }
        }

        val beatIndex = when {
            last == null -> 0L
            periodHintMs == null -> last.beatIndex + 1L
            else -> {
                val ratio = (songPositionMs - last.songPositionMs) / periodHintMs!!
                val inferredStep = ratio.roundToInt().coerceIn(1, MAXIMUM_INFERRED_BEAT_STEP)
                val residual = abs(ratio - inferredStep)
                last.beatIndex + if (residual <= INFERRED_STEP_TOLERANCE) inferredStep else 1
            }
        }
        taps.addLast(TapPoint(songPositionMs.toDouble(), realtimeMs, beatIndex))
        while (taps.size > windowSize) taps.removeFirst()

        val result = estimate() ?: return null
        periodHintMs = 60_000.0 / result.bpm
        return result
    }

    private fun addFirstTap(songPositionMs: Long, realtimeMs: Long): TapTempoResult? {
        taps.addLast(TapPoint(songPositionMs.toDouble(), realtimeMs, 0L))
        return null
    }

    private fun estimate(): TapTempoResult? {
        if (taps.size < minimumTaps) return null
        var points = taps.toList()
        var periodMs = repeatedMedianSlope(points) ?: return null
        if (!periodMs.isValidPeriod()) return null

        val intercepts = points.map { it.songPositionMs - periodMs * it.beatIndex }
        val interceptMedian = intercepts.median()
        val residuals = intercepts.map { abs(it - interceptMedian) }
        val residualMad = residuals.median()
        val residualLimit = min(
            periodMs * MAXIMUM_RESIDUAL_PERIOD_RATIO,
            max(
                periodMs * MINIMUM_RESIDUAL_PERIOD_RATIO,
                max(MINIMUM_RESIDUAL_MS, residualMad * MAD_SCALE * MAD_LIMIT),
            ),
        )
        val inliers = points.filter { point ->
            abs(point.songPositionMs - periodMs * point.beatIndex - interceptMedian) <= residualLimit
        }
        val requiredInliers = max(3, ceil(points.size * MINIMUM_INLIER_RATIO).toInt())
        if (inliers.size < requiredInliers) return null
        points = inliers
        leastSquaresSlope(points)?.takeIf { it.isValidPeriod() }?.let { periodMs = it }

        val bpm = (60_000.0 / periodMs * 10.0).roundToInt().div(10f)
        if (bpm !in minimumBpm..maximumBpm) return null
        val roundedPeriodMs = 60_000.0 / bpm

        var sumSin = 0.0
        var sumCos = 0.0
        points.forEach { point ->
            val angle = TWO_PI * positiveModulo(point.songPositionMs, roundedPeriodMs) / roundedPeriodMs
            sumSin += sin(angle)
            sumCos += cos(angle)
        }
        val meanAngle = atan2(sumSin, sumCos)
        val phaseMs = meanAngle / TWO_PI * roundedPeriodMs
        val centeredOffsetMs = centeredModulo(phaseMs, roundedPeriodMs)
        val jitterMs = points
            .map { point -> abs(centeredModulo(point.songPositionMs - centeredOffsetMs, roundedPeriodMs)) }
            .median()
            .toFloat()
        val phaseConsistency = (hypot(sumSin, sumCos) / points.size)
            .toFloat()
            .coerceIn(0f, 1f)
        val maximumJitterMs = min(MAXIMUM_JITTER_MS, roundedPeriodMs * MAXIMUM_JITTER_PERIOD_RATIO)
        if (phaseConsistency < MINIMUM_PHASE_CONSISTENCY || jitterMs > maximumJitterMs) return null

        return TapTempoResult(
            bpm = bpm,
            beatOffsetMs = centeredModulo(centeredOffsetMs, roundedPeriodMs).roundToLong(),
            tapCount = taps.size,
            jitterMs = jitterMs,
            phaseConsistency = phaseConsistency,
        )
    }

    private fun repeatedMedianSlope(points: List<TapPoint>): Double? {
        if (points.size < 2) return null
        val pointSlopes = points.indices.mapNotNull { first ->
            points.indices
                .asSequence()
                .filter { it != first }
                .mapNotNull { second ->
                    val beatDelta = points[second].beatIndex - points[first].beatIndex
                    if (beatDelta == 0L) null
                    else (points[second].songPositionMs - points[first].songPositionMs) / beatDelta
                }
                .filter { it > 0.0 && it.isFinite() }
                .toList()
                .takeIf(List<Double>::isNotEmpty)
                ?.median()
        }
        return pointSlopes.takeIf(List<Double>::isNotEmpty)?.median()
    }

    private fun leastSquaresSlope(points: List<TapPoint>): Double? {
        if (points.size < 2) return null
        val meanBeat = points.map(TapPoint::beatIndex).average()
        val meanPosition = points.map(TapPoint::songPositionMs).average()
        var covariance = 0.0
        var beatVariance = 0.0
        points.forEach { point ->
            val beatDelta = point.beatIndex - meanBeat
            covariance += beatDelta * (point.songPositionMs - meanPosition)
            beatVariance += beatDelta * beatDelta
        }
        return if (beatVariance > 0.0) covariance / beatVariance else null
    }

    private fun Double.isValidPeriod(): Boolean =
        isFinite() && this in (60_000.0 / maximumBpm)..(60_000.0 / minimumBpm)

    private fun List<Double>.median(): Double {
        val sorted = sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    private fun positiveModulo(value: Double, modulus: Double): Double =
        ((value % modulus) + modulus) % modulus

    private fun centeredModulo(value: Double, modulus: Double): Double =
        positiveModulo(value + modulus / 2.0, modulus) - modulus / 2.0

    private companion object {
        const val TWO_PI = PI * 2.0
        const val TAP_DEBOUNCE_MS = 80L
        const val DEFAULT_SESSION_GAP_MS = 2_500L
        const val MAXIMUM_SESSION_GAP_MS = 8_000L
        const val SESSION_GAP_PERIODS = 1.5
        const val MINIMUM_TRANSPORT_TOLERANCE_MS = 220.0
        const val TRANSPORT_TOLERANCE_RATIO = .45
        const val DUPLICATE_TAP_RATIO = .38
        const val MAXIMUM_INFERRED_BEAT_STEP = 4
        const val INFERRED_STEP_TOLERANCE = .38
        const val MAD_SCALE = 1.4826
        const val MAD_LIMIT = 3.0
        const val MINIMUM_RESIDUAL_MS = 45.0
        const val MINIMUM_RESIDUAL_PERIOD_RATIO = .10
        const val MAXIMUM_RESIDUAL_PERIOD_RATIO = .35
        const val MINIMUM_INLIER_RATIO = .70
        const val MINIMUM_PHASE_CONSISTENCY = .72f
        const val MAXIMUM_JITTER_MS = 150.0
        const val MAXIMUM_JITTER_PERIOD_RATIO = .24
    }
}
