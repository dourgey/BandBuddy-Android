package com.lonelyme.bandbuddy.engine

import android.media.AudioAttributes
import android.media.MediaPlayer
import android.media.PlaybackParams
import android.os.SystemClock
import com.lonelyme.bandbuddy.data.StemRecord
import com.lonelyme.bandbuddy.data.StemType
import com.lonelyme.bandbuddy.data.TrackState
import java.io.Closeable
import kotlin.math.abs

/** Six-stem transport with one timing authority and guarded drift correction. */
class MultiStemPlayer : Closeable {
    private val players = linkedMapOf<StemType, MediaPlayer>()
    private var tracks: Map<StemType, TrackState> = emptyMap()
    private var masterVolume = 1f
    private var driftViolationCount = 0
    private var driftOutliers: Set<StemType> = emptySet()
    private var lastDriftCorrectionAtMs = 0L

    @get:Synchronized
    val isLoaded: Boolean get() = players.isNotEmpty()
    @get:Synchronized
    val isPlaying: Boolean get() = runCatching { players.values.firstOrNull()?.isPlaying == true }.getOrDefault(false)
    @get:Synchronized
    val positionMs: Long get() = runCatching { players.values.firstOrNull()?.currentPosition?.toLong() ?: 0 }.getOrDefault(0)
    @get:Synchronized
    val durationMs: Long get() = runCatching { players.values.firstOrNull()?.duration?.toLong() ?: 0 }.getOrDefault(0)

    @Synchronized
    fun load(stems: List<StemRecord>, positionMs: Long, speed: Float) {
        close()
        require(stems.map { it.type }.toSet().size == StemType.entries.size) { "需要完整的六个分轨" }
        stems.sortedBy { it.type.ordinal }.forEach { stem ->
            players[stem.type] = MediaPlayer().apply {
                setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build())
                setDataSource(stem.path)
                prepare()
                seekTo(positionMs.coerceAtLeast(0), MediaPlayer.SEEK_CLOSEST)
                playbackParams = PlaybackParams().setSpeed(speed).setPitch(1f)
                if (isPlaying) pause()
            }
        }
        applyMix()
    }

    @Synchronized
    fun play() {
        val target = positionMs
        players.values.forEach { player ->
            if (abs(player.currentPosition.toLong() - target) > 30) player.seekTo(target, MediaPlayer.SEEK_CLOSEST_SYNC)
        }
        resetDriftGuard()
        players.values.forEach(MediaPlayer::start)
    }

    @Synchronized
    fun pause() {
        players.values.forEach { if (it.isPlaying) it.pause() }
        resetDriftGuard()
    }

    @Synchronized
    fun seekTo(positionMs: Long) {
        val target = positionMs.coerceIn(0, durationMs.coerceAtLeast(0))
        players.values.forEach { it.seekTo(target, MediaPlayer.SEEK_CLOSEST) }
        resetDriftGuard()
    }

    @Synchronized
    fun setSpeed(speed: Float) {
        val safeSpeed = speed.coerceIn(.5f, 1.5f)
        players.values.forEach { player ->
            val wasPlaying = player.isPlaying
            player.playbackParams = PlaybackParams().setSpeed(safeSpeed).setPitch(1f)
            if (!wasPlaying && player.isPlaying) player.pause()
        }
        resetDriftGuard()
    }

    @Synchronized
    fun setMix(tracks: Map<StemType, TrackState>, masterVolume: Float) {
        val soloType = StemType.entries.firstOrNull { tracks[it]?.solo == true }
        this.tracks = StemType.entries.associateWith { type ->
            (tracks[type] ?: TrackState()).let { it.copy(solo = it.solo && type == soloType) }
        }
        this.masterVolume = masterVolume
        applyMix()
    }

    /**
     * Corrects only a large, persistent decoder drift.
     *
     * MediaPlayer positions naturally differ by a few AAC/audio-buffer frames,
     * especially while time stretching. Seeking every 500 ms for those small
     * differences caused the audible gaps at 0.5x/0.75x/1.25x/1.5x. We use the
     * median clock, require the same outlier for three checks, and rate-limit a
     * real correction so ordinary position jitter never interrupts playback.
     */
    @Synchronized
    fun synchronize(maxDriftMs: Long = 180, minCorrectionIntervalMs: Long = 10_000) {
        if (players.size < 2 || players.values.first().isPlaying.not()) return
        val positions = players.mapValues { (_, player) ->
            runCatching { player.currentPosition.toLong() }.getOrNull()
        }
        if (positions.values.any { it == null }) return
        val reference = positions.values.filterNotNull().sorted()[positions.size / 2]
        val currentOutliers = positions
            .filterValues { position -> position != null && abs(position - reference) > maxDriftMs }
            .keys

        if (currentOutliers.isEmpty()) {
            resetDriftGuard()
            return
        }
        if (currentOutliers == driftOutliers) {
            driftViolationCount += 1
        } else {
            driftOutliers = currentOutliers
            driftViolationCount = 1
        }

        val now = SystemClock.elapsedRealtime()
        if (driftViolationCount < 3 || now - lastDriftCorrectionAtMs < minCorrectionIntervalMs) return
        currentOutliers.forEach { type ->
            players[type]?.seekTo(reference, MediaPlayer.SEEK_CLOSEST)
        }
        lastDriftCorrectionAtMs = now
        resetDriftGuard()
    }

    private fun resetDriftGuard() {
        driftViolationCount = 0
        driftOutliers = emptySet()
    }

    private fun applyMix() {
        val hasSolo = tracks.values.any { it.solo && !it.muted }
        players.forEach { (type, player) ->
            val state = tracks[type] ?: TrackState()
            val audible = !state.muted && (!hasSolo || state.solo)
            val gain = if (audible) (state.volume * masterVolume).coerceIn(0f, 1f) else 0f
            player.setVolume(gain, gain)
        }
    }

    @Synchronized
    override fun close() {
        players.values.forEach { runCatching { it.stop() }; it.release() }
        players.clear()
        resetDriftGuard()
    }
}
