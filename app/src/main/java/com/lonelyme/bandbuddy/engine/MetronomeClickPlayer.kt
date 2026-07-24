package com.lonelyme.bandbuddy.engine

import android.media.AudioManager
import android.media.ToneGenerator
import android.os.SystemClock
import android.util.Log
import java.io.Closeable
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import kotlin.math.roundToInt

/**
 * Produces the metronome on Android's native tone path.
 *
 * A short, start/flush/stop streaming [android.media.AudioTrack] is dropped by
 * some vendor audio mixers, even though the track reports PLAYING. A single
 * warmed-up [ToneGenerator] stays attached to STREAM_MUSIC and makes every
 * count-in and running beat audible alongside all six stem players.
 */
class MetronomeClickPlayer : Closeable {
    private companion object {
        const val TAG = "MetronomeClick"
        const val CLICK_DURATION_MS = 70
        const val RETRY_DELAY_MS = 2_000L
    }

    private val executor = Executors.newSingleThreadExecutor { task ->
        Thread(task, "BandBuddyMetronome").apply { isDaemon = true }
    }

    @Volatile
    private var closed = false
    private var generator: ToneGenerator? = null
    private var generatorVolume = -1
    private var nextInitializationAttemptAtMs = 0L

    /** Initializes the native output before the first scheduled beat. */
    fun warmUp() = submit {
        getOrCreateGenerator(volumePercent = 100)
    }

    fun play(accented: Boolean, volume: Float = 1f) = submit {
        val volumePercent = (volume.coerceIn(.2f, 1f) * 100f).roundToInt()
        val output = getOrCreateGenerator(volumePercent) ?: return@submit
        val tone = if (accented) ToneGenerator.TONE_PROP_BEEP2 else ToneGenerator.TONE_PROP_BEEP
        val started = runCatching {
            output.stopTone()
            output.startTone(tone, CLICK_DURATION_MS)
        }.getOrElse {
            discardGenerator(output)
            Log.w(TAG, "Native metronome output failed; will retry", it)
            false
        }
        if (!started) {
            discardGenerator(output)
            Log.w(TAG, "Native metronome output rejected a click; will retry")
        }
    }

    private fun submit(block: () -> Unit) {
        if (closed) return
        try {
            executor.execute {
                if (!closed) block()
            }
        } catch (_: RejectedExecutionException) {
            // close() raced this click; there is no output left to use.
        }
    }

    private fun getOrCreateGenerator(volumePercent: Int): ToneGenerator? {
        generator?.takeIf { generatorVolume == volumePercent }?.let { return it }
        generator?.let {
            runCatching { it.stopTone() }
            it.release()
            generator = null
        }

        val now = SystemClock.elapsedRealtime()
        if (now < nextInitializationAttemptAtMs) return null
        nextInitializationAttemptAtMs = now + RETRY_DELAY_MS

        return runCatching {
            ToneGenerator(AudioManager.STREAM_MUSIC, volumePercent).also {
                generator = it
                generatorVolume = volumePercent
                nextInitializationAttemptAtMs = 0L
            }
        }.getOrElse {
            Log.w(TAG, "Could not initialize native metronome output; will retry", it)
            null
        }
    }

    private fun discardGenerator(failed: ToneGenerator) {
        if (generator !== failed) return
        generator = null
        generatorVolume = -1
        nextInitializationAttemptAtMs = SystemClock.elapsedRealtime() + RETRY_DELAY_MS
        runCatching { failed.stopTone() }
        failed.release()
    }

    override fun close() {
        closed = true
        executor.shutdownNow()
        generator?.let {
            runCatching { it.stopTone() }
            it.release()
        }
        generator = null
    }
}
