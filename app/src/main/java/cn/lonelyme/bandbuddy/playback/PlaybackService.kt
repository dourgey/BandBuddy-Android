package cn.lonelyme.bandbuddy.playback

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaMetadata
import android.media.session.MediaSession
import android.media.session.PlaybackState
import android.os.Binder
import android.os.Handler
import android.os.IBinder
import android.graphics.drawable.Icon
import android.os.Looper
import android.os.SystemClock
import androidx.core.content.ContextCompat
import cn.lonelyme.bandbuddy.MainActivity
import cn.lonelyme.bandbuddy.R
import cn.lonelyme.bandbuddy.data.LocalSongRepository
import cn.lonelyme.bandbuddy.data.PracticeState
import cn.lonelyme.bandbuddy.data.SongRecord
import cn.lonelyme.bandbuddy.data.SongStatus
import cn.lonelyme.bandbuddy.data.normalized
import cn.lonelyme.bandbuddy.engine.MetronomeClickPlayer
import cn.lonelyme.bandbuddy.engine.MetronomeTiming
import cn.lonelyme.bandbuddy.engine.MultiStemPlayer

/**
 * Owns the six-stem transport independently from Compose. The active media
 * session supplies notification, lock-screen, headset and Bluetooth controls.
 */
class PlaybackService : Service() {
    companion object {
        private const val CHANNEL_ID = "practice_playback"
        private const val NOTIFICATION_ID = 6402
        private const val ACTION_TOGGLE = "cn.lonelyme.bandbuddy.playback.TOGGLE"
        private const val ACTION_BACK = "cn.lonelyme.bandbuddy.playback.BACK"
        private const val ACTION_FORWARD = "cn.lonelyme.bandbuddy.playback.FORWARD"
        private const val ACTION_STOP = "cn.lonelyme.bandbuddy.playback.STOP"
        private const val ACTION_INTERNAL_ACTIVE = "cn.lonelyme.bandbuddy.playback.INTERNAL_ACTIVE"
        private const val PREFS = "playback_session"
        private const val CURRENT_SONG = "current_song"
        private const val WAS_PLAYING = "was_playing"

        fun lastSongId(context: Context): String? = context.applicationContext
            .getSharedPreferences(PREFS, MODE_PRIVATE)
            .getString(CURRENT_SONG, null)

        fun stopIfCurrent(context: Context, songId: String) {
            val appContext = context.applicationContext
            val preferences = appContext.getSharedPreferences(PREFS, MODE_PRIVATE)
            if (preferences.getString(CURRENT_SONG, null) == songId) {
                preferences.edit().remove(CURRENT_SONG).putBoolean(WAS_PLAYING, false).apply()
                ContextCompat.startForegroundService(
                    appContext,
                    Intent(appContext, PlaybackService::class.java).setAction(ACTION_STOP)
                )
            }
        }
    }

    inner class PlaybackBinder : Binder() {
        val service: PlaybackService get() = this@PlaybackService
    }

    private val binder = PlaybackBinder()
    private val repository by lazy { LocalSongRepository(applicationContext) }
    private val player = MultiStemPlayer()
    private val metronomeClickDelegate = lazy(LazyThreadSafetyMode.NONE) {
        runCatching { MetronomeClickPlayer() }.getOrNull()
    }
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var mediaSession: MediaSession
    private lateinit var audioManager: AudioManager
    private lateinit var focusRequest: AudioFocusRequest
    private var currentSong: SongRecord? = null
    private var practice = PracticeState()
    private var lastPersistedAt = 0L
    private var playbackGeneration = 0L
    private var countInGeneration = 0L
    private var countInBeat = 0
    private var countInTotal = 0
    private var countInRemainingValue = 0
    private var scheduledMetronomeBeat = 0L

    val songId: String? get() = currentSong?.id
    val isLoaded: Boolean get() = player.isLoaded
    val isPlaying: Boolean get() = player.isPlaying
    val countInRemaining: Int get() = countInRemainingValue
    val isCountingIn: Boolean get() = countInGeneration != 0L
    val isPlayingOrCountingIn: Boolean get() = player.isPlaying || isCountingIn
    val positionMs: Long get() = player.positionMs
    val durationMs: Long get() = player.durationMs

    private val countInTicker = object : Runnable {
        override fun run() {
            val generation = countInGeneration
            if (generation == 0L || generation != playbackGeneration || !player.isLoaded) return
            if (countInBeat >= countInTotal) {
                countInGeneration = 0
                countInRemainingValue = 0
                beginPlayback(generation)
                return
            }
            countInRemainingValue = countInTotal - countInBeat
            playMetronomeClick(accented = countInBeat % 4 == 0)
            countInBeat += 1
            updatePlaybackState()
            getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification())
            val intervalMs = (60_000.0 / practice.metronomeBpm / practice.speed)
                .toLong()
                .coerceAtLeast(50)
            handler.postDelayed(this, intervalMs)
        }
    }

    private val metronomeTicker = object : Runnable {
        override fun run() {
            if (!player.isPlaying || !practice.metronomeEnabled) return
            playMetronomeClick(accented = Math.floorMod(scheduledMetronomeBeat, 4L) == 0L)
            scheduleNextMetronomeBeat()
        }
    }

    private val ticker = object : Runnable {
        override fun run() {
            if (player.isLoaded) {
                player.synchronize()
                val position = player.positionMs
                if (practice.loopEnabled && practice.loopB > practice.loopA && position >= practice.loopB) {
                    player.seekTo(practice.loopA)
                    startMetronome()
                }
                updatePlaybackState()
                if (player.isPlaying) {
                    getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification())
                }
                if (SystemClock.elapsedRealtime() - lastPersistedAt >= 5_000) persist()
            }
            handler.postDelayed(this, 500)
        }
    }

    override fun onCreate() {
        super.onCreate()
        metronomeClickDelegate.value?.warmUp()
        audioManager = getSystemService(AudioManager::class.java)
        focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build())
            .setOnAudioFocusChangeListener { focus ->
                if (focus == AudioManager.AUDIOFOCUS_LOSS || focus == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT) pause()
            }
            .build()
        mediaSession = MediaSession(this, "BandBuddyPractice").apply {
            setCallback(object : MediaSession.Callback() {
                override fun onPlay() = play()
                override fun onPause() = pause()
                override fun onStop() = stopSession()
                override fun onSeekTo(pos: Long) = seekTo(pos)
                override fun onSkipToPrevious() = seekTo((positionMs - 5_000).coerceAtLeast(0))
                override fun onSkipToNext() = seekTo((positionMs + 5_000).coerceAtMost(durationMs))
            }, handler)
            isActive = true
        }
        createChannel()
        handler.post(ticker)
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onUnbind(intent: Intent?): Boolean {
        persist()
        if (!isPlayingOrCountingIn) stopSelf()
        return true
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                startForeground(NOTIFICATION_ID, notification())
                stopSession()
                return START_NOT_STICKY
            }
            ACTION_TOGGLE, ACTION_BACK, ACTION_FORWARD -> {
                startForeground(NOTIFICATION_ID, notification())
                restoreIfNeeded()
                when (intent.action) {
                    ACTION_TOGGLE -> if (isPlayingOrCountingIn) pause() else play()
                    ACTION_BACK -> seekTo((positionMs - 5_000).coerceAtLeast(0))
                    ACTION_FORWARD -> seekTo((positionMs + 5_000).coerceAtMost(durationMs))
                }
                if (!isPlayingOrCountingIn) {
                    stopForeground(STOP_FOREGROUND_DETACH)
                    getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification())
                }
            }
            ACTION_INTERNAL_ACTIVE -> Unit
            null -> if (getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean(WAS_PLAYING, false) && !player.isPlaying) {
                startForeground(NOTIFICATION_ID, notification())
                restoreIfNeeded()
                if (player.isLoaded && audioManager.requestAudioFocus(focusRequest) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                    val generation = ++playbackGeneration
                    beginPlayback(generation)
                } else {
                    stopForeground(STOP_FOREGROUND_REMOVE)
                }
            }
        }
        return START_STICKY
    }

    @Synchronized
    fun load(song: SongRecord) {
        require(song.status == SongStatus.READY) { "歌曲尚未完成分轨" }
        if (currentSong?.id == song.id && player.isLoaded) return
        persist()
        cancelScheduledPlayback()
        player.close()
        currentSong = song
        practice = song.practice.normalized()
        player.load(song.stems, practice.positionMs, practice.speed)
        player.setMix(practice.tracks)
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(CURRENT_SONG, song.id).apply()
        updateMetadata()
        updatePlaybackState()
    }

    fun updatePractice(next: PracticeState) {
        val normalized = next.normalized()
        val speedChanged = normalized.speed != practice.speed
        val metronomeChanged = normalized.metronomeEnabled != practice.metronomeEnabled ||
            normalized.metronomeBpm != practice.metronomeBpm ||
            normalized.metronomeOffsetMs != practice.metronomeOffsetMs ||
            speedChanged
        practice = normalized
        player.setMix(normalized.tracks)
        if (speedChanged) player.setSpeed(normalized.speed)
        if (isCountingIn && !normalized.metronomeEnabled) {
            val generation = playbackGeneration
            cancelCountIn()
            beginPlayback(generation)
        } else if (metronomeChanged && player.isPlaying) {
            if (normalized.metronomeEnabled) startMetronome() else stopMetronome()
        }
        persist()
    }

    fun play() {
        if (!player.isLoaded) restoreIfNeeded()
        if (!player.isLoaded) return
        if (audioManager.requestAudioFocus(focusRequest) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) return
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(WAS_PLAYING, true).apply()
        startService(Intent(this, PlaybackService::class.java).setAction(ACTION_INTERNAL_ACTIVE))
        val generation = ++playbackGeneration
        cancelCountIn()
        stopMetronome()
        mediaSession.isActive = true
        updatePlaybackState()
        startForeground(NOTIFICATION_ID, notification())
        if (practice.metronomeEnabled && practice.countInBeats > 0) {
            countInGeneration = generation
            countInBeat = 0
            countInTotal = practice.countInBeats
            countInRemainingValue = practice.countInBeats
            handler.post(countInTicker)
        } else {
            beginPlayback(generation)
        }
    }

    fun pause() {
        playbackGeneration += 1
        cancelCountIn()
        stopMetronome()
        if (player.isLoaded) player.pause()
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(WAS_PLAYING, false).apply()
        persist()
        updatePlaybackState()
        stopForeground(STOP_FOREGROUND_DETACH)
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification())
    }

    fun seekTo(position: Long) {
        if (player.isLoaded) player.seekTo(position)
        if (player.isPlaying && practice.metronomeEnabled) startMetronome()
        updatePlaybackState()
    }

    private fun beginPlayback(generation: Long) {
        if (generation != playbackGeneration || !player.isLoaded) return
        player.play()
        if (practice.metronomeEnabled) startMetronome()
        mediaSession.isActive = true
        updatePlaybackState()
        startForeground(NOTIFICATION_ID, notification())
    }

    private fun startMetronome() {
        stopMetronome()
        if (!player.isPlaying || !practice.metronomeEnabled) return
        scheduleNextMetronomeBeat()
    }

    private fun scheduleNextMetronomeBeat() {
        handler.removeCallbacks(metronomeTicker)
        val next = MetronomeTiming.nextBeat(
            songPositionMs = player.positionMs,
            bpm = practice.metronomeBpm,
            beatOffsetMs = practice.metronomeOffsetMs,
            playbackRate = practice.speed
        ) ?: return
        scheduledMetronomeBeat = next.beatIndex
        handler.postDelayed(metronomeTicker, next.delayMs)
    }

    private fun stopMetronome() {
        handler.removeCallbacks(metronomeTicker)
    }

    private fun playMetronomeClick(accented: Boolean) {
        runCatching {
            metronomeClickDelegate.value?.play(
                accented = accented,
                volume = practice.metronomeVolume
            )
        }
    }

    fun previewMetronome() {
        playMetronomeClick(accented = true)
    }

    private fun cancelCountIn() {
        handler.removeCallbacks(countInTicker)
        countInGeneration = 0
        countInBeat = 0
        countInTotal = 0
        countInRemainingValue = 0
    }

    private fun cancelScheduledPlayback() {
        playbackGeneration += 1
        cancelCountIn()
        stopMetronome()
    }

    private fun restoreIfNeeded() {
        if (player.isLoaded) return
        val id = getSharedPreferences(PREFS, MODE_PRIVATE).getString(CURRENT_SONG, null) ?: return
        repository.load().firstOrNull { it.id == id && it.status == SongStatus.READY }?.let { runCatching { load(it) } }
    }

    private fun persist() {
        val song = currentSong ?: return
        val saved = practice.copy(positionMs = player.positionMs).normalized()
        practice = saved
        repository.updateSong(song.id) { it.copy(practice = saved) }
        lastPersistedAt = SystemClock.elapsedRealtime()
    }

    private fun updateMetadata() {
        val song = currentSong ?: return
        mediaSession.setMetadata(MediaMetadata.Builder()
            .putString(MediaMetadata.METADATA_KEY_TITLE, song.title)
            .putString(MediaMetadata.METADATA_KEY_ARTIST, song.artist)
            .putLong(MediaMetadata.METADATA_KEY_DURATION, song.durationMs)
            .build())
    }

    private fun updatePlaybackState() {
        val state = when {
            player.isPlaying -> PlaybackState.STATE_PLAYING
            isCountingIn -> PlaybackState.STATE_BUFFERING
            else -> PlaybackState.STATE_PAUSED
        }
        mediaSession.setPlaybackState(PlaybackState.Builder()
            .setActions(
                PlaybackState.ACTION_PLAY or PlaybackState.ACTION_PAUSE or PlaybackState.ACTION_PLAY_PAUSE or
                    PlaybackState.ACTION_SEEK_TO or PlaybackState.ACTION_SKIP_TO_PREVIOUS or PlaybackState.ACTION_SKIP_TO_NEXT or PlaybackState.ACTION_STOP
            )
            .setState(state, player.positionMs, practice.speed, SystemClock.elapsedRealtime())
            .build())
    }

    private fun notification(): Notification {
        val song = currentSong
        val contentIntent = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
                .putExtra("open_song_id", song?.id),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val style = Notification.MediaStyle().setMediaSession(mediaSession.sessionToken).setShowActionsInCompactView(0, 1, 2)
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(song?.title ?: "BandBuddy")
            .setContentText(if (isCountingIn) "预备拍 · 还剩 $countInRemaining 拍" else song?.artist ?: "本地六轨练习")
            .setContentIntent(contentIntent)
            .setDeleteIntent(serviceIntent(ACTION_STOP, 4))
            .setCategory(Notification.CATEGORY_TRANSPORT)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .setOnlyAlertOnce(true)
            .setShowWhen(false)
            .setOngoing(isPlayingOrCountingIn)
            .addAction(Notification.Action.Builder(Icon.createWithResource(this, R.drawable.ic_seek_back), "后退 5 秒", serviceIntent(ACTION_BACK, 1)).build())
            .addAction(Notification.Action.Builder(Icon.createWithResource(this, if (isPlayingOrCountingIn) R.drawable.ic_pause else R.drawable.ic_play), if (isPlayingOrCountingIn) "暂停" else "播放", serviceIntent(ACTION_TOGGLE, 2)).build())
            .addAction(Notification.Action.Builder(Icon.createWithResource(this, R.drawable.ic_seek_forward), "前进 5 秒", serviceIntent(ACTION_FORWARD, 3)).build())
            .setStyle(style)
            .build()
    }

    private fun serviceIntent(action: String, requestCode: Int): PendingIntent = PendingIntent.getForegroundService(
        this,
        requestCode,
        Intent(this, PlaybackService::class.java).setAction(action),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
    )

    private fun createChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "练习播放", NotificationManager.IMPORTANCE_LOW).apply {
                description = "后台播放、锁屏和蓝牙控制"
                setSound(null, null)
            }
        )
    }

    private fun stopSession() {
        cancelScheduledPlayback()
        player.pause()
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(WAS_PLAYING, false).apply()
        persist()
        audioManager.abandonAudioFocusRequest(focusRequest)
        mediaSession.isActive = false
        stopForeground(STOP_FOREGROUND_REMOVE)
        getSystemService(NotificationManager::class.java).cancel(NOTIFICATION_ID)
        stopSelf()
    }

    override fun onDestroy() {
        handler.removeCallbacks(ticker)
        cancelScheduledPlayback()
        persist()
        getSystemService(NotificationManager::class.java).cancel(NOTIFICATION_ID)
        player.close()
        if (metronomeClickDelegate.isInitialized()) {
            runCatching { metronomeClickDelegate.value?.close() }
        }
        audioManager.abandonAudioFocusRequest(focusRequest)
        mediaSession.release()
        super.onDestroy()
    }
}
