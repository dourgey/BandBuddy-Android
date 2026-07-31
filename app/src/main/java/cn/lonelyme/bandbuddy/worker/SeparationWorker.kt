package cn.lonelyme.bandbuddy.worker

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import cn.lonelyme.bandbuddy.data.LocalSongRepository
import cn.lonelyme.bandbuddy.data.SongRecord
import cn.lonelyme.bandbuddy.data.SongStatus
import cn.lonelyme.bandbuddy.engine.LongSongSeparator
import cn.lonelyme.bandbuddy.model.ModelStore
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class SeparationWorker(context: Context, parameters: WorkerParameters) : CoroutineWorker(context, parameters) {
    companion object {
        private const val CHANNEL_ID = "local_separation"
        private const val INPUT_SONG_ID = "song_id"
        private val singleJob = Mutex()

        fun enqueue(context: Context, songId: String, replace: Boolean = false) {
            val request = OneTimeWorkRequestBuilder<SeparationWorker>()
                .setInputData(workDataOf(INPUT_SONG_ID to songId))
                .addTag("separation-$songId")
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork(
                "separation-$songId", if (replace) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP, request
            )
        }

        fun cancel(context: Context, songId: String) = WorkManager.getInstance(context).cancelUniqueWork("separation-$songId")
    }

    override suspend fun doWork(): Result = singleJob.withLock {
        val songId = inputData.getString(INPUT_SONG_ID) ?: return@withLock Result.failure()
        val repository = LocalSongRepository(applicationContext)
        val song = repository.load().firstOrNull { it.id == songId } ?: return@withLock Result.failure()
        if (!ModelStore(applicationContext).isCurrentReady()) {
            update(repository, songId) {
                it.copy(
                    status = SongStatus.QUEUED,
                    jobStage = "等待下载分轨模型",
                    progress = 0,
                    error = null,
                )
            }
            return@withLock Result.success()
        }
        val processingStartedAt = System.currentTimeMillis()
        setForeground(notification(song, "准备音频", 1))
        update(repository, songId) {
            it.copy(
                status = SongStatus.PROCESSING,
                jobStage = "准备音频",
                progress = 1,
                error = null,
                processingStartedAt = processingStartedAt
            )
        }
        return@withLock runCatching {
            val output = LongSongSeparator(applicationContext).separate(
                song,
                onProgress = { progress ->
                    update(repository, songId) { it.copy(status = SongStatus.PROCESSING, jobStage = progress.stage, progress = progress.percent, error = null) }
                    setProgressAsync(workDataOf("stage" to progress.stage, "progress" to progress.percent))
                    setForegroundAsync(notification(song, progress.stage, progress.percent))
                },
                isCancelled = { isStopped }
            )
            update(repository, songId) { current ->
                val analyzedPractice = output.beatGrid?.let { analysis ->
                    current.practice.copy(
                        metronomeBpm = analysis.bpm,
                        metronomeOffsetMs = analysis.beatOffsetMs,
                        metronomeAnalysisDone = true,
                        metronomeAnalysisVersion = output.beatGridAnalysisVersion,
                        metronomeConfidence = analysis.confidence
                    )
                } ?: current.practice.copy(
                    metronomeAnalysisDone = true,
                    metronomeAnalysisVersion = 0,
                    metronomeConfidence = 0f
                )
                current.copy(
                    stems = output.stems,
                    status = SongStatus.READY,
                    jobStage = "可练习",
                    progress = 100,
                    error = null,
                    processingStartedAt = null,
                    practice = analyzedPractice
                )
            }
            Result.success()
        }.getOrElse { error ->
            val cancelled = isStopped || error.message == "CANCELLED"
            update(repository, songId) {
                it.copy(
                    status = SongStatus.FAILED,
                    jobStage = if (cancelled) "已取消" else "分轨失败",
                    error = if (cancelled) "任务已取消" else error.message ?: "分轨失败",
                    processingStartedAt = null
                )
            }
            Result.failure()
        }
    }

    private fun update(repository: LocalSongRepository, id: String, transform: (SongRecord) -> SongRecord) {
        repository.updateSong(id, transform)
    }

    private fun notification(song: SongRecord, stage: String, progress: Int): ForegroundInfo {
        val manager = applicationContext.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL_ID, "本地分轨", NotificationManager.IMPORTANCE_LOW))
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(cn.lonelyme.bandbuddy.R.drawable.ic_notification)
            .setContentTitle(song.title)
            .setContentText(if (progress in 1..99) "$progress% · 正在本地分轨" else stage)
            .setContentIntent(PendingIntent.getActivity(
                applicationContext,
                0,
                Intent(applicationContext, cn.lonelyme.bandbuddy.MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            ))
            .addAction(
                cn.lonelyme.bandbuddy.R.drawable.ic_close,
                "取消",
                WorkManager.getInstance(applicationContext).createCancelPendingIntent(id)
            )
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setProgress(100, progress.coerceIn(0, 100), false)
            .build()
        return ForegroundInfo(song.id.hashCode(), notification, android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
    }
}
