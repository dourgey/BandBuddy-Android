package cn.lonelyme.bandbuddy.worker

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.StatFs
import androidx.core.app.NotificationCompat
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import cn.lonelyme.bandbuddy.BuildConfig
import cn.lonelyme.bandbuddy.MainActivity
import cn.lonelyme.bandbuddy.R
import cn.lonelyme.bandbuddy.engine.ModelRuntime
import cn.lonelyme.bandbuddy.model.ModelSpec
import cn.lonelyme.bandbuddy.model.ModelStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.concurrent.TimeUnit

class ModelDownloadWorker(
    context: Context,
    parameters: WorkerParameters,
) : CoroutineWorker(context, parameters) {
    companion object {
        private const val UNIQUE_WORK = "model-download"
        private const val TAG = "model-download"
        private const val INPUT_SESSION = "session"
        private const val CHANNEL_ID = "model_download"
        private const val MAX_REDIRECTS = 6
        private const val BUFFER_BYTES = 256 * 1024
        private const val PROGRESS_UPDATE_BYTES = 1024 * 1024
        private const val STORAGE_RESERVE_BYTES = 32L * 1024L * 1024L
        private val CONTENT_RANGE =
            Regex("""bytes\s+(\d+)-(\d+)/(\d+)""", RegexOption.IGNORE_CASE)

        fun enqueue(context: Context) {
            val store = ModelStore(context)
            if (store.isCurrentReady()) return
            val session = store.beginDownload()
            val request = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .build()
                )
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.SECONDS)
                .setInputData(workDataOf(INPUT_SESSION to session))
                .addTag(TAG)
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork(
                UNIQUE_WORK,
                ExistingWorkPolicy.REPLACE,
                request,
            )
        }

        fun pause(context: Context) {
            val store = ModelStore(context)
            store.activeSession()?.let(store::markPaused)
            WorkManager.getInstance(context).cancelUniqueWork(UNIQUE_WORK)
        }

        fun delete(context: Context): Long {
            val store = ModelStore(context)
            val removedBytes = store.deleteAll()
            WorkManager.getInstance(context).cancelUniqueWork(UNIQUE_WORK)
            ModelRuntime.clearCompiledCache(context)
            return removedBytes
        }
    }

    private val store = ModelStore(applicationContext)

    override suspend fun doWork(): Result {
        val session = inputData.getString(INPUT_SESSION) ?: return Result.failure()
        if (!store.isActiveSession(session)) return Result.success()
        if (store.isCurrentReady()) return Result.success()

        setForeground(notification("等待下载", store.partialModelFile().length(), indeterminate = true))
        return try {
            val target = store.currentModelFile()
            // Recover cleanly if the process was killed after the atomic move
            // but before installation metadata was committed.
            val candidate = withContext(Dispatchers.IO) {
                target.takeIf { it.isFile && it.length() == ModelSpec.BYTES }
                    ?: download(session)
            }
            ensureActiveSession(session)
            store.markVerifying(session)
            setForeground(notification("正在校验完整性", ModelSpec.BYTES, indeterminate = true))

            val coroutineContext = currentCoroutineContext()
            val actualHash = withContext(Dispatchers.IO) {
                store.sha256(candidate) {
                    coroutineContext.ensureActive()
                    ensureActiveSession(session)
                }
            }
            if (!actualHash.equals(ModelSpec.SHA256, ignoreCase = true)) {
                candidate.delete()
                throw ModelDownloadException("模型校验失败，请重新下载", retryable = false)
            }

            ensureActiveSession(session)
            if (candidate != target) installAtomically(candidate, target)
            store.markInstalled(session)
            ModelRuntime.clearCompiledCache(applicationContext)
            setForeground(notification("模型已安装", ModelSpec.BYTES, indeterminate = false))
            Result.success()
        } catch (cancelled: CancellationException) {
            if (store.isActiveSession(session)) store.markPaused(session)
            throw cancelled
        } catch (_: StaleSessionException) {
            Result.success()
        } catch (error: Throwable) {
            val message = error.message ?: "模型下载失败"
            val retryable = (error as? ModelDownloadException)?.retryable
                ?: (error is IOException)
            if (store.isActiveSession(session) && retryable && runAttemptCount < 3) {
                store.markQueued(session, "下载中断，将自动重试 · $message")
                Result.retry()
            } else {
                store.markFailed(session, message)
                Result.failure()
            }
        }
    }

    private suspend fun download(session: String): File {
        val directory = store.currentDirectory()
        check(directory.exists() || directory.mkdirs()) { "无法创建模型目录" }
        val partial = store.partialModelFile()
        if (partial.length() > ModelSpec.BYTES) partial.delete()
        ensureFreeSpace(directory, ModelSpec.BYTES - partial.length())

        var offset = partial.length()
        var connection = openConnection(offset)
        var responseCode = connection.responseCode
        if (responseCode == 416) {
            connection.disconnect()
            if (offset == ModelSpec.BYTES) return partial
            partial.delete()
            offset = 0L
            connection = openConnection(0L)
            responseCode = connection.responseCode
        }
        if (offset > 0L && responseCode == HttpURLConnection.HTTP_OK) {
            connection.disconnect()
            RandomAccessFile(partial, "rw").use { it.setLength(0L) }
            offset = 0L
            connection = openConnection(0L)
            responseCode = connection.responseCode
        }

        validateResponse(connection, responseCode, offset)
        store.markDownloading(session, offset)
        setForeground(notification("正在下载模型", offset, indeterminate = false))

        try {
            connection.inputStream.buffered(BUFFER_BYTES).use { input ->
                RandomAccessFile(partial, "rw").use { output ->
                    output.seek(offset)
                    val buffer = ByteArray(BUFFER_BYTES)
                    var downloaded = offset
                    var lastPublished = offset
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        ensureActiveSession(session)
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        downloaded += read
                        if (downloaded > ModelSpec.BYTES) {
                            throw ModelDownloadException("服务器返回的模型大小不正确", retryable = false)
                        }
                        if (downloaded - lastPublished >= PROGRESS_UPDATE_BYTES.toLong()) {
                            lastPublished = downloaded
                            store.markDownloading(session, downloaded)
                            setProgress(workDataOf("downloaded" to downloaded, "total" to ModelSpec.BYTES))
                            setForeground(notification("正在下载模型", downloaded, indeterminate = false))
                        }
                    }
                    output.fd.sync()
                }
            }
        } finally {
            connection.disconnect()
        }

        if (partial.length() != ModelSpec.BYTES) {
            throw ModelDownloadException(
                "模型下载不完整（${formatBytes(partial.length())} / ${formatBytes(ModelSpec.BYTES)}）",
                retryable = true,
            )
        }
        store.markDownloading(session, ModelSpec.BYTES)
        return partial
    }

    private fun openConnection(offset: Long): HttpURLConnection {
        var url = URL(ModelSpec.downloadUrl)
        repeat(MAX_REDIRECTS + 1) { redirectCount ->
            val connection = (url.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 20_000
                readTimeout = 30_000
                requestMethod = "GET"
                setRequestProperty("Accept-Encoding", "identity")
                setRequestProperty("User-Agent", "BandBuddy/${BuildConfig.VERSION_NAME} Android")
                if (offset > 0L) setRequestProperty("Range", "bytes=$offset-")
            }
            val code = connection.responseCode
            if (code !in 300..399) return connection
            val location = connection.getHeaderField("Location")
                ?: throw ModelDownloadException("模型下载地址重定向无效", retryable = true)
            connection.disconnect()
            if (redirectCount == MAX_REDIRECTS) {
                throw ModelDownloadException("模型下载重定向过多", retryable = true)
            }
            url = URL(url, location)
        }
        error("unreachable")
    }

    private fun validateResponse(
        connection: HttpURLConnection,
        responseCode: Int,
        offset: Long,
    ) {
        if (responseCode !in listOf(HttpURLConnection.HTTP_OK, HttpURLConnection.HTTP_PARTIAL)) {
            val retryable = responseCode == 408 || responseCode == 429 || responseCode >= 500
            throw ModelDownloadException("模型服务器返回 HTTP $responseCode", retryable)
        }
        if (offset > 0L) {
            if (responseCode != HttpURLConnection.HTTP_PARTIAL) {
                throw ModelDownloadException("服务器不支持断点续传", retryable = true)
            }
            val contentRange = connection.getHeaderField("Content-Range").orEmpty()
            val match = CONTENT_RANGE.matchEntire(contentRange)
                ?: throw ModelDownloadException("服务器返回的分段信息无效", retryable = true)
            val start = match.groupValues[1].toLong()
            val total = match.groupValues[3].toLong()
            if (start != offset || total != ModelSpec.BYTES) {
                throw ModelDownloadException("服务器返回的模型版本不匹配", retryable = false)
            }
        } else {
            val responseBytes = connection.contentLengthLong
            if (responseBytes > 0L && responseBytes != ModelSpec.BYTES) {
                throw ModelDownloadException("服务器返回的模型大小不正确", retryable = false)
            }
        }
    }

    private fun ensureFreeSpace(directory: File, remainingBytes: Long) {
        val available = StatFs(directory.absolutePath).availableBytes
        if (available < remainingBytes + STORAGE_RESERVE_BYTES) {
            throw ModelDownloadException(
                "存储空间不足，至少还需要 ${formatBytes(remainingBytes + STORAGE_RESERVE_BYTES)}",
                retryable = false,
            )
        }
    }

    private fun installAtomically(partial: File, target: File) {
        try {
            Files.move(
                partial.toPath(),
                target.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(partial.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun ensureActiveSession(session: String) {
        if (isStopped) throw CancellationException("Model download stopped")
        if (!store.isActiveSession(session)) throw StaleSessionException()
    }

    private fun notification(stage: String, downloaded: Long, indeterminate: Boolean): ForegroundInfo {
        val manager = applicationContext.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "分轨模型下载", NotificationManager.IMPORTANCE_LOW)
        )
        val percent = ((downloaded * 100L) / ModelSpec.BYTES).toInt().coerceIn(0, 100)
        val content = if (indeterminate) stage
        else "$stage · ${formatBytes(downloaded)} / ${formatBytes(ModelSpec.BYTES)}"
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("BandBuddy 分轨模型")
            .setContentText(content)
            .setContentIntent(
                PendingIntent.getActivity(
                    applicationContext,
                    0,
                    Intent(applicationContext, MainActivity::class.java),
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
                )
            )
            .addAction(
                R.drawable.ic_close,
                "暂停",
                WorkManager.getInstance(applicationContext).createCancelPendingIntent(id),
            )
            .setOnlyAlertOnce(true)
            .setOngoing(downloaded < ModelSpec.BYTES)
            .setProgress(100, percent, indeterminate)
            .build()
        return ForegroundInfo(
            "model-download".hashCode(),
            notification,
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
        )
    }

    private fun formatBytes(bytes: Long): String =
        if (bytes >= 1024L * 1024L) "%.1f MB".format(bytes / (1024.0 * 1024.0))
        else "%.0f KB".format(bytes / 1024.0)

    private class StaleSessionException : RuntimeException()

    private class ModelDownloadException(
        message: String,
        val retryable: Boolean,
    ) : IOException(message)

}
