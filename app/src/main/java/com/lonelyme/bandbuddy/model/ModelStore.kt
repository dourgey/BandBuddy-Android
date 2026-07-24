package com.lonelyme.bandbuddy.model

import android.content.Context
import androidx.core.content.edit
import java.io.File
import java.security.MessageDigest
import java.util.UUID

enum class ModelInstallPhase {
    NOT_INSTALLED,
    UPDATE_AVAILABLE,
    QUEUED,
    DOWNLOADING,
    VERIFYING,
    PAUSED,
    FAILED,
    READY,
}

data class ModelInstallSnapshot(
    val phase: ModelInstallPhase,
    val targetVersion: String = ModelSpec.VERSION,
    val installedVersion: String? = null,
    val downloadedBytes: Long = 0,
    val totalBytes: Long = ModelSpec.BYTES,
    val error: String? = null,
) {
    val isReady: Boolean
        get() = phase == ModelInstallPhase.READY

    val isActive: Boolean
        get() = phase == ModelInstallPhase.QUEUED ||
            phase == ModelInstallPhase.DOWNLOADING ||
            phase == ModelInstallPhase.VERIFYING

    val progress: Float
        get() = if (totalBytes <= 0) 0f
        else (downloadedBytes.toDouble() / totalBytes).toFloat().coerceIn(0f, 1f)
}

/**
 * Owns the versioned on-device model layout and its small persistent state.
 *
 * Layout:
 * files/models/<model-id>/<version>/<model-file>
 * files/models/<model-id>/<version>/<model-file>.part
 */
class ModelStore(context: Context) {
    companion object {
        private const val PREFERENCES = "bandbuddy_model_store_v1"
        private const val KEY_PHASE = "phase"
        private const val KEY_INSTALLED_VERSION = "installed_version"
        private const val KEY_INSTALLED_SHA256 = "installed_sha256"
        private const val KEY_INSTALLED_BYTES = "installed_bytes"
        private const val KEY_DOWNLOADED_BYTES = "downloaded_bytes"
        private const val KEY_ERROR = "error"
        private const val KEY_SESSION = "download_session"
    }

    private val appContext = context.applicationContext
    private val preferences = appContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
    private val modelRoot = File(appContext.filesDir, "models/${ModelSpec.ID}")

    fun currentDirectory(): File = File(modelRoot, ModelSpec.VERSION)

    fun currentModelFile(): File = File(currentDirectory(), ModelSpec.FILE_NAME)

    fun partialModelFile(): File = File(currentDirectory(), "${ModelSpec.FILE_NAME}.part")

    fun isCurrentReady(): Boolean {
        val file = currentModelFile()
        return file.isFile &&
            file.length() == ModelSpec.BYTES &&
            preferences.getString(KEY_INSTALLED_VERSION, null) == ModelSpec.VERSION &&
            preferences.getString(KEY_INSTALLED_SHA256, null) == ModelSpec.SHA256 &&
            preferences.getLong(KEY_INSTALLED_BYTES, -1L) == ModelSpec.BYTES
    }

    fun snapshot(): ModelInstallSnapshot {
        if (isCurrentReady()) {
            return ModelInstallSnapshot(
                phase = ModelInstallPhase.READY,
                installedVersion = ModelSpec.VERSION,
                downloadedBytes = ModelSpec.BYTES,
            )
        }

        val installedVersion = preferences.getString(KEY_INSTALLED_VERSION, null)
            ?.takeIf { version -> versionFile(version).isFile }
        val partialBytes = partialModelFile().takeIf(File::isFile)?.length()
            ?.coerceIn(0L, ModelSpec.BYTES)
            ?: 0L
        val persistedPhase = preferences.getString(KEY_PHASE, null)
            ?.let { value -> runCatching { ModelInstallPhase.valueOf(value) }.getOrNull() }
        val phase = when {
            persistedPhase == ModelInstallPhase.QUEUED -> ModelInstallPhase.QUEUED
            persistedPhase == ModelInstallPhase.DOWNLOADING -> ModelInstallPhase.DOWNLOADING
            persistedPhase == ModelInstallPhase.VERIFYING -> ModelInstallPhase.VERIFYING
            persistedPhase == ModelInstallPhase.FAILED -> ModelInstallPhase.FAILED
            persistedPhase == ModelInstallPhase.PAUSED -> ModelInstallPhase.PAUSED
            installedVersion != null && installedVersion != ModelSpec.VERSION ->
                ModelInstallPhase.UPDATE_AVAILABLE
            partialBytes > 0L -> ModelInstallPhase.PAUSED
            else -> ModelInstallPhase.NOT_INSTALLED
        }
        return ModelInstallSnapshot(
            phase = phase,
            installedVersion = installedVersion,
            downloadedBytes = if (phase == ModelInstallPhase.VERIFYING) ModelSpec.BYTES else partialBytes,
            error = preferences.getString(KEY_ERROR, null),
        )
    }

    fun beginDownload(): String {
        val session = UUID.randomUUID().toString()
        currentDirectory().mkdirs()
        preferences.edit(commit = true) {
            putString(KEY_SESSION, session)
            putString(KEY_PHASE, ModelInstallPhase.QUEUED.name)
            putLong(KEY_DOWNLOADED_BYTES, partialModelFile().length().coerceAtMost(ModelSpec.BYTES))
            remove(KEY_ERROR)
        }
        return session
    }

    fun isActiveSession(session: String): Boolean =
        preferences.getString(KEY_SESSION, null) == session

    fun activeSession(): String? = preferences.getString(KEY_SESSION, null)

    fun markQueued(session: String, error: String? = null) {
        updateActiveSession(session) {
            putString(KEY_PHASE, ModelInstallPhase.QUEUED.name)
            putLong(KEY_DOWNLOADED_BYTES, partialModelFile().length().coerceAtMost(ModelSpec.BYTES))
            if (error == null) remove(KEY_ERROR) else putString(KEY_ERROR, error)
        }
    }

    fun markDownloading(session: String, downloadedBytes: Long) {
        updateActiveSession(session) {
            putString(KEY_PHASE, ModelInstallPhase.DOWNLOADING.name)
            putLong(KEY_DOWNLOADED_BYTES, downloadedBytes.coerceIn(0L, ModelSpec.BYTES))
            remove(KEY_ERROR)
        }
    }

    fun markVerifying(session: String) {
        updateActiveSession(session) {
            putString(KEY_PHASE, ModelInstallPhase.VERIFYING.name)
            putLong(KEY_DOWNLOADED_BYTES, ModelSpec.BYTES)
            remove(KEY_ERROR)
        }
    }

    fun markPaused(session: String) {
        updateActiveSession(session) {
            putString(KEY_PHASE, ModelInstallPhase.PAUSED.name)
            putLong(KEY_DOWNLOADED_BYTES, partialModelFile().length().coerceAtMost(ModelSpec.BYTES))
            remove(KEY_ERROR)
        }
    }

    fun markFailed(session: String, error: String) {
        updateActiveSession(session) {
            putString(KEY_PHASE, ModelInstallPhase.FAILED.name)
            putLong(KEY_DOWNLOADED_BYTES, partialModelFile().length().coerceAtMost(ModelSpec.BYTES))
            putString(KEY_ERROR, error)
        }
    }

    fun markInstalled(session: String) {
        check(isActiveSession(session)) { "Model download session is no longer active" }
        val file = currentModelFile()
        check(file.isFile && file.length() == ModelSpec.BYTES) { "Verified model file is missing" }
        preferences.edit(commit = true) {
            putString(KEY_PHASE, ModelInstallPhase.READY.name)
            putString(KEY_INSTALLED_VERSION, ModelSpec.VERSION)
            putString(KEY_INSTALLED_SHA256, ModelSpec.SHA256)
            putLong(KEY_INSTALLED_BYTES, ModelSpec.BYTES)
            putLong(KEY_DOWNLOADED_BYTES, ModelSpec.BYTES)
            remove(KEY_ERROR)
            remove(KEY_SESSION)
        }
        deleteSupersededVersions()
    }

    fun invalidateDownload() {
        preferences.edit(commit = true) {
            remove(KEY_SESSION)
            remove(KEY_ERROR)
            remove(KEY_DOWNLOADED_BYTES)
            remove(KEY_PHASE)
        }
    }

    fun deleteAll(): Long {
        invalidateDownload()
        val removedBytes = modelRoot.walkTopDown()
            .filter(File::isFile)
            .sumOf(File::length)
        if (modelRoot.exists()) modelRoot.deleteRecursively()
        preferences.edit(commit = true) {
            remove(KEY_INSTALLED_VERSION)
            remove(KEY_INSTALLED_SHA256)
            remove(KEY_INSTALLED_BYTES)
        }
        return removedBytes
    }

    fun sha256(file: File, onBytesRead: (Long) -> Unit = {}): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        var total = 0L
        file.inputStream().buffered(buffer.size).use { input ->
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
                total += read
                onBytesRead(total)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private fun versionFile(version: String): File =
        File(File(modelRoot, version), ModelSpec.FILE_NAME)

    private fun updateActiveSession(
        session: String,
        update: android.content.SharedPreferences.Editor.() -> Unit,
    ) {
        if (!isActiveSession(session)) return
        preferences.edit(commit = true, action = update)
    }

    private fun deleteSupersededVersions() {
        modelRoot.listFiles().orEmpty()
            .filter { file -> file.isDirectory && file.name != ModelSpec.VERSION }
            .forEach { directory -> runCatching { directory.deleteRecursively() } }
    }
}
