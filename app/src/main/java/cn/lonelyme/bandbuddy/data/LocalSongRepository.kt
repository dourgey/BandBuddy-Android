package cn.lonelyme.bandbuddy.data

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import cn.lonelyme.bandbuddy.engine.WaveformPeaks
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

enum class SongStatus { QUEUED, PROCESSING, READY, FAILED }

enum class StemType(val fileName: String, val label: String) {
    VOCALS("vocals.wav", "人声"),
    DRUMS("drums.wav", "鼓"),
    BASS("bass.wav", "贝斯"),
    GUITAR("guitar.wav", "吉他"),
    PIANO("piano.wav", "钢琴"),
    OTHER("other.wav", "其他");

    companion object {
        fun detect(name: String): StemType? {
            val value = name.lowercase()
            return entries.firstOrNull { type ->
                when (type) {
                    VOCALS -> listOf("vocals", "vocal", "voice", "人声").any(value::contains)
                    DRUMS -> listOf("drums", "drum", "鼓").any(value::contains)
                    BASS -> listOf("bass", "贝斯").any(value::contains)
                    GUITAR -> listOf("guitar", "吉他").any(value::contains)
                    PIANO -> listOf("piano", "keys", "keyboard", "钢琴").any(value::contains)
                    OTHER -> listOf("other", "others", "其他").any(value::contains)
                }
            }
        }
    }
}

data class StemRecord(val type: StemType, val path: String)

data class TrackState(
    val muted: Boolean = false,
    val solo: Boolean = false,
    val volume: Float = 1f
)

data class PracticeState(
    val positionMs: Long = 0,
    val speed: Float = 1f,
    val loopA: Long = 0,
    val loopB: Long = 0,
    val loopEnabled: Boolean = false,
    val metronomeEnabled: Boolean = false,
    val metronomeBpm: Float = 120f,
    val metronomeOffsetMs: Long = 0,
    val metronomeVolume: Float = 1f,
    val metronomeAnalysisDone: Boolean = false,
    val metronomeAnalysisVersion: Int = 0,
    val metronomeConfidence: Float = 0f,
    val countInBeats: Int = 0,
    val tracks: Map<StemType, TrackState> = StemType.entries.associateWith { TrackState() }
)

fun normalizeBeatOffsetMs(offsetMs: Long, bpm: Float): Long {
    if (!bpm.isFinite() || bpm <= 0f) return 0
    val beatDurationMs = 60_000.0 / bpm
    val normalized = ((offsetMs + beatDurationMs / 2) % beatDurationMs + beatDurationMs) % beatDurationMs -
        beatDurationMs / 2
    return normalized.toLong()
}

fun PracticeState.normalized(): PracticeState {
    val soloType = StemType.entries.firstOrNull { tracks[it]?.solo == true }
    val normalizedTracks = StemType.entries.associateWith { type ->
        val state = tracks[type] ?: TrackState()
        val solo = state.solo && type == soloType
        state.copy(
            muted = state.muted && !solo,
            solo = solo,
            volume = state.volume.coerceIn(0f, 1f)
        )
    }
    val bpm = metronomeBpm.takeIf(Float::isFinite)?.coerceIn(20f, 400f) ?: 120f
    return copy(
        positionMs = positionMs.coerceAtLeast(0),
        speed = speed.takeIf(Float::isFinite)?.coerceIn(.5f, 1.5f) ?: 1f,
        loopA = loopA.coerceAtLeast(0),
        loopB = loopB.coerceAtLeast(0),
        loopEnabled = loopEnabled && loopB > loopA,
        metronomeBpm = bpm,
        metronomeOffsetMs = normalizeBeatOffsetMs(metronomeOffsetMs, bpm),
        metronomeVolume = metronomeVolume.takeIf(Float::isFinite)?.coerceIn(.2f, 1f) ?: 1f,
        metronomeAnalysisVersion = metronomeAnalysisVersion.coerceAtLeast(0),
        metronomeConfidence = metronomeConfidence.takeIf(Float::isFinite)?.coerceIn(0f, 1f) ?: 0f,
        countInBeats = countInBeats.takeIf { it == 4 || it == 8 } ?: 0,
        tracks = normalizedTracks
    )
}

fun PracticeState.withTrackState(type: StemType, state: TrackState): PracticeState {
    val selected = state.copy(muted = state.muted && !state.solo)
    val updatedTracks = StemType.entries.associateWith { candidate ->
        when {
            candidate == type -> selected
            selected.solo -> (tracks[candidate] ?: TrackState()).copy(solo = false)
            else -> tracks[candidate] ?: TrackState()
        }
    }
    return copy(tracks = updatedTracks).normalized()
}

data class SongRecord(
    val id: String,
    val title: String,
    val artist: String,
    val durationMs: Long,
    val sourcePath: String?,
    val stems: List<StemRecord>,
    val lyricsPath: String? = null,
    val status: SongStatus,
    val jobStage: String,
    val progress: Int,
    val error: String? = null,
    val favorite: Boolean = false,
    val importedAt: Long = System.currentTimeMillis(),
    val processingStartedAt: Long? = null,
    val practice: PracticeState = PracticeState()
)

class LocalSongRepository(private val context: Context) {
    companion object {
        private val storageLock = Any()
    }

    private val preferences = context.getSharedPreferences("bandbuddy_library_v2", Context.MODE_PRIVATE)
    private val songsRoot = File(context.filesDir, "songs").apply { mkdirs() }

    fun importSource(uri: Uri): SongRecord {
        val id = UUID.randomUUID().toString()
        val name = displayName(uri) ?: "未命名歌曲"
        val extension = name.substringAfterLast('.', "audio").take(8).lowercase()
        val directory = File(songsRoot, "$id/source").apply { mkdirs() }
        val destination = File(directory, "original.$extension")
        copyUri(uri, destination)
        val metadata = metadata(destination)
        return SongRecord(
            id = id,
            title = metadata.title ?: name.substringBeforeLast('.', name),
            artist = metadata.artist ?: "未知艺人",
            durationMs = metadata.durationMs,
            sourcePath = destination.absolutePath,
            stems = emptyList(),
            status = SongStatus.QUEUED,
            jobStage = "等待本地分轨",
            progress = 0
        )
    }

    fun importStems(uris: List<Uri>): SongRecord {
        require(uris.isNotEmpty()) { "请选择分轨文件" }
        val detected = uris.map { uri ->
            val name = displayName(uri) ?: uri.lastPathSegment.orEmpty()
            (StemType.detect(name) ?: error("无法识别分轨：$name")) to uri
        }
        val duplicates = detected.groupBy { it.first }.filterValues { it.size > 1 }.keys
        require(duplicates.isEmpty()) { "存在重复声部：${duplicates.joinToString { it.label }}" }
        val missing = StemType.entries - detected.map { it.first }.toSet()
        require(missing.isEmpty()) { "缺少分轨：${missing.joinToString { it.label }}" }

        return importStems(detected.toMap())
    }

    fun importStems(mapping: Map<StemType, Uri>): SongRecord {
        val missing = StemType.entries - mapping.keys
        require(missing.isEmpty() && mapping.size == StemType.entries.size) { "请为六个声部分别选择一个文件" }
        require(mapping.values.toSet().size == StemType.entries.size) { "同一个文件不能映射到多个声部" }

        val id = UUID.randomUUID().toString()
        val directory = File(songsRoot, "$id/stems").apply { mkdirs() }
        val stems = mapping.map { (type, uri) ->
            val destination = File(directory, type.fileName)
            copyUri(uri, destination)
            StemRecord(type, destination.absolutePath)
        }.sortedBy { it.type.ordinal }
        WaveformPeaks.generateFromAudioFiles(
            stems.map { File(it.path) },
            File(directory.parentFile, "peaks/mix.f32le")
        )
        val firstMetadata = metadata(File(stems.first().path))
        val firstName = displayName(mapping.values.first()).orEmpty()
        val inferredTitle = firstName.substringBeforeLast('.', firstName)
            .replace(Regex("(?i)[ _.-]*(vocals?|drums?|bass|guitar|piano|keys|other)$"), "")
            .ifBlank { "导入的分轨" }
        return SongRecord(
            id = id,
            title = firstMetadata.title ?: inferredTitle,
            artist = firstMetadata.artist ?: "未知艺人",
            durationMs = firstMetadata.durationMs,
            sourcePath = null,
            stems = stems,
            status = SongStatus.READY,
            jobStage = "可练习",
            progress = 100
        )
    }

    fun nameOf(uri: Uri): String = displayName(uri) ?: uri.lastPathSegment ?: "未命名音频"

    fun importLyrics(songId: String, uri: Uri): SongRecord {
        require(load().any { it.id == songId }) { "歌曲不存在" }
        val directory = File(songsRoot, "$songId/lyrics").apply { mkdirs() }
        val candidate = File(directory, "lyrics.import")
        val destination = File(directory, "lyrics.lrc")
        return try {
            copyUri(uri, candidate)
            require(LrcLyrics.parse(candidate).isNotEmpty()) { "LRC 中没有可用的时间轴歌词" }
            if (destination.exists()) destination.delete()
            check(candidate.renameTo(destination)) { "无法保存歌词文件" }
            updateSong(songId) { current -> current.copy(lyricsPath = destination.absolutePath) }
                ?: error("歌曲不存在")
        } finally {
            candidate.delete()
            File(directory, "${candidate.name}.part").delete()
        }
    }

    fun removeLyrics(songId: String): SongRecord? = updateSong(songId) { song ->
        song.lyricsPath?.let(::File)?.delete()
        song.copy(lyricsPath = null)
    }

    fun load(): List<SongRecord> = synchronized(storageLock) { loadUnlocked() }

    private fun loadUnlocked(): List<SongRecord> = runCatching {
        val array = JSONArray(preferences.getString("songs", "[]"))
        buildList {
            for (index in 0 until array.length()) {
                decodeSong(array.getJSONObject(index))?.let(::add)
            }
        }.sortedByDescending(SongRecord::importedAt)
    }.getOrDefault(emptyList())

    fun save(songs: List<SongRecord>) {
        synchronized(storageLock) { saveUnlocked(songs) }
    }

    private fun saveUnlocked(songs: List<SongRecord>) {
        val array = JSONArray()
        songs.forEach { array.put(encodeSong(it)) }
        preferences.edit().putString("songs", array.toString()).commit()
    }

    fun updateSong(id: String, transform: (SongRecord) -> SongRecord): SongRecord? = synchronized(storageLock) {
        val songs = loadUnlocked()
        val current = songs.firstOrNull { it.id == id } ?: return@synchronized null
        val updated = transform(current)
        saveUnlocked(songs.map { if (it.id == id) updated else it })
        updated
    }

    fun addSong(song: SongRecord) = synchronized(storageLock) {
        saveUnlocked(listOf(song) + loadUnlocked().filterNot { it.id == song.id })
    }

    fun delete(song: SongRecord) {
        synchronized(storageLock) {
            val directory = File(songsRoot, song.id).canonicalFile
            if (directory.parentFile == songsRoot.canonicalFile && directory.exists()) directory.deleteRecursively()
            saveUnlocked(loadUnlocked().filterNot { it.id == song.id })
        }
    }

    fun cleanupOrphans(): Long = synchronized(storageLock) {
        val songs = loadUnlocked()
        val referenced = songs.map(SongRecord::id).toSet()
        var removed = songsRoot.listFiles().orEmpty().filter { it.isDirectory && it.name !in referenced }.sumOf { directory ->
            val bytes = sizeOf(directory)
            val canonical = directory.canonicalFile
            if (canonical.parentFile == songsRoot.canonicalFile) canonical.deleteRecursively()
            bytes
        }
        songs.filter { it.status != SongStatus.PROCESSING && it.status != SongStatus.QUEUED }.forEach { song ->
            val root = File(songsRoot, song.id).canonicalFile
            if (root.parentFile == songsRoot.canonicalFile) {
                listOf(File(root, "stems.part"), File(root, "normalized")).forEach { temporary ->
                    removed += sizeOf(temporary)
                    if (temporary.exists()) temporary.deleteRecursively()
                }
            }
        }
        removed
    }

    private fun sizeOf(file: File): Long = when {
        !file.exists() -> 0
        file.isFile -> file.length()
        else -> file.walkTopDown().filter(File::isFile).sumOf(File::length)
    }

    private fun copyUri(uri: Uri, destination: File) {
        val temporary = File(destination.parentFile, "${destination.name}.part")
        context.contentResolver.openInputStream(uri)?.use { input ->
            temporary.outputStream().use(input::copyTo)
        } ?: error("无法读取所选文件")
        if (destination.exists()) destination.delete()
        check(temporary.renameTo(destination)) { "无法保存导入文件" }
    }

    private fun displayName(uri: Uri): String? = context.contentResolver
        .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }

    private data class AudioMetadata(val title: String?, val artist: String?, val durationMs: Long)

    private fun metadata(file: File): AudioMetadata = runCatching {
        MediaMetadataRetriever().use { retriever ->
            retriever.setDataSource(file.absolutePath)
            AudioMetadata(
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_TITLE),
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_ARTIST),
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0
            )
        }
    }.getOrDefault(AudioMetadata(null, null, 0))

    private fun encodeSong(song: SongRecord) = JSONObject().apply {
        put("id", song.id); put("title", song.title); put("artist", song.artist)
        put("durationMs", song.durationMs); put("sourcePath", song.sourcePath); put("lyricsPath", song.lyricsPath)
        put("status", song.status.name); put("jobStage", song.jobStage); put("progress", song.progress)
        put("error", song.error); put("favorite", song.favorite); put("importedAt", song.importedAt)
        song.processingStartedAt?.let { put("processingStartedAt", it) }
        put("stems", JSONArray().apply { song.stems.forEach { put(JSONObject().put("type", it.type.name).put("path", it.path)) } })
        put("practice", JSONObject().apply {
            put("positionMs", song.practice.positionMs); put("speed", song.practice.speed.toDouble())
            put("loopA", song.practice.loopA); put("loopB", song.practice.loopB)
            put("loopEnabled", song.practice.loopEnabled)
            put("metronomeEnabled", song.practice.metronomeEnabled)
            put("metronomeBpm", song.practice.metronomeBpm.toDouble())
            put("metronomeOffsetMs", song.practice.metronomeOffsetMs)
            put("metronomeVolume", song.practice.metronomeVolume.toDouble())
            put("metronomeAnalysisDone", song.practice.metronomeAnalysisDone)
            put("metronomeAnalysisVersion", song.practice.metronomeAnalysisVersion)
            put("metronomeConfidence", song.practice.metronomeConfidence.toDouble())
            put("countInBeats", song.practice.countInBeats)
            put("tracks", JSONObject().apply { song.practice.tracks.forEach { (type, state) ->
                put(type.name, JSONObject().put("muted", state.muted).put("solo", state.solo).put("volume", state.volume.toDouble()))
            } })
        })
    }

    private fun decodeSong(value: JSONObject): SongRecord? {
        val stems = buildList {
            val array = value.optJSONArray("stems") ?: JSONArray()
            for (index in 0 until array.length()) {
                val item = array.getJSONObject(index)
                val path = item.getString("path")
                if (File(path).isFile) add(StemRecord(StemType.valueOf(item.getString("type")), path))
            }
        }
        val sourcePath = value.optString("sourcePath").takeIf { it.isNotBlank() && it != "null" }
        val lyricsPath = value.optString("lyricsPath").takeIf { it.isNotBlank() && it != "null" && File(it).isFile }
        if (sourcePath == null && stems.isEmpty()) return null
        val practiceJson = value.optJSONObject("practice") ?: JSONObject()
        val tracksJson = practiceJson.optJSONObject("tracks") ?: JSONObject()
        val legacyMasterGain = practiceJson
            .optDouble("masterVolume", 1.0)
            .toFloat()
            .takeIf(Float::isFinite)
            ?.coerceIn(0f, 1f)
            ?: 1f
        val tracks = StemType.entries.associateWith { type ->
            val state = tracksJson.optJSONObject(type.name) ?: JSONObject()
            TrackState(
                muted = state.optBoolean("muted"),
                solo = state.optBoolean("solo"),
                volume = state.optDouble("volume", 1.0).toFloat() * legacyMasterGain,
            )
        }
        return SongRecord(
            id = value.getString("id"), title = value.getString("title"), artist = value.optString("artist", "未知艺人"),
            durationMs = value.optLong("durationMs"), sourcePath = sourcePath, stems = stems, lyricsPath = lyricsPath,
            status = if (stems.size == StemType.entries.size) SongStatus.READY else SongStatus.valueOf(value.optString("status", "QUEUED")),
            jobStage = value.optString("jobStage", "等待本地分轨"), progress = value.optInt("progress"),
            error = value.optString("error").takeIf { it.isNotBlank() && it != "null" }, favorite = value.optBoolean("favorite"),
            importedAt = value.optLong("importedAt", System.currentTimeMillis()),
            processingStartedAt = value.optLong("processingStartedAt").takeIf { it > 0 },
            practice = PracticeState(
                positionMs = practiceJson.optLong("positionMs"), speed = practiceJson.optDouble("speed", 1.0).toFloat(),
                loopA = practiceJson.optLong("loopA"), loopB = practiceJson.optLong("loopB"),
                loopEnabled = practiceJson.optBoolean("loopEnabled"),
                metronomeEnabled = practiceJson.optBoolean("metronomeEnabled"),
                metronomeBpm = practiceJson.optDouble("metronomeBpm", 120.0).toFloat(),
                metronomeOffsetMs = practiceJson.optLong("metronomeOffsetMs"),
                metronomeVolume = practiceJson.optDouble("metronomeVolume", 1.0).toFloat(),
                metronomeAnalysisDone = practiceJson.optBoolean("metronomeAnalysisDone"),
                metronomeAnalysisVersion = practiceJson.optInt("metronomeAnalysisVersion"),
                metronomeConfidence = practiceJson.optDouble("metronomeConfidence", 0.0).toFloat(),
                countInBeats = practiceJson.optInt("countInBeats"),
                tracks = tracks
            ).normalized()
        )
    }
}
