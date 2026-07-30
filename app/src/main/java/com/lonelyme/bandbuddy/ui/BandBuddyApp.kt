package com.lonelyme.bandbuddy.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.net.Uri
import android.os.IBinder
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.basicMarquee
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items as gridItems
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.lonelyme.bandbuddy.BuildConfig
import com.lonelyme.bandbuddy.data.LocalSongRepository
import com.lonelyme.bandbuddy.data.LrcLyrics
import com.lonelyme.bandbuddy.data.LyricLine
import com.lonelyme.bandbuddy.data.PracticeState
import com.lonelyme.bandbuddy.data.SongRecord
import com.lonelyme.bandbuddy.data.SongStatus
import com.lonelyme.bandbuddy.data.StemType
import com.lonelyme.bandbuddy.data.TrackState
import com.lonelyme.bandbuddy.data.withTrackState
import com.lonelyme.bandbuddy.engine.AudioDecoder
import com.lonelyme.bandbuddy.engine.BeatGridDetector
import com.lonelyme.bandbuddy.engine.ModelRuntime
import com.lonelyme.bandbuddy.engine.WaveformPeaks
import com.lonelyme.bandbuddy.model.ModelInstallPhase
import com.lonelyme.bandbuddy.model.ModelInstallSnapshot
import com.lonelyme.bandbuddy.model.ModelSpec
import com.lonelyme.bandbuddy.model.ModelStore
import com.lonelyme.bandbuddy.playback.PlaybackService
import com.lonelyme.bandbuddy.worker.ModelDownloadWorker
import com.lonelyme.bandbuddy.worker.SeparationWorker
import com.lonelyme.bandbuddy.ui.theme.BlueChip
import com.lonelyme.bandbuddy.ui.theme.Cream
import com.lonelyme.bandbuddy.ui.theme.GreenChip
import com.lonelyme.bandbuddy.ui.theme.Ink
import com.lonelyme.bandbuddy.ui.theme.Line
import com.lonelyme.bandbuddy.ui.theme.MutedInk
import com.lonelyme.bandbuddy.ui.theme.Paper
import com.lonelyme.bandbuddy.ui.theme.Sand
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.log10
import kotlin.math.roundToInt

internal enum class AppScreen { LIBRARY, PRACTICE, SETTINGS }
private enum class LibraryFilter(val label: String) { ALL("全部"), READY("可练习"), ACTIVE("处理中"), FAILED("失败"), FAVORITE("收藏") }
private val TabletBreakpoint = 600.dp
private val WidePracticeBreakpoint = 680.dp
private val TabletRailWidth = 92.dp

@Composable
fun BandBuddyApp(openSongId: String? = null, onOpenSongConsumed: () -> Unit = {}) {
    val context = LocalContext.current.applicationContext
    val repository = remember { LocalSongRepository(context) }
    val modelStore = remember { ModelStore(context) }
    val scope = rememberCoroutineScope()
    var songs by remember { mutableStateOf(repository.load()) }
    var screen by remember { mutableStateOf(AppScreen.LIBRARY) }
    var selectedId by remember { mutableStateOf(PlaybackService.lastSongId(context)) }
    var showImport by remember { mutableStateOf(false) }
    var pendingStemUris by remember { mutableStateOf<List<Uri>>(emptyList()) }
    var pendingSourceSong by remember { mutableStateOf<SongRecord?>(null) }
    var editingSong by remember { mutableStateOf<SongRecord?>(null) }
    var pendingLyricsSongId by remember { mutableStateOf<String?>(null) }
    var message by remember { mutableStateOf<String?>(null) }
    var modelState by remember { mutableStateOf(modelStore.snapshot()) }
    var runtimeStatus by remember { mutableStateOf<String?>(null) }

    fun replaceSong(updated: SongRecord) {
        repository.updateSong(updated.id) { updated }
        songs = repository.load()
    }

    fun addSong(song: SongRecord) {
        repository.addSong(song)
        songs = repository.load()
    }

    fun openLastPractice() {
        val rememberedId = PlaybackService.lastSongId(context)
        val target = songs.firstOrNull { it.id == rememberedId && it.status == SongStatus.READY }
            ?: songs.firstOrNull { it.status == SongStatus.READY }
        if (target == null) {
            message = "还没有可练习的歌曲"
        } else {
            selectedId = target.id
            screen = AppScreen.PRACTICE
        }
    }

    val sourcePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@rememberLauncherForActivityResult
        if (!modelStore.isCurrentReady()) {
            screen = AppScreen.SETTINGS
            message = "请先在设置中下载分轨模型"
            return@rememberLauncherForActivityResult
        }
        scope.launch {
            runCatching { withContext(Dispatchers.IO) { repository.importSource(uri) } }
                .onSuccess { pendingSourceSong = it }
                .onFailure { message = it.message ?: "导入失败" }
        }
    }
    val stemsPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isEmpty()) return@rememberLauncherForActivityResult
        if (uris.size != StemType.entries.size) message = "请选择恰好六个分轨文件"
        else pendingStemUris = uris
    }
    val lyricsPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val songId = pendingLyricsSongId
        pendingLyricsSongId = null
        if (uri == null || songId == null) return@rememberLauncherForActivityResult
        scope.launch {
            runCatching { withContext(Dispatchers.IO) { repository.importLyrics(songId, uri) } }
                .onSuccess { updated ->
                    songs = repository.load()
                    editingSong = updated
                    message = "LRC 歌词已导入"
                }
                .onFailure { message = it.message ?: "歌词导入失败" }
        }
    }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) { repository.cleanupOrphans() }
        var enqueuedForVersion: String? = null
        while (true) {
            val (refreshed, refreshedModel) = withContext(Dispatchers.IO) {
                repository.load() to modelStore.snapshot()
            }
            if (refreshed != songs) songs = refreshed
            modelState = refreshedModel
            if (refreshedModel.isReady) {
                if (runtimeStatus == null) {
                    runtimeStatus = withContext(Dispatchers.IO) {
                        runCatching { ModelRuntime(context).statusSummary() }
                            .getOrElse { "运行环境不可用 · ${it.message ?: it.javaClass.simpleName}" }
                    }
                }
                if (enqueuedForVersion != refreshedModel.targetVersion) {
                    refreshed
                        .filter { it.status == SongStatus.QUEUED && it.sourcePath != null }
                        .forEach { SeparationWorker.enqueue(context, it.id) }
                    enqueuedForVersion = refreshedModel.targetVersion
                }
            } else {
                runtimeStatus = null
                enqueuedForVersion = null
            }
            delay(700)
        }
    }
    LaunchedEffect(message) { if (message != null) { delay(3200); message = null } }
    LaunchedEffect(openSongId, songs) {
        val requested = openSongId?.let { id -> songs.firstOrNull { it.id == id && it.status == SongStatus.READY } }
        if (requested != null) {
            selectedId = requested.id
            screen = AppScreen.PRACTICE
            onOpenSongConsumed()
        }
    }

    val selected = songs.firstOrNull { it.id == selectedId }
    Surface(Modifier.fillMaxSize(), color = Paper) {
        BoxWithConstraints(Modifier.fillMaxSize()) {
            val isTablet = maxWidth >= TabletBreakpoint
            Box(Modifier.fillMaxSize()) {
                Row(Modifier.fillMaxSize()) {
                    if (isTablet) {
                        TabletNavigationRail(
                            selected = screen,
                            onLibrary = { screen = AppScreen.LIBRARY },
                            onPractice = ::openLastPractice,
                            onSettings = { screen = AppScreen.SETTINGS },
                        )
                    }
                    Box(Modifier.weight(1f).fillMaxHeight()) {
                        when (screen) {
                            AppScreen.LIBRARY -> LibraryScreen(
                                songs = songs,
                                isTablet = isTablet,
                                onImport = { showImport = true },
                                onOpen = { song ->
                                    if (song.status == SongStatus.READY) { selectedId = song.id; screen = AppScreen.PRACTICE }
                                    else message = song.error ?: song.jobStage
                                },
                                onFavorite = { replaceSong(it.copy(favorite = !it.favorite)) },
                                onEdit = { editingSong = it },
                                onDelete = { song ->
                                    SeparationWorker.cancel(context, song.id)
                                    PlaybackService.stopIfCurrent(context, song.id)
                                    scope.launch(Dispatchers.IO) { repository.delete(song) }
                                    songs = songs.filterNot { it.id == song.id }
                                },
                                onRetry = { song ->
                                    replaceSong(
                                        song.copy(
                                            status = SongStatus.QUEUED,
                                            jobStage = "等待本地分轨",
                                            progress = 0,
                                            error = null,
                                            processingStartedAt = null
                                        )
                                    )
                                    if (modelState.isReady) {
                                        SeparationWorker.enqueue(context, song.id, replace = true)
                                    } else {
                                        screen = AppScreen.SETTINGS
                                        message = "请先在设置中下载分轨模型"
                                    }
                                },
                                onCancel = { song ->
                                    SeparationWorker.cancel(context, song.id)
                                    replaceSong(
                                        song.copy(
                                            status = SongStatus.FAILED,
                                            jobStage = "已取消",
                                            error = "任务已取消",
                                            processingStartedAt = null
                                        )
                                    )
                                },
                                onSettings = { screen = AppScreen.SETTINGS },
                                onPractice = ::openLastPractice
                            )
                            AppScreen.PRACTICE -> if (selected != null) PracticeScreen(
                                song = selected,
                                isTablet = isTablet,
                                onBack = { screen = AppScreen.LIBRARY },
                                onUpdate = ::replaceSong,
                                onLibrary = { screen = AppScreen.LIBRARY }
                            ) else screen.also { screen = AppScreen.LIBRARY }
                            AppScreen.SETTINGS -> SettingsScreen(
                                model = modelState,
                                runtimeStatus = runtimeStatus,
                                songs = songs,
                                isTablet = isTablet,
                                onBack = { screen = AppScreen.LIBRARY },
                                onDownloadModel = {
                                    ModelDownloadWorker.enqueue(context)
                                    modelState = modelStore.snapshot()
                                    message = "已开始下载分轨模型"
                                },
                                onPauseModel = {
                                    ModelDownloadWorker.pause(context)
                                    modelState = modelStore.snapshot()
                                    message = "模型下载已暂停，可稍后继续"
                                },
                                onDeleteModel = {
                                    scope.launch {
                                        val removed = withContext(Dispatchers.IO) {
                                            ModelDownloadWorker.delete(context)
                                        }
                                        modelState = modelStore.snapshot()
                                        runtimeStatus = null
                                        message = "已删除分轨模型，释放 ${formatBytes(removed)}"
                                    }
                                },
                                onCleanup = {
                                    scope.launch {
                                        val removed = withContext(Dispatchers.IO) { repository.cleanupOrphans() }
                                        message = if (removed > 0) "已清理 ${formatBytes(removed)} 临时文件" else "没有需要清理的临时文件"
                                    }
                                }
                            )
                        }
                    }
                }
                message?.let {
                    ToastBanner(
                        text = it,
                        isTablet = isTablet,
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .padding(start = if (isTablet) TabletRailWidth else 0.dp),
                    )
                }
            }
        }
    }

    if (showImport) ImportDialog(
        onDismiss = { showImport = false },
        onSource = {
            showImport = false
            if (modelState.isReady) {
                sourcePicker.launch(arrayOf("audio/*"))
            } else {
                screen = AppScreen.SETTINGS
                message = "请先在设置中下载分轨模型"
            }
        },
        onStems = { showImport = false; stemsPicker.launch(arrayOf("audio/*")) }
    )
    if (pendingStemUris.isNotEmpty()) StemMappingDialog(
        uris = pendingStemUris,
        nameOf = repository::nameOf,
        onDismiss = { pendingStemUris = emptyList() },
        onConfirm = { mapping ->
            pendingStemUris = emptyList()
            scope.launch {
                runCatching { withContext(Dispatchers.IO) { repository.importStems(mapping) } }
                    .onSuccess { addSong(it); message = "六个分轨已导入，可以开始练习" }
                    .onFailure { message = it.message ?: "分轨导入失败" }
            }
        }
    )
    pendingSourceSong?.let { song ->
        EditSongDialog(
            song = song,
            heading = "确认歌曲信息",
            confirmLabel = "开始本地分轨",
            allowLyricsImport = false,
            onDismiss = {
                pendingSourceSong = null
                scope.launch(Dispatchers.IO) { repository.delete(song) }
            },
            onConfirm = { title, artist ->
                val confirmed = song.copy(title = title, artist = artist)
                pendingSourceSong = null
                addSong(confirmed)
                if (modelStore.isCurrentReady()) {
                    SeparationWorker.enqueue(context, confirmed.id)
                    message = "已加入本地分轨队列"
                } else {
                    screen = AppScreen.SETTINGS
                    message = "歌曲已加入队列，请先下载分轨模型"
                }
            }
        )
    }
    editingSong?.let { song ->
        EditSongDialog(
            song = song,
            onDismiss = { editingSong = null },
            onImportLyrics = {
                pendingLyricsSongId = song.id
                lyricsPicker.launch(arrayOf("text/plain", "text/*", "application/octet-stream"))
            },
            onRemoveLyrics = {
                repository.removeLyrics(song.id)?.let { updated ->
                    songs = repository.load()
                    editingSong = updated
                    message = "歌词已移除"
                }
            },
            onConfirm = { title, artist ->
                replaceSong(song.copy(title = title, artist = artist))
                editingSong = null
            }
        )
    }
}

@Composable
internal fun LibraryScreen(
    songs: List<SongRecord>,
    isTablet: Boolean,
    onImport: () -> Unit,
    onOpen: (SongRecord) -> Unit,
    onFavorite: (SongRecord) -> Unit,
    onEdit: (SongRecord) -> Unit,
    onDelete: (SongRecord) -> Unit,
    onRetry: (SongRecord) -> Unit,
    onCancel: (SongRecord) -> Unit,
    onSettings: () -> Unit,
    onPractice: () -> Unit
) {
    var query by remember { mutableStateOf("") }
    var filter by remember { mutableStateOf(LibraryFilter.ALL) }
    val searchKeys = remember(songs) { songs.associate { it.id to PinyinSearch.keyOf(it) } }
    val filtered = songs.filter { song ->
        PinyinSearch.matches(searchKeys.getValue(song.id), query) && when (filter) {
            LibraryFilter.ALL -> true
            LibraryFilter.READY -> song.status == SongStatus.READY
            LibraryFilter.ACTIVE -> song.status == SongStatus.PROCESSING || song.status == SongStatus.QUEUED
            LibraryFilter.FAILED -> song.status == SongStatus.FAILED
            LibraryFilter.FAVORITE -> song.favorite
        }
    }
    if (isTablet) {
        LazyVerticalGrid(
            columns = GridCells.Adaptive(minSize = 300.dp),
            modifier = Modifier.fillMaxSize().statusBarsPadding(),
            contentPadding = PaddingValues(horizontal = 32.dp, vertical = 24.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item(span = { GridItemSpan(maxLineSpan) }) {
                Column {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                "曲库",
                                fontFamily = FontFamily.Serif,
                                fontSize = 36.sp,
                                lineHeight = 40.sp,
                                fontWeight = FontWeight.Black,
                            )
                            Text(
                                "整理歌曲、分轨与下一次排练",
                                color = MutedInk,
                                fontSize = 13.sp,
                                modifier = Modifier.padding(top = 5.dp),
                            )
                        }
                        Button(
                            onClick = onImport,
                            modifier = Modifier.height(48.dp),
                            shape = RoundedCornerShape(15.dp),
                            contentPadding = PaddingValues(horizontal = 20.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Ink),
                        ) {
                            Text("＋  导入歌曲", fontSize = 13.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                    Spacer(Modifier.height(26.dp))
                    OutlinedTextField(
                        value = query, onValueChange = { query = it }, singleLine = true,
                        placeholder = { Text("搜索歌曲、艺人或拼音", fontSize = 13.sp) },
                        leadingIcon = { Text("⌕", fontSize = 22.sp, color = MutedInk) },
                        shape = RoundedCornerShape(18.dp),
                        modifier = Modifier.widthIn(max = 680.dp).fillMaxWidth().heightIn(min = 56.dp),
                    )
                    Spacer(Modifier.height(14.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                        LibraryFilter.entries.forEach { item ->
                            MiniButton(item.label, filter == item) { filter = item }
                        }
                    }
                    Spacer(Modifier.height(34.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("我的歌曲", fontSize = 20.sp, fontWeight = FontWeight.Bold)
                        Spacer(Modifier.weight(1f))
                        Text("${filtered.size} 首", fontSize = 12.sp, color = MutedInk)
                    }
                }
            }
            if (filtered.isEmpty()) {
                item(span = { GridItemSpan(maxLineSpan) }) {
                    EmptyLibrary(hasQuery = query.isNotBlank(), onImport = onImport)
                }
            }
            gridItems(filtered, key = SongRecord::id) { song ->
                SongCard(song, onOpen, onFavorite, onEdit, onDelete, onRetry, onCancel)
            }
        }
    } else {
        Column(Modifier.fillMaxSize().statusBarsPadding()) {
            LazyColumn(
                Modifier.weight(1f),
                contentPadding = PaddingValues(horizontal = 26.dp, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                item {
                    BrandRow(onSettings)
                    Spacer(Modifier.height(18.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        OutlinedTextField(
                            value = query, onValueChange = { query = it }, singleLine = true,
                            placeholder = { Text("搜索歌曲、艺人或拼音", fontSize = 13.sp) },
                            leadingIcon = { Text("⌕", fontSize = 22.sp, color = MutedInk) },
                            shape = RoundedCornerShape(18.dp),
                            modifier = Modifier.weight(1f).heightIn(min = 56.dp)
                        )
                        Spacer(Modifier.width(10.dp))
                        Button(
                            onClick = onImport, modifier = Modifier.height(48.dp), shape = RoundedCornerShape(15.dp),
                            contentPadding = PaddingValues(horizontal = 15.dp), colors = ButtonDefaults.buttonColors(containerColor = Ink)
                        ) { Text("＋  导入", fontSize = 13.sp, fontWeight = FontWeight.Bold) }
                    }
                    Spacer(Modifier.height(14.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        LibraryFilter.entries.forEach { item -> MiniButton(item.label, filter == item) { filter = item } }
                    }
                    Spacer(Modifier.height(38.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("我的歌曲", fontSize = 19.sp, fontWeight = FontWeight.Bold)
                        Spacer(Modifier.weight(1f)); Text("${filtered.size} 首", fontSize = 12.sp, color = MutedInk)
                    }
                    Spacer(Modifier.height(13.dp))
                }
                if (filtered.isEmpty()) item { EmptyLibrary(hasQuery = query.isNotBlank(), onImport = onImport) }
                items(filtered, key = SongRecord::id) { song -> SongCard(song, onOpen, onFavorite, onEdit, onDelete, onRetry, onCancel) }
            }
            BottomBar(AppScreen.LIBRARY, onLibrary = {}, onPractice = onPractice)
        }
    }
}

@Composable
private fun BrandRow(onSettings: () -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        AppIcon(Modifier.size(34.dp))
        Spacer(Modifier.width(9.dp)); Text("BandBuddy", fontSize = 21.sp, fontFamily = FontFamily.Serif, fontWeight = FontWeight.Black)
        Spacer(Modifier.weight(1f))
        Text("⚙", fontSize = 21.sp, color = MutedInk, modifier = Modifier.clip(CircleShape).clickable(onClick = onSettings).padding(8.dp))
    }
}

@Composable
private fun EmptyLibrary(hasQuery: Boolean, onImport: () -> Unit) {
    Column(Modifier.fillMaxWidth().padding(top = 48.dp, bottom = 80.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        AppIcon(Modifier.size(92.dp))
        Spacer(Modifier.height(24.dp))
        Text(if (hasQuery) "没有匹配的歌曲" else "曲库还是空的", fontSize = 18.sp, fontWeight = FontWeight.Bold)
        Text(if (hasQuery) "换个关键词试试" else "导入一首歌曲，或直接导入六个已有分轨", color = MutedInk, fontSize = 13.sp, modifier = Modifier.padding(top = 7.dp))
        if (!hasQuery) TextButton(onClick = onImport, modifier = Modifier.padding(top = 10.dp)) { Text("开始导入", color = Ink, fontWeight = FontWeight.Bold) }
    }
}

@Composable
private fun SongCard(
    song: SongRecord,
    onOpen: (SongRecord) -> Unit,
    onFavorite: (SongRecord) -> Unit,
    onEdit: (SongRecord) -> Unit,
    onDelete: (SongRecord) -> Unit,
    onRetry: (SongRecord) -> Unit,
    onCancel: (SongRecord) -> Unit
) {
    var menu by remember { mutableStateOf(false) }
    Card(
        modifier = Modifier.fillMaxWidth().clickable { onOpen(song) },
        shape = RoundedCornerShape(17.dp), colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(Modifier.padding(horizontal = 13.dp, vertical = 11.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                RecordArt(song.title.take(1), Modifier.size(43.dp), accentFor(song.id))
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(song.title, fontWeight = FontWeight.Bold, fontSize = 15.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(song.artist, color = MutedInk, fontSize = 12.sp, maxLines = 1)
                }
                Text(formatDuration(song.durationMs), color = MutedInk, fontSize = 12.sp)
                Spacer(Modifier.width(8.dp))
                Text(if (song.favorite) "♥" else "♡", color = if (song.favorite) Color(0xFF9F503F) else MutedInk,
                    modifier = Modifier.clickable { onFavorite(song) }.padding(5.dp))
                Text("⋯", color = MutedInk, fontSize = 18.sp, modifier = Modifier.clickable { menu = !menu }.padding(5.dp))
            }
            if (song.status != SongStatus.READY) {
                Row(Modifier.padding(start = 55.dp, top = 7.dp), verticalAlignment = Alignment.CenterVertically) {
                    StatusPill(song)
                    if (song.progress > 0) {
                        Spacer(Modifier.width(9.dp)); LinearProgressIndicator(
                            progress = { song.progress / 100f }, modifier = Modifier.weight(1f).height(2.dp), color = Ink, trackColor = Line
                        )
                    }
                }
            }
            if (menu) {
                HorizontalDivider(Modifier.padding(top = 9.dp), color = Line)
                Row(Modifier.align(Alignment.End)) {
                    TextButton(onClick = { menu = false; onEdit(song) }) { Text("编辑信息", color = Ink, fontSize = 12.sp) }
                    if (song.status == SongStatus.FAILED && song.sourcePath != null) TextButton(onClick = { menu = false; onRetry(song) }) { Text("重试", color = Ink, fontSize = 12.sp) }
                    if (song.status == SongStatus.PROCESSING || song.status == SongStatus.QUEUED) TextButton(onClick = { menu = false; onCancel(song) }) { Text("取消任务", color = MutedInk, fontSize = 12.sp) }
                    TextButton(onClick = { menu = false; onDelete(song) }) { Text("从本机删除", color = Color(0xFF9F3E36), fontSize = 12.sp) }
                }
            }
        }
    }
}

@Composable
private fun StatusPill(song: SongRecord) {
    val (background, foreground) = if (song.status == SongStatus.FAILED) Color(0xFFF7E6E3) to Color(0xFF9F3E36) else Sand to MutedInk
    val status = when (song.status) {
        SongStatus.PROCESSING -> {
            val percent = song.progress.coerceIn(1, 99)
            "$percent% · ${separationEta(song, percent)}"
        }
        else -> song.error ?: song.jobStage
    }
    Text(status, maxLines = 1, fontSize = 10.sp, color = foreground,
        modifier = Modifier.clip(RoundedCornerShape(8.dp)).background(background).padding(horizontal = 8.dp, vertical = 4.dp))
}

@Composable
private fun PracticeScreen(
    song: SongRecord,
    isTablet: Boolean,
    onBack: () -> Unit,
    onUpdate: (SongRecord) -> Unit,
    onLibrary: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var engine by remember(song.id) { mutableStateOf<PlaybackService?>(null) }
    var practice by remember(song.id) { mutableStateOf(song.practice) }
    val latestPractice by rememberUpdatedState(practice)
    var position by remember(song.id) { mutableLongStateOf(song.practice.positionMs) }
    var loaded by remember(song.id) { mutableStateOf(false) }
    var playing by remember(song.id) { mutableStateOf(false) }
    var songPlaying by remember(song.id) { mutableStateOf(false) }
    var countInRemaining by remember(song.id) { mutableIntStateOf(0) }
    var loadError by remember { mutableStateOf<String?>(null) }
    var beatAnalysisRunning by remember(song.id) { mutableStateOf(false) }
    var waveform by remember(song.id) { mutableStateOf(FloatArray(0)) }
    var lyrics by remember(song.id) { mutableStateOf<List<LyricLine>>(emptyList()) }

    fun update(next: PracticeState) {
        practice = next
        engine?.updatePractice(next)
        onUpdate(song.copy(practice = next.copy(positionMs = position)))
    }

    fun requestBeatAnalysis() {
        if (beatAnalysisRunning) return
        beatAnalysisRunning = true
        scope.launch {
            val result = try {
                withContext(Dispatchers.IO) {
                    val drumSource = song.stems.firstOrNull { it.type == StemType.DRUMS }
                        ?.path
                        ?.let(::File)
                        ?.takeIf(File::isFile)
                    val bassSource = song.stems.firstOrNull { it.type == StemType.BASS }
                        ?.path
                        ?.let(::File)
                        ?.takeIf(File::isFile)
                    val sources = buildList {
                        listOf(StemType.DRUMS, StemType.BASS, StemType.OTHER, StemType.GUITAR).forEach { type ->
                            song.stems.firstOrNull { it.type == type }
                                ?.path
                                ?.let(::File)
                                ?.takeIf(File::isFile)
                                ?.let(::add)
                        }
                        song.sourcePath
                            ?.let(::File)
                            ?.takeIf(File::isFile)
                            ?.let(::add)
                    }.distinctBy { it.absolutePath }
                    if (sources.isEmpty()) error("没有可用于节拍分析的音频")
                    val analysisDirectory = File(context.applicationContext.cacheDir, "beat-analysis").apply { mkdirs() }
                    fun cleanDecoded(file: File) {
                        file.delete()
                        File(analysisDirectory, "${file.name}.decoded").delete()
                        File(analysisDirectory, "${file.name}.part").delete()
                    }
                    val jointResult = if (drumSource != null && bassSource != null) {
                        val drumsFile = File(analysisDirectory, "${song.id}-drums.f32le")
                        val bassFile = File(analysisDirectory, "${song.id}-bass.f32le")
                        try {
                            runCatching {
                                val decoder = AudioDecoder()
                                val drums = decoder.decode(drumSource, drumsFile, maxDurationSeconds = 180)
                                val bass = decoder.decode(bassSource, bassFile, maxDurationSeconds = 180)
                                BeatGridDetector.detectFromDrumsAndBassPcm(
                                    drumsFile = drums.file,
                                    drumsFrames = drums.frames,
                                    bassFile = bass.file,
                                    bassFrames = bass.frames
                                )
                            }.getOrNull()
                        } finally {
                            cleanDecoded(drumsFile)
                            cleanDecoded(bassFile)
                        }
                    } else {
                        null
                    }
                    jointResult ?: sources.firstNotNullOfOrNull { source ->
                        val decodedFile = File(analysisDirectory, "${song.id}-fallback.f32le")
                        try {
                            runCatching {
                                val decoded = AudioDecoder().decode(source, decodedFile, maxDurationSeconds = 180)
                                BeatGridDetector.detectFromStereoFloatPcm(decoded.file, decoded.frames)
                            }.getOrNull()
                        } finally {
                            cleanDecoded(decodedFile)
                        }
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: Throwable) {
                null
            }
            val current = latestPractice
            update(
                result?.let { analysis ->
                    current.copy(
                        metronomeBpm = analysis.bpm,
                        metronomeOffsetMs = analysis.beatOffsetMs,
                        metronomeAnalysisDone = true,
                        metronomeAnalysisVersion = BeatGridDetector.ANALYSIS_VERSION,
                        metronomeConfidence = analysis.confidence
                    )
                } ?: current.copy(
                    metronomeAnalysisDone = true,
                    metronomeAnalysisVersion = BeatGridDetector.ANALYSIS_VERSION,
                    metronomeConfidence = 0f
                )
            )
            beatAnalysisRunning = false
        }
    }

    DisposableEffect(song.id) {
        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                engine = (binder as? PlaybackService.PlaybackBinder)?.service
            }
            override fun onServiceDisconnected(name: ComponentName?) {
                engine = null
                loaded = false
                songPlaying = false
            }
        }
        context.bindService(Intent(context, PlaybackService::class.java), connection, Context.BIND_AUTO_CREATE)
        onDispose {
            runCatching { context.unbindService(connection) }
            engine = null
        }
    }
    LaunchedEffect(engine, song.id) {
        val service = engine ?: return@LaunchedEffect
        runCatching { withContext(Dispatchers.IO) { service.load(song) } }
            .onSuccess { loaded = service.isLoaded; service.updatePractice(practice) }
            .onFailure { loadError = it.message ?: "播放器加载失败" }
    }
    LaunchedEffect(song.id) { waveform = withContext(Dispatchers.IO) { WaveformPeaks.load(song) } }
    LaunchedEffect(song.id) {
        if (latestPractice.metronomeAnalysisVersion < BeatGridDetector.ANALYSIS_VERSION) requestBeatAnalysis()
    }
    LaunchedEffect(song.lyricsPath) {
        lyrics = withContext(Dispatchers.IO) {
            song.lyricsPath?.let(::File)?.takeIf(File::isFile)?.let { runCatching { LrcLyrics.parse(it) }.getOrDefault(emptyList()) }
                ?: emptyList()
        }
    }
    LaunchedEffect(engine, loaded, playing) {
        val service = engine ?: return@LaunchedEffect
        while (service.isLoaded) {
            position = service.positionMs
            playing = service.isPlayingOrCountingIn
            songPlaying = service.isPlaying
            countInRemaining = service.countInRemaining
            delay(150)
        }
    }

    val playbackDuration = engine?.durationMs?.takeIf { it > 0 } ?: song.durationMs
    val onSeek: (Long) -> Unit = { target ->
        position = target
        engine?.seekTo(target)
    }
    val onPlay: () -> Unit = {
        engine?.let { if (it.isPlayingOrCountingIn) it.pause() else it.play() }
    }
    val onSkipBack: () -> Unit = {
        engine?.seekTo((position - 5000).coerceAtLeast(0))
    }
    val onSkipForward: () -> Unit = {
        engine?.seekTo((position + 5000).coerceAtMost(engine?.durationMs ?: song.durationMs))
    }

    if (isTablet) {
        BoxWithConstraints(Modifier.fillMaxSize().statusBarsPadding()) {
            val useTwoPane = maxWidth >= WidePracticeBreakpoint
            Column(Modifier.fillMaxSize()) {
                if (useTwoPane) {
                    Row(
                        modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 28.dp, vertical = 20.dp),
                        horizontalArrangement = Arrangement.spacedBy(26.dp),
                    ) {
                        LazyColumn(
                            modifier = Modifier.weight(.9f).fillMaxHeight(),
                            contentPadding = PaddingValues(bottom = 20.dp),
                        ) {
                            item {
                                PracticeHero(
                                    song = song,
                                    lyrics = lyrics,
                                    position = position,
                                    duration = playbackDuration,
                                    waveform = waveform,
                                    playing = playing,
                                    loadError = loadError,
                                    showBack = false,
                                    expanded = true,
                                    onBack = onBack,
                                    onFavorite = { onUpdate(song.copy(favorite = !song.favorite)) },
                                    onSeek = onSeek,
                                )
                            }
                        }
                        LazyColumn(
                            modifier = Modifier.weight(1.1f).fillMaxHeight(),
                            contentPadding = PaddingValues(bottom = 20.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            item { MixerHeading(expanded = true) }
                            items(StemType.entries) { type ->
                                TrackRow(type, practice.tracks[type] ?: TrackState()) { state ->
                                    update(practice.withTrackState(type, state))
                                }
                            }
                            item {
                                Spacer(Modifier.height(8.dp))
                                LoopAndSpeed(
                                    practice = practice,
                                    position = position,
                                    beatAnalysisRunning = beatAnalysisRunning,
                                    isSongPlaying = songPlaying,
                                    onDetectBeatGrid = ::requestBeatAnalysis,
                                    onPreviewMetronome = { engine?.previewMetronome() },
                                    readPlaybackPosition = {
                                        engine?.takeIf(PlaybackService::isPlaying)?.positionMs
                                    },
                                    onChange = ::update,
                                )
                            }
                        }
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.weight(1f),
                        contentPadding = PaddingValues(horizontal = 30.dp, vertical = 16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        item {
                            PracticeHero(
                                song = song,
                                lyrics = lyrics,
                                position = position,
                                duration = playbackDuration,
                                waveform = waveform,
                                playing = playing,
                                loadError = loadError,
                                showBack = false,
                                expanded = false,
                                onBack = onBack,
                                onFavorite = { onUpdate(song.copy(favorite = !song.favorite)) },
                                onSeek = onSeek,
                            )
                            Spacer(Modifier.height(20.dp))
                            MixerHeading(expanded = true)
                        }
                        items(StemType.entries) { type ->
                            TrackRow(type, practice.tracks[type] ?: TrackState()) { state ->
                                update(practice.withTrackState(type, state))
                            }
                        }
                        item {
                            Spacer(Modifier.height(10.dp))
                            LoopAndSpeed(
                                practice = practice,
                                position = position,
                                beatAnalysisRunning = beatAnalysisRunning,
                                isSongPlaying = songPlaying,
                                onDetectBeatGrid = ::requestBeatAnalysis,
                                onPreviewMetronome = { engine?.previewMetronome() },
                                readPlaybackPosition = {
                                    engine?.takeIf(PlaybackService::isPlaying)?.positionMs
                                },
                                onChange = ::update,
                            )
                            Spacer(Modifier.height(12.dp))
                        }
                    }
                }
                TransportBar(
                    song = song,
                    loaded = loaded,
                    playing = playing,
                    position = position,
                    countInRemaining = countInRemaining,
                    expanded = true,
                    onPlay = onPlay,
                    onBack = onSkipBack,
                    onForward = onSkipForward,
                )
            }
        }
    } else {
        Column(Modifier.fillMaxSize().statusBarsPadding()) {
            LazyColumn(
                Modifier.weight(1f),
                contentPadding = PaddingValues(horizontal = 24.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                item {
                    PracticeHero(
                        song = song,
                        lyrics = lyrics,
                        position = position,
                        duration = playbackDuration,
                        waveform = waveform,
                        playing = playing,
                        loadError = loadError,
                        showBack = true,
                        expanded = false,
                        onBack = onBack,
                        onFavorite = { onUpdate(song.copy(favorite = !song.favorite)) },
                        onSeek = onSeek,
                    )
                    Spacer(Modifier.height(19.dp))
                    MixerHeading(expanded = false)
                    Spacer(Modifier.height(5.dp))
                }
                items(StemType.entries) { type ->
                    TrackRow(type, practice.tracks[type] ?: TrackState()) { state ->
                        update(practice.withTrackState(type, state))
                    }
                }
                item {
                    Spacer(Modifier.height(12.dp))
                    LoopAndSpeed(
                        practice = practice,
                        position = position,
                        beatAnalysisRunning = beatAnalysisRunning,
                        isSongPlaying = songPlaying,
                        onDetectBeatGrid = ::requestBeatAnalysis,
                        onPreviewMetronome = { engine?.previewMetronome() },
                        readPlaybackPosition = {
                            engine?.takeIf(PlaybackService::isPlaying)?.positionMs
                        },
                        onChange = ::update,
                    )
                    Spacer(Modifier.height(12.dp))
                }
            }
            TransportBar(
                song = song,
                loaded = loaded,
                playing = playing,
                position = position,
                countInRemaining = countInRemaining,
                expanded = false,
                onPlay = onPlay,
                onBack = onSkipBack,
                onForward = onSkipForward,
            )
            BottomBar(AppScreen.PRACTICE, onLibrary = onLibrary, onPractice = {})
        }
    }
}

@Composable
private fun PracticeHero(
    song: SongRecord,
    lyrics: List<LyricLine>,
    position: Long,
    duration: Long,
    waveform: FloatArray,
    playing: Boolean,
    loadError: String?,
    showBack: Boolean,
    expanded: Boolean,
    onBack: () -> Unit,
    onFavorite: () -> Unit,
    onSeek: (Long) -> Unit,
) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        if (showBack) {
            Text("‹", fontSize = 35.sp, modifier = Modifier.clickable(onClick = onBack).padding(end = 8.dp))
        } else {
            Text("练习室", color = MutedInk, fontSize = 12.sp, fontWeight = FontWeight.Bold)
        }
        Spacer(Modifier.weight(1f))
        Text(
            if (song.favorite) "♥" else "♡",
            color = if (song.favorite) Color(0xFF9F503F) else MutedInk,
            fontSize = 18.sp,
            modifier = Modifier.clickable(onClick = onFavorite).padding(6.dp),
        )
    }
    Spacer(Modifier.height(if (expanded) 18.dp else 16.dp))
    if (expanded) {
        Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
            AnimatedRecordArt(song.title.take(1), Modifier.size(148.dp), accentFor(song.id), playing)
            Text(
                song.title,
                fontFamily = FontFamily.Serif,
                fontSize = 34.sp,
                lineHeight = 37.sp,
                fontWeight = FontWeight.Black,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.padding(top = 18.dp),
            )
            Text(song.artist, color = MutedInk, fontSize = 15.sp, modifier = Modifier.padding(top = 5.dp))
            LyricsMarquee(lyrics, position, Modifier.padding(top = 13.dp))
        }
    } else {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(
                    song.title,
                    fontFamily = FontFamily.Serif,
                    fontSize = 30.sp,
                    lineHeight = 33.sp,
                    fontWeight = FontWeight.Black,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(song.artist, color = MutedInk, fontSize = 15.sp, modifier = Modifier.padding(top = 5.dp))
                LyricsMarquee(lyrics, position, Modifier.padding(top = 13.dp))
            }
            AnimatedRecordArt(song.title.take(1), Modifier.size(112.dp), accentFor(song.id), playing)
        }
    }
    Spacer(Modifier.height(if (expanded) 32.dp else 28.dp))
    RealTimeline(position, duration, waveform, onSeek)
    loadError?.let {
        Text(it, color = Color(0xFF9F3E36), fontSize = 12.sp, modifier = Modifier.padding(top = 8.dp))
    }
}

@Composable
private fun MixerHeading(expanded: Boolean) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column {
            Text("混音器", fontWeight = FontWeight.Bold, fontSize = if (expanded) 20.sp else 17.sp)
            if (expanded) {
                Text("逐轨调整你的排练声场", color = MutedInk, fontSize = 10.sp, modifier = Modifier.padding(top = 2.dp))
            }
        }
        Spacer(Modifier.weight(1f))
        Text("M 静音 · S 独奏", fontSize = 10.sp, color = MutedInk)
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun LyricsMarquee(lines: List<LyricLine>, positionMs: Long, modifier: Modifier = Modifier) {
    val current = lines.lastOrNull { it.timestampMs <= positionMs }?.text
        ?: if (lines.isEmpty()) "♪ 暂无歌词，可在曲库的编辑信息中导入 LRC" else "♪ 前奏"
    Text(
        text = current,
        color = MutedInk,
        fontSize = 11.sp,
        maxLines = 1,
        modifier = modifier
            .fillMaxWidth()
            .basicMarquee(iterations = Int.MAX_VALUE)
    )
}

@Composable
private fun AnimatedRecordArt(
    letter: String,
    modifier: Modifier,
    accent: Color,
    playing: Boolean
) {
    val rotation = remember { Animatable(0f) }
    val lightTransition = rememberInfiniteTransition(label = "vinyl-light")
    val lightPulse by lightTransition.animateFloat(
        initialValue = .35f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 850),
            repeatMode = RepeatMode.Reverse
        ),
        label = "vinyl-light-pulse"
    )
    LaunchedEffect(playing) {
        if (playing) {
            while (true) {
                rotation.animateTo(
                    targetValue = rotation.value + 360f,
                    animationSpec = tween(durationMillis = 3_800, easing = LinearEasing)
                )
                rotation.snapTo(rotation.value % 360f)
            }
        }
    }

    Box(modifier, contentAlignment = Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            if (playing) {
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            accent.copy(alpha = .46f * lightPulse),
                            accent.copy(alpha = .12f * lightPulse),
                            Color.Transparent
                        ),
                        center = center,
                        radius = size.minDimension / 2
                    ),
                    radius = size.minDimension / 2
                )
                drawCircle(
                    color = accent.copy(alpha = .25f * lightPulse),
                    radius = size.minDimension * (.43f + .035f * lightPulse),
                    style = Stroke(width = 2.5f)
                )
            }
        }
        RecordArt(
            letter = letter,
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp)
                .graphicsLayer { rotationZ = rotation.value },
            accent = accent
        )
        Canvas(Modifier.fillMaxSize().padding(11.dp)) {
            drawArc(
                color = Color.White.copy(alpha = if (playing) .38f * lightPulse else .16f),
                startAngle = 212f,
                sweepAngle = 42f,
                useCenter = false,
                topLeft = Offset(size.width * .16f, size.height * .16f),
                size = Size(size.width * .68f, size.height * .68f),
                style = Stroke(width = 2.2f, cap = StrokeCap.Round)
            )
        }
    }
}

@Composable
private fun RealTimeline(position: Long, duration: Long, waveform: FloatArray, onSeek: (Long) -> Unit) {
    Column {
        Box(Modifier.fillMaxWidth().height(40.dp), contentAlignment = Alignment.Center) {
            Canvas(Modifier.fillMaxWidth().height(28.dp)) {
                if (waveform.isEmpty()) {
                    drawLine(Line, Offset(0f, size.height / 2), Offset(size.width, size.height / 2), 1f)
                } else {
                    val progress = if (duration > 0) position.toFloat() / duration.toFloat() else 0f
                    var x = 0f
                    while (x < size.width) {
                        val index = (x / size.width * waveform.size).toInt().coerceIn(waveform.indices)
                        val half = (waveform[index].coerceIn(.03f, 1f) * size.height * .46f)
                        drawLine(if (x / size.width <= progress) Ink else Line, Offset(x, size.height / 2 - half), Offset(x, size.height / 2 + half), 1.5f)
                        x += 3f
                    }
                }
            }
            Slider(
                value = if (duration > 0) position.toFloat().coerceIn(0f, duration.toFloat()) else 0f,
                onValueChange = { onSeek(it.toLong()) }, valueRange = 0f..duration.coerceAtLeast(1).toFloat(),
                colors = SliderDefaults.colors(thumbColor = Ink, activeTrackColor = Color.Transparent, inactiveTrackColor = Color.Transparent),
                modifier = Modifier.fillMaxWidth().height(36.dp)
            )
        }
        Row(Modifier.fillMaxWidth()) { Text(formatDuration(position), fontSize = 10.sp, color = MutedInk); Spacer(Modifier.weight(1f)); Text(formatDuration(duration), fontSize = 10.sp, color = MutedInk) }
    }
}

@Composable
private fun TrackRow(type: StemType, state: TrackState, onChange: (TrackState) -> Unit) {
    val color = when (type) { StemType.VOCALS -> Cream; StemType.DRUMS -> BlueChip; StemType.BASS -> GreenChip; StemType.GUITAR -> Color(0xFFFFE9D8); StemType.PIANO -> Color(0xFFF0E9F5); StemType.OTHER -> Sand }
    Card(shape = RoundedCornerShape(14.dp), colors = CardDefaults.cardColors(containerColor = Color.White)) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 11.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            StemIcon(type, color, Modifier.size(31.dp))
            Spacer(Modifier.width(9.dp)); Text(type.label, fontSize = 13.sp, fontWeight = FontWeight.Bold, modifier = Modifier.width(39.dp))
            TinyToggle("M", state.muted) {
                onChange(state.copy(muted = !state.muted, solo = false))
            }
            Spacer(Modifier.width(5.dp)); TinyToggle("S", state.solo) {
                onChange(state.copy(solo = !state.solo, muted = false))
            }
            PaddleVolumeSlider(
                value = state.volume,
                onValueChange = { onChange(state.copy(volume = it)) },
                muted = state.muted,
                solo = state.solo,
                contentDescription = "${type.label}音量",
                modifier = Modifier.weight(1f).height(38.dp),
            )
            Text(gainLabel(state.volume), fontSize = 8.sp, color = MutedInk, modifier = Modifier.width(34.dp))
        }
    }
}

@Composable
private fun StemIcon(type: StemType, background: Color, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier.clip(RoundedCornerShape(9.dp)).background(background),
        contentAlignment = Alignment.Center
    ) {
        Canvas(Modifier.size(20.dp)) {
            val stroke = 1.6.dp.toPx()
            val lineStyle = Stroke(width = stroke, cap = StrokeCap.Round)
            when (type) {
                StemType.VOCALS -> {
                    drawRoundRect(
                        color = Ink,
                        topLeft = Offset(size.width * .36f, size.height * .08f),
                        size = Size(size.width * .28f, size.height * .52f),
                        cornerRadius = CornerRadius(size.width * .14f),
                        style = lineStyle
                    )
                    drawArc(
                        color = Ink,
                        startAngle = 0f,
                        sweepAngle = 180f,
                        useCenter = false,
                        topLeft = Offset(size.width * .25f, size.height * .28f),
                        size = Size(size.width * .5f, size.height * .43f),
                        style = lineStyle
                    )
                    drawLine(Ink, Offset(size.width * .5f, size.height * .71f), Offset(size.width * .5f, size.height * .9f), stroke, StrokeCap.Round)
                    drawLine(Ink, Offset(size.width * .36f, size.height * .9f), Offset(size.width * .64f, size.height * .9f), stroke, StrokeCap.Round)
                }
                StemType.DRUMS -> {
                    drawOval(
                        color = Ink,
                        topLeft = Offset(size.width * .18f, size.height * .32f),
                        size = Size(size.width * .64f, size.height * .46f),
                        style = lineStyle
                    )
                    drawLine(Ink, Offset(size.width * .24f, size.height * .78f), Offset(size.width * .18f, size.height * .92f), stroke, StrokeCap.Round)
                    drawLine(Ink, Offset(size.width * .76f, size.height * .78f), Offset(size.width * .82f, size.height * .92f), stroke, StrokeCap.Round)
                    drawLine(Ink, Offset(size.width * .18f, size.height * .08f), Offset(size.width * .58f, size.height * .43f), stroke, StrokeCap.Round)
                    drawLine(Ink, Offset(size.width * .82f, size.height * .08f), Offset(size.width * .42f, size.height * .43f), stroke, StrokeCap.Round)
                }
                StemType.BASS -> {
                    drawCircle(Ink, radius = size.width * .22f, center = Offset(size.width * .33f, size.height * .68f), style = lineStyle)
                    drawCircle(Ink, radius = size.width * .13f, center = Offset(size.width * .54f, size.height * .58f), style = lineStyle)
                    drawLine(Ink, Offset(size.width * .46f, size.height * .51f), Offset(size.width * .82f, size.height * .13f), stroke * 1.5f, StrokeCap.Round)
                    drawLine(Ink, Offset(size.width * .74f, size.height * .18f), Offset(size.width * .9f, size.height * .08f), stroke, StrokeCap.Round)
                    drawLine(Ink, Offset(size.width * .25f, size.height * .67f), Offset(size.width * .77f, size.height * .16f), stroke * .55f, StrokeCap.Round)
                }
                StemType.GUITAR -> {
                    drawCircle(Ink, radius = size.width * .2f, center = Offset(size.width * .34f, size.height * .7f), style = lineStyle)
                    drawCircle(Ink, radius = size.width * .16f, center = Offset(size.width * .53f, size.height * .55f), style = lineStyle)
                    drawCircle(Ink, radius = size.width * .055f, center = Offset(size.width * .43f, size.height * .62f), style = lineStyle)
                    drawLine(Ink, Offset(size.width * .55f, size.height * .47f), Offset(size.width * .82f, size.height * .13f), stroke * 1.4f, StrokeCap.Round)
                    drawLine(Ink, Offset(size.width * .77f, size.height * .16f), Offset(size.width * .91f, size.height * .07f), stroke, StrokeCap.Round)
                }
                StemType.PIANO -> {
                    drawRoundRect(
                        color = Ink,
                        topLeft = Offset(size.width * .08f, size.height * .24f),
                        size = Size(size.width * .84f, size.height * .55f),
                        cornerRadius = CornerRadius(size.width * .05f),
                        style = lineStyle
                    )
                    for (index in 1..4) {
                        val x = size.width * (.08f + .84f * index / 5f)
                        drawLine(Ink, Offset(x, size.height * .5f), Offset(x, size.height * .79f), stroke * .65f)
                    }
                    listOf(.22f, .39f, .72f).forEach { x ->
                        drawRect(Ink, topLeft = Offset(size.width * x, size.height * .24f), size = Size(size.width * .075f, size.height * .26f))
                    }
                }
                StemType.OTHER -> {
                    listOf(.25f, .5f, .75f).forEachIndexed { index, y ->
                        drawLine(Ink, Offset(size.width * .12f, size.height * y), Offset(size.width * .88f, size.height * y), stroke, StrokeCap.Round)
                        val knobX = listOf(.36f, .68f, .48f)[index]
                        drawCircle(Ink, radius = stroke * 1.45f, center = Offset(size.width * knobX, size.height * y))
                    }
                }
            }
        }
    }
}

@Composable
private fun TinyToggle(label: String, active: Boolean, onClick: () -> Unit) {
    Box(Modifier.size(27.dp).clip(RoundedCornerShape(7.dp)).background(if (active) Ink else Paper).border(1.dp, if (active) Ink else Line, RoundedCornerShape(7.dp)).clickable(onClick = onClick), contentAlignment = Alignment.Center) {
        Text(label, fontSize = 10.sp, color = if (active) Color.White else Ink, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun LoopAndSpeed(
    practice: PracticeState,
    position: Long,
    beatAnalysisRunning: Boolean,
    isSongPlaying: Boolean,
    onDetectBeatGrid: () -> Unit,
    onPreviewMetronome: () -> Unit,
    readPlaybackPosition: () -> Long?,
    onChange: (PracticeState) -> Unit
) {
    var showMetronomeSettings by remember { mutableStateOf(false) }
    Card(shape = RoundedCornerShape(15.dp), colors = CardDefaults.cardColors(containerColor = Color.White)) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("循环", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(10.dp))
                MiniButton("设 A", practice.loopA > 0) {
                    onChange(practice.copy(loopA = position, loopB = practice.loopB.takeIf { it > position } ?: 0, loopEnabled = practice.loopEnabled && practice.loopB > position))
                }
                Spacer(Modifier.width(5.dp)); MiniButton("设 B", practice.loopB > practice.loopA) {
                    if (position > practice.loopA) onChange(practice.copy(loopB = position))
                }
                Spacer(Modifier.width(5.dp)); MiniButton(if (practice.loopEnabled) "已开启" else "开启", practice.loopEnabled) {
                    if (practice.loopB > practice.loopA) onChange(practice.copy(loopEnabled = !practice.loopEnabled))
                }
                Spacer(Modifier.weight(1f)); Text(if (practice.loopB > practice.loopA) "${formatDuration(practice.loopA)}–${formatDuration(practice.loopB)}" else "未设置", fontSize = 9.sp, color = MutedInk)
            }
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("速度", fontSize = 12.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.width(12.dp))
                listOf(.5f, .8f, 1f, 1.2f, 1.5f).forEach { speed ->
                    val label = if (speed == 1f) "1x" else "${speed}x"
                    MiniButton(label, practice.speed == speed) { onChange(practice.copy(speed = speed)) }
                    if (speed != 1.5f) Spacer(Modifier.width(5.dp))
                }
            }
            Spacer(Modifier.height(13.dp))
            HorizontalDivider(color = Line)
            Spacer(Modifier.height(11.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                MetronomeIcon(Modifier.size(17.dp), if (practice.metronomeEnabled) Ink else MutedInk)
                Spacer(Modifier.width(6.dp))
                Text("节拍器", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                Text(
                    "${formatBpm(practice.metronomeBpm)} BPM",
                    fontSize = 9.sp,
                    color = MutedInk,
                    modifier = Modifier.padding(start = 8.dp)
                )
                Spacer(Modifier.weight(1f))
                MiniButton(if (practice.metronomeEnabled) "关闭" else "开启", practice.metronomeEnabled) {
                    onChange(practice.copy(metronomeEnabled = !practice.metronomeEnabled))
                }
                Spacer(Modifier.width(6.dp))
                MiniButton("设置", showMetronomeSettings) {
                    showMetronomeSettings = true
                }
            }
        }
    }
    if (showMetronomeSettings) {
        MetronomeSettingsDialog(
            practice = practice,
            beatAnalysisRunning = beatAnalysisRunning,
            isSongPlaying = isSongPlaying,
            onDetectBeatGrid = onDetectBeatGrid,
            onPreview = onPreviewMetronome,
            readPlaybackPosition = readPlaybackPosition,
            onChange = onChange,
            onDismiss = { showMetronomeSettings = false }
        )
    }
}

@Composable
private fun MiniButton(text: String, active: Boolean, onClick: () -> Unit) {
    Text(text, fontSize = 10.sp, color = if (active) Color.White else Ink, fontWeight = FontWeight.Medium,
        modifier = Modifier.clip(RoundedCornerShape(8.dp)).background(if (active) Ink else Sand).clickable(onClick = onClick).padding(horizontal = 9.dp, vertical = 6.dp))
}

@Composable
private fun TransportBar(
    song: SongRecord,
    loaded: Boolean,
    playing: Boolean,
    position: Long,
    countInRemaining: Int,
    expanded: Boolean,
    onPlay: () -> Unit,
    onBack: () -> Unit,
    onForward: () -> Unit
) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(Ink)
            .padding(horizontal = if (expanded) 32.dp else 20.dp, vertical = if (expanded) 13.dp else 11.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RecordArt(song.title.take(1), Modifier.size(if (expanded) 42.dp else 36.dp), accentFor(song.id))
        Spacer(Modifier.width(if (expanded) 12.dp else 9.dp))
        Column(Modifier.weight(1f).widthIn(max = 360.dp)) {
            Text(song.title, color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1)
            Text(
                if (countInRemaining > 0) "预备拍 · $countInRemaining" else formatDuration(position),
                color = Color(0xFFBFB8B0),
                fontSize = 9.sp
            )
        }
        if (expanded) {
            Text("后退 5 秒", color = Color(0xFFDDD5CC), fontSize = 10.sp)
        }
        Text("−5", color = Color.White, fontSize = 11.sp, modifier = Modifier.clickable(onClick = onBack).padding(10.dp))
        Box(
            Modifier
                .size(if (expanded) 46.dp else 42.dp)
                .clip(CircleShape)
                .background(Paper)
                .clickable(enabled = loaded, onClick = onPlay),
            contentAlignment = Alignment.Center,
        ) {
            Text(if (playing) "Ⅱ" else "▶", color = Ink, fontSize = 16.sp)
        }
        Text("+5", color = Color.White, fontSize = 11.sp, modifier = Modifier.clickable(onClick = onForward).padding(10.dp))
        if (expanded) {
            Text("前进 5 秒", color = Color(0xFFDDD5CC), fontSize = 10.sp)
        }
    }
}

@Composable
private fun SettingsScreen(
    model: ModelInstallSnapshot,
    runtimeStatus: String?,
    songs: List<SongRecord>,
    isTablet: Boolean,
    onBack: () -> Unit,
    onDownloadModel: () -> Unit,
    onPauseModel: () -> Unit,
    onDeleteModel: () -> Unit,
    onCleanup: () -> Unit,
) {
    val context = LocalContext.current
    var showLicenses by remember { mutableStateOf(false) }
    var showDisclaimer by remember { mutableStateOf(false) }
    var confirmModelDelete by remember { mutableStateOf(false) }
    val storageBytes = songs.sumOf { song ->
        (song.sourcePath?.let { File(it).length() } ?: 0) +
            (song.lyricsPath?.let { File(it).length() } ?: 0) +
            song.stems.sumOf { File(it.path).length() }
    }
    val separationRunning = songs.any { it.status == SongStatus.PROCESSING }
    if (isTablet) {
        BoxWithConstraints(Modifier.fillMaxSize().statusBarsPadding()) {
            val useTwoPane = maxWidth >= 660.dp
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 32.dp, vertical = 24.dp),
            ) {
                item {
                    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.TopCenter) {
                        Column(Modifier.widthIn(max = 1040.dp).fillMaxWidth()) {
                            Text(
                                "设置",
                                fontFamily = FontFamily.Serif,
                                fontSize = 36.sp,
                                lineHeight = 40.sp,
                                fontWeight = FontWeight.Black,
                            )
                            Text(
                                "管理本地模型、存储与应用信息",
                                color = MutedInk,
                                fontSize = 13.sp,
                                modifier = Modifier.padding(top = 5.dp),
                            )
                            Spacer(Modifier.height(30.dp))
                            if (useTwoPane) {
                                Row(
                                    Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(24.dp),
                                    verticalAlignment = Alignment.Top,
                                ) {
                                    Column(Modifier.weight(.9f)) {
                                        Text("设备与模型", fontSize = 18.sp, fontWeight = FontWeight.Bold)
                                        Text(
                                            "分轨能力保存在这台设备上",
                                            color = MutedInk,
                                            fontSize = 10.sp,
                                            modifier = Modifier.padding(top = 3.dp, bottom = 12.dp),
                                        )
                                        ModelManagementCard(
                                            model = model,
                                            runtimeStatus = runtimeStatus,
                                            canDelete = !separationRunning,
                                            onDownload = onDownloadModel,
                                            onPause = onPauseModel,
                                            onDelete = { confirmModelDelete = true },
                                        )
                                        OutlinedButton(
                                            onClick = onCleanup,
                                            shape = RoundedCornerShape(12.dp),
                                            contentPadding = PaddingValues(horizontal = 14.dp, vertical = 7.dp),
                                            modifier = Modifier.padding(top = 18.dp),
                                        ) {
                                            Text("清理临时文件", color = Ink, fontSize = 11.sp)
                                        }
                                        Text(
                                            "BandBuddy ${BuildConfig.VERSION_NAME} · 本地六轨练习机",
                                            color = MutedInk,
                                            fontSize = 10.sp,
                                            modifier = Modifier.padding(top = 28.dp, bottom = 20.dp),
                                        )
                                    }
                                    Column(Modifier.weight(1.1f)) {
                                        Text("应用信息", fontSize = 18.sp, fontWeight = FontWeight.Bold)
                                        Text(
                                            "隐私、音频引擎与许可",
                                            color = MutedInk,
                                            fontSize = 10.sp,
                                            modifier = Modifier.padding(top = 3.dp, bottom = 12.dp),
                                        )
                                        SettingsInformation(
                                            songsCount = songs.size,
                                            storageBytes = storageBytes,
                                            inCard = true,
                                            onShowDisclaimer = { showDisclaimer = true },
                                            onShowLicenses = { showLicenses = true },
                                        )
                                    }
                                }
                            } else {
                                Text("设备与模型", fontSize = 18.sp, fontWeight = FontWeight.Bold)
                                Spacer(Modifier.height(12.dp))
                                ModelManagementCard(
                                    model = model,
                                    runtimeStatus = runtimeStatus,
                                    canDelete = !separationRunning,
                                    onDownload = onDownloadModel,
                                    onPause = onPauseModel,
                                    onDelete = { confirmModelDelete = true },
                                )
                                Spacer(Modifier.height(26.dp))
                                Text("应用信息", fontSize = 18.sp, fontWeight = FontWeight.Bold)
                                Spacer(Modifier.height(8.dp))
                                SettingsInformation(
                                    songsCount = songs.size,
                                    storageBytes = storageBytes,
                                    inCard = true,
                                    onShowDisclaimer = { showDisclaimer = true },
                                    onShowLicenses = { showLicenses = true },
                                )
                                OutlinedButton(
                                    onClick = onCleanup,
                                    shape = RoundedCornerShape(12.dp),
                                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 7.dp),
                                    modifier = Modifier.padding(top = 18.dp),
                                ) {
                                    Text("清理临时文件", color = Ink, fontSize = 11.sp)
                                }
                                Text(
                                    "BandBuddy ${BuildConfig.VERSION_NAME} · 本地六轨练习机",
                                    color = MutedInk,
                                    fontSize = 10.sp,
                                    modifier = Modifier.padding(top = 28.dp, bottom = 20.dp),
                                )
                            }
                        }
                    }
                }
            }
        }
    } else {
        LazyColumn(
            Modifier.fillMaxSize().statusBarsPadding(),
            contentPadding = PaddingValues(horizontal = 27.dp, vertical = 16.dp),
        ) {
            item {
                Text("‹", fontSize = 35.sp, modifier = Modifier.clickable(onClick = onBack))
                Spacer(Modifier.height(28.dp))
                Text("设置", fontFamily = FontFamily.Serif, fontSize = 35.sp, fontWeight = FontWeight.Black)
                Spacer(Modifier.height(28.dp))
                ModelManagementCard(
                    model = model,
                    runtimeStatus = runtimeStatus,
                    canDelete = !separationRunning,
                    onDownload = onDownloadModel,
                    onPause = onPauseModel,
                    onDelete = { confirmModelDelete = true },
                )
                Spacer(Modifier.height(16.dp))
                SettingsInformation(
                    songsCount = songs.size,
                    storageBytes = storageBytes,
                    inCard = false,
                    onShowDisclaimer = { showDisclaimer = true },
                    onShowLicenses = { showLicenses = true },
                )
                OutlinedButton(
                    onClick = onCleanup,
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 7.dp),
                    modifier = Modifier.padding(top = 18.dp)
                ) { Text("清理临时文件", color = Ink, fontSize = 11.sp) }
                Spacer(Modifier.height(44.dp))
                Text(
                    "BandBuddy ${BuildConfig.VERSION_NAME} · 本地六轨练习机",
                    color = MutedInk,
                    fontSize = 11.sp,
                    modifier = Modifier.padding(bottom = 20.dp),
                )
            }
        }
    }
    if (confirmModelDelete) {
        AlertDialog(
            onDismissRequest = { confirmModelDelete = false },
            containerColor = Paper,
            shape = RoundedCornerShape(22.dp),
            title = { Text("删除分轨模型？", fontFamily = FontFamily.Serif, fontWeight = FontWeight.Black) },
            text = { Text("会释放 ${formatBytes(ModelSpec.BYTES)} 左右空间。以后仍可从 ModelScope 重新下载。", color = MutedInk) },
            confirmButton = {
                TextButton(onClick = {
                    confirmModelDelete = false
                    onDeleteModel()
                }) { Text("删除", color = Color(0xFF9D3D31), fontWeight = FontWeight.Bold) }
            },
            dismissButton = {
                TextButton(onClick = { confirmModelDelete = false }) { Text("取消", color = MutedInk) }
            },
        )
    }
    if (showLicenses) {
        val notices = remember { context.assets.open("THIRD_PARTY_NOTICES.txt").bufferedReader().use { it.readText() } }
        AlertDialog(
            onDismissRequest = { showLicenses = false },
            containerColor = Paper,
            shape = RoundedCornerShape(22.dp),
            title = { Text("开源许可", fontFamily = FontFamily.Serif, fontWeight = FontWeight.Black, fontSize = 22.sp) },
            text = { LazyColumn(Modifier.height(410.dp)) { item { Text(notices, color = MutedInk, fontSize = 10.sp, lineHeight = 15.sp) } } },
            confirmButton = { TextButton(onClick = { showLicenses = false }) { Text("完成", color = Ink, fontWeight = FontWeight.Bold) } }
        )
    }
    if (showDisclaimer) {
        UsageDisclaimerDialog(onDismiss = { showDisclaimer = false })
    }
}

@Composable
private fun SettingsInformation(
    songsCount: Int,
    storageBytes: Long,
    inCard: Boolean,
    onShowDisclaimer: () -> Unit,
    onShowLicenses: () -> Unit,
) {
    val content: @Composable () -> Unit = {
        SettingRow("本地曲库", "$songsCount 首 · ${formatBytes(storageBytes)}")
        SettingRow("隐私", "仅联网下载模型；歌曲、分轨和练习数据不会上传")
        SettingRow("音频引擎", "六轨同步校正 · 44.1 kHz · 锁屏与蓝牙控制")
        SettingRow("分轨说明", "吉他和钢琴为实验性声部，复杂编曲可能出现串音")
        SettingRow("使用与免责", "版权授权 · 分轨误差 · 听力安全 · 数据保管", onShowDisclaimer)
        SettingRow("开源许可", "Demucs · LiteRT · Qualcomm QNN · AndroidX", onShowLicenses)
    }
    if (inCard) {
        Card(
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(18.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(Modifier.padding(horizontal = 18.dp, vertical = 4.dp)) {
                content()
            }
        }
    } else {
        content()
    }
}

@Composable
private fun ModelManagementCard(
    model: ModelInstallSnapshot,
    runtimeStatus: String?,
    canDelete: Boolean,
    onDownload: () -> Unit,
    onPause: () -> Unit,
    onDelete: () -> Unit,
) {
    val status = when (model.phase) {
        ModelInstallPhase.NOT_INSTALLED -> "尚未下载"
        ModelInstallPhase.UPDATE_AVAILABLE -> "已安装 ${model.installedVersion}，可更新"
        ModelInstallPhase.QUEUED -> model.error ?: "等待网络"
        ModelInstallPhase.DOWNLOADING -> "正在下载 ${formatBytes(model.downloadedBytes)} / ${formatBytes(model.totalBytes)}"
        ModelInstallPhase.VERIFYING -> "下载完成，正在校验完整性"
        ModelInstallPhase.PAUSED -> "已暂停于 ${formatBytes(model.downloadedBytes)}，可断点续传"
        ModelInstallPhase.FAILED -> model.error ?: "下载失败"
        ModelInstallPhase.READY -> runtimeStatus ?: "模型已安装"
    }
    Card(
        colors = CardDefaults.cardColors(containerColor = Color.White),
        shape = RoundedCornerShape(18.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("分轨模型", fontSize = 14.sp, fontWeight = FontWeight.Bold)
            Text(
                "HTDemucs 6-Stem · v${model.targetVersion} · ${formatBytes(model.totalBytes)}",
                color = MutedInk,
                fontSize = 10.sp,
                modifier = Modifier.padding(top = 5.dp),
            )
            Text(
                status,
                color = if (model.phase == ModelInstallPhase.FAILED) Color(0xFF9D3D31) else MutedInk,
                fontSize = 11.sp,
                modifier = Modifier.padding(top = 8.dp),
            )
            if (model.isActive || model.downloadedBytes > 0L && !model.isReady) {
                LinearProgressIndicator(
                    progress = { model.progress },
                    modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                    color = Ink,
                    trackColor = Sand,
                )
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(9.dp),
                modifier = Modifier.padding(top = 13.dp),
            ) {
                when {
                    model.isActive -> OutlinedButton(
                        onClick = onPause,
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 7.dp),
                    ) { Text("暂停", color = Ink, fontSize = 11.sp) }
                    model.isReady -> OutlinedButton(
                        onClick = onDelete,
                        enabled = canDelete,
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 7.dp),
                    ) { Text(if (canDelete) "删除模型" else "分轨中不可删除", fontSize = 11.sp) }
                    else -> Button(
                        onClick = onDownload,
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 7.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Ink),
                    ) {
                        Text(
                            if (model.downloadedBytes > 0L) "继续下载" else "下载模型",
                            color = Color.White,
                            fontSize = 11.sp,
                        )
                    }
                }
            }
            Text("来源：ModelScope · 固定版本 ${ModelSpec.REVISION}", color = MutedInk, fontSize = 9.sp)
        }
    }
}

@Composable
private fun SettingRow(title: String, value: String, onClick: (() -> Unit)? = null) {
    val rowModifier = if (onClick == null) Modifier else Modifier.clickable(onClick = onClick)
    Column(rowModifier.fillMaxWidth().padding(vertical = 12.dp)) { Text(title, fontSize = 14.sp, fontWeight = FontWeight.Bold); Text(value, fontSize = 11.sp, color = MutedInk, modifier = Modifier.padding(top = 5.dp)); HorizontalDivider(Modifier.padding(top = 12.dp), color = Line) }
}

@Composable
private fun ImportDialog(onDismiss: () -> Unit, onSource: () -> Unit, onStems: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss, containerColor = Paper, shape = RoundedCornerShape(22.dp),
        title = { Text("导入到曲库", fontFamily = FontFamily.Serif, fontWeight = FontWeight.Black, fontSize = 22.sp) },
        text = { Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            ImportOption("导入歌曲", "MP3、M4A、WAV 或 FLAC；随后在本机分轨", onSource)
            ImportOption("导入已有六轨", "选择 vocals / drums / bass / guitar / piano / other", onStems)
        } },
        confirmButton = {}, dismissButton = { TextButton(onClick = onDismiss) { Text("取消", color = MutedInk) } }
    )
}

@Composable
private fun ImportOption(title: String, detail: String, onClick: () -> Unit) {
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Color.White).clickable(onClick = onClick).padding(13.dp)) {
        Text(title, fontSize = 14.sp, fontWeight = FontWeight.Bold); Text(detail, fontSize = 10.sp, color = MutedInk, modifier = Modifier.padding(top = 4.dp))
    }
}

@Composable
private fun StemMappingDialog(
    uris: List<Uri>,
    nameOf: (Uri) -> String,
    onDismiss: () -> Unit,
    onConfirm: (Map<StemType, Uri>) -> Unit
) {
    val names = remember(uris) { uris.associateWith(nameOf) }
    val automatic = remember(uris) {
        uris.mapNotNull { uri -> StemType.detect(names.getValue(uri))?.let { it to uri } }
            .groupBy({ it.first }, { it.second })
            .filterValues { it.size == 1 }
            .mapValues { it.value.single() }
    }
    var mapping by remember(uris) { mutableStateOf(automatic) }
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Paper,
        shape = RoundedCornerShape(22.dp),
        title = { Text("确认六轨映射", fontFamily = FontFamily.Serif, fontWeight = FontWeight.Black, fontSize = 22.sp) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("已按文件名预填；点选任一行可手动更改。", color = MutedInk, fontSize = 10.sp, modifier = Modifier.padding(bottom = 5.dp))
                StemType.entries.forEach { type ->
                    var expanded by remember(type, uris) { mutableStateOf(false) }
                    Box {
                        Row(
                            Modifier.fillMaxWidth().clip(RoundedCornerShape(11.dp)).background(Color.White)
                                .clickable { expanded = true }.padding(horizontal = 11.dp, vertical = 9.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(type.label, fontWeight = FontWeight.Bold, fontSize = 12.sp, modifier = Modifier.width(45.dp))
                            Text(mapping[type]?.let(names::getValue) ?: "选择文件", color = if (mapping[type] == null) Color(0xFF9F3E36) else MutedInk,
                                fontSize = 10.sp, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                            Text("⌄", color = MutedInk, fontSize = 13.sp)
                        }
                        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                            uris.forEach { uri ->
                                val usedByAnother = mapping.any { (mappedType, mappedUri) -> mappedType != type && mappedUri == uri }
                                DropdownMenuItem(
                                    text = { Text(names.getValue(uri), fontSize = 11.sp, maxLines = 1) },
                                    enabled = !usedByAnother,
                                    onClick = { mapping = mapping.filterValues { it != uri } + (type to uri); expanded = false }
                                )
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(enabled = mapping.size == StemType.entries.size && mapping.values.toSet().size == StemType.entries.size, onClick = { onConfirm(mapping) }) {
                Text("导入六轨", color = if (mapping.size == StemType.entries.size) Ink else MutedInk, fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消", color = MutedInk) } }
    )
}

@Composable
private fun EditSongDialog(
    song: SongRecord,
    heading: String = "编辑歌曲信息",
    confirmLabel: String = "保存",
    allowLyricsImport: Boolean = true,
    onDismiss: () -> Unit,
    onImportLyrics: (() -> Unit)? = null,
    onRemoveLyrics: (() -> Unit)? = null,
    onConfirm: (String, String) -> Unit
) {
    var title by remember(song.id) { mutableStateOf(song.title) }
    var artist by remember(song.id) { mutableStateOf(song.artist) }
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Paper,
        shape = RoundedCornerShape(22.dp),
        title = { Text(heading, fontFamily = FontFamily.Serif, fontWeight = FontWeight.Black, fontSize = 22.sp) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(title, { title = it }, label = { Text("歌名") }, singleLine = true, shape = RoundedCornerShape(13.dp), modifier = Modifier.fillMaxWidth())
                OutlinedTextField(artist, { artist = it }, label = { Text("艺人") }, singleLine = true, shape = RoundedCornerShape(13.dp), modifier = Modifier.fillMaxWidth())
                if (allowLyricsImport && onImportLyrics != null) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        OutlinedButton(
                            onClick = onImportLyrics,
                            shape = RoundedCornerShape(12.dp),
                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 7.dp)
                        ) {
                            Text(if (song.lyricsPath == null) "从文件导入 LRC 歌词" else "重新导入 LRC 歌词", color = Ink, fontSize = 11.sp)
                        }
                        if (song.lyricsPath != null && onRemoveLyrics != null) {
                            TextButton(onClick = onRemoveLyrics) { Text("移除歌词", color = MutedInk, fontSize = 11.sp) }
                        }
                    }
                    Text(
                        if (song.lyricsPath == null) "支持带时间轴的 .lrc 文件" else "已导入，练习室将按播放进度滚动显示",
                        color = MutedInk,
                        fontSize = 9.sp
                    )
                }
            }
        },
        confirmButton = { TextButton(enabled = title.isNotBlank(), onClick = { onConfirm(title.trim(), artist.trim().ifBlank { "未知艺人" }) }) { Text(confirmLabel, color = Ink, fontWeight = FontWeight.Bold) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消", color = MutedInk) } }
    )
}

@Composable
internal fun TabletNavigationRail(
    selected: AppScreen,
    onLibrary: () -> Unit,
    onPractice: () -> Unit,
    onSettings: () -> Unit,
) {
    Column(
        modifier = Modifier
            .width(TabletRailWidth)
            .fillMaxHeight()
            .statusBarsPadding()
            .background(Ink)
            .navigationBarsPadding()
            .padding(horizontal = 10.dp, vertical = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        AppIcon(Modifier.size(40.dp))
        Text(
            "BandBuddy",
            color = Color(0xFFF7F0E7),
            fontFamily = FontFamily.Serif,
            fontSize = 11.sp,
            fontWeight = FontWeight.Black,
            modifier = Modifier.padding(top = 7.dp),
        )
        Spacer(Modifier.height(34.dp))
        TabletRailItem(
            icon = "●",
            label = "曲库",
            selected = selected == AppScreen.LIBRARY,
            onClick = onLibrary,
        )
        Spacer(Modifier.height(10.dp))
        TabletRailItem(
            icon = "♫",
            label = "练习室",
            selected = selected == AppScreen.PRACTICE,
            onClick = onPractice,
        )
        Spacer(Modifier.weight(1f))
        TabletRailItem(
            icon = "⚙",
            label = "设置",
            selected = selected == AppScreen.SETTINGS,
            onClick = onSettings,
        )
    }
}

@Composable
private fun TabletRailItem(
    icon: String,
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    val foreground = if (selected) Ink else Color(0xFFC9C0B7)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(if (selected) Paper else Color.Transparent)
            .clickable(onClick = onClick)
            .padding(vertical = 11.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(icon, color = foreground, fontSize = 18.sp)
        Text(
            label,
            color = foreground,
            fontSize = 10.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
            modifier = Modifier.padding(top = 3.dp),
        )
    }
}

@Composable
private fun BottomBar(selected: AppScreen, onLibrary: () -> Unit, onPractice: () -> Unit) {
    val dark = selected == AppScreen.PRACTICE
    val background = if (dark) Ink else Paper
    val border = if (dark) Color(0xFF393633) else Line
    Row(Modifier.fillMaxWidth().background(background).border(1.dp, border).navigationBarsPadding().padding(vertical = 7.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
        NavItem("●", "曲库", selected == AppScreen.LIBRARY, dark, onLibrary)
        NavItem("♫", "练习室", selected == AppScreen.PRACTICE, dark, onPractice)
    }
}

@Composable
private fun NavItem(icon: String, label: String, selected: Boolean, dark: Boolean, onClick: () -> Unit) {
    val selectedColor = if (dark) Color(0xFFF8F2E8) else Ink
    val normalColor = if (dark) Color(0xFFBFB8B0) else MutedInk
    Column(Modifier.clip(RoundedCornerShape(12.dp)).clickable(onClick = onClick).padding(horizontal = 38.dp, vertical = 2.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(icon, fontSize = 17.sp, color = if (selected) selectedColor else normalColor)
        Text(label, fontSize = 10.sp, color = if (selected) selectedColor else normalColor, fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal)
    }
}

@Composable
private fun ToastBanner(text: String, isTablet: Boolean, modifier: Modifier) {
    Text(
        text,
        color = Color.White,
        fontSize = 12.sp,
        modifier = modifier
            .navigationBarsPadding()
            .padding(bottom = if (isTablet) 22.dp else 70.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(Ink)
            .padding(horizontal = 15.dp, vertical = 10.dp),
    )
}

@Composable
private fun RecordArt(letter: String, modifier: Modifier, accent: Color) {
    Box(modifier, contentAlignment = Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            val radius = size.minDimension / 2
            drawCircle(Color(0xFF181715), radius)
            listOf(.42f, .60f, .78f).forEach { drawCircle(Color(0xFF393733), radius * it, style = Stroke(1f)) }
            drawCircle(accent, radius * .34f); drawCircle(Paper, radius * .055f)
        }
        if (letter.isNotBlank()) Text(letter.uppercase(), fontSize = 9.sp, fontWeight = FontWeight.Black, color = Ink)
    }
}

@Composable
private fun accentFor(id: String): Color = listOf(Color(0xFFD9B06F), Color(0xFF98B9C3), Color(0xFFAA9AC1), Color(0xFFA9B68C), Color(0xFFC99373))[kotlin.math.abs(id.hashCode()) % 5]
private fun separationEta(song: SongRecord, percent: Int): String {
    val startedAt = song.processingStartedAt ?: return "正在估算剩余时间"
    val elapsedMs = (System.currentTimeMillis() - startedAt).coerceAtLeast(0)
    if (elapsedMs < 5_000 || percent < 2) return "正在估算剩余时间"
    val remainingMs = (elapsedMs.toDouble() * (100 - percent) / percent)
        .toLong()
        .coerceIn(1_000, 24 * 60 * 60 * 1_000L)
    if (remainingMs < 60_000) return "预计还需不到 1 分钟"
    val minutes = (remainingMs + 59_999) / 60_000
    return if (minutes < 60) {
        "预计还需约 $minutes 分钟"
    } else {
        val hours = minutes / 60
        val rest = minutes % 60
        if (rest == 0L) "预计还需约 $hours 小时" else "预计还需约 $hours 小时 $rest 分钟"
    }
}
private fun formatBpm(bpm: Float): String =
    if (kotlin.math.abs(bpm - bpm.roundToInt()) < .05f) bpm.roundToInt().toString() else "%.1f".format(bpm)
private fun formatDuration(milliseconds: Long): String { val seconds = (milliseconds.coerceAtLeast(0) / 1000); return "%d:%02d".format(seconds / 60, seconds % 60) }
private fun formatBytes(bytes: Long): String = when { bytes >= 1_073_741_824 -> "%.1f GB".format(bytes / 1_073_741_824.0); bytes >= 1_048_576 -> "%.0f MB".format(bytes / 1_048_576.0); else -> "%.0f KB".format(bytes / 1024.0) }
private fun gainLabel(gain: Float): String = if (gain <= .001f) "−∞" else "${(20 * log10(gain)).roundToInt()} dB"
