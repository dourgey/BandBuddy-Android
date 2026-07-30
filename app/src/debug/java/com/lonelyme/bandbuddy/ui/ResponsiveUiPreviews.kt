package com.lonelyme.bandbuddy.ui

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.lonelyme.bandbuddy.data.SongRecord
import com.lonelyme.bandbuddy.data.SongStatus
import com.lonelyme.bandbuddy.ui.theme.BandBuddyTheme

@Preview(name = "手机版 · 曲库", widthDp = 390, heightDp = 844, showBackground = true)
@Composable
private fun PhoneLibraryPreview() {
    BandBuddyTheme {
        LibraryPreviewContent(isTablet = false)
    }
}

@Preview(name = "Pad 版 · 曲库", widthDp = 1024, heightDp = 768, showBackground = true)
@Composable
private fun TabletLibraryPreview() {
    BandBuddyTheme {
        Row(Modifier.fillMaxSize()) {
            TabletNavigationRail(
                selected = AppScreen.LIBRARY,
                onLibrary = {},
                onPractice = {},
                onSettings = {},
            )
            LibraryPreviewContent(isTablet = true)
        }
    }
}

@Composable
private fun LibraryPreviewContent(isTablet: Boolean) {
    val songs = remember {
        listOf(
            SongRecord(
                id = "preview-ready",
                title = "夜航",
                artist = "BandBuddy Session",
                durationMs = 243_000,
                sourcePath = null,
                stems = emptyList(),
                status = SongStatus.READY,
                jobStage = "可以练习",
                progress = 100,
                favorite = true,
            ),
            SongRecord(
                id = "preview-processing",
                title = "沿海公路",
                artist = "排练室 Demo",
                durationMs = 198_000,
                sourcePath = null,
                stems = emptyList(),
                status = SongStatus.PROCESSING,
                jobStage = "正在本地分轨",
                progress = 62,
                processingStartedAt = System.currentTimeMillis() - 35_000,
            ),
            SongRecord(
                id = "preview-ready-two",
                title = "凌晨四点",
                artist = "周末乐队",
                durationMs = 276_000,
                sourcePath = null,
                stems = emptyList(),
                status = SongStatus.READY,
                jobStage = "可以练习",
                progress = 100,
            ),
            SongRecord(
                id = "preview-ready-three",
                title = "回声",
                artist = "Live Take",
                durationMs = 224_000,
                sourcePath = null,
                stems = emptyList(),
                status = SongStatus.READY,
                jobStage = "可以练习",
                progress = 100,
            ),
        )
    }
    LibraryScreen(
        songs = songs,
        isTablet = isTablet,
        onImport = {},
        onOpen = {},
        onFavorite = {},
        onEdit = {},
        onDelete = {},
        onRetry = {},
        onCancel = {},
        onSettings = {},
        onPractice = {},
    )
}
