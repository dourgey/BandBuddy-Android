package com.lonelyme.bandbuddy.ui

import androidx.compose.foundation.Image
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import com.lonelyme.bandbuddy.R

/** The same artwork used by README.md and the launcher resources. */
@Composable
internal fun AppIcon(modifier: Modifier = Modifier) {
    Image(
        painter = painterResource(R.drawable.icon_launcher_art),
        contentDescription = null,
        modifier = modifier,
        contentScale = ContentScale.Fit,
    )
}
