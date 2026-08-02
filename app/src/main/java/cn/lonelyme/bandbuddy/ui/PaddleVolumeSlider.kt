package cn.lonelyme.bandbuddy.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsDraggedAsState
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import cn.lonelyme.bandbuddy.ui.theme.Ink

private val PaddleOlive = Color(0xFFA9B58D)
private val PaddleOliveEdge = Color(0xFF7D8967)
private val PaddleMuted = Color(0xFFE5E2DE)
private val PaddleMutedEdge = Color(0xFFC9C4BE)
private val PaddleSolo = Color(0xFF24211E)
private val PaddleSoloEdge = Color(0xFF0F0E0D)
private val PaddleTrack = Color(0xFFE9E5E0)

/**
 * A compact mixer fader whose thumb borrows the silhouette of a guitar pick.
 *
 * Muted and solo tracks use color only; the thumb deliberately has no visible
 * label so the fader stays calm and legible at mixer-row size.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun PaddleVolumeSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    muted: Boolean,
    solo: Boolean,
    contentDescription: String,
    modifier: Modifier = Modifier,
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isDragged by interactionSource.collectIsDraggedAsState()
    val isPressed by interactionSource.collectIsPressedAsState()
    val interactionProgress by animateFloatAsState(
        targetValue = if (isDragged || isPressed) 1f else 0f,
        animationSpec = tween(durationMillis = 150),
        label = "paddle-fader-feedback",
    )
    val fill = when {
        muted -> PaddleMuted
        solo -> PaddleSolo
        else -> PaddleOlive
    }
    val edge = when {
        muted -> PaddleMutedEdge
        solo -> PaddleSoloEdge
        else -> PaddleOliveEdge
    }
    val activeTrack = if (muted) Color(0xFFBEB9B3) else Ink

    Slider(
        value = value.coerceIn(0f, 1f),
        onValueChange = onValueChange,
        modifier = modifier.semantics { this.contentDescription = contentDescription },
        interactionSource = interactionSource,
        thumb = {
            PaddleThumb(
                fill = fill,
                edge = edge,
                interactionProgress = interactionProgress,
            )
        },
        track = { sliderState ->
            PaddleTrack(
                sliderState = sliderState,
                activeColor = activeTrack,
            )
        },
    )
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun PaddleTrack(
    sliderState: SliderState,
    activeColor: Color,
) {
    val range = sliderState.valueRange
    val rangeLength = range.endInclusive - range.start
    val fraction = if (rangeLength == 0f) {
        0f
    } else {
        ((sliderState.value - range.start) / rangeLength).coerceIn(0f, 1f)
    }
    Canvas(Modifier.fillMaxWidth().height(5.dp)) {
        val radius = size.height / 2f
        drawRoundRect(
            color = PaddleTrack,
            size = size,
            cornerRadius = CornerRadius(radius),
        )
        if (fraction > 0f) {
            drawRoundRect(
                color = activeColor,
                size = Size(size.width * fraction, size.height),
                cornerRadius = CornerRadius(radius),
            )
        }
    }
}

@Composable
private fun PaddleThumb(
    fill: Color,
    edge: Color,
    interactionProgress: Float,
) {
    Canvas(Modifier.size(width = 34.dp, height = 36.dp)) {
        val center = Offset(size.width / 2f, size.height * .46f)
        if (interactionProgress > 0f) {
            drawCircle(
                color = edge.copy(alpha = .10f * interactionProgress),
                radius = size.width * .49f,
                center = center,
                style = Stroke(width = 1.dp.toPx()),
            )
            drawCircle(
                color = edge.copy(alpha = .18f * interactionProgress),
                radius = size.width * .41f,
                center = center,
                style = Stroke(width = 1.dp.toPx()),
            )
        }

        val shadow = paddlePath(size, yOffset = 1.4.dp.toPx())
        drawPath(shadow, Color.Black.copy(alpha = .14f))

        val paddle = paddlePath(size)
        drawPath(paddle, fill)
        drawPath(
            path = paddle,
            color = edge,
            style = Stroke(width = 1.dp.toPx()),
        )
        drawCircle(
            color = Color.White.copy(alpha = .18f),
            radius = size.width * .055f,
            center = Offset(size.width * .38f, size.height * .27f),
        )
    }
}

private fun paddlePath(size: Size, yOffset: Float = 0f): Path = Path().apply {
    moveTo(size.width * .50f, size.height * .94f + yOffset)
    cubicTo(
        size.width * .43f,
        size.height * .91f + yOffset,
        size.width * .19f,
        size.height * .74f + yOffset,
        size.width * .16f,
        size.height * .48f + yOffset,
    )
    cubicTo(
        size.width * .13f,
        size.height * .20f + yOffset,
        size.width * .28f,
        size.height * .10f + yOffset,
        size.width * .50f,
        size.height * .10f + yOffset,
    )
    cubicTo(
        size.width * .72f,
        size.height * .10f + yOffset,
        size.width * .87f,
        size.height * .20f + yOffset,
        size.width * .84f,
        size.height * .48f + yOffset,
    )
    cubicTo(
        size.width * .81f,
        size.height * .74f + yOffset,
        size.width * .57f,
        size.height * .91f + yOffset,
        size.width * .50f,
        size.height * .94f + yOffset,
    )
    close()
}
