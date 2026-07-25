package com.lonelyme.bandbuddy.ui

import android.os.SystemClock
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.disabled
import androidx.compose.ui.semantics.onClick
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.lonelyme.bandbuddy.data.PracticeState
import com.lonelyme.bandbuddy.data.normalizeBeatOffsetMs
import com.lonelyme.bandbuddy.engine.BeatGridDetector
import com.lonelyme.bandbuddy.engine.TapTempoEstimator
import com.lonelyme.bandbuddy.ui.theme.Ink
import com.lonelyme.bandbuddy.ui.theme.Line
import com.lonelyme.bandbuddy.ui.theme.MutedInk
import com.lonelyme.bandbuddy.ui.theme.Paper
import com.lonelyme.bandbuddy.ui.theme.Sand
import kotlin.math.roundToInt

@Composable
internal fun MetronomeSettingsDialog(
    practice: PracticeState,
    beatAnalysisRunning: Boolean,
    isSongPlaying: Boolean,
    onDetectBeatGrid: () -> Unit,
    onPreview: () -> Unit,
    readPlaybackPosition: () -> Long?,
    onChange: (PracticeState) -> Unit,
    onDismiss: () -> Unit,
) {
    val tapEstimator = remember { TapTempoEstimator() }
    var tapCount by remember { mutableIntStateOf(0) }

    fun resetTapSession() {
        tapEstimator.reset()
        tapCount = 0
    }

    fun setManualTempo(value: Float) {
        resetTapSession()
        val bpm = value.coerceIn(20f, 400f)
        onChange(
            practice.copy(
                metronomeBpm = bpm,
                metronomeOffsetMs = normalizeBeatOffsetMs(practice.metronomeOffsetMs, bpm),
                metronomeConfidence = 0f,
            )
        )
    }

    fun setManualOffset(value: Long) {
        resetTapSession()
        onChange(
            practice.copy(
                metronomeOffsetMs = normalizeBeatOffsetMs(value, practice.metronomeBpm),
                metronomeConfidence = 0f,
            )
        )
    }

    LaunchedEffect(isSongPlaying, practice.speed) {
        if (!isSongPlaying || tapCount > 0) resetTapSession()
    }

    val offsetLimit = (30_000f / practice.metronomeBpm).roundToInt().coerceAtLeast(1)
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 18.dp)
                .heightIn(max = 660.dp),
            shape = RoundedCornerShape(28.dp),
            color = Paper,
            shadowElevation = 10.dp,
        ) {
            LazyColumn(
                contentPadding = PaddingValues(18.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                item {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(42.dp)
                                .clip(RoundedCornerShape(13.dp))
                                .background(Sand),
                            contentAlignment = Alignment.Center,
                        ) {
                            MetronomeIcon(Modifier.size(22.dp))
                        }
                        Spacer(Modifier.weight(1f))
                        MetronomeToggle(
                            enabled = practice.metronomeEnabled,
                            onToggle = {
                                onChange(practice.copy(metronomeEnabled = !practice.metronomeEnabled))
                            },
                        )
                        Spacer(Modifier.width(9.dp))
                        CloseButton(onClick = onDismiss)
                    }
                }

                item {
                    ControlCard {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            TempoStepButton(
                                symbol = "−",
                                description = "BPM 减一",
                                onClick = { setManualTempo(practice.metronomeBpm - 1f) },
                            )
                            Column(
                                modifier = Modifier.weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                            ) {
                                EditableTempoValue(
                                    value = practice.metronomeBpm,
                                    onValidValue = ::setManualTempo,
                                )
                                Text(
                                    text = "BPM",
                                    color = MutedInk,
                                    fontSize = 9.sp,
                                    fontWeight = FontWeight.Bold,
                                    letterSpacing = 1.5.sp,
                                )
                            }
                            TempoStepButton(
                                symbol = "+",
                                description = "BPM 加一",
                                onClick = { setManualTempo(practice.metronomeBpm + 1f) },
                            )
                        }
                        Slider(
                            value = practice.metronomeBpm,
                            onValueChange = {
                                setManualTempo((it * 10f).roundToInt() / 10f)
                            },
                            valueRange = 20f..400f,
                            modifier = Modifier.fillMaxWidth().height(32.dp),
                            colors = metronomeSliderColors(),
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            AutoTempoButton(
                                running = beatAnalysisRunning,
                                confidence = practice.metronomeConfidence,
                                onClick = {
                                    resetTapSession()
                                    onDetectBeatGrid()
                                },
                                modifier = Modifier.width(70.dp),
                            )
                            TapTempoButton(
                                enabled = isSongPlaying,
                                tapCount = tapCount,
                                onClick = {
                                    val position = readPlaybackPosition()
                                    if (position == null) {
                                        resetTapSession()
                                    } else {
                                        val result = tapEstimator.tap(
                                            songPositionMs = position,
                                            realtimeMs = SystemClock.elapsedRealtime(),
                                            playbackSpeed = practice.speed,
                                        )
                                        tapCount = tapEstimator.tapCount
                                        result?.let { measured ->
                                            onChange(
                                                practice.copy(
                                                    metronomeBpm = measured.bpm,
                                                    metronomeOffsetMs = measured.beatOffsetMs,
                                                    metronomeAnalysisDone = true,
                                                    metronomeAnalysisVersion = BeatGridDetector.ANALYSIS_VERSION,
                                                    metronomeConfidence = 0f,
                                                )
                                            )
                                        }
                                    }
                                },
                                modifier = Modifier.weight(1f),
                            )
                            PreviewBeatButton(
                                onClick = onPreview,
                                modifier = Modifier.width(54.dp),
                            )
                        }
                    }
                }

                item {
                    ControlCard {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            BeatOffsetIcon(Modifier.size(21.dp), MutedInk)
                            Spacer(Modifier.width(9.dp))
                            NudgeButton("−10", "拍点提前 10 毫秒") {
                                setManualOffset(practice.metronomeOffsetMs - 10L)
                            }
                            Column(
                                modifier = Modifier.weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                            ) {
                                Text(
                                    text = signedOffset(practice.metronomeOffsetMs),
                                    color = Ink,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Black,
                                    textAlign = TextAlign.Center,
                                )
                                Text("ms", color = MutedInk, fontSize = 8.sp)
                            }
                            NudgeButton("+10", "拍点延后 10 毫秒") {
                                setManualOffset(practice.metronomeOffsetMs + 10L)
                            }
                            Spacer(Modifier.width(7.dp))
                            ResetOffsetButton { setManualOffset(0L) }
                        }
                        Slider(
                            value = practice.metronomeOffsetMs
                                .coerceIn(-offsetLimit.toLong(), offsetLimit.toLong())
                                .toFloat(),
                            onValueChange = { setManualOffset(it.roundToInt().toLong()) },
                            valueRange = -offsetLimit.toFloat()..offsetLimit.toFloat(),
                            modifier = Modifier.fillMaxWidth().height(31.dp),
                            colors = metronomeSliderColors(),
                        )
                    }
                }

                item {
                    ControlCard {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            VolumeIcon(Modifier.size(21.dp), MutedInk)
                            Slider(
                                value = practice.metronomeVolume,
                                onValueChange = {
                                    onChange(practice.copy(metronomeVolume = it))
                                },
                                valueRange = .2f..1f,
                                modifier = Modifier.weight(1f).height(31.dp).padding(horizontal = 8.dp),
                                colors = metronomeSliderColors(),
                            )
                            Text(
                                text = "${(practice.metronomeVolume * 100).roundToInt()}%",
                                color = MutedInk,
                                fontSize = 9.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.width(34.dp),
                                textAlign = TextAlign.End,
                            )
                        }
                    }
                }

                item {
                    ControlCard {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            CountInIcon(Modifier.size(23.dp), MutedInk)
                            Spacer(Modifier.weight(1f))
                            CountInSelector(
                                selected = practice.countInBeats,
                                onSelected = { onChange(practice.copy(countInBeats = it)) },
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun EditableTempoValue(
    value: Float,
    onValidValue: (Float) -> Unit,
) {
    val focusManager = LocalFocusManager.current
    val formattedValue = tempoValue(value)
    var hasFocus by remember { mutableStateOf(false) }
    var fieldValue by remember {
        mutableStateOf(TextFieldValue(formattedValue))
    }

    LaunchedEffect(formattedValue, hasFocus) {
        if (!hasFocus && fieldValue.text != formattedValue) {
            fieldValue = TextFieldValue(formattedValue)
        }
    }

    fun submit() {
        val bpm = fieldValue.text
            .toFloatOrNull()
            ?.coerceIn(MINIMUM_BPM, MAXIMUM_BPM)
            ?.let { (it * 10f).roundToInt() / 10f }
        if (bpm == null) {
            fieldValue = TextFieldValue(formattedValue)
        } else {
            fieldValue = TextFieldValue(tempoValue(bpm))
            onValidValue(bpm)
        }
        focusManager.clearFocus()
    }

    BasicTextField(
        value = fieldValue,
        onValueChange = { candidate ->
            val normalized = candidate.text.replace(',', '.')
            if (BPM_INPUT_PATTERN.matches(normalized)) {
                fieldValue = candidate.copy(text = normalized)
                normalized.toFloatOrNull()
                    ?.takeIf { it in MINIMUM_BPM..MAXIMUM_BPM }
                    ?.let { onValidValue((it * 10f).roundToInt() / 10f) }
            }
        },
        modifier = Modifier
            .width(132.dp)
            .height(48.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(Sand.copy(alpha = if (hasFocus) 1f else .52f))
            .border(
                width = 1.dp,
                color = if (hasFocus) Ink.copy(alpha = .72f) else Line.copy(alpha = .55f),
                shape = RoundedCornerShape(14.dp),
            )
            .onFocusChanged { state ->
                if (state.isFocused && !hasFocus) {
                    fieldValue = fieldValue.copy(
                        selection = TextRange(0, fieldValue.text.length),
                    )
                } else if (!state.isFocused && hasFocus) {
                    fieldValue = TextFieldValue(tempoValue(value))
                }
                hasFocus = state.isFocused
            }
            .semantics {
                contentDescription = "BPM，点击输入"
            },
        singleLine = true,
        textStyle = TextStyle(
            color = Ink,
            fontFamily = FontFamily.Serif,
            fontSize = 36.sp,
            fontWeight = FontWeight.Black,
            lineHeight = 38.sp,
            textAlign = TextAlign.Center,
        ),
        keyboardOptions = KeyboardOptions(
            keyboardType = KeyboardType.Decimal,
            imeAction = ImeAction.Done,
        ),
        keyboardActions = KeyboardActions(onDone = { submit() }),
        cursorBrush = SolidColor(Ink),
        decorationBox = { innerTextField ->
            Box(
                modifier = Modifier.fillMaxWidth().height(48.dp),
                contentAlignment = Alignment.Center,
            ) {
                innerTextField()
            }
        },
    )
}

@Composable
private fun ControlCard(content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(Color.White)
            .border(1.dp, Line.copy(alpha = .72f), RoundedCornerShape(18.dp))
            .padding(horizontal = 14.dp, vertical = 12.dp),
        content = content,
    )
}

@Composable
private fun MetronomeToggle(enabled: Boolean, onToggle: () -> Unit) {
    Box(
        modifier = Modifier
            .width(50.dp)
            .height(29.dp)
            .clip(RoundedCornerShape(15.dp))
            .background(if (enabled) Ink else Line)
            .toggleable(
                value = enabled,
                role = Role.Switch,
                onValueChange = { onToggle() },
            )
            .semantics {
                contentDescription = "节拍器"
                stateDescription = if (enabled) "开启" else "关闭"
            }
            .padding(3.dp),
        contentAlignment = if (enabled) Alignment.CenterEnd else Alignment.CenterStart,
    ) {
        Box(Modifier.size(23.dp).clip(CircleShape).background(Paper))
    }
}

@Composable
private fun CloseButton(onClick: () -> Unit) {
    Canvas(
        Modifier
            .size(36.dp)
            .clip(CircleShape)
            .background(Sand)
            .clickable(onClick = onClick)
            .semantics {
                contentDescription = "关闭"
                role = Role.Button
            }
            .padding(11.dp)
    ) {
        val stroke = 1.8.dp.toPx()
        drawLine(Ink, Offset.Zero, Offset(size.width, size.height), stroke, StrokeCap.Round)
        drawLine(Ink, Offset(size.width, 0f), Offset(0f, size.height), stroke, StrokeCap.Round)
    }
}

@Composable
private fun TempoStepButton(symbol: String, description: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(42.dp)
            .clip(CircleShape)
            .background(Sand)
            .clickable(onClick = onClick)
            .semantics {
                contentDescription = description
                role = Role.Button
            },
        contentAlignment = Alignment.Center,
    ) {
        Text(symbol, color = Ink, fontSize = 22.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun AutoTempoButton(
    running: Boolean,
    confidence: Float,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .height(60.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(if (running) Ink else Sand)
            .clickable(enabled = !running, onClick = onClick)
            .semantics {
                contentDescription = "自动测定 BPM 和拍点"
                role = Role.Button
            }
            .padding(vertical = 7.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        AutoDetectIcon(
            modifier = Modifier.size(19.dp),
            color = if (running) Paper else Ink,
        )
        Text(
            text = if (running) "•••" else "AUTO",
            color = if (running) Paper else Ink,
            fontSize = 8.sp,
            fontWeight = FontWeight.Black,
            letterSpacing = .7.sp,
        )
        if (!running && confidence > 0f) {
            Text(
                text = "${(confidence * 100).roundToInt()}%",
                color = MutedInk,
                fontSize = 7.sp,
                lineHeight = 7.sp,
            )
        }
    }
}

@Composable
private fun TapTempoButton(
    enabled: Boolean,
    tapCount: Int,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .height(60.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(if (enabled) Ink else Sand)
            .pointerInput(enabled) {
                if (enabled) {
                    detectTapGestures(onPress = { onClick() })
                }
            }
            .semantics {
                contentDescription = "连续点击手动测定 BPM 和拍点"
                stateDescription = if (enabled) "播放中可用" else "需要播放歌曲"
                role = Role.Button
                if (enabled) {
                    onClick {
                        onClick()
                        true
                    }
                } else {
                    disabled()
                }
            },
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "TAP",
            color = if (enabled) Paper else MutedInk,
            fontSize = 17.sp,
            fontWeight = FontWeight.Black,
            letterSpacing = 1.6.sp,
        )
        Row(
            modifier = Modifier.padding(top = 5.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            repeat(4) { index ->
                Box(
                    Modifier
                        .size(4.dp)
                        .clip(CircleShape)
                        .background(
                            when {
                                index < tapCount -> if (enabled) Color(0xFFA9B58D) else MutedInk
                                enabled -> Paper.copy(alpha = .28f)
                                else -> Line
                            }
                        )
                )
            }
        }
    }
}

@Composable
private fun PreviewBeatButton(onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .height(60.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(Sand)
            .clickable(onClick = onClick)
            .semantics {
                contentDescription = "试听节拍"
                role = Role.Button
            },
        contentAlignment = Alignment.Center,
    ) {
        Canvas(Modifier.size(20.dp)) {
            val path = Path().apply {
                moveTo(size.width * .30f, size.height * .18f)
                lineTo(size.width * .82f, size.height * .50f)
                lineTo(size.width * .30f, size.height * .82f)
                close()
            }
            drawPath(path, Ink)
        }
    }
}

@Composable
private fun NudgeButton(label: String, description: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .height(34.dp)
            .width(43.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(Sand)
            .clickable(onClick = onClick)
            .semantics {
                contentDescription = description
                role = Role.Button
            },
        contentAlignment = Alignment.Center,
    ) {
        Text(label, color = Ink, fontSize = 9.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun ResetOffsetButton(onClick: () -> Unit) {
    Canvas(
        Modifier
            .size(34.dp)
            .clip(CircleShape)
            .background(Sand)
            .clickable(onClick = onClick)
            .semantics {
                contentDescription = "拍点归零"
                role = Role.Button
            }
            .padding(8.dp)
    ) {
        val stroke = 1.6.dp.toPx()
        drawArc(
            color = Ink,
            startAngle = 35f,
            sweepAngle = 285f,
            useCenter = false,
            style = Stroke(width = stroke, cap = StrokeCap.Round),
        )
        val path = Path().apply {
            moveTo(size.width * .20f, size.height * .31f)
            lineTo(size.width * .12f, size.height * .08f)
            lineTo(size.width * .38f, size.height * .13f)
            close()
        }
        drawPath(path, Ink)
    }
}

@Composable
private fun CountInSelector(selected: Int, onSelected: (Int) -> Unit) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Sand)
            .padding(3.dp),
        horizontalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        listOf(0, 4, 8).forEach { beats ->
            Box(
                modifier = Modifier
                    .size(width = 52.dp, height = 34.dp)
                    .clip(RoundedCornerShape(9.dp))
                    .background(if (selected == beats) Ink else Color.Transparent)
                    .clickable { onSelected(beats) }
                    .semantics {
                        contentDescription = if (beats == 0) "关闭预备拍" else "$beats 拍预备拍"
                        role = Role.Button
                    },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = beats.toString(),
                    color = if (selected == beats) Paper else Ink,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                )
            }
        }
    }
}

@Composable
internal fun MetronomeIcon(modifier: Modifier = Modifier, color: Color = Ink) {
    Canvas(modifier) {
        val stroke = 1.5.dp.toPx()
        drawLine(color, Offset(size.width * .33f, size.height * .12f), Offset(size.width * .16f, size.height * .88f), stroke, StrokeCap.Round)
        drawLine(color, Offset(size.width * .67f, size.height * .12f), Offset(size.width * .84f, size.height * .88f), stroke, StrokeCap.Round)
        drawLine(color, Offset(size.width * .16f, size.height * .88f), Offset(size.width * .84f, size.height * .88f), stroke, StrokeCap.Round)
        drawLine(color, Offset(size.width * .5f, size.height * .7f), Offset(size.width * .68f, size.height * .25f), stroke, StrokeCap.Round)
        drawCircle(color, radius = stroke * 1.25f, center = Offset(size.width * .69f, size.height * .23f))
        drawLine(color, Offset(size.width * .27f, size.height * .68f), Offset(size.width * .73f, size.height * .68f), stroke * .75f, StrokeCap.Round)
    }
}

@Composable
private fun AutoDetectIcon(modifier: Modifier = Modifier, color: Color) {
    Canvas(modifier) {
        val stroke = 1.5.dp.toPx()
        val heights = listOf(.28f, .58f, .82f, .46f)
        heights.forEachIndexed { index, height ->
            val x = size.width * (.16f + index * .18f)
            drawLine(
                color,
                Offset(x, size.height * (1f - height) / 2f),
                Offset(x, size.height * (1f + height) / 2f),
                stroke,
                StrokeCap.Round,
            )
        }
        drawCircle(
            color = color,
            radius = size.width * .16f,
            center = Offset(size.width * .77f, size.height * .70f),
            style = Stroke(stroke),
        )
        drawLine(
            color,
            Offset(size.width * .87f, size.height * .80f),
            Offset(size.width * .97f, size.height * .90f),
            stroke,
            StrokeCap.Round,
        )
    }
}

@Composable
private fun BeatOffsetIcon(modifier: Modifier = Modifier, color: Color) {
    Canvas(modifier) {
        val stroke = 1.4.dp.toPx()
        drawLine(color, Offset(size.width * .18f, size.height * .18f), Offset(size.width * .18f, size.height * .82f), stroke, StrokeCap.Round)
        drawLine(color, Offset(size.width * .82f, size.height * .18f), Offset(size.width * .82f, size.height * .82f), stroke, StrokeCap.Round)
        drawLine(color, Offset(size.width * .50f, size.height * .10f), Offset(size.width * .50f, size.height * .90f), stroke * 1.4f, StrokeCap.Round)
        drawLine(color, Offset(size.width * .28f, size.height * .50f), Offset(size.width * .72f, size.height * .50f), stroke, StrokeCap.Round)
        drawLine(color, Offset(size.width * .28f, size.height * .50f), Offset(size.width * .38f, size.height * .40f), stroke, StrokeCap.Round)
        drawLine(color, Offset(size.width * .72f, size.height * .50f), Offset(size.width * .62f, size.height * .60f), stroke, StrokeCap.Round)
    }
}

@Composable
private fun VolumeIcon(modifier: Modifier = Modifier, color: Color) {
    Canvas(modifier) {
        val stroke = 1.5.dp.toPx()
        val speaker = Path().apply {
            moveTo(size.width * .12f, size.height * .40f)
            lineTo(size.width * .34f, size.height * .40f)
            lineTo(size.width * .55f, size.height * .20f)
            lineTo(size.width * .55f, size.height * .80f)
            lineTo(size.width * .34f, size.height * .60f)
            lineTo(size.width * .12f, size.height * .60f)
            close()
        }
        drawPath(speaker, color)
        drawArc(
            color,
            startAngle = -48f,
            sweepAngle = 96f,
            useCenter = false,
            topLeft = Offset(size.width * .48f, size.height * .29f),
            size = androidx.compose.ui.geometry.Size(size.width * .27f, size.height * .42f),
            style = Stroke(stroke, cap = StrokeCap.Round),
        )
        drawArc(
            color,
            startAngle = -48f,
            sweepAngle = 96f,
            useCenter = false,
            topLeft = Offset(size.width * .43f, size.height * .18f),
            size = androidx.compose.ui.geometry.Size(size.width * .45f, size.height * .64f),
            style = Stroke(stroke, cap = StrokeCap.Round),
        )
    }
}

@Composable
private fun CountInIcon(modifier: Modifier = Modifier, color: Color) {
    Canvas(modifier) {
        repeat(4) { index ->
            val angle = -PI_OVER_TWO + index * PI_OVER_TWO
            val center = Offset(
                x = (size.width * .50f + kotlin.math.cos(angle) * size.width * .28f).toFloat(),
                y = (size.height * .50f + kotlin.math.sin(angle) * size.height * .28f).toFloat(),
            )
            drawCircle(
                color = if (index == 0) color else color.copy(alpha = .35f),
                radius = size.width * .075f,
                center = center,
            )
        }
    }
}

@Composable
private fun metronomeSliderColors() = SliderDefaults.colors(
    thumbColor = Ink,
    activeTrackColor = Ink,
    inactiveTrackColor = Line,
)

private fun tempoValue(bpm: Float): String =
    if (kotlin.math.abs(bpm - bpm.roundToInt()) < .05f) {
        bpm.roundToInt().toString()
    } else {
        "%.1f".format(bpm)
    }

private fun signedOffset(offsetMs: Long): String =
    if (offsetMs >= 0L) "+$offsetMs" else offsetMs.toString()

private val BPM_INPUT_PATTERN = Regex("""\d{0,3}(?:\.\d?)?""")
private const val MINIMUM_BPM = 20f
private const val MAXIMUM_BPM = 400f
private const val PI_OVER_TWO = Math.PI / 2.0
