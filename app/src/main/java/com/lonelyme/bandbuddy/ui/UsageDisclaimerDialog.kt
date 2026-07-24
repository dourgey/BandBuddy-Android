package com.lonelyme.bandbuddy.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.lonelyme.bandbuddy.ui.theme.Ink
import com.lonelyme.bandbuddy.ui.theme.Line
import com.lonelyme.bandbuddy.ui.theme.MutedInk
import com.lonelyme.bandbuddy.ui.theme.Paper

private data class DisclaimerItem(
    val title: String,
    val body: String,
)

private val disclaimerItems = listOf(
    DisclaimerItem(
        title = "音频与版权",
        body = "请只导入、处理和导出您拥有合法权利或已获授权的音频。BandBuddy 不提供音频内容；因未经授权使用作品产生的争议和责任由使用者承担。",
    ),
    DisclaimerItem(
        title = "分轨与分析结果",
        body = "分轨、节拍和波形由算法自动生成，可能出现串音、漏音、相位变化、音质损失或识别偏差。结果仅用于练习和辅助参考，不保证达到专业制作、演出或出版标准。",
    ),
    DisclaimerItem(
        title = "听力与设备安全",
        body = "长时间或高音量播放可能损伤听力，也可能造成扬声器失真。请从较低音量开始；使用耳机、蓝牙音箱或外接设备时，请同时检查系统和设备增益。如有不适请立即停止播放。",
    ),
    DisclaimerItem(
        title = "本地数据与备份",
        body = "歌曲、分轨和练习记录保存在本机。卸载应用、清理存储、系统故障、设备损坏或手动删除都可能造成数据丢失，请自行备份重要原始音频和导出文件。",
    ),
    DisclaimerItem(
        title = "网络、性能与兼容性",
        body = "模型下载依赖第三方网络服务；处理速度和可用功能会受网络、系统版本、设备性能及硬件兼容性影响，无法保证在所有设备和环境中持续可用。",
    ),
    DisclaimerItem(
        title = "责任范围",
        body = "在适用法律允许的范围内，开发者不对超出设计用途的使用、未备份数据、设备故障或第三方服务中断造成的间接损失负责；法律规定不得免除的责任不受本条影响。",
    ),
)

@Composable
internal fun UsageDisclaimerDialog(onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Paper,
        shape = RoundedCornerShape(22.dp),
        title = {
            Text(
                text = "使用与免责",
                fontFamily = FontFamily.Serif,
                fontWeight = FontWeight.Black,
                fontSize = 22.sp,
            )
        },
        text = {
            LazyColumn(
                modifier = Modifier.heightIn(max = 430.dp),
                verticalArrangement = Arrangement.spacedBy(13.dp),
            ) {
                item {
                    Text(
                        text = "BandBuddy 是本地音乐练习工具。以下内容说明功能边界，不影响您依法享有的权利。",
                        color = MutedInk,
                        fontSize = 11.sp,
                        lineHeight = 17.sp,
                    )
                    HorizontalDivider(
                        modifier = Modifier.padding(top = 13.dp),
                        color = Line,
                    )
                }
                items(disclaimerItems) { item ->
                    Column {
                        Text(
                            text = item.title,
                            color = Ink,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = item.body,
                            color = MutedInk,
                            fontSize = 11.sp,
                            lineHeight = 17.sp,
                            modifier = Modifier.padding(top = 4.dp),
                        )
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text(
                    text = "我知道了",
                    color = Ink,
                    fontWeight = FontWeight.Bold,
                )
            }
        },
    )
}
