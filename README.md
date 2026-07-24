<p align="center">
  <img src="icon.png" alt="BandBuddy 图标" width="128" height="128">
</p>

<h1 align="center">BandBuddy</h1>

<p align="center">在 Android 手机上本地完成六轨分离，并把歌曲变成可循环、可调速、可独奏的练习伴奏。</p>

## 下载

从 [GitHub Releases](https://github.com/dourgey/BandBuddy-Android/releases/latest) 下载最新 APK。当前构建仅包含 `arm64-v8a`，最低支持 Android 10（API 29）。

安装后请进入“设置”下载约 112 MiB 的固定版本 HTDemucs 模型。模型下载完成并校验通过后，才能开始自动分轨。

> 当前版本依赖 Qualcomm HTP NPU 和设备提供的 FastRPC 运行时，不会静默回退到 CPU。项目已在 Snapdragon 8 Gen 3（SM8650）设备上验证；其他芯片或未开放 HTP 运行时的设备可能无法安装或执行分轨。

## 功能

- 导入 MP3、M4A、WAV、FLAC 等本地歌曲，在设备上分离人声、鼓、贝斯、吉他、钢琴和其他声部。
- 直接导入已有的六个分轨文件，并按文件名自动识别声部。
- 六轨混音器：单轨音量、静音、独奏和总音量控制。
- 练习工具：0.5×–1.5× 调速、A/B 循环、自动节拍分析、节拍器、拍点微调和 4/8 拍预备拍。
- 波形与播放进度显示，支持带时间轴的 LRC 歌词。
- 本地曲库、搜索、拼音搜索、收藏、歌曲信息编辑和失败任务重试。
- 模型断点下载、SHA-256 校验、分轨进度通知和任务取消。

## 分轨与本地存储

BandBuddy 使用固定的 `htdemucs_6s` 六轨模型。输入音频会统一解码为 44.1 kHz 双声道 PCM，再以 7.8 秒窗口和 25% overlap 进行长音频推理。最终六轨会连续编码为 160 kbps AAC/M4A；只有六个文件全部成功后，临时目录才会原子提交为正式结果。取消或失败时不会把半成品加入曲库。

歌曲、分轨、歌词和练习记录均保存在 App 私有目录。卸载 App 或清理应用数据会删除这些内容，请保留重要原始音频的独立备份。

## 从源码构建

环境要求：

- Android Studio（建议使用其内置 JDK）
- Android SDK 36.1
- Android NDK `27.2.12479018`
- CMake `3.22.1`

克隆并构建：

```bash
git clone https://github.com/dourgey/BandBuddy-Android.git
cd BandBuddy-Android
./gradlew assembleRelease
```

Windows PowerShell：

```powershell
git clone https://github.com/dourgey/BandBuddy-Android.git
Set-Location BandBuddy-Android
.\gradlew.bat assembleRelease
```

APK 输出到 `app/build/outputs/apk/release/`。仓库中的 release 构建为可直接安装的测试版本，当前使用 Android debug signing config；正式上架应用商店前请替换为妥善保管的产品签名。

默认模型仓库为 `Zzzzzzorz/BandBuddy-HTDemucs-6s`。如需在构建时更换 ModelScope 仓库，可设置 Gradle 属性：

```properties
BAND_BUDDY_MODELSCOPE_REPOSITORY=owner/repository
```

## 测试

运行 JVM 单元测试：

```bash
./gradlew testDebugUnitTest
```

NPU、QNN、原生 DSP 和完整六轨推理需要兼容的 Qualcomm 真机。详细的模型转换、数值验证、性能结果与真机测试命令见 [HTDemucs 端侧适配文档](docs/htdemucs-on-device-porting.md)。

## 技术栈

- Kotlin、Jetpack Compose、WorkManager
- LiteRT / TensorFlow Lite
- Qualcomm QNN HTP Delegate
- C++20、JNI、Android NDK
- Android MediaCodec / MediaMuxer

## 许可与声明

项目许可见 [LICENSE](LICENSE)。HTDemucs、TensorFlow Lite、Kiss FFT、PocketFFT、QNN 运行时及相关组件的版权与许可信息见 [THIRD_PARTY_NOTICES.txt](app/src/main/assets/THIRD_PARTY_NOTICES.txt)。

请只处理您拥有合法权利或已获授权的音频。自动分轨与节拍分析可能产生串音、漏音、相位变化或识别偏差，结果适合练习和辅助参考，不保证达到专业制作、演出或出版标准。
