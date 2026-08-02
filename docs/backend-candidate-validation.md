# HTDemucs NPU/GPU 后端候选实机验证

本文记录 BandBuddy 在 Xiaomi 14（Snapdragon 8 Gen 3 / SM8650）上对完整
7.8 秒 `htdemucs_6s` 图进行的 FP16/FP32 后端搜索。验证日期为
2026-07-24，模型 SHA-256 为
`a9fcc89e84aa65313e0540b582e710007ed12064969a0d49a3c85e49f1ae4e3d`，
运行时为 TensorFlow Lite 2.17.0 与 Qualcomm QNN 2.48.0。

## 验收规则

每个能完成推理的候选必须处理 F1 和“乌云典当记”各一个真实 343,980
sample 双声道窗口，并与固定官方 Torch 权重逐样本比较：

```text
all-stem correlation >= 0.99999
all-stem SNR         >= 50 dB
输出必须全部有限，peak < 10
```

性能只在质量门禁通过后参与选择。图能 finalize、输出没有 NaN，均不能代替
Torch 质量门禁。

## 候选矩阵

| 候选 | 实际委托 | 实机结果 |
|---|---:|---|
| HTP 全图 FP16 | 3497 节点 / 3 分区 | `qnn_graph_finalize` 失败，Error 1002；VTCM 装载失败 |
| HTP 宽覆盖 FP16 | 2586 / 4 | Transformer 与两个超长卷积留 CPU，仍在 finalize 阶段 Error 1002 |
| HTP 卷积专用 FP16 | 94 / 48 | 图准备 141.52 秒；NSP buffer mapping 8003，首次执行 Error 6001 |
| GPU 全图 FP16 | 3490 / 6 | OpenCL 图编译显存不足，finalize Error 6022 |
| GPU 全图 FP32 | 3490 / 6 | `softmax_phase2_image2darray` 编译显存不足，Error 6022 |
| GPU 宽覆盖 FP16 | 2580 / 8 | `reduce_buffer_subgroup_atomic` 编译显存不足，Error 6022 |
| GPU 宽覆盖 FP32 | 2580 / 8 | `zero_pad_image_float` 编译显存不足，Error 6022 |
| GPU 卷积专用 FP16 | 92 / 48 | F1 推理 12.56 秒、PSS 4.87 GB、peak 0.111，随后被系统杀死 |
| GPU 卷积专用 FP32 | 92 / 48 | 两段均完成，但质量严重失败，且速度/内存均劣于生产方案 |
| 生产混合 HTP FP16 + CPU FP32 | 11 / 7 | 两段均完成并通过 Torch 门禁 |

GPU 首轮测试曾报 `Invalid OpenCL driver path`。目标设备把
`libOpenCL.so` 列为公共 native library，但 targetSdk 36 应用仍需显式声明。
debug manifest 加入可选 `uses-native-library` 后，QNN 成功识别
`OpenCL 3.0 QUALCOMM build 0762.36.1`，上表 GPU 结论均来自驱动成功加载后的
真实图编译/推理，不是把配置问题误判为 GPU 不支持。

## 可执行候选的质量与性能

| 候选 | F1 corr / SNR | Wuyun corr / SNR | F1 / Wuyun 推理 | PSS |
|---|---:|---:|---:|---:|
| 生产混合方案 | 0.9999986304 / 55.62 dB | 0.9999991524 / 57.70 dB | 7.10 / 6.87 秒 | 1.43–1.51 GB |
| GPU 卷积 FP32 | 0.0094965 / -0.09 dB | -0.0001626 / -0.08 dB | 12.42 / 11.22 秒 | 4.86 GB |

生产方案冷图准备为 20.96 秒；同一图缓存命中时为 0.74 秒。GPU 卷积 FP32
准备为 7.88 秒，但其推理慢 63%–75%、内存约为 3.2 倍，而且没有通过质量
门禁，因此不具备部署价值。

## 最终选择

生产运行时保持：

```text
HTP FP16/HMX:
586, 867,
2639, 2645, 2777, 2784, 2923, 2929,
3157, 3164, 3303

其余 3493 个节点:
XNNPACK FP32，4 线程
```

这是本轮唯一同时满足可执行性、两段真实音频质量门禁、内存与速度要求的
方案。GroupNorm、注意力和其他归约算子继续留在 FP32 CPU；当前 QNN 2.48
与 SM8650 驱动组合下，不应把它们或全部卷积直接扩大到 HTP/GPU。

## 复现

设备候选入口为
[`InferenceBackendCandidateInstrumentedTest.kt`](../app/src/androidTest/java/cn/lonelyme/bandbuddy/InferenceBackendCandidateInstrumentedTest.kt)。
候选必须逐个启动 instrumentation 进程，避免 delegate/cache/显存状态互相污染：

```powershell
adb shell am instrument -w -r `
  -e class cn.lonelyme.bandbuddy.InferenceBackendCandidateInstrumentedTest#exportsTwoRealWindowsForTorchGate `
  -e backendCandidate production `
  cn.lonelyme.bandbuddy.test/androidx.test.runner.AndroidJUnitRunner
```

Android 14 的 app-specific external storage 会拒绝应用读取由 `adb shell`
直接创建的文件。校准输入应先推到 `/data/local/tmp`，再以应用 UID 复制：

```powershell
adb push build\calibration\device-8s\f1.interleaved.f32le /data/local/tmp/bandbuddy-f1.interleaved.f32le
adb shell chmod 644 /data/local/tmp/bandbuddy-f1.interleaved.f32le
adb shell "run-as cn.lonelyme.bandbuddy sh -c 'mkdir -p files/backend-matrix-input && cp /data/local/tmp/bandbuddy-f1.interleaved.f32le files/backend-matrix-input/f1.interleaved.f32le'"
```

输出拉回后，用固定官方 Torch 权重验收：

```powershell
python tools\validate_backend_candidates.py `
  --poc-root C:\path\to\BandBuddy-android-poc `
  --model-root build\torch-model-cache `
  --candidate production=build\device-output\backend-matrix\production `
  --output build\qnn-model\backend-candidate-matrix-validation.json
```

本轮完整成功候选报告为
[`backend-candidate-matrix-validation.json`](../build/qnn-model/backend-candidate-matrix-validation.json)；
失败候选的 instrumentation 与 logcat 证据位于
`build/backend-matrix/logs/`。
