package com.lonelyme.bandbuddy.model

import com.lonelyme.bandbuddy.BuildConfig

/**
 * Immutable description of the model ABI expected by this app build.
 *
 * A future model release gets a new VERSION, REVISION, SHA256 and CACHE_TOKEN.
 * The old on-device version is kept until the replacement has been downloaded
 * and verified successfully.
 */
object ModelSpec {
    const val ID = "htdemucs-6s-core"
    const val VERSION = "1.0.0"
    const val REVISION = "v1.0.0"
    const val FILE_NAME = "htdemucs_6s.core.tflite"
    const val BYTES = 117_784_760L
    const val SHA256 = "a9fcc89e84aa65313e0540b582e710007ed12064969a0d49a3c85e49f1ae4e3d"
    const val CACHE_TOKEN = "bandbuddy-htdemucs-6s-7p8s-mixed-a9fcc89e-v1"

    val repository: String
        get() = BuildConfig.MODELSCOPE_REPOSITORY

    val downloadUrl: String
        get() = "https://www.modelscope.cn/models/$repository/resolve/$REVISION/$FILE_NAME"
}
