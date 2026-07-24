package com.lonelyme.bandbuddy.engine

import java.nio.FloatBuffer

/**
 * Native boundary for the fixed HTDemucs core. The methods intentionally expose
 * no UI concerns: all model-domain DSP lives below this bridge.
 */
object NativeDemucsBridge {
    init { System.loadLibrary("bandbuddy_audio") }

    external fun contractSummary(): String
    external fun supportsCurrentAbi(): Boolean
    external fun preprocess(mix: FloatBuffer, specChannels: FloatBuffer): Boolean
    external fun postprocess(frequencyChannels: FloatBuffer, timeWaveform: FloatBuffer, output: FloatBuffer): Boolean
}
