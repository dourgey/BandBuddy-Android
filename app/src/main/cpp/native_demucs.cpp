#include <jni.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

namespace {
constexpr int kSampleRate = 44100;
constexpr int kChannels = 2;
constexpr int kFftSize = 4096;
constexpr int kHopSize = 1024;
constexpr int kBins = 2048;
constexpr int kSources = 6;
constexpr int kOuterLeft = 1536;
constexpr int kCenterPad = 2048;
constexpr float kPi = 3.14159265358979323846f;

int reflectIndex(int index, int length) {
    while (index < 0 || index >= length) {
        if (index < 0) index = -index;
        if (index >= length) index = 2 * length - 2 - index;
    }
    return index;
}

struct FftPlan {
    std::vector<int> reversed;
    std::vector<float> cosine;
    std::vector<float> sine;
    std::vector<float> window;

    FftPlan()
        : reversed(kFftSize),
          cosine(kFftSize / 2),
          sine(kFftSize / 2),
          window(kFftSize) {
        constexpr int bits = 12;
        for (int value = 0; value < kFftSize; ++value) {
            int source = value;
            int destination = 0;
            for (int bit = 0; bit < bits; ++bit) {
                destination = (destination << 1) | (source & 1);
                source >>= 1;
            }
            reversed[value] = destination;
        }
        for (int index = 0; index < kFftSize / 2; ++index) {
            const float angle = 2.f * kPi * static_cast<float>(index) / kFftSize;
            cosine[index] = std::cos(angle);
            sine[index] = std::sin(angle);
        }
        for (int index = 0; index < kFftSize; ++index) {
            window[index] = .5f - .5f * std::cos(
                2.f * kPi * static_cast<float>(index) / kFftSize);
        }
    }
};

const FftPlan& fftPlan() {
    static const FftPlan plan;
    return plan;
}

void fft(std::vector<float>& real, std::vector<float>& imaginary, bool inverse) {
    const auto& plan = fftPlan();
    for (int index = 0; index < kFftSize; ++index) {
        const int reversed = plan.reversed[index];
        if (index < reversed) {
            std::swap(real[index], real[reversed]);
            std::swap(imaginary[index], imaginary[reversed]);
        }
    }
    for (int length = 2; length <= kFftSize; length <<= 1) {
        const int half = length / 2;
        const int twiddleStep = kFftSize / length;
        for (int offset = 0; offset < kFftSize; offset += length) {
            for (int index = 0; index < half; ++index) {
                const int twiddle = index * twiddleStep;
                const float wr = plan.cosine[twiddle];
                const float wi = inverse ? plan.sine[twiddle] : -plan.sine[twiddle];
                const int evenIndex = offset + index;
                const int oddIndex = evenIndex + half;
                const float oddReal = real[oddIndex] * wr - imaginary[oddIndex] * wi;
                const float oddImaginary = real[oddIndex] * wi + imaginary[oddIndex] * wr;
                const float evenReal = real[evenIndex];
                const float evenImaginary = imaginary[evenIndex];
                real[evenIndex] = evenReal + oddReal;
                imaginary[evenIndex] = evenImaginary + oddImaginary;
                real[oddIndex] = evenReal - oddReal;
                imaginary[oddIndex] = evenImaginary - oddImaginary;
            }
        }
    }
    if (inverse) {
        constexpr float scale = 1.f / static_cast<float>(kFftSize);
        for (int index = 0; index < kFftSize; ++index) {
            real[index] *= scale;
            imaginary[index] *= scale;
        }
    }
}

jlong bufferCapacity(JNIEnv* env, jobject buffer) {
    if (buffer == nullptr || env->GetDirectBufferAddress(buffer) == nullptr) return -1;
    return env->GetDirectBufferCapacity(buffer);
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_cn_lonelyme_bandbuddy_engine_NativeDemucsBridge_contractSummary(JNIEnv* env, jobject) {
    return env->NewStringUTF("htdemucs_6s: 44.1kHz stereo, variable window, 4096 FFT, 1024 hop, 6 stems");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_cn_lonelyme_bandbuddy_engine_NativeDemucsBridge_supportsCurrentAbi(JNIEnv*, jobject) {
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_cn_lonelyme_bandbuddy_engine_NativeDemucsBridge_preprocess(
        JNIEnv* env, jobject, jobject mixBuffer, jobject specBuffer) {
    const jlong mixCount = bufferCapacity(env, mixBuffer);
    const jlong specCount = bufferCapacity(env, specBuffer);
    if (mixCount <= 0 || mixCount % kChannels != 0 || specCount <= 0 ||
        specCount % (4 * kBins) != 0) return JNI_FALSE;
    const int samples = static_cast<int>(mixCount / kChannels);
    const int frames = static_cast<int>(specCount / (4 * kBins));
    if (samples < kSampleRate || frames != (samples + kHopSize - 1) / kHopSize) return JNI_FALSE;
    const int outerLength = frames * kHopSize + 2 * kOuterLeft;
    const auto* mix = static_cast<const float*>(env->GetDirectBufferAddress(mixBuffer));
    auto* spec = static_cast<float*>(env->GetDirectBufferAddress(specBuffer));
    const auto& plan = fftPlan();
    const float normalization = std::sqrt(static_cast<float>(kFftSize));

    const auto processChannel = [&](int channel) {
        std::vector<float> real(kFftSize);
        std::vector<float> imaginary(kFftSize, 0.f);
        for (int outputFrame = 0; outputFrame < frames; ++outputFrame) {
            const int fullFrame = outputFrame + 2;
            const int outerStart = fullFrame * kHopSize - kCenterPad;
            for (int sample = 0; sample < kFftSize; ++sample) {
                const int outerIndex = reflectIndex(outerStart + sample, outerLength);
                const int rawIndex = reflectIndex(outerIndex - kOuterLeft, samples);
                real[sample] = mix[channel * samples + rawIndex] * plan.window[sample];
                imaginary[sample] = 0.f;
            }
            fft(real, imaginary, false);
            for (int bin = 0; bin < kBins; ++bin) {
                const int realComponent = channel * 2;
                const int imagComponent = realComponent + 1;
                const int realIndex = (realComponent * kBins + bin) * frames + outputFrame;
                const int imagIndex = (imagComponent * kBins + bin) * frames + outputFrame;
                spec[realIndex] = real[bin] / normalization;
                spec[imagIndex] = imaginary[bin] / normalization;
            }
        }
    };
    std::thread secondChannel(processChannel, 1);
    processChannel(0);
    secondChannel.join();
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_cn_lonelyme_bandbuddy_engine_NativeDemucsBridge_postprocess(
        JNIEnv* env, jobject, jobject frequencyBuffer, jobject timeBuffer, jobject outputBuffer) {
    const jlong frequencyCount = bufferCapacity(env, frequencyBuffer);
    const jlong waveformCount = bufferCapacity(env, timeBuffer);
    const jlong outputCount = bufferCapacity(env, outputBuffer);
    constexpr int frequencyPlane = kSources * 4 * kBins;
    constexpr int waveformPlane = kSources * kChannels;
    if (frequencyCount <= 0 || frequencyCount % frequencyPlane != 0 ||
        waveformCount <= 0 || waveformCount % waveformPlane != 0 ||
        outputCount != waveformCount) return JNI_FALSE;
    const int frames = static_cast<int>(frequencyCount / frequencyPlane);
    const int samples = static_cast<int>(waveformCount / waveformPlane);
    if (samples < kSampleRate || frames != (samples + kHopSize - 1) / kHopSize) return JNI_FALSE;
    const int fullFrames = frames + 4;
    const int reconstructedLength = (fullFrames - 1) * kHopSize + kFftSize;
    const auto* frequency = static_cast<const float*>(env->GetDirectBufferAddress(frequencyBuffer));
    const auto* time = static_cast<const float*>(env->GetDirectBufferAddress(timeBuffer));
    auto* output = static_cast<float*>(env->GetDirectBufferAddress(outputBuffer));
    const auto& plan = fftPlan();
    const float inverseNormalization = std::sqrt(static_cast<float>(kFftSize));
    const int cropStart = kCenterPad + kOuterLeft;
    std::vector<float> inverseEnvelope(reconstructedLength, 0.f);
    for (int frame = 0; frame < fullFrames; ++frame) {
        const int offset = frame * kHopSize;
        for (int sample = 0; sample < kFftSize; ++sample) {
            inverseEnvelope[offset + sample] += plan.window[sample] * plan.window[sample];
        }
    }
    for (float& value : inverseEnvelope) {
        value = value > 1e-10f ? 1.f / value : 0.f;
    }

    std::atomic<int> nextPlane{0};
    const auto processPlanes = [&]() {
        std::vector<float> real(kFftSize);
        std::vector<float> imaginary(kFftSize);
        std::vector<float> reconstructed(reconstructedLength);
        while (true) {
            const int plane = nextPlane.fetch_add(1, std::memory_order_relaxed);
            if (plane >= kSources * kChannels) return;
            const int source = plane / kChannels;
            const int channel = plane % kChannels;
            std::fill(reconstructed.begin(), reconstructed.end(), 0.f);
            for (int fullFrame = 0; fullFrame < fullFrames; ++fullFrame) {
                std::fill(real.begin(), real.end(), 0.f);
                std::fill(imaginary.begin(), imaginary.end(), 0.f);
                if (fullFrame >= 2 && fullFrame < frames + 2) {
                    const int modelFrame = fullFrame - 2;
                    const int realComponent = source * 4 + channel * 2;
                    const int imagComponent = realComponent + 1;
                    for (int bin = 0; bin < kBins; ++bin) {
                        const int realIndex = (realComponent * kBins + bin) * frames + modelFrame;
                        const int imagIndex = (imagComponent * kBins + bin) * frames + modelFrame;
                        real[bin] = frequency[realIndex];
                        imaginary[bin] = frequency[imagIndex];
                    }
                    for (int bin = 1; bin < kBins; ++bin) {
                        real[kFftSize - bin] = real[bin];
                        imaginary[kFftSize - bin] = -imaginary[bin];
                    }
                }
                fft(real, imaginary, true);
                const int offset = fullFrame * kHopSize;
                for (int sample = 0; sample < kFftSize; ++sample) {
                    reconstructed[offset + sample] +=
                        real[sample] * inverseNormalization * plan.window[sample];
                }
            }
            const int base = plane * samples;
            for (int sample = 0; sample < samples; ++sample) {
                const int reconstructedIndex = cropStart + sample;
                output[base + sample] = time[base + sample] +
                    reconstructed[reconstructedIndex] * inverseEnvelope[reconstructedIndex];
            }
        }
    };
    constexpr int workerCount = 4;
    std::vector<std::thread> workers;
    workers.reserve(workerCount - 1);
    for (int worker = 1; worker < workerCount; ++worker) workers.emplace_back(processPlanes);
    processPlanes();
    for (auto& worker : workers) worker.join();
    return JNI_TRUE;
}
