import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

val modelScopeRepository = providers.gradleProperty("BAND_BUDDY_MODELSCOPE_REPOSITORY")
    .orElse("Zzzzzzorz/BandBuddy-HTDemucs-6s")
val releaseKeystorePropertiesFile = rootProject.file("keystore.properties")
val releaseKeystoreProperties = Properties().apply {
    if (releaseKeystorePropertiesFile.isFile) {
        releaseKeystorePropertiesFile.inputStream().use(::load)
    }
}

android {
    namespace = "cn.lonelyme.bandbuddy"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "cn.lonelyme.bandbuddy"
        // The product target is Android 10 and above. Keeping this at 29 also
        // lets the connected physical device run the development build.
        minSdk = 29
        targetSdk = 36
        versionCode = 7
        versionName = "1.3.3"
        buildConfigField("String", "MODELSCOPE_REPOSITORY", "\"${modelScopeRepository.get()}\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++20"
            }
        }
    }

    signingConfigs {
        create("release") {
            releaseKeystoreProperties.getProperty("storeFile")?.let { storeFile = file(it) }
            releaseKeystoreProperties.getProperty("storePassword")?.let { storePassword = it }
            releaseKeystoreProperties.getProperty("keyAlias")?.let { keyAlias = it }
            releaseKeystoreProperties.getProperty("keyPassword")?.let { keyPassword = it }
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    packaging {
        jniLibs {
            // The CDSP daemon cannot mmap a skeleton directly from inside the
            // APK. Extract the matching QNN HTP skel beside the app libraries.
            useLegacyPackaging = true
        }
    }
    ndkVersion = "27.2.12479018"
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation("androidx.work:work-runtime-ktx:2.10.1")
    // Compatible devices accelerate the model with Qualcomm's QNN delegate on
    // the HTP NPU. Other devices use TensorFlow Lite's XNNPACK CPU backend.
    // The QNN versions are pinned together so the Java delegate and bundled
    // runtime use one ABI.
    // Qualcomm's delegate currently binds to the stable org.tensorflow.lite
    // Java Delegate API (the same runtime used by Qualcomm's Android samples).
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.48.0")
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.48.0")
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}
