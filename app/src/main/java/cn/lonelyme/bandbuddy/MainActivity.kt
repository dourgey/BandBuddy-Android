package cn.lonelyme.bandbuddy

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.content.Intent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.runtime.mutableStateOf
import cn.lonelyme.bandbuddy.ui.BandBuddyApp
import cn.lonelyme.bandbuddy.ui.theme.BandBuddyTheme

class MainActivity : ComponentActivity() {
    private val requestedSongId = mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedSongId.value = intent.getStringExtra("open_song_id")
        enableEdgeToEdge()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 41)
        }
        setContent { BandBuddyTheme { BandBuddyApp(requestedSongId.value) { requestedSongId.value = null } } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        requestedSongId.value = intent.getStringExtra("open_song_id")
    }
}
