package com.lonelyme.bandbuddy.ui

import android.icu.text.Transliterator
import com.lonelyme.bandbuddy.data.SongRecord
import java.text.Normalizer
import java.util.Locale

internal data class SongSearchKey(
    val original: String,
    val pinyin: String,
    val initials: String
)

internal object PinyinSearch {
    private val hanToLatin by lazy {
        runCatching { Transliterator.getInstance("Han-Latin; Latin-ASCII; Lower()") }.getOrNull()
    }

    fun keyOf(song: SongRecord): SongSearchKey {
        val source = "${song.title} ${song.artist}"
        val transliterated = hanToLatin?.transliterate(source) ?: source
        val words = asciiWords(transliterated)
        return SongSearchKey(
            original = compact(source),
            pinyin = words.joinToString(""),
            initials = words.mapNotNull(String::firstOrNull).joinToString("")
        )
    }

    fun matches(key: SongSearchKey, query: String): Boolean {
        if (query.isBlank()) return true
        val compactQuery = compact(query)
        if (compactQuery.isBlank()) return true
        if (key.original.contains(compactQuery)) return true

        val transliteratedQuery = hanToLatin?.transliterate(query) ?: query
        val pinyinQuery = asciiWords(transliteratedQuery).joinToString("")
        return pinyinQuery.isNotBlank() &&
            (key.pinyin.contains(pinyinQuery) || key.initials.contains(pinyinQuery))
    }

    private fun compact(value: String): String = value
        .lowercase(Locale.ROOT)
        .filter(Char::isLetterOrDigit)

    private fun asciiWords(value: String): List<String> {
        val ascii = Normalizer.normalize(value, Normalizer.Form.NFD)
            .replace(Regex("""\p{M}+"""), "")
            .lowercase(Locale.ROOT)
        return Regex("""[a-z0-9]+""").findAll(ascii).map(MatchResult::value).toList()
    }
}
