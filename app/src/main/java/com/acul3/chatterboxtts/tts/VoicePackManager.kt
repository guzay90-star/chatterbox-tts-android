package com.acul3.chatterboxtts.tts

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.FileOutputStream
import java.io.ByteArrayOutputStream
import java.util.UUID
import java.util.zip.ZipInputStream

/** ZIP packs contain real precomputed speaker and vocoder conditioning vectors. */
data class VoicePack(val id: String, val title: String)

class VoicePackManager(private val context: Context) {
    companion object {
        const val DEFAULT_ID = "default"
        val SIZES = linkedMapOf(
            "speaker_emb.bin" to 1024,
            "cond_speech_tokens.bin" to 600,
            "prompt_tokens.bin" to 300,
            "prompt_token_len.bin" to 4,
            "xvector.bin" to 768,
            "prompt_mel.bin" to 100480
        )
    }
    private val root: File get() = File(context.filesDir, "voicepacks").apply { mkdirs() }

    fun list(): List<VoicePack> {
        val installed = root.listFiles()?.filter { folder ->
            folder.isDirectory && SIZES.all { (name, size) -> File(folder, name).length() == size.toLong() }
        }?.map { folder ->
            VoicePack(folder.name, File(folder, "name.txt").takeIf { it.isFile }?.readText()?.trim()?.take(50)?.ifBlank { folder.name } ?: folder.name)
        }?.sortedBy { it.title } ?: emptyList()
        return listOf(VoicePack(DEFAULT_ID, "Встроенный голос")) + installed
    }

    fun readAsset(packId: String, filename: String): ByteArray {
        require(filename in SIZES) { "Недопустимый параметр голоса: $filename" }
        if (packId == DEFAULT_ID) return context.assets.open(filename).use { it.readBytes() }
        require(packId.matches(Regex("[0-9a-f-]{36}"))) { "Недопустимый голосовой профиль" }
        val file = File(File(root, packId), filename)
        require(file.isFile && file.length() == SIZES.getValue(filename).toLong()) { "Голосовой пакет повреждён: $filename" }
        return file.readBytes()
    }

    fun importZip(uri: Uri): VoicePack {
        val id = UUID.randomUUID().toString()
        val target = File(root, id)
        check(target.mkdirs()) { "Не удалось создать каталог голоса" }
        try {
            val seen = HashSet<String>()
            var title = "Мой голос ${id.take(8)}"
            context.contentResolver.openInputStream(uri).use { fileStream ->
                requireNotNull(fileStream) { "Невозможно открыть файл" }
                ZipInputStream(fileStream).use { stream ->
                    while (true) {
                        val entry = stream.nextEntry ?: break
                        val name = entry.name
                        require(!entry.isDirectory && (name in SIZES || name == "name.txt")) {
                            "В архиве должен быть только набор голосовых параметров, без каталогов"
                        }
                        require(seen.add(name)) { "Повторяющийся файл $name" }
                        val limit = if (name == "name.txt") 200 else SIZES.getValue(name)
                        val collector = ByteArrayOutputStream(minOf(limit, 4096))
                        val block = ByteArray(4096)
                        var copied = 0
                        while (true) {
                            val read = stream.read(block, 0, minOf(block.size, limit + 1 - copied))
                            if (read == -1) break
                            if (read == 0) continue
                            copied += read
                            require(copied <= limit) { "Недопустимый размер файла $name" }
                            collector.write(block, 0, read)
                        }
                        val bytes = collector.toByteArray()
                        if (name == "name.txt") {
                            title = bytes.toString(Charsets.UTF_8).trim().replace(Regex("[\\r\\n]"), " ").take(50)
                            require(title.isNotBlank()) { "Укажите название голоса" }
                        } else {
                            require(bytes.size == limit) { "Недопустимый размер $name" }
                            FileOutputStream(File(target, name)).use { it.write(bytes) }
                        }
                        stream.closeEntry()
                    }
                }
            }
            require(SIZES.keys.all { it in seen }) { "Голосовой пакет неполный" }
            File(target, "name.txt").writeText(title)
            return VoicePack(id, title)
        } catch (t: Throwable) {
            target.deleteRecursively()
            throw t
        }
    }
}
