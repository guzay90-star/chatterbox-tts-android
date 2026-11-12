package com.acul3.chatterboxtts.tts

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class TTSState(
    val inputText: String = "Hello, welcome to Chatterbox text to speech!",
    val selectedLanguage: String = "ru",
    val voicePacks: List<VoicePack> = listOf(VoicePack("default", "Встроенный голос")),
    val selectedVoicePack: String = "default",
    val styleId: String = "natural",
    val speed: Float = 1f,
    val pitch: Float = 0f,
    val expression: Float = 0.5f,
    val isGenerating: Boolean = false,
    val progress: Float = 0f,
    val progressMessage: String = "",
    val error: String? = null,
    val audioReady: Boolean = false,
    val isPlaying: Boolean = false,
    val modelStatus: String = "Models not loaded. Will download on first generation (~2.6GB)."
)

class TTSViewModel : ViewModel() {
    private val _state = MutableStateFlow(TTSState())
    val state: StateFlow<TTSState> = _state.asStateFlow()

    // VOICE_V2_PATCH
    private var preferences: SharedPreferences? = null
    private var ttsEngine: ChatterboxTTS? = null
    private var audioPlayer: AudioPlayer? = null
    private var lastAudioData: ByteArray? = null

    fun updateText(text: String) {
        _state.update { it.copy(inputText = text) }
    }

    fun updateLanguage(lang: String) {
        _state.update { it.copy(selectedLanguage = lang) }
    }

    fun refreshVoicePacks(context: Context) {
        val packs = VoicePackManager(context).list()
        if (preferences == null) {
            preferences = context.getSharedPreferences("voice-controls-v2", Context.MODE_PRIVATE)
            val pref = preferences!!
            val styleId = pref.getString("style", "natural") ?: "natural"
            val selected = pref.getString("voice", "default") ?: "default"
            val style = VoiceStyles.get(styleId)
            _state.update { it.copy(
                voicePacks = packs,
                selectedVoicePack = selected.takeIf { id -> packs.any { pack -> pack.id == id } } ?: "default",
                styleId = style.id,
                speed = pref.getFloat("speed", style.speed).coerceIn(0.65f, 1.5f),
                pitch = pref.getFloat("pitch", style.pitchSemitones).coerceIn(-7f, 7f),
                expression = pref.getFloat("expression", 0.5f).coerceIn(0.25f, 1.5f)
            ) }
        } else {
            _state.update { it.copy(voicePacks = packs) }
        }
    }

    private fun persist() {
        val v = _state.value
        preferences?.edit()?.putString("style", v.styleId)
            ?.putString("voice", v.selectedVoicePack)
            ?.putFloat("speed", v.speed)
            ?.putFloat("pitch", v.pitch)
            ?.putFloat("expression", v.expression)
            ?.apply()
    }

    fun setVoicePack(id: String) {
        if (_state.value.isGenerating || _state.value.voicePacks.none { it.id == id }) return
        _state.update { it.copy(selectedVoicePack = id) }
        persist()
    }
    fun setStyle(id: String) {
        if (_state.value.isGenerating) return
        val style = VoiceStyles.get(id)
        _state.update { it.copy(styleId = style.id, speed = style.speed, pitch = style.pitchSemitones) }
        persist()
    }
    fun setSpeed(value: Float) {
        if (_state.value.isGenerating) return
        _state.update { it.copy(speed = value.coerceIn(0.65f, 1.5f)) }
        persist()
    }
    fun setPitch(value: Float) {
        if (_state.value.isGenerating) return
        _state.update { it.copy(pitch = value.coerceIn(-7f, 7f)) }
        persist()
    }
    fun setExpression(value: Float) {
        if (_state.value.isGenerating) return
        _state.update { it.copy(expression = value.coerceIn(0.25f, 1.5f)) }
        persist()
    }
    fun importVoicePack(context: Context, uri: Uri) {
        if (_state.value.isGenerating) return
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val pack = VoicePackManager(context).importZip(uri)
                _state.update { it.copy(voicePacks = VoicePackManager(context).list(), selectedVoicePack = pack.id, error = null) }
                persist()
            } catch (e: Exception) {
                _state.update { it.copy(error = "Не удалось импортировать голос: ${e.message}") }
            }
        }
    }

    suspend fun generate(context: Context) {
        val currentState = _state.value
        if (currentState.isGenerating || currentState.inputText.isBlank()) return

        _state.update {
            it.copy(
                isGenerating = true,
                progress = 0f,
                progressMessage = "Initializing...",
                error = null,
                audioReady = false
            )
        }

        try {
            withContext(Dispatchers.IO) {
                // Initialize engine if needed
                if (ttsEngine == null) {
                    _state.update { it.copy(progressMessage = "Setting up TTS engine...") }
                    ttsEngine = ChatterboxTTS(context)
                }

                val engine = ttsEngine!!

                // Download models if needed
                _state.update { it.copy(progressMessage = "Checking models...") }
                engine.ensureModelsReady { progress, message ->
                    _state.update {
                        it.copy(
                            progress = progress,
                            progressMessage = message,
                            modelStatus = message
                        )
                    }
                }

                // Load models
                _state.update {
                    it.copy(
                        progress = 0.3f,
                        progressMessage = "Loading models..."
                    )
                }
                engine.loadModels { progress, message ->
                    _state.update {
                        it.copy(
                            progress = 0.3f + progress * 0.2f,
                            progressMessage = message
                        )
                    }
                }

                // Generate
                _state.update {
                    it.copy(
                        progress = 0.5f,
                        progressMessage = "Generating speech..."
                    )
                }

                val selectedStyle = VoiceStyles.get(currentState.styleId)
                val audioData = engine.generate(
                    text = currentState.inputText,
                    language = currentState.selectedLanguage,
                    voicePackId = currentState.selectedVoicePack,
                    expression = currentState.expression,
                    speed = currentState.speed,
                    pitchSemitones = currentState.pitch,
                    warmth = selectedStyle.warmth,
                    texture = selectedStyle.texture
                ) { progress, message ->
                    _state.update {
                        it.copy(
                            progress = 0.5f + progress * 0.5f,
                            progressMessage = message
                        )
                    }
                }

                lastAudioData = audioData

                _state.update {
                    it.copy(
                        isGenerating = false,
                        progress = 1f,
                        progressMessage = "Done! Generated ${audioData.size / 1024}KB of audio.",
                        audioReady = true,
                        modelStatus = "Models loaded and ready."
                    )
                }
            }
        } catch (e: Throwable) {
            Log.e("TTSViewModel", "Generation failed", e)
            _state.update {
                it.copy(
                    isGenerating = false,
                    progress = 0f,
                    progressMessage = "",
                    error = "Generation failed: ${e.message}\n${e.stackTraceToString().take(500)}"
                )
            }
        }
    }

    fun playAudio(context: Context) {
        val data = lastAudioData ?: return
        if (audioPlayer == null) {
            audioPlayer = AudioPlayer()
        }
        audioPlayer?.play(data, Constants.S3GEN_SR) {
            _state.update { it.copy(isPlaying = false) }
        }
        _state.update { it.copy(isPlaying = true) }
    }

    fun stopPlayback() {
        audioPlayer?.stop()
        _state.update { it.copy(isPlaying = false) }
    }

    fun saveWavToUri(context: Context, uri: Uri) {
        val data = lastAudioData ?: throw IllegalStateException("No audio data")
        val stream = context.contentResolver.openOutputStream(uri) ?: throw IllegalStateException("Cannot open output file")
        stream.use { it.write(data) }
    }

    fun saveWavToFile(file: File) {
        val data = lastAudioData ?: throw IllegalStateException("No audio data")
        AudioPlayer.writeWavFile(file, data, Constants.S3GEN_SR)
    }

    override fun onCleared() {
        super.onCleared()
        audioPlayer?.release()
        ttsEngine?.close()
    }
}
