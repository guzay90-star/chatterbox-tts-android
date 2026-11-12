package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import com.acul3.chatterboxtts.models.PteModel
import com.acul3.chatterboxtts.models.T3Decoder
import com.acul3.chatterboxtts.models.VocoderPipeline
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Main TTS engine class. Orchestrates the full Chatterbox pipeline:
 * Text → Tokenize → Condition → T3 Decode → Vocoder → Audio
 *
 * Pipeline:
 *   1. Load conditioning assets from disk
 *   2. t3_cond_speech_emb.pte:  cond_speech_tokens (1,150) → cond_speech_emb (1,150,1024)
 *   3. t3_cond_enc.pte:         (speaker_emb, cond_speech_emb, emotion_adv) → cond_emb (1,34,1024)
 *   4. Tokenize text → pad to (1, 258) with SOT=255 + 256 padded + EOT=0
 *   5. t3_prefill.pte:          (cond_emb, text_tokens) → (logits, kv_flat)
 *   6. T3 decode loop:          autoregressive speech token generation
 *   7. VocoderPipeline:         speech tokens → audio
 */
class ChatterboxTTS(private val context: Context) {

    companion object {
        private const val TAG = "ChatterboxTTS"
    }

    // VOICE_V2_PATCH — voice pack and expression are applied before each generation.
    private val voicePackManager = VoicePackManager(context)
    private var activePackId = VoicePackManager.DEFAULT_ID
    private var activeExpression = 0.5f
    private val modelManager = ModelManager(context)
    private var tokenizer: TextTokenizer? = null

    // Models
    private var condSpeechEmbModel: PteModel? = null   // t3_cond_speech_emb.pte
    private var condEncModel: PteModel? = null          // t3_cond_enc.pte
    private var t3Decoder: T3Decoder? = null
    private var vocoderPipeline: VocoderPipeline? = null

    // Cached conditioning tensors (loaded from assets)
    private var speakerEmb: Tensor? = null          // (1, 256) float32
    private var condSpeechTokens: Tensor? = null    // (1, 150) int64
    private var emotionAdv: Tensor? = null          // (1, 1, 1) float32

    // Cached pre-computed cond embedding (result of cond_enc, valid for default voice)
    private var cachedCondEmb: Tensor? = null       // (1, 34, 1024) float32

    suspend fun ensureModelsReady(onProgress: (Float, String) -> Unit) {
        if (!modelManager.allModelsReady()) {
            modelManager.ensureModelsDownloaded(onProgress)
        } else {
            onProgress(1f, "All models cached.")
        }
    }

    fun loadModels(onProgress: (Float, String) -> Unit) {
        if (t3Decoder != null && cachedCondEmb != null) {
            onProgress(1f, "Models already loaded")
            return
        }

        // Initialize tokenizer
        onProgress(0f, "Loading tokenizer...")
        tokenizer = TextTokenizer(context)

        // Keep model descriptors, but do NOT load all seven models together.
        // Prefill and decoder can consume several gigabytes of memory.
        onProgress(0.1f, "Preparing models for lazy loading...")
        condSpeechEmbModel = PteModel(modelManager.getModelPath("t3_cond_speech_emb.pte"))
        condEncModel = PteModel(modelManager.getModelPath("t3_cond_enc.pte"))
        val prefill = PteModel(modelManager.getModelPath("t3_prefill.pte"))
        val decode = PteModel(modelManager.getModelPath("t3_decode.pte"))
        t3Decoder = T3Decoder(context, prefill, decode)
        val s3gen = PteModel(modelManager.getModelPath("s3gen_encoder.pte"))
        val cfm = PteModel(modelManager.getModelPath("cfm_step.pte"))
        val hifigan = PteModel(modelManager.getModelPath("hifigan.pte"))
        vocoderPipeline = VocoderPipeline(context, s3gen, cfm, hifigan)

        // Load default voice conditioning from assets
        onProgress(0.9f, "Loading default voice...")
        loadDefaultConditioning()

        // Pre-compute cond_emb (speaker-specific, can be cached)
        onProgress(0.95f, "Pre-computing conditioning embedding...")
        try {
            precomputeCondEmb()
        } catch (e: Exception) {
            Log.e(TAG, "precomputeCondEmb FAILED", e)
            throw RuntimeException("Conditioning failed: ${e.message}\n${e.stackTraceToString().take(300)}", e)
        }

        onProgress(1f, "All models loaded!")
        Log.i(TAG, "All models loaded successfully")
    }

    /**
     * Generate speech audio from text.
     *
     * @param text Input text to speak
     * @param language Language code (e.g., "en", "fr", "ms")
     * @param onProgress Progress callback (0-1, message)
     * @return WAV byte array
     */
    fun generate(
        text: String,
        language: String,
        voicePackId: String = VoicePackManager.DEFAULT_ID,
        expression: Float = 0.5f,
        speed: Float = 1f,
        pitchSemitones: Float = 0f,
        warmth: Float = 0f,
        texture: Float = 0f,
        onProgress: (Float, String) -> Unit
    ): ByteArray {
        val tok = tokenizer ?: throw IllegalStateException("Tokenizer not loaded")
        prepareVoice(voicePackId, expression)
        vocoderPipeline?.selectVoice(voicePackId)
        val condEmb = cachedCondEmb ?: throw IllegalStateException("Conditioning not computed")

        // Step 1: Tokenize text
        onProgress(0f, "Tokenizing text...")
        val rawTokenIds = tok.encode(text, language).drop(1).dropLast(1)
        Log.i(TAG, "Tokenized: ${rawTokenIds.size} raw tokens")

        // Pad/truncate to exactly 256 positions, then wrap with SOT=255 and EOT=0
        // Total: 1 SOT + 256 tokens + 1 EOT = 258
        val textSeq = IntArray(Constants.TEXT_SEQ_LEN) { 0 }  // default padding = EOT
        textSeq[0] = Constants.SOT_TEXT
        val copyLen = minOf(rawTokenIds.size, Constants.MAX_TEXT_LEN)
        for (i in 0 until copyLen) {
            textSeq[1 + i] = rawTokenIds[i]
        }
        textSeq[Constants.TEXT_SEQ_LEN - 1] = Constants.EOT_TEXT

        val textTensor = Tensor.fromBlob(textSeq, longArrayOf(1, Constants.TEXT_SEQ_LEN.toLong()))

        // Step 2: T3 decode (autoregressive speech token generation)
        onProgress(0.05f, "Starting T3 decode...")
        val speechTokens = t3Decoder!!.decode(
            condEmbedding = condEmb,
            textTokens = textTensor
        ) { progress, message ->
            onProgress(0.05f + progress * 0.55f, message)
        }
        Log.i(TAG, "T3 decode produced ${speechTokens.size} speech tokens")

        // Step 3: Vocoder (speech tokens → audio)
        onProgress(0.6f, "Running vocoder...")
        val audioSamples = vocoderPipeline!!.synthesize(
            speechTokens = speechTokens
        ) { progress, message ->
            onProgress(0.6f + progress * 0.35f, message)
        }

        // Step 4: Convert to WAV
        onProgress(0.95f, "Encoding WAV...")
        val adjustedSamples = VoiceEffects.process(audioSamples, speed, pitchSemitones, warmth, texture)
        val wavData = AudioPlayer.floatToWav(adjustedSamples, Constants.S3GEN_SR)
        val durationSec = adjustedSamples.size.toFloat() / Constants.S3GEN_SR

        onProgress(1f, "Done! ${String.format("%.1f", durationSec)}s of audio generated.")
        Log.i(TAG, "Generated ${wavData.size} bytes WAV (${durationSec}s)")

        return wavData
    }

    /**
     * Load default voice conditioning tensors from bundled binary assets.
     */
    private fun loadDefaultConditioning() {
        try {
            speakerEmb = loadFloatAsset("speaker_emb.bin", longArrayOf(1, 256))
            condSpeechTokens = loadIntAsset("cond_speech_tokens.bin", longArrayOf(1, 150))
            emotionAdv = loadFloatAsset("emotion_adv.bin", longArrayOf(1, 1, 1))
            Log.i(TAG, "Loaded conditioning tensors from assets")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load conditioning assets: ${e.message}", e)
            throw RuntimeException("Conditioning assets missing or corrupted: ${e.message}", e)
        }
    }

    /**
     * Pre-compute the conditioning embedding (cached once per voice).
     * Flow: cond_speech_tokens → t3_cond_speech_emb → cond_speech_emb
     *       (speaker_emb, cond_speech_emb, emotion_adv) → t3_cond_enc → cond_emb
     */
    private fun precomputeCondEmb() {
        val cst = condSpeechTokens ?: throw IllegalStateException("condSpeechTokens not loaded")
        val spk = speakerEmb ?: throw IllegalStateException("speakerEmb not loaded")
        val emo = emotionAdv ?: throw IllegalStateException("emotionAdv not loaded")

        // Step a: embed cond speech tokens (1, 150) → (1, 150, 1024)
        Log.i(TAG, "Running t3_cond_speech_emb...")
        val condSpeechEmb = try {
            condSpeechEmbModel!!.load()
            val tensor = condSpeechEmbModel!!.forward(EValue.from(cst))[0].toTensor()
            Tensor.fromBlob(tensor.dataAsFloatArray, tensor.shape())
        } finally {
            condSpeechEmbModel?.close()
        }
        Log.i(TAG, "cond_speech_emb shape: ${condSpeechEmb.shape().contentToString()}")

        // Step b: encode conditioning (speaker_emb, cond_speech_emb, emotion_adv) → (1, 34, 1024)
        Log.i(TAG, "Running t3_cond_enc...")
        try {
            condEncModel!!.load()
            val encoded = condEncModel!!.forward(
                EValue.from(spk), EValue.from(condSpeechEmb), EValue.from(emo)
            )[0].toTensor()
            cachedCondEmb = Tensor.fromBlob(encoded.dataAsFloatArray, encoded.shape())
        } finally {
            condEncModel?.close()
        }
        Log.i(TAG, "cond_emb shape: ${cachedCondEmb!!.shape().contentToString()}")
    }

    /** Switches neural conditioning only when actual speaker or expression changes. */
    private fun prepareVoice(packId: String, expression: Float) {
        val adjustedExpression = expression.coerceIn(0.25f, 1.5f)
        if (cachedCondEmb != null && activePackId == packId && activeExpression == adjustedExpression) return
        val speakerBytes = voicePackManager.readAsset(packId, "speaker_emb.bin")
        val tokenBytes = voicePackManager.readAsset(packId, "cond_speech_tokens.bin")
        speakerEmb = tensorFromFloatBytes(speakerBytes, longArrayOf(1, 256))
        condSpeechTokens = tensorFromIntBytes(tokenBytes, longArrayOf(1, 150))
        emotionAdv = Tensor.fromBlob(floatArrayOf(adjustedExpression), longArrayOf(1, 1, 1))
        cachedCondEmb = null
        precomputeCondEmb()
        activePackId = packId
        activeExpression = adjustedExpression
    }

    private fun tensorFromFloatBytes(bytes: ByteArray, shape: LongArray): Tensor {
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val data = FloatArray(fb.remaining())
        fb.get(data)
        return Tensor.fromBlob(data, shape)
    }

    private fun tensorFromIntBytes(bytes: ByteArray, shape: LongArray): Tensor {
        val ib = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
        val data = IntArray(ib.remaining())
        ib.get(data)
        return Tensor.fromBlob(data, shape)
    }

    // ── Asset loading helpers ─────────────────────────────────────────────────

    fun loadFloatAsset(name: String, shape: LongArray): Tensor {
        val bytes = context.assets.open(name).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4)
        buffer.asFloatBuffer().get(floats)
        return Tensor.fromBlob(floats, shape)
    }

    fun loadIntAsset(name: String, shape: LongArray): Tensor {
        val bytes = context.assets.open(name).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val ints = IntArray(bytes.size / 4)
        buffer.asIntBuffer().get(ints)
        // ExecuTorch Android: int[] creates INT32 tensor (matches re-exported models)
        return Tensor.fromBlob(ints, shape)
    }

    fun close() {
        condSpeechEmbModel?.close()
        condEncModel?.close()
    }
}
