package com.acul3.chatterboxtts.models

import android.util.Log
import com.acul3.chatterboxtts.tts.Constants
import com.acul3.chatterboxtts.tts.SpeechSampler
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor

/**
 * T3 prefill + autoregressive decode loop.
 * Produces speech tokens from text + conditioning embeddings.
 *
 * Model interfaces (V4):
 *   t3_prefill.pte:
 *     inputs:  cond_emb (1,34,1024) float, text_tokens (1,258) int32
 *     outputs: logits (1,8194) float, kv_flat (79441920,) float
 *              kv_flat shape = (30, 2, 1, 16, 1293, 64) — split dim1 → keys/values
 *
 *   t3_decode.pte (V4):
 *     inputs:  prev_token (1,1) int32, step_idx () int32 scalar,
 *              kv_k (30,1,16,1293,64) float, kv_v (30,1,16,1293,64) float
 *     outputs: logits (1,8194) float,
 *              kv_k (30,1,16,1293,64) float, kv_v (30,1,16,1293,64) float
 */
class T3Decoder(
    private val prefillModel: PteModel,
    private val decodeModel: PteModel
) {

    companion object {
        private const val TAG = "T3Decoder"
        // KV flat size from prefill: 30 * 2 * 1 * 16 * 1293 * 64 = 79441920
        // Each layer stores its key block followed by its value block.
        private const val KV_HALF = 30 * 1 * 16 * 1293 * 64  // 39720960
        private val KV_SHAPE = longArrayOf(30, 1, 16, 1293, 64)
    }

    /**
     * Run the full T3 decode pipeline.
     *
     * @param condEmbedding Conditioning embedding tensor (1, 34, 1024)
     * @param textTokens Text token IDs (1, 258) int32
     * @param onProgress Callback with (progress 0-1, message)
     * @return Array of speech token IDs
     */
    fun decode(
        condEmbedding: Tensor,
        textTokens: Tensor,
        maxSteps: Int = Constants.MAX_DECODE_STEPS,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): LongArray {
        Log.i(TAG, "Starting T3 prefill...")
        onProgress(0f, "Running T3 prefill...")

        val initial = runPrefill(condEmbedding, textTokens)
        var currentLogits = initial.first
        var kvK = initial.second
        var kvV = initial.third

        Log.i(TAG, "Prefill done, starting decode loop...")
        onProgress(0.05f, "Prefill complete. Decoding speech tokens...")

        if (!decodeModel.isLoaded()) decodeModel.load()
        try {
        // Autoregressive decode
        val speechTokens = mutableListOf<Long>()
        require(maxSteps in 1..Constants.MAX_DECODE_STEPS)

        // First token is sampled from prefill logits
        for (step in 0 until maxSteps) {
            val logitsData = currentLogits.dataAsFloatArray

            // Take logits for the last position (shape is (1, SPEECH_VOCAB) already)
            val vocabLogits = if (logitsData.size > Constants.SPEECH_VOCAB) {
                logitsData.copyOfRange(logitsData.size - Constants.SPEECH_VOCAB, logitsData.size)
            } else {
                logitsData
            }

            val token = SpeechSampler.sampleToken(
                logits = vocabLogits,
                previousTokens = speechTokens.map { it.toInt() }
            ).toLong()

            // Check for EOS
            if (token == Constants.EOT_SPEECH.toLong()) {
                Log.i(TAG, "EOS token at step $step")
                break
            }

            speechTokens.add(token)

            // Progress update
            val progress = 0.05f + (step.toFloat() / maxSteps) * 0.95f
            if (step % 50 == 0) {
                onProgress(progress, "Decode step $step: ${speechTokens.size} tokens generated")
            }

            if (step + 1 == maxSteps) break

            // Prepare decode inputs
            // prev_token: (1, 1) int32
            val tokenTensor = Tensor.fromBlob(
                intArrayOf(token.toInt()),
                longArrayOf(1, 1)
            )
            // step_idx: () int32 scalar (0-indexed, step=0 is first generated token)
            val stepTensor = Tensor.fromBlob(
                intArrayOf(step),
                longArrayOf()  // scalar = empty shape
            )

            // Run decode step V4
            val decodeOutputs = decodeModel.forward(
                EValue.from(tokenTensor),
                EValue.from(stepTensor),
                EValue.from(kvK),
                EValue.from(kvV)
            )

            currentLogits = decodeOutputs[0].toTensor()
            kvK = decodeOutputs[1].toTensor()
            kvV = decodeOutputs[2].toTensor()
        }

        Log.i(TAG, "Decode complete: ${speechTokens.size} speech tokens")
        onProgress(1f, "Generated ${speechTokens.size} speech tokens")

        return speechTokens.toLongArray()
        } finally {
            decodeModel.close()
        }
    }
    private fun runPrefill(cond: Tensor, text: Tensor): Triple<Tensor, Tensor, Tensor> {
        if (!prefillModel.isLoaded()) prefillModel.load()
        try {
            val outputs = prefillModel.forward(EValue.from(cond), EValue.from(text))
            val logits = outputs[0].toTensor()
            // Own the output before releasing the multi-gigabyte prefill module.
            val ownedLogits = Tensor.fromBlob(logits.dataAsFloatArray, logits.shape())
            val (keys, values) = splitKvCache(outputs[1].toTensor())
            return Triple(ownedLogits, keys, values)
        } finally {
            prefillModel.close()
        }
    }

    /** The export flattens [layer, key/value, batch, head, time, dim]. */
    private fun splitKvCache(flat: Tensor): Pair<Tensor, Tensor> {
        val data = flat.dataAsFloatArray
        require(data.size == KV_HALF * 2) { "Unexpected KV cache size: ${data.size}" }
        val layerSize = KV_HALF / Constants.N_LAYERS
        // Direct buffers avoid two additional 159 MB Java heap arrays.
        val keys = Tensor.allocateFloatBuffer(KV_HALF)
        val values = Tensor.allocateFloatBuffer(KV_HALF)
        for (layer in 0 until Constants.N_LAYERS) {
            keys.put(data, layer * 2 * layerSize, layerSize)
            values.put(data, (layer * 2 + 1) * layerSize, layerSize)
        }
        keys.rewind()
        values.rewind()
        return Tensor.fromBlob(keys, KV_SHAPE) to Tensor.fromBlob(values, KV_SHAPE)
    }

}
