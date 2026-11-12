package com.acul3.chatterboxtts.models

import android.content.Context
import android.util.Log
import com.acul3.chatterboxtts.tts.Constants
import com.acul3.chatterboxtts.tts.SpeechSampler
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.TensorBufferAccess
import java.nio.FloatBuffer

/**
 * RAM-efficient T3 prefill / decode for ExecuTorch 1.1.0.
 * The prefill tensor contains ~318 MB of interleaved K/V state.
 * It must never be converted to FloatArray on Android's capped Java heap.
 * Decode K/V outputs are copied into SharedMemory / mmap rather than retained
 * in Java-owned direct ByteBuffers, which also consume Android heap quota.
 */
class T3Decoder(
    private val context: Context,
    private val prefillModel: PteModel,
    private val decodeModel: PteModel
) {
    companion object {
        private const val TAG = "T3Decoder"
        private const val KV_HALF = 30 * 1 * 16 * 1293 * 64
        private const val KV_TOTAL = KV_HALF * 2
        private val KV_SHAPE = longArrayOf(30, 1, 16, 1293, 64)
    }

    private data class PrefillState(val logits: Tensor, val memory: KvMappedMemory)

    fun decode(
        condEmbedding: Tensor,
        textTokens: Tensor,
        maxSteps: Int = Constants.MAX_DECODE_STEPS,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): LongArray {
        require(maxSteps in 1..Constants.MAX_DECODE_STEPS)
        Log.i(TAG, "Starting prefill with mapped KV cache")
        onProgress(0f, "Running T3 prefill...")
        val initial = runPrefill(condEmbedding, textTokens)
        try {
            var logits = initial.logits
            val memory = initial.memory
            val keys = memory.keysTensor(KV_SHAPE)
            val values = memory.valuesTensor(KV_SHAPE)
            val result = ArrayList<Long>(maxSteps)
            // Load the decode model only after unloading prefill.
            decodeModel.load()
            try {
                onProgress(0.05f, "Prefill complete. Decoding speech tokens...")
                for (step in 0 until maxSteps) {
                    val data = logits.dataAsFloatArray // only 8,194 floats, never the KV tensor
                    val vocabLogits = if (data.size > Constants.SPEECH_VOCAB) {
                        data.copyOfRange(data.size - Constants.SPEECH_VOCAB, data.size)
                    } else data
                    val token = SpeechSampler.sampleToken(
                        logits = vocabLogits,
                        previousTokens = result.map { it.toInt() }
                    ).toLong()
                    if (token == Constants.EOT_SPEECH.toLong()) {
                        Log.i(TAG, "EOS at step $step")
                        break
                    }
                    result.add(token)
                    if (step % 25 == 0) {
                        val progress = 0.05f + 0.95f * step.toFloat() / maxSteps
                        onProgress(progress, "Decode step $step: ${result.size} tokens")
                    }
                    if (step + 1 == maxSteps) break
                    // Calling a separate method lets large decode output buffers become
                    // unreachable BEFORE the next iteration allocates new JNI outputs.
                    logits = decodeOne(token.toInt(), step, keys, values, memory)
                }
                onProgress(1f, "Generated ${result.size} speech tokens")
                return result.toLongArray()
            } finally {
                decodeModel.close()
            }
        } finally {
            initial.memory.close()
        }
    }

    private fun runPrefill(cond: Tensor, text: Tensor): PrefillState {
        var memory: KvMappedMemory? = null
        try {
            prefillModel.load()
            val outputs = prefillModel.forward(EValue.from(cond), EValue.from(text))
            check(outputs.size >= 2) { "T3 prefill returned only ${outputs.size} values" }
            val rawLogits = outputs[0].toTensor()
            val ownedLogits = Tensor.fromBlob(rawLogits.dataAsFloatArray, rawLogits.shape())
            val flat = outputs[1].toTensor()
            check(flat.numel() == KV_TOTAL.toLong()) { "Unexpected KV size: ${flat.numel()}" }
            val source = TensorBufferAccess.floatBuffer(flat)
            memory = KvMappedMemory(context, KV_HALF)
            val layerSize = KV_HALF / Constants.N_LAYERS
            // Prefill layout: [layer, 2 (K/V), batch, heads, time, headDim].
            // Decoder layout: K layers followed by V layers.
            for (layer in 0 until Constants.N_LAYERS) {
                val sourceStart = layer * layerSize * 2
                val targetStart = layer * layerSize
                memory.copyLayerFrom(source, sourceStart, memory.keys, targetStart, layerSize)
                memory.copyLayerFrom(source, sourceStart + layerSize, memory.values, targetStart, layerSize)
            }
            Log.i(TAG, "Prefill complete; KV mapped outside Java heap")
            return PrefillState(ownedLogits, memory)
        } catch (t: Throwable) {
            memory?.close()
            throw t
        } finally {
            prefillModel.close()
        }
    }

    private fun decodeOne(
        token: Int,
        step: Int,
        keys: Tensor,
        values: Tensor,
        memory: KvMappedMemory
    ): Tensor {
        val tokenTensor = Tensor.fromBlob(intArrayOf(token), longArrayOf(1, 1))
        val stepTensor = Tensor.fromBlob(intArrayOf(step), longArrayOf())
        val outputs = decodeModel.forward(
            EValue.from(tokenTensor), EValue.from(stepTensor),
            EValue.from(keys), EValue.from(values)
        )
        check(outputs.size >= 3) { "T3 decode returned only ${outputs.size} values" }
        // Copy small logits, and bulk-copy the 2 x 159 MB K/V outputs into mapped
        // memory without any large FloatArray or Java-owned KV buffer.
        val outputLogits = outputs[0].toTensor()
        val ownedLogits = Tensor.fromBlob(outputLogits.dataAsFloatArray, outputLogits.shape())
        val outK: FloatBuffer = TensorBufferAccess.floatBuffer(outputs[1].toTensor())
        val outV: FloatBuffer = TensorBufferAccess.floatBuffer(outputs[2].toTensor())
        memory.copyAllFrom(outK, memory.keys)
        memory.copyAllFrom(outV, memory.values)
        return ownedLogits
    }
}
