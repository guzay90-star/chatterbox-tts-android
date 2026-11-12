package com.acul3.chatterboxtts.tts

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.tanh

/** Offline PCM post-processing. Styles modify the same speaker; they do not clone a voice. */
data class VoiceStyle(
    val id: String,
    val name: String,
    val description: String,
    val pitchSemitones: Float,
    val speed: Float,
    val warmth: Float,
    val texture: Float
)

object VoiceStyles {
    val all = listOf(
        VoiceStyle("natural", "Оригинальный", "Без обработки", 0f, 1f, 0f, 0f),
        VoiceStyle("deep", "Низкий", "Более низкий и размеренный", -3f, 0.94f, 0.25f, 0f),
        VoiceStyle("grandpa", "Дедушка", "Низкий, медленный, с лёгкой хрипотцой", -3.5f, 0.85f, 0.48f, 0.23f),
        VoiceStyle("radio", "Диктор", "Ровный и тёплый", -1f, 0.96f, 0.3f, 0.03f),
        VoiceStyle("bright", "Звонкий", "Выше и быстрее", 2.5f, 1.08f, 0f, 0f),
        VoiceStyle("cartoon", "Мультяшный", "Высокий комический голос", 4.5f, 1.17f, 0f, 0f)
    )
    fun get(id: String): VoiceStyle = all.firstOrNull { it.id == id } ?: all.first()
}

object VoiceEffects {
    /** Speed changes tempo; pitch is independently adjusted through resampling + overlap-add stretching. */
    fun process(
        input: FloatArray,
        speed: Float,
        pitchSemitones: Float,
        warmth: Float,
        texture: Float
    ): FloatArray {
        if (input.isEmpty()) return input
        val s = speed.coerceIn(0.65f, 1.5f)
        val p = pitchSemitones.coerceIn(-7f, 7f)
        val w = warmth.coerceIn(0f, 1f)
        val t = texture.coerceIn(0f, 1f)
        if (s == 1f && p == 0f && w == 0f && t == 0f) return input
        val pitchFactor = 2.0.pow((p / 12f).toDouble()).toFloat()
        val pitched = if (abs(pitchFactor - 1f) < 0.0001f) input else resample(input, pitchFactor)
        val targetRate = s / pitchFactor
        val output = if (abs(targetRate - 1f) < 0.0001f) pitched.copyOf() else stretch(pitched, targetRate)
        var low = 0f
        var peak = 0f
        for (i in output.indices) {
            val v = output[i]
            low += 0.10f * (v - low)
            val warm = v * (1f - w * 0.36f) + low * w * 0.36f
            val rough = tanh((warm * 2.1f).toDouble()).toFloat() / 0.97045f
            val textured = warm * (1f - t * 0.38f) + rough * t * 0.38f
            output[i] = textured
            peak = maxOf(peak, abs(textured))
        }
        if (peak > 1f) {
            val gain = 0.98f / peak
            for (i in output.indices) output[i] *= gain
        }
        return output
    }

    private fun resample(input: FloatArray, ratio: Float): FloatArray {
        val count = (input.size / ratio).roundToInt().coerceAtLeast(1)
        val result = FloatArray(count)
        for (i in 0 until count) {
            val x = (i.toFloat() * ratio).coerceIn(0f, (input.size - 1).toFloat())
            val a = x.toInt()
            val b = min(a + 1, input.lastIndex)
            result[i] = input[a] * (1f - (x - a)) + input[b] * (x - a)
        }
        return result
    }

    /** WSOLA-inspired granular overlap-add, entirely on-device with bounded allocations. */
    private fun stretch(input: FloatArray, rate: Float): FloatArray {
        if (input.size < 2048) return resample(input, rate)
        val window = 1024
        val hop = 256
        val outputSize = (input.size / rate).roundToInt().coerceAtLeast(1)
        val result = FloatArray(outputSize + window)
        val weights = FloatArray(outputSize + window)
        val hann = FloatArray(window) { idx ->
            (0.5 - 0.5 * cos(2.0 * PI * (idx + 0.5) / window)).toFloat()
        }
        val frames = (outputSize + hop - 1) / hop + 1
        var previousStart = 0
        for (frame in 0 until frames) {
            val outPos = frame * hop
            if (outPos >= outputSize) break
            val nominal = (frame.toFloat() * hop * rate).roundToInt().coerceIn(0, input.size - window)
            var source = nominal
            if (frame > 0) {
                val minCandidate = (nominal - 64).coerceAtLeast(0)
                val maxCandidate = (nominal + 64).coerceAtMost(input.size - window)
                var bestCost = Double.POSITIVE_INFINITY
                // Compare the existing 768-sample overlap with candidate grains.
                for (candidate in minCandidate..maxCandidate step 4) {
                    var cost = 0.0
                    for (k in 0 until window - hop step 8) {
                        val previousSample = input[previousStart + hop + k]
                        val delta = previousSample - input[candidate + k]
                        cost += (delta * delta).toDouble()
                    }
                    if (cost < bestCost) {
                        bestCost = cost
                        source = candidate
                    }
                }
            }
            previousStart = source
            val n = min(window, result.size - outPos)
            for (j in 0 until n) {
                val weight = hann[j]
                result[outPos + j] += input[source + j] * weight
                weights[outPos + j] += weight
            }
        }
        val finished = FloatArray(outputSize)
        for (i in finished.indices) finished[i] = if (weights[i] > 0.0001f) result[i] / weights[i] else 0f
        return finished
    }
}
