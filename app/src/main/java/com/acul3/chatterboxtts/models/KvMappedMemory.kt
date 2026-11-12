package com.acul3.chatterboxtts.models

import android.content.Context
import android.os.Build
import android.os.SharedMemory
import org.pytorch.executorch.Tensor
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

/** Holds the two KV tensors outside the Android Java heap. */
internal class KvMappedMemory(private val context: Context, private val elementsPerCache: Int) : Closeable {
    private val bytesPerCache = Math.multiplyExact(elementsPerCache, 4)
    private val totalBytes = Math.multiplyExact(bytesPerCache, 2)
    private var sharedMemory: SharedMemory? = null
    private var fileBacking: RandomAccessFile? = null
    private val mapping: ByteBuffer = createMapping().order(ByteOrder.nativeOrder())

    val keys: FloatBuffer = floatView(0)
    val values: FloatBuffer = floatView(bytesPerCache)

    private fun createMapping(): ByteBuffer {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            try {
                val memory = SharedMemory.create("ChatterboxT3KV", totalBytes)
                val buffer = memory.mapReadWrite()
                sharedMemory = memory
                return buffer
            } catch (_: Exception) {
                // Older/vendor devices may have a smaller ashmem limit; map a local file instead.
                sharedMemory?.close()
                sharedMemory = null
            }
        }
        val temp = File.createTempFile("chatterbox-kv-", ".bin", context.cacheDir)
        val raf = RandomAccessFile(temp, "rw")
        try {
            raf.setLength(totalBytes.toLong())
            val mapped = raf.channel.map(FileChannel.MapMode.READ_WRITE, 0, totalBytes.toLong())
            fileBacking = raf
            // The mapping survives unlink on Linux, leaving no persistent 318 MB file.
            temp.delete()
            return mapped
        } catch (ex: Exception) {
            raf.close()
            temp.delete()
            throw ex
        }
    }

    private fun floatView(byteOffset: Int): FloatBuffer {
        val duplicate = mapping.duplicate().order(ByteOrder.nativeOrder())
        duplicate.position(byteOffset)
        duplicate.limit(byteOffset + bytesPerCache)
        return duplicate.slice().order(ByteOrder.nativeOrder()).asFloatBuffer()
    }

    fun copyLayerFrom(src: FloatBuffer, srcStart: Int, dest: FloatBuffer, destStart: Int, layerSize: Int) {
        val source = src.duplicate()
        source.position(srcStart)
        source.limit(srcStart + layerSize)
        val target = dest.duplicate()
        target.position(destStart)
        target.put(source)
    }

    fun copyAllFrom(src: FloatBuffer, dest: FloatBuffer) {
        require(src.capacity() == elementsPerCache) { "Invalid decoded cache: ${src.capacity()}" }
        val source = src.duplicate().apply { clear() }
        val target = dest.duplicate().apply { clear() }
        target.put(source)
    }

    fun keysTensor(shape: LongArray): Tensor = Tensor.fromBlob(keys, shape)
    fun valuesTensor(shape: LongArray): Tensor = Tensor.fromBlob(values, shape)

    override fun close() {
        if (sharedMemory != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            try { SharedMemory.unmap(mapping) } catch (_: Exception) { }
            sharedMemory?.close()
        }
        fileBacking?.close()
        fileBacking = null
        sharedMemory = null
    }
}
