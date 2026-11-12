package org.pytorch.executorch;

import java.nio.Buffer;
import java.nio.FloatBuffer;

/**
 * ExecuTorch v1.1.0 does not expose Tensor.getRawDataBuffer() publicly.
 * This package-local bridge prevents a 318 MB Java float[] allocation when reading KV cache.
 * This helper is tightly coupled to org.pytorch:executorch-android:1.1.0.
 */
public final class TensorBufferAccess {
    private TensorBufferAccess() { }

    public static FloatBuffer floatBuffer(Tensor tensor) {
        Buffer data = tensor.getRawDataBuffer();
        if (!(data instanceof FloatBuffer)) {
            throw new IllegalArgumentException("Expected FLOAT32 tensor, got " + tensor.dtype());
        }
        FloatBuffer result = ((FloatBuffer) data).duplicate();
        result.clear();
        return result;
    }
}
