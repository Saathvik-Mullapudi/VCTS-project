# verify_tflite.py
"""Check that an exported TFLite model loads and exposes quantized tensors."""

from pathlib import Path

import numpy as np
import tensorflow as tf


def find_tflite_model():
    candidates = sorted(Path(".").rglob("*.tflite"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No .tflite file found under the project folder.")
    return candidates[0]


model_path = find_tflite_model()
print(f"Loading: {model_path}")

interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

print("Model loaded")
print(f"Input dtype : {inp['dtype']}")
print(f"Input shape : {inp['shape']}")
print(f"Output dtype: {out['dtype']}")
print(f"Output shape: {out['shape']}")

input_shape = inp["shape"]
if inp["dtype"] == np.uint8:
    dummy = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
else:
    dummy = np.random.random(size=input_shape).astype(inp["dtype"])

interpreter.set_tensor(inp["index"], dummy)
interpreter.invoke()
result = interpreter.get_tensor(out["index"])
print(f"Inference OK. First 5 output values: {result.ravel()[:5]}")
