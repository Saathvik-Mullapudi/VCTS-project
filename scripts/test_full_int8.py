
"""Test the full integer quantized TFLite model."""

import numpy as np
import tensorflow as tf

model_path = "models/yolov8n_saved_model/yolov8n_full_integer_quant.tflite"
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
dummy = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
interpreter.set_tensor(inp["index"], dummy)
interpreter.invoke()
result = interpreter.get_tensor(out["index"])
print(f"Inference OK. First 5 output values: {result.ravel()[:5]}")
