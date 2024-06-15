import onnxruntime as rt
import numpy as np
import time

yolon_path = "yolon_best.onnx"

random_image = np.random.rand(1, 3, 640, 640).astype(np.float32)

session = rt.InferenceSession(yolon_path, providers=['CPUExecutionProvider'])

start = time.perf_counter()
predictions = session.run(None, {'images': random_image})
end = (time.perf_counter() - start) * 1000
print("first one: ", end)

total = 0
for i in range(0,100):
    start = time.perf_counter()
    predictions = session.run(None, {'images': random_image})
    end = (time.perf_counter() - start) * 1000
    total += end

print(total / 100)