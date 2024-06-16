import onnxruntime as rt
import numpy as np
import time

yolon_path = "yolon_best.onnx"
nanodet_path = "nanodet.onnx"
int8_dyn_path = "yolon_int8_dyn.onnx"
int8_path = "yolon_int8.onnx"
damoyolo_path = "damoyolo_Ns.onnx"

def run_bench(session_p, name):
    random_image = np.random.rand(1, 3, 640, 640).astype(np.float32)
    total = 0
    for i in range(0,100):
        start = time.perf_counter()
        predictions = session_p.run(None, {name: random_image})
        end = (time.perf_counter() - start) * 1000
        total += end

    print(total / 100, "ms average over 100 runs")

random_image = np.random.rand(1, 3, 640, 640).astype(np.float32)

session = rt.InferenceSession(yolon_path, providers=['CPUExecutionProvider'])

start = time.perf_counter()
predictions = session.run(None, {'images': random_image})
end = (time.perf_counter() - start) * 1000
print("first one: ", end)

run_bench(session, 'images')

del session

session_damoyolo = rt.InferenceSession(damoyolo_path, providers=['CPUExecutionProvider'])

print("damoyolo:")
# run_bench(session_damoyolo, 'images')

random_image = np.random.rand(1, 3, 416, 416).astype(np.float32)
total = 0
for i in range(0,100):
    start = time.perf_counter()
    predictions = session_damoyolo.run(None, {'images': random_image})
    end = (time.perf_counter() - start) * 1000
    total += end

print(total / 100, "ms average over 100 runs")

del session_damoyolo

session_nanodet = rt.InferenceSession(nanodet_path, providers=['CPUExecutionProvider'])

print("nanodet:")
run_bench(session_nanodet, 'data')

del session_nanodet

session_int8_dyn = rt.InferenceSession(int8_dyn_path, providers=['CPUExecutionProvider'])

print("int8 dynamic yolo:")
run_bench(session_int8_dyn, 'images')

del session_int8_dyn

session_int8 = rt.InferenceSession(int8_path, providers=['CPUExecutionProvider'])

print("int8 yolo:")
run_bench(session_int8, 'images')
