import onnxruntime as rt
import numpy as np
import time

yolon_path = "yolon_best.onnx"
nanodet_path = "nanodet.onnx"
int8_dyn_path = "yolon_int8_dyn.onnx"
int8_path = "yolon_int8.onnx"
damoyolo_path = "damoyolo_Ns.onnx"
yolon_320_path = "yolo320.onnx"
quantized_yolon320_path = "yolon320_int8_dyn.onnx"
model_int8_path_320 = 'yolon_int8_static_320.onnx'

def run_bench(session_p, name, img_size=640):
    random_image = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    total = 0
    for i in range(0,100):
        start = time.perf_counter()
        predictions = session_p.run(None, {name: random_image})
        end = (time.perf_counter() - start) * 1000
        total += end

    print(total / 100, "ms average over 100 runs")

# random_image = np.random.rand(1, 3, 320, 320).astype(np.float32)


session_320 = rt.InferenceSession(yolon_320_path, providers=['CPUExecutionProvider'])

print("yolo320:")
run_bench(session_320, 'images', 320)

del session_320

session_320_int8_dyn = rt.InferenceSession(quantized_yolon320_path, providers=['CPUExecutionProvider'])

print("int8dyn320:")
run_bench(session_320_int8_dyn, 'images', 320)

del session_320_int8_dyn

session_320_int8_static = rt.InferenceSession(model_int8_path_320, providers=['CPUExecutionProvider'])

print("int8static320:")
run_bench(session_320_int8_static, 'images', 320)

del session_320_int8_static









exit()
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
