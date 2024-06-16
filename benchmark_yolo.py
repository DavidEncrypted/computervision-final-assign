import onnxruntime as rt
import numpy as np
import time

yolon_path = "yolon_best.onnx"
nanodet_path = "nanodet.onnx"
int8_dyn_path = "yolon_int8_dyn.onnx"
int8_path = "yolon_int8.onnx"

sess_options = rt.SessionOptions()
sess_options.enable_profiling = True
sess_options.intra_op_num_threads = 1

def run_bench(session_p, name, num=100):
    random_image = np.random.rand(1, 3, 640, 640).astype(np.float32)
    total = 0
    for i in range(0,num):
        start = time.perf_counter()
        predictions = session_p.run(None, {name: random_image})
        end = (time.perf_counter() - start) * 1000
        total += end

    print(total / num, f"ms average over {num} runs")

session_int8_dyn = rt.InferenceSession(int8_dyn_path, providers=['CPUExecutionProvider'], sess_options=sess_options)

print("int8 dynamic yolo:")

while (True):
    start = time.time()
    run_bench(session_int8_dyn, 'images', 1)
    end = (time.time() - start)
    time.sleep(1-end)

