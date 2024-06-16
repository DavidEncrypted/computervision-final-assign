import onnxruntime as rt
import numpy as np
import time

yolon_path = "yolon_best.onnx"
nanodet_path = "nanodet.onnx"
int8_dyn_path = "yolon_int8_dyn.onnx"
int8_path = "yolon_int8.onnx"
damoyolo_path = "damoyolo_Ns.onnx"

model_int8_path_320 = 'yolon_int8_static_320.onnx'

sess_options = rt.SessionOptions()
sess_options.enable_profiling = False
sess_options.intra_op_num_threads = 1

def run_bench(session_p, name, num=100):
    random_image = np.random.rand(1, 3, 320, 320).astype(np.float32)
    total = 0
    for i in range(0,num):
        start = time.perf_counter()
        predictions = session_p.run(None, {name: random_image})
        end = (time.perf_counter() - start) * 1000
        total += end

    print(total / num, f"ms average over {num} runs")

session_int8_static_320 = rt.InferenceSession(model_int8_path_320, providers=['CPUExecutionProvider'], sess_options=sess_options)

print("int8 static 320 yolo:")

while (True):
    start = time.time()
    run_bench(session_int8_static_320, 'images', 1)
    end = (time.time() - start)
    time.sleep(1-end)

