"""
Minimal TensorRT inference wrapper for the plate detector engine.

Handles dynamic input shapes (buffer reallocation), FP16/FP32 dtypes, and
asynchronous CUDA copies on a dedicated stream.

Requires: tensorrt, pycuda (pycuda.autoinit initializes the CUDA context).

Sanity-check an engine:
    python deploy/trt_infer.py [--engine outputs/deploy/plate_detector_fp16.engine]
"""

import argparse

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTDetector:
    def __init__(self, engine_path, input_shape=(1, 3, 640, 640)):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                break
        if self.input_name is None:
            raise RuntimeError("Engine has no input tensor.")

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self._allocate_buffers(input_shape)

    def _allocate_buffers(self, input_shape):
        self.input_shape = list(input_shape)
        self.context.set_input_shape(self.input_name, self.input_shape)
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self.input_shape if is_input else list(self.context.get_tensor_shape(name))
            host = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(device))
            self.context.set_tensor_address(name, int(device))
            entry = {"name": name, "dtype": dtype, "shape": shape, "host": host, "device": device}
            (self.inputs if is_input else self.outputs).append(entry)

    def _ensure_shape(self, shape):
        if list(shape) != self.input_shape:
            self._allocate_buffers(shape)

    def infer(self, img_array):
        """Run inference on a preprocessed (1, 3, H, W) array. Returns output list."""
        img_array = np.ascontiguousarray(img_array)
        self._ensure_shape(img_array.shape)
        inp = self.inputs[0]
        if img_array.dtype != inp["dtype"]:
            img_array = img_array.astype(inp["dtype"])

        np.copyto(inp["host"], img_array.ravel())
        cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        return [out["host"].reshape(self.context.get_tensor_shape(out["name"])) for out in self.outputs]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sanity-check a TensorRT engine")
    p.add_argument("--engine", default="outputs/deploy/plate_detector_fp16.engine")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    detector = TRTDetector(args.engine, (1, 3, args.imgsz, args.imgsz))
    dummy = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(detector.inputs[0]["dtype"])
    outs = detector.infer(dummy)
    print("Engine OK. Output shapes:", [o.shape for o in outs])
