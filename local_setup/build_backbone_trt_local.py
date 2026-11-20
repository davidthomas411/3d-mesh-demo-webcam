"""Build a batch-1 FP16 TensorRT engine of the DINOv3 backbone at a chosen input size (upstream script is fixed at 512).
usage: python build_backbone_trt_local.py 384"""
import os, sys

size = int(sys.argv[1]) if len(sys.argv) > 1 else 384
os.environ["IMG_SIZE"] = str(size)
os.environ["USE_TRT_BACKBONE"] = "0"
os.environ["USE_COMPILE"] = "0"

import convert_backbone_tensorrt as c

c.IMAGE_SIZE = (size, size)
c.OUTPUT_SIZE = (size // 16, size // 16)
c.ONNX_PATH = os.path.join(c.TRT_OUTPUT_DIR, f"backbone_dinov3_{size}.onnx")
c.TRT_PATH = os.path.join(c.TRT_OUTPUT_DIR, f"backbone_dinov3_fp16_{size}.engine")

backbone = c.load_backbone()
assert c.step1_export_onnx(backbone, [1])
del backbone
import torch, gc
gc.collect(); torch.cuda.empty_cache()
assert c.step2_convert_tensorrt([1])
print("ENGINE_OK", c.TRT_PATH)
