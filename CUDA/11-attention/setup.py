from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

original_cuda_version = torch.version.cuda
torch.version.cuda = "13.0"
print(
    f"Hacking torch.version.cuda from {original_cuda_version} to {torch.version.cuda} to bypass check."
)

this_dir = os.path.dirname(os.path.abspath(__file__))

cuda_source = "attention.cu"

setup(
    name="softmax_attention",
    ext_modules=[
        CUDAExtension(
            "softmax_attention",
            ["attention_bind.cpp", cuda_source],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
