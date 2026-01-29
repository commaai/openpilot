#!/bin/bash

AMD=1 AMD_LLVM=1 python -m pytest -n=1 test/test_ops.py test/test_dtype.py test/test_dtype_alu.py test/test_linearizer.py test/test_randomness.py test/test_jit.py test/test_graph.py test/test_multitensor.py --durations=20
AMD=1 AMD_LLVM=0 python -m pytest -n=1 test/test_ops.py test/test_dtype.py test/test_dtype_alu.py test/test_linearizer.py test/test_randomness.py test/test_jit.py test/test_graph.py test/test_multitensor.py --durations=20

CNT=1 AMD_LLVM=0 DEBUG=2 FP8E4M3=1 HALF=0 BFLOAT16=0 SHOULD_USE_TC=1 python extra/gemm/simple_matmul.py
CNT=1 AMD_LLVM=0 DEBUG=2 FP8E4M3=0 HALF=1 BFLOAT16=0 SHOULD_USE_TC=1 python extra/gemm/simple_matmul.py
CNT=1 AMD_LLVM=0 DEBUG=2 FP8E4M3=0 HALF=0 BFLOAT16=1 SHOULD_USE_TC=1 python extra/gemm/simple_matmul.py

CNT=1 AMD_LLVM=1 DEBUG=2 FP8E4M3=0 HALF=1 BFLOAT16=0 SHOULD_USE_TC=1 python extra/gemm/simple_matmul.py
CNT=1 AMD_LLVM=1 DEBUG=2 FP8E4M3=0 HALF=0 BFLOAT16=1 SHOULD_USE_TC=1 python extra/gemm/simple_matmul.py
CNT=1 AMD_LLVM=1 DEBUG=2 FP8E4M3=1 HALF=0 BFLOAT16=0 SHOULD_USE_TC=1 python extra/gemm/simple_matmul.py