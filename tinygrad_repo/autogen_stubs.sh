#!/bin/bash -e

# setup instructions for clang2py
if [[ ! $(clang2py -V) ]]; then
  pushd .
  cd /tmp
  sudo apt-get install -y --no-install-recommends clang
  pip install --upgrade pip setuptools
  pip install clang==14.0.6
  git clone https://github.com/nimlgen/ctypeslib.git
  cd ctypeslib
  pip install .
  clang2py -V
  popd
fi

BASE=tinygrad/runtime/autogen/

fixup() {
  sed -i '1s/^/# mypy: ignore-errors\n/' $1
  sed -i 's/ *$//' $1
  grep FIXME_STUB $1 || true
}

patch_dlopen() {
  path=$1; shift
  name=$1; shift
  cat <<EOF | sed -i "/import ctypes.*/r /dev/stdin" $path
PATHS_TO_TRY = [
$(for p in "$@"; do echo "  $p,"; done)
]
def _try_dlopen_$name():
  library = ctypes.util.find_library("$name")
  if library:
    try: return ctypes.CDLL(library)
    except OSError: pass
  for candidate in PATHS_TO_TRY:
    try: return ctypes.CDLL(candidate)
    except OSError: pass
  return None
EOF
}

generate_opencl() {
  clang2py /usr/include/CL/cl.h -o $BASE/opencl.py -l /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -k cdefstum
  fixup $BASE/opencl.py
  # hot patches
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/opencl.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libOpenCL.so.1')\ctypes.CDLL(ctypes.util.find_library('OpenCL'))\g" $BASE/opencl.py
  python3 -c "import tinygrad.runtime.autogen.opencl"
}

generate_hip() {
  clang2py /opt/rocm/include/hip/hip_ext.h /opt/rocm/include/hip/hiprtc.h \
  /opt/rocm/include/hip/hip_runtime_api.h /opt/rocm/include/hip/driver_types.h \
  --clang-args="-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -x c++" -o $BASE/hip.py -l /opt/rocm/lib/libamdhip64.so
  echo "hipDeviceProp_t = hipDeviceProp_tR0600" >> $BASE/hip.py
  echo "hipGetDeviceProperties = hipGetDevicePropertiesR0600" >> $BASE/hip.py
  fixup $BASE/hip.py
  # we can trust HIP is always at /opt/rocm/lib
  #sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/hip.py
  #sed -i "s\ctypes.CDLL('/opt/rocm/lib/libhiprtc.so')\ctypes.CDLL(ctypes.util.find_library('hiprtc'))\g" $BASE/hip.py
  #sed -i "s\ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')\ctypes.CDLL(ctypes.util.find_library('amdhip64'))\g" $BASE/hip.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/hip.py
  sed -i "s\'/opt/rocm/\os.getenv('ROCM_PATH', '/opt/rocm/')+'/\g" $BASE/hip.py
  python3 -c "import tinygrad.runtime.autogen.hip"
}

generate_comgr() {
  clang2py /opt/rocm/include/amd_comgr/amd_comgr.h \
  --clang-args="-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -x c++" -o $BASE/comgr.py -l /opt/rocm/lib/libamd_comgr.so
  fixup $BASE/comgr.py
  sed -i "s\import ctypes\import ctypes, ctypes.util, os\g" $BASE/comgr.py
  patch_dlopen $BASE/comgr.py amd_comgr "'/opt/rocm/lib/libamd_comgr.so'" "os.getenv('ROCM_PATH', '')+'/lib/libamd_comgr.so'" "'/usr/local/lib/libamd_comgr.dylib'" "'/opt/homebrew/lib/libamd_comgr.dylib'"
  sed -i "s\ctypes.CDLL('/opt/rocm/lib/libamd_comgr.so')\_try_dlopen_amd_comgr()\g" $BASE/comgr.py
  python3 -c "import tinygrad.runtime.autogen.comgr"
}

generate_kfd() {
  clang2py /usr/include/linux/kfd_ioctl.h -o $BASE/kfd.py -k cdefstum

  fixup $BASE/kfd.py
  sed -i "s/import ctypes/import ctypes, os/g" $BASE/kfd.py
  sed -i "s/import fcntl, functools/import functools/g" $BASE/kfd.py
  sed -i "/import functools/a from tinygrad.runtime.support.hcq import FileIOInterface" $BASE/kfd.py
  sed -i "s/def _do_ioctl(__idir, __base, __nr, __user_struct, __fd, \*\*kwargs):/def _do_ioctl(__idir, __base, __nr, __user_struct, __fd:FileIOInterface, \*\*kwargs):/g" $BASE/kfd.py
  sed -i "s/fcntl.ioctl(__fd, (__idir<<30)/__fd.ioctl((__idir<<30)/g" $BASE/kfd.py
  sed -i "s/!!/not not /g" $BASE/kfd.py
  python3 -c "import tinygrad.runtime.autogen.kfd"
}

generate_cuda() {
  clang2py /usr/include/cuda.h --clang-args="-D__CUDA_API_VERSION_INTERNAL" -o $BASE/cuda.py -l /usr/lib/x86_64-linux-gnu/libcuda.so
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/cuda.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so')\ctypes.CDLL(ctypes.util.find_library('cuda'))\g" $BASE/cuda.py
  fixup $BASE/cuda.py
  python3 -c "import tinygrad.runtime.autogen.cuda"
}

generate_nvrtc() {
  clang2py /usr/local/cuda/include/nvrtc.h /usr/local/cuda/include/nvJitLink.h -o $BASE/nvrtc.py -l /usr/local/cuda/lib64/libnvrtc.so -l /usr/local/cuda/lib64/libnvJitLink.so
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/nvrtc.py
  sed -i "s\ctypes.CDLL('/usr/local/cuda/lib64/libnvrtc.so')\ctypes.CDLL(ctypes.util.find_library('nvrtc'))\g" $BASE/nvrtc.py
  sed -i "s\ctypes.CDLL('/usr/local/cuda/lib64/libnvJitLink.so')\ctypes.CDLL(ctypes.util.find_library('nvJitLink'))\g" $BASE/nvrtc.py
  fixup $BASE/nvrtc.py
  python3 -c "import tinygrad.runtime.autogen.nvrtc"
}

generate_nv() {
  NVKERN_COMMIT_HASH=81fe4fb417c8ac3b9bdcc1d56827d116743892a5
  NVKERN_SRC=/tmp/open-gpu-kernel-modules-$NVKERN_COMMIT_HASH
  if [ ! -d "$NVKERN_SRC" ]; then
    git clone https://github.com/NVIDIA/open-gpu-kernel-modules $NVKERN_SRC
    pushd .
    cd $NVKERN_SRC
    git reset --hard $NVKERN_COMMIT_HASH
    popd
  fi

  clang2py -k cdefstum \
    extra/nv_gpu_driver/clc6c0qmd.h \
    extra/nv_gpu_driver/clcec0qmd.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/cl0000.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/cl0080.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/cl2080.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/cl2080_notification.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc56f.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc86f.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc96f.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc761.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/cl83de.h \
    $NVKERN_SRC/src/nvidia/generated/g_allclasses.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc6c0.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clcdc0.h \
    $NVKERN_SRC/kernel-open/nvidia-uvm/clc6b5.h \
    $NVKERN_SRC/kernel-open/nvidia-uvm/clc9b5.h \
    $NVKERN_SRC/kernel-open/nvidia-uvm/uvm_ioctl.h \
    $NVKERN_SRC/kernel-open/nvidia-uvm/uvm_linux_ioctl.h \
    $NVKERN_SRC/kernel-open/nvidia-uvm/hwref/ampere/ga100/dev_fault.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv_escape.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl-numbers.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl-numa.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv-unix-nvos-params-wrappers.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/alloc/alloc_channel.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/nvos.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl0000/*.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl0080/*.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl2080/*.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl83de/*.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrlcb33.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrla06c.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl90f1.h \
    --clang-args="-include $NVKERN_SRC/src/common/sdk/nvidia/inc/nvtypes.h -I$NVKERN_SRC/src/common/inc -I$NVKERN_SRC/kernel-open/nvidia-uvm -I$NVKERN_SRC/kernel-open/common/inc -I$NVKERN_SRC/src/common/sdk/nvidia/inc -I$NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include -I$NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl" \
    -o $BASE/nv_gpu.py
  fixup $BASE/nv_gpu.py
  sed -i "s\(0000000001)\1\g" $BASE/nv_gpu.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/nv_gpu.py
  sed -i 's/#\?\s\([A-Za-z0-9_]\+\) = MW ( \([0-9]\+\) : \([0-9]\+\) )/\1 = (\2 , \3)/' $BASE/nv_gpu.py # NVC6C0_QMDV03_00 processing
  sed -i 's/#\sdef NVC6C0_QMD\([A-Za-z0-9_()]\+\):/def NVC6C0_QMD\1:/' $BASE/nv_gpu.py
  sed -i 's/#\sdef NVCEC0_QMD\([A-Za-z0-9_()]\+\):/def NVCEC0_QMD\1:/' $BASE/nv_gpu.py
  sed -E -i -n '/^def (NVCEC0_QMDV05_00_RELEASE)(_ENABLE)\(i\):/{p;s//\1'"0"'\2=\1\2(0)\n\1'"1"'\2=\1\2(1)/;H;b};p;${x;s/^\n//;p}' "$BASE/nv_gpu.py"
  sed -i 's/#\s*return MW(\([0-9i()*+]\+\):\([0-9i()*+]\+\))/    return (\1 , \2)/' $BASE/nv_gpu.py
  sed -i 's/#\?\s*\(.*\)\s*=\s*\(NV\)\?BIT\(32\)\?\s*(\s*\([0-9]\+\)\s*)/\1 = (1 << \4)/' $BASE/nv_gpu.py # name = BIT(x) -> name = (1 << x)
  sed -i "s/UVM_\([A-Za-z0-9_]\+\) = \['i', '(', '\([0-9]\+\)', ')'\]/UVM_\1 = \2/" $BASE/nv_gpu.py # UVM_name = ['i', '(', '<num>', ')'] -> UVM_name = <num>

  # Parse status codes
  sed -n '1i\
nv_status_codes = {}
/^NV_STATUS_CODE/ { s/^NV_STATUS_CODE(\([^,]*\), *\([^,]*\), *"\([^"]*\)") *.*$/\1 = \2\nnv_status_codes[\1] = "\3"/; p }' $NVKERN_SRC/src/common/sdk/nvidia/inc/nvstatuscodes.h >> $BASE/nv_gpu.py
  python3 -c "import tinygrad.runtime.autogen.nv_gpu"

  clang2py -k cdefstum \
    $NVKERN_SRC/src/nvidia/inc/kernel/gpu/fsp/kern_fsp_cot_payload.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/common/inc/gsp/gspifpub.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/common/inc/gsp/gsp_fw_wpr_meta.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/common/inc/gsp/gsp_fw_sr_meta.h \
    $NVKERN_SRC/src/nvidia/inc/kernel/gpu/gsp/gsp_init_args.h \
    $NVKERN_SRC/src/nvidia/inc/kernel/gpu/gsp/gsp_init_args.h \
    $NVKERN_SRC/src/common/uproc/os/common/include/libos_init_args.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/common/inc/rmRiscvUcode.h \
    $NVKERN_SRC/src/common/shared/msgq/inc/msgq/msgq_priv.h \
    $NVKERN_SRC/src/nvidia/inc/kernel/vgpu/rpc_headers.h \
    $NVKERN_SRC/src/nvidia/inc/kernel/vgpu/rpc_global_enums.h \
    $NVKERN_SRC/src/nvidia/generated/g_rpc-structures.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/common/inc/fsp/fsp_nvdm_format.h \
    extra/nv_gpu_driver/g_rpc-message-header.h \
    extra/nv_gpu_driver/gsp_static_config.h \
    extra/nv_gpu_driver/vbios.h \
    extra/nv_gpu_driver/pci_exp_table.h \
    --clang-args="-DRPC_MESSAGE_STRUCTURES -DRPC_STRUCTURES -include $NVKERN_SRC/src/common/sdk/nvidia/inc/nvtypes.h -I$NVKERN_SRC/src/nvidia/generated -I$NVKERN_SRC/src/common/inc -I$NVKERN_SRC/src/nvidia/inc -I$NVKERN_SRC/src/nvidia/interface/ -I$NVKERN_SRC/src/nvidia/inc/kernel -I$NVKERN_SRC/src/nvidia/inc/libraries -I$NVKERN_SRC/src/nvidia/arch/nvalloc/common/inc -I$NVKERN_SRC/kernel-open/nvidia-uvm -I$NVKERN_SRC/kernel-open/common/inc -I$NVKERN_SRC/src/common/sdk/nvidia/inc -I$NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include -I$NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl" \
    -o $BASE/nv/nv.py

  fixup $BASE/nv/nv.py
  python3 -c "import tinygrad.runtime.autogen.nv.nv"
}

generate_amd() {
  # clang2py broken when pass -x c++ to prev headers
  clang2py -k cdefstum \
    extra/hip_gpu_driver/sdma_registers.h \
    extra/hip_gpu_driver/nvd.h \
    extra/hip_gpu_driver/gc_11_0_0_offset.h \
    extra/hip_gpu_driver/sienna_cichlid_ip_offset.h \
    --clang-args="-I/opt/rocm/include -x c++" \
    -o $BASE/amd_gpu.py

  fixup $BASE/amd_gpu.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/amd_gpu.py
  python3 -c "import tinygrad.runtime.autogen.amd_gpu"
}

generate_hsa() {
  clang2py \
    /opt/rocm/include/hsa/hsa.h \
    /opt/rocm/include/hsa/hsa_ext_amd.h \
    /opt/rocm/include/hsa/amd_hsa_signal.h \
    /opt/rocm/include/hsa/amd_hsa_queue.h \
    /opt/rocm/include/hsa/amd_hsa_kernel_code.h \
    /opt/rocm/include/hsa/hsa_ext_finalize.h /opt/rocm/include/hsa/hsa_ext_image.h \
    /opt/rocm/include/hsa/hsa_ven_amd_aqlprofile.h \
    --clang-args="-I/opt/rocm/include" \
    -o $BASE/hsa.py -l /opt/rocm/lib/libhsa-runtime64.so

  fixup $BASE/hsa.py
  sed -i "s\import ctypes\import ctypes, ctypes.util, os\g" $BASE/hsa.py
  sed -i "s\ctypes.CDLL('/opt/rocm/lib/libhsa-runtime64.so')\ctypes.CDLL(os.getenv('ROCM_PATH')+'/lib/libhsa-runtime64.so' if os.getenv('ROCM_PATH') else ctypes.util.find_library('hsa-runtime64'))\g" $BASE/hsa.py
  python3 -c "import tinygrad.runtime.autogen.hsa"
}

generate_io_uring() {
  clang2py -k cdefstum \
    /usr/include/liburing.h \
    /usr/include/linux/io_uring.h \
    -o $BASE/io_uring.py

  sed -r '/^#define __NR_io_uring/ s/^#define __(NR_io_uring[^ ]+) (.*)$/\1 = \2/; t; d' /usr/include/asm-generic/unistd.h >> $BASE/io_uring.py # io_uring syscalls numbers
  fixup $BASE/io_uring.py
}

generate_ib() {
  clang2py -k cdefstum \
    /usr/include/infiniband/verbs.h \
    /usr/include/infiniband/verbs_api.h \
    /usr/include/infiniband/ib_user_ioctl_verbs.h \
    /usr/include/rdma/ib_user_verbs.h \
    -o $BASE/ib.py

  sed -i "s\import ctypes\import ctypes, ctypes.util\g" "$BASE/ib.py"
  sed -i "s\FIXME_STUB\libibverbs\g" "$BASE/ib.py"
  sed -i "s\FunctionFactoryStub()\ctypes.CDLL(ctypes.util.find_library('ibverbs'), use_errno=True)\g" "$BASE/ib.py"

  fixup $BASE/ib.py
}

generate_llvm() {
  INC="$(llvm-config-14 --includedir)"
  clang2py -k cdefstum \
    $(find "$INC/llvm-c/" -type f -name '*.h' | sort) \
    "$INC/llvm/Config/Targets.def" \
    "$INC/llvm/Config/AsmPrinters.def" \
    "$INC/llvm/Config/AsmParsers.def" \
    "$INC/llvm/Config/Disassemblers.def" \
    --clang-args="$(llvm-config-14 --cflags)" \
    -o "$BASE/llvm.py"

  sed -i "s\import ctypes\import ctypes, tinygrad.runtime.support.llvm as llvm_support\g" "$BASE/llvm.py"
  sed -i "s\FIXME_STUB\llvm\g" "$BASE/llvm.py"
  sed -i "s\FunctionFactoryStub()\ctypes.CDLL(llvm_support.LLVM_PATH)\g" "$BASE/llvm.py"

  fixup "$BASE/llvm.py"
}

generate_kgsl() {
  clang2py extra/qcom_gpu_driver/msm_kgsl.h -o $BASE/kgsl.py -k cdefstum
  fixup $BASE/kgsl.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/kgsl.py
  sed -nE 's/#define ([A-Za-z0-9_]+)_SHIFT\s*[^\S\r\n]*[0-9]*$/def \1(val): return (val << \1_SHIFT) \& \1_MASK/p' extra/qcom_gpu_driver/msm_kgsl.h >> $BASE/kgsl.py
  sed -i "s\fcntl.ioctl(__fd, (__idir<<30)\__fd.ioctl((__idir<<30)\g" $BASE/kgsl.py
  python3 -c "import tinygrad.runtime.autogen.kgsl"
}

generate_adreno() {
  clang2py extra/qcom_gpu_driver/a6xx.xml.h -o $BASE/adreno.py -k cestum
  sed -nE 's/#define ([A-Za-z0-9_]+)__SHIFT\s*[^\S\r\n]*[0-9]*$/def \1(val): return (val << \1__SHIFT) \& \1__MASK/p' extra/qcom_gpu_driver/a6xx.xml.h >> $BASE/adreno.py
  fixup $BASE/adreno.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/adreno.py
  python3 -c "import tinygrad.runtime.autogen.adreno"
}

generate_qcom() {
  clang2py -k cdefstum \
    extra/dsp/include/ion.h \
    extra/dsp/include/msm_ion.h \
    extra/dsp/include/adsprpc_shared.h \
    extra/dsp/include/remote_default.h \
    extra/dsp/include/apps_std.h \
    -o $BASE/qcom_dsp.py

  fixup $BASE/qcom_dsp.py
  python3 -c "import tinygrad.runtime.autogen.qcom_dsp"
}

generate_pci() {
  clang2py -k cdefstum \
    /usr/include/linux/pci_regs.h \
    -o $BASE/pci.py
  fixup $BASE/pci.py
}

generate_vfio() {
  clang2py -k cdefstum \
    /usr/include/linux/vfio.h \
    -o $BASE/vfio.py
  fixup $BASE/vfio.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/vfio.py
  sed -i "s\import fcntl, functools\import functools" $BASE/vfio.py
  sed -i "s\import ctypes,os\a from tinygrad.runtime.support import FileIOInterface\g" $BASE/vfio.py
  sed -i "s\fcntl.ioctl(__fd, (__idir<<30)\return __fd.ioctl((__idir<<30)\g" $BASE/vfio.py
}

generate_am() {
  AMKERN_COMMIT_HASH=ceb12c04e2b5b53ec0779362831f5ee40c4921e4
  AMKERN_SRC=/tmp/ROCK-Kernel-Driver-$AMKERN_COMMIT_HASH
  if [ ! -d "$AMKERN_SRC" ]; then
    git clone https://github.com/ROCm/ROCK-Kernel-Driver $AMKERN_SRC --depth 1
  fi
  AMKERN_AMD=$AMKERN_SRC/drivers/gpu/drm/amd/
  AMKERN_INC=$AMKERN_AMD/include/

  clang2py -k cdefstum \
    extra/amdpci/headers/v11_structs.h \
    extra/amdpci/headers/v12_structs.h \
    extra/amdpci/headers/amdgpu_vm.h \
    extra/amdpci/headers/discovery.h \
    extra/amdpci/headers/amdgpu_ucode.h \
    extra/amdpci/headers/psp_gfx_if.h \
    extra/amdpci/headers/amdgpu_psp.h \
    extra/amdpci/headers/amdgpu_irq.h \
    extra/amdpci/headers/amdgpu_doorbell.h \
    $AMKERN_INC/soc15_ih_clientid.h \
    --clang-args="-include stdint.h" \
    -o $BASE/am/am.py
  fixup $BASE/am/am.py
  sed -i "s\(int64_t)\ \g" $BASE/am/am.py
  sed -i "s\AMDGPU_PTE_MTYPE_VG10(2)\AMDGPU_PTE_MTYPE_VG10(0, 2)\g" $BASE/am/am.py # incorrect parsing (TODO: remove when clang2py is gone).

  clang2py -k cdefstum \
    $AMKERN_AMD/amdkfd/kfd_pm4_headers_ai.h \
    $AMKERN_AMD/amdgpu/soc15d.h \
    -o $BASE/am/pm4_soc15.py
  fixup $BASE/am/pm4_soc15.py

  clang2py -k cdefstum \
    $AMKERN_AMD/amdkfd/kfd_pm4_headers_ai.h \
    $AMKERN_AMD/amdgpu/nvd.h \
    -o $BASE/am/pm4_nv.py
  fixup $BASE/am/pm4_nv.py

  clang2py -k cdefstum \
    extra/hip_gpu_driver/sdma_registers.h \
    $AMKERN_AMD/amdgpu/vega10_sdma_pkt_open.h \
    --clang-args="-I/opt/rocm/include -x c++" \
    -o $BASE/am/sdma_4_0_0.py
  fixup $BASE/am/sdma_4_0_0.py

  clang2py -k cdefstum \
    extra/hip_gpu_driver/sdma_registers.h \
    $AMKERN_AMD/amdgpu/navi10_sdma_pkt_open.h \
    --clang-args="-I/opt/rocm/include -x c++" \
    -o $BASE/am/sdma_5_0_0.py
  fixup $BASE/am/sdma_5_0_0.py

  clang2py -k cdefstum \
    extra/hip_gpu_driver/sdma_registers.h \
    $AMKERN_AMD/amdgpu/sdma_v6_0_0_pkt_open.h \
    --clang-args="-I/opt/rocm/include -x c++" \
    -o $BASE/am/sdma_6_0_0.py
  fixup $BASE/am/sdma_6_0_0.py

  clang2py -k cdefstum \
    $AMKERN_AMD/pm/swsmu/inc/pmfw_if/smu_v13_0_0_ppsmc.h \
    $AMKERN_AMD/pm/swsmu/inc/pmfw_if/smu13_driver_if_v13_0_0.h \
    extra/amdpci/headers/amdgpu_smu.h \
    -o $BASE/am/smu_v13_0_0.py
  fixup $BASE/am/smu_v13_0_0.py

  clang2py -k cdefstum \
    $AMKERN_AMD/pm/swsmu/inc/pmfw_if/smu_v14_0_0_pmfw.h \
    $AMKERN_AMD/pm/swsmu/inc/pmfw_if/smu_v14_0_2_ppsmc.h \
    $AMKERN_AMD/pm/swsmu/inc/pmfw_if/smu14_driver_if_v14_0.h \
    extra/amdpci/headers/amdgpu_smu.h \
    --clang-args="-include stdint.h" \
    -o $BASE/am/smu_v14_0_2.py
  fixup $BASE/am/smu_v14_0_2.py
}

generate_sqtt() {
  clang2py -k cdefstum \
    extra/sqtt/sqtt.h \
    -o $BASE/sqtt.py
  fixup $BASE/sqtt.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/sqtt.py
  python3 -c "import tinygrad.runtime.autogen.sqtt"

  ROCPROF_COMMIT_HASH=dd0485100971522cc4cd8ae136bdda431061a04d
  ROCPROF_SRC=/tmp/rocprof-trace-decoder-$ROCPROF_COMMIT_HASH
  if [ ! -d "$ROCPROF_SRC" ]; then
    git clone https://github.com/ROCm/rocprof-trace-decoder $ROCPROF_SRC
    pushd .
    cd $ROCPROF_SRC
    git reset --hard $ROCPROF_COMMIT_HASH
    popd
  fi

  clang2py -k cdefstum \
    $ROCPROF_SRC/include/rocprof_trace_decoder.h \
    $ROCPROF_SRC/include/trace_decoder_instrument.h \
    $ROCPROF_SRC/include/trace_decoder_types.h \
    -o $BASE/rocprof.py
  fixup $BASE/rocprof.py
  sed -i '1s/^/# pylint: skip-file\n/' $BASE/rocprof.py
  sed -i "s/import ctypes/import ctypes, ctypes.util/g" $BASE/rocprof.py
  patch_dlopen $BASE/rocprof.py rocprof-trace-decoder "'/usr/local/lib/librocprof-trace-decoder.so'" "'/usr/local/lib/librocprof-trace-decoder.dylib'"
  sed -i "s/def _try_dlopen_rocprof-trace-decoder():/def _try_dlopen_rocprof_trace_decoder():/g" $BASE/rocprof.py
  sed -i "s|FunctionFactoryStub()|_try_dlopen_rocprof_trace_decoder()|g" $BASE/rocprof.py
}

generate_webgpu() {
  clang2py extra/webgpu/webgpu.h -o $BASE/webgpu.py
  fixup $BASE/webgpu.py
  sed -i "s/FIXME_STUB/webgpu/g" "$BASE/webgpu.py"
  sed -i "s/FunctionFactoryStub()/ctypes.CDLL(webgpu_support.WEBGPU_PATH)/g" "$BASE/webgpu.py"
  sed -i "s/import ctypes/import ctypes, tinygrad.runtime.support.webgpu as webgpu_support/g" "$BASE/webgpu.py"
  python3 -c "import tinygrad.runtime.autogen.webgpu"
}

generate_libusb() {
  clang2py -k cdefstum \
    /usr/include/libusb-1.0/libusb.h \
    -o $BASE/libusb.py

  fixup $BASE/libusb.py
  sed -i "s\import ctypes\import ctypes, ctypes.util, os\g" $BASE/libusb.py
  sed -i "s/FIXME_STUB/libusb/g" "$BASE/libusb.py"
  sed -i "s/libusb_le16_to_cpu = libusb_cpu_to_le16//g" "$BASE/libusb.py"
  sed -i "s/FunctionFactoryStub()/None if (lib_path:=os.getenv('LIBUSB_PATH', ctypes.util.find_library('usb-1.0'))) is None else ctypes.CDLL(lib_path)/g" "$BASE/libusb.py"
  python3 -c "import tinygrad.runtime.autogen.libusb"
}

generate_mesa() {
  MESA_TAG="mesa-25.2.4"
  MESA_SRC=/tmp/mesa-$MESA_TAG
  TINYMESA_TAG=tinymesa-32dc66c
  TINYMESA_DIR=/tmp/tinymesa-$MESA_TAG-$TINYMESA_TAG/
  TINYMESA_SO=$TINYMESA_DIR/libtinymesa_cpu.so
  if [ ! -d "$MESA_SRC" ]; then
    git clone --depth 1 --branch $MESA_TAG https://gitlab.freedesktop.org/mesa/mesa.git $MESA_SRC
    pushd .
    cd $MESA_SRC
    git reset --hard $MESA_COMMIT_HASH
    # clang 14 doesn't support packed enums
    sed -i "s/enum \w\+ \(\w\+\);$/uint8_t \1;/" $MESA_SRC/src/nouveau/headers/nv_device_info.h
    sed -i "s/enum \w\+ \(\w\+\);$/uint8_t \1;/" $MESA_SRC/src/nouveau/compiler/nak.h
    sed -i "s/nir_instr_type \(\w\+\);/uint8_t \1;/" $MESA_SRC/src/compiler/nir/nir.h
    mkdir -p gen/util/format
    python3 src/util/format/u_format_table.py src/util/format/u_format.yaml --enums > gen/util/format/u_format_gen.h
    python3 src/compiler/nir/nir_opcodes_h.py > gen/nir_opcodes.h
    python3 src/compiler/nir/nir_intrinsics_h.py --outdir gen
    python3 src/compiler/nir/nir_intrinsics_indices_h.py --outdir gen
    python3 src/compiler/nir/nir_builder_opcodes_h.py > gen/nir_builder_opcodes.h
    python3 src/compiler/nir/nir_intrinsics_h.py --outdir gen
    python3 src/compiler/builtin_types_h.py gen/builtin_types.h
    popd
  fi

  if [ ! -d "$TINYMESA_DIR" ]; then
    mkdir $TINYMESA_DIR
    curl -L https://github.com/sirhcm/tinymesa/releases/download/$TINYMESA_TAG/libtinymesa_cpu-$MESA_TAG-linux-amd64.so -o $TINYMESA_SO
  fi

  clang2py -k cdefstu \
    $MESA_SRC/src/compiler/nir/nir.h \
    $MESA_SRC/src/compiler/nir/nir_builder.h \
    $MESA_SRC/src/compiler/nir/nir_shader_compiler_options.h \
    $MESA_SRC/src/compiler/nir/nir_serialize.h \
    $MESA_SRC/gen/nir_intrinsics.h \
    $MESA_SRC/src/nouveau/headers/nv_device_info.h \
    $MESA_SRC/src/nouveau/compiler/nak.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_passmgr.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_misc.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_type.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_init.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_nir.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_struct.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_jit_types.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_flow.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_const.h \
    $MESA_SRC/src/compiler/glsl_types.h \
    $MESA_SRC/src/util/blob.h \
    $MESA_SRC/src/util/ralloc.h \
    --clang-args="-DHAVE_ENDIAN_H -DHAVE_STRUCT_TIMESPEC -DHAVE_PTHREAD -I$MESA_SRC/src -I$MESA_SRC/include -I$MESA_SRC/gen -I$MESA_SRC/src/compiler/nir -I$MESA_SRC/src/gallium/auxiliary -I$MESA_SRC/src/gallium/include -I$(llvm-config-20 --includedir)" \
    -l $TINYMESA_SO \
    -o $BASE/mesa.py

  LVP_NIR_OPTIONS=$(./extra/mesa/lvp_nir_options.sh $MESA_SRC)

  fixup $BASE/mesa.py
  patch_dlopen $BASE/mesa.py tinymesa_cpu "(BASE:=os.getenv('MESA_PATH', f\"/usr{'/local/' if helpers.OSX else '/'}lib\"))+'/libtinymesa_cpu'+(EXT:='.dylib' if helpers.OSX else '.so')" "f'{BASE}/libtinymesa{EXT}'" "'/opt/homebrew/lib/libtinymesa_cpu.dylib'" "'/opt/homebrew/lib/libtinymesa.dylib'"
  echo "lvp_nir_options = gzip.decompress(base64.b64decode('$LVP_NIR_OPTIONS'))" >> $BASE/mesa.py
  sed -i "/in_dll/s/.*/try: &\nexcept (AttributeError, ValueError): pass/" $BASE/mesa.py
  sed -i "s/import ctypes/import ctypes, ctypes.util, os, gzip, base64, subprocess, tinygrad.helpers as helpers/" $BASE/mesa.py
  sed -i "s/ctypes.CDLL('.\+')/(dll := _try_dlopen_tinymesa_cpu())/" $BASE/mesa.py
  echo "def __getattr__(nm): raise AttributeError('LLVMpipe requires tinymesa_cpu' if 'tinymesa_cpu' not in dll._name else f'attribute {nm} not found') if dll else FileNotFoundError(f'libtinymesa not found (MESA_PATH={BASE}). See https://github.com/sirhcm/tinymesa ($TINYMESA_TAG, $MESA_TAG)')" >> $BASE/mesa.py
  sed -i "s/ctypes.glsl_base_type/glsl_base_type/" $BASE/mesa.py
  # bitfield bug in clang2py
  sed -i "s/('fp_fast_math', ctypes.c_bool, 9)/('fp_fast_math', ctypes.c_uint32, 9)/" $BASE/mesa.py
  sed -i "s/('\(\w\+\)', pipe_shader_type, 8)/('\1', ctypes.c_ubyte)/" $BASE/mesa.py
  sed -i "s/\([0-9]\+\)()/\1/" $BASE/mesa.py
  sed -i '/struct_nir_builder._pack_ = 1 # source:False/d' "$BASE/mesa.py"
  python3 -c "import tinygrad.runtime.autogen.mesa"
}

if [ "$1" == "opencl" ]; then generate_opencl
elif [ "$1" == "hip" ]; then generate_hip
elif [ "$1" == "comgr" ]; then generate_comgr
elif [ "$1" == "cuda" ]; then generate_cuda
elif [ "$1" == "nvrtc" ]; then generate_nvrtc
elif [ "$1" == "hsa" ]; then generate_hsa
elif [ "$1" == "kfd" ]; then generate_kfd
elif [ "$1" == "nv" ]; then generate_nv
elif [ "$1" == "amd" ]; then generate_amd
elif [ "$1" == "am" ]; then generate_am
elif [ "$1" == "sqtt" ]; then generate_sqtt
elif [ "$1" == "qcom" ]; then generate_qcom
elif [ "$1" == "io_uring" ]; then generate_io_uring
elif [ "$1" == "ib" ]; then generate_ib
elif [ "$1" == "llvm" ]; then generate_llvm
elif [ "$1" == "kgsl" ]; then generate_kgsl
elif [ "$1" == "adreno" ]; then generate_adreno
elif [ "$1" == "pci" ]; then generate_pci
elif [ "$1" == "vfio" ]; then generate_vfio
elif [ "$1" == "webgpu" ]; then generate_webgpu
elif [ "$1" == "libusb" ]; then generate_libusb
elif [ "$1" == "mesa" ]; then generate_mesa
elif [ "$1" == "all" ]; then generate_opencl; generate_hip; generate_comgr; generate_cuda; generate_nvrtc; generate_hsa; generate_kfd; generate_nv; generate_amd; generate_io_uring; generate_am; generate_webgpu; generate_mesa
else echo "usage: $0 <type>"
fi
