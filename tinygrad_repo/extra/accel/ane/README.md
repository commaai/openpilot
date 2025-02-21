# The Apple Neural Engine

The Apple Neural Engine is a fancy DMA Engine that is based around convolutions. We don't have all the details worked out yet, but we can do some things with it. At its core, it runs through 0x300 ops in an hwx file. See `aneregs` for the registers used in each op.

It operates out of RAM or its 4MB L2 cache. The L2 "cache" appears to be manually managed, and only applies to the input and output, not the weights. The weights are usually included in the program, and it's unclear where they are copied to.

The 16 cores likely refer to the 16 wide Kernel DMA engine. They claim 11 TOPS total, which would be 687.5 GOPS/core. Perhaps it's a 32x32 MAC running at 335 MHz. That clock speed matches the cycle count time ratio from the debug perf stats.

It works with 5D Tensors, you specify the stride for the latter 4. All strides must be a multiple of 0x40 bytes
* Column (width)    -- aneRegs.Common.InDim.Win / aneRegs.Common.OutDim.Wout
* Row    (height)   -- aneRegs.Common.InDim.Hin / aneRegs.Common.OutDim.Hout
* Plane  (channels) -- aneRegs.Common.Cin.Cin / aneRegs.Common.Cout.Cout
* Depth
* Group  (batch)    -- aneRegs.Common.GroupConvCfg.NumGroups

It works with 3 data types
* UInt8
* Int8
* Float16

The ops have several parts
* Header -- The base addresses for the DMA engines
* KernelDMASrc -- 16x wide DMA engine for the weights/bias/nonlinearity
* Common -- Specifies the parameters for the convolution
* TileDMASrc -- Input DMA engine
* L2 -- Use the L2 cache for Source/Result instead of RAM
* NE -- Configure Kernel/MAC/Post
* TileDMADst -- Output DMA engine

It can work with 8 base addresses for the DMA streams per OP
* 2x Read, both used for things like sum
* 1x Write
* 1x T?
* 4x Kernel, though only the first one seems used

## Normal Flow for ANE Usage

* Keras/ONNX model -> coremltools
* CoreML model -> Espresso
* net.plist -> ANECompiler
* model.hwx -> ANEServices
* AppleH11ANEInterface, an IOKit interface to the kernel

## hwx file?

This is a Mach-O file. We haven't figured out all the details, but the ops are at 0x4000. See `hwx_parse.py`

## amfid

Sadly disabling amfi breaks things like vscode. You can runtime patch

```
# MacOS 12.4

smol :: ~/fun/tinygrad » sha1sum /usr/libexec/amfid 
0f7e7f7e41408f83d7ebc7564a3828f41cb2ab58  /usr/libexec/amfid

# with patching +0x8e38

(lldb) image list
[  0] 04B6DF6C-6068-3F18-81A7-978985574387 0x0000000102ad0000 /usr/libexec/amfid 
(lldb) p *(unsigned int *)0x102ad8e38=0xd2800000
```

This disables the entitlement check, then you don't need a bootarg. I wish Apple made a better way to do this.

## Extracting ANEServices.framework

```
# install xcode and 
sudo xcode-select --switch /Applications/Xcode.app
# xcode also contains ANEServices.tbd
brew install keith/formulae/dyld-shared-cache-extractor
dyld-shared-cache-extractor /System/Library/dyld/dyld_shared_cache_arm64e /tmp/libraries
cp /tmp/libraries/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler .
cp /tmp/libraries/System/Library/PrivateFrameworks/ANEServices.framework/Versions/A/ANEServices .
cp /tmp/libraries/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine .
```

## Other work

```
# sadly also relies on ZinIrRegBitPrintOutDebug
https://github.com/antgroup-arclab/ANETools.git

# sadly looks like we do actually need a direct connection to run hwx files, aned is at the espresso level
* frame #0: 0x00000001c250fecc AppleNeuralEngine`-[_ANEDaemonConnection loadModel:sandboxExtension:options:qos:withReply:]
(lldb) po $x2
_ANEModel: { modelURL=file:///var/folders/l8/38vj8bm52_gfgsqgdn__sh2w0000gn/T/test_F48D9B88-A68D-476F-ADC8-32BDAF9A2498.mlmodelc/ : key={"isegment":0,"inputs":{"image":{"shape":[1,1,1,64,1]},"image2":{"shape":[1,1,1,64,1]}},"outputs":{"probs":{"shape":[1,1,1,64,1]}}} : string_id=0x00000000 : program=(null) : state=1 : programHandle=0 : intermediateBufferHandle=0 : queueDepth=0 : attr={
} : perfStatsMask=0} 
```

## Choices

* Disable amfid (breaks vscode)
* Patch amfid to allow restricted entitlements
* Sign with a "provisioning profile" to allow the entitlement
* Patch the ANE kext to not require a special entitlement (this is ideal, as we don't need to resign python)
