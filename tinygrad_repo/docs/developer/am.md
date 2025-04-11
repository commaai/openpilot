# AM Driver

AM driver is a userspace driver targeting AMD's 7900XTX. You only need tinygrad to send compute tasks to your GPU!

## How to run?
Make sure that amdgpu module is unloaded and just run tinygrad with `AMD=1`!

Optional requirements:

* System without IOMMU for P2P / SDMA support
* vfio-pci module for IRQ handling

## Environment Variables

| Variable | Possible Value(s) | Description |
|----------|------------------|-------------|
| AM_RESET | [1] | Performs a full GPU reset (reloading all firmware and IP blocks) |
| AM_DEBUG | [0-4] | Sets the level of additional debugging information |

## AM Driver Details

### Compute & SDMA Queues

AM binds compute queues directly to MEC (bypassing MES). Tinygrad uses only one compute queue, which is bound at `pipe=0 queue=0`. Similarly, the single SDMA queue is bound at `engine=0 queue=0`.

### Boot

The GPU being passed can be in one of several states:
1. Not initialized
2. Initialized by amdgpu
3. Initialized by AM

The first and second states require a full GPU setup since their states are unknown. The second state also requires a mode1 reset to reinitialize all components.

The third state can be set up partially to optimize boot time. In this case, only the GFX and SDMA IPs need to be initialized. To enable this, AM uses a separate boot memory that is guaranteed not to be overwritten. This physical memory is utilized for all blocks that are initialized only during the initial AM boot. To determine if the GPU is in the third state, AM uses `regSCRATCH_REG7` as a flag.

### VM Management

Each AM device sets up only a single `VMID=0` and one page directory. The page directory used is 3-level and thus supports up to 512GB of virtual addresses. All AM devices are located in one virtual address space.