from __future__ import annotations

from ctypes import (
    CDLL,
    POINTER,
    cast,
    c_void_p,
    c_size_t,
    c_int,
    c_uint32,
    c_ulong,
    c_char_p,
)

# Constants for memory mapping
PROT_READ = 1
PROT_WRITE = 2
MAP_SHARED = 1
MAP_FIXED = 0x10
O_RDWR = 2
O_SYNC = 0x1000
O_CLOEXEC = 0x80000

# Function prototypes
libc = CDLL('libc.so.6')
mmap = libc.mmap
mmap.argtypes = [c_void_p, c_size_t, c_int, c_int, c_int, c_ulong]
mmap.restype = c_void_p

munmap = libc.munmap
munmap.argtypes = [c_void_p, c_size_t]
munmap.restype = c_int

open_rest = libc.open
open_rest.argtypes = [c_char_p, c_int, c_int]
open_rest.restype = c_int

close_rest = libc.close
close_rest.argtypes = [c_int]
close_rest.restype = c_int

class GPIOChip:
    def __init__(self, gpio_nr: int) -> None:
        self.gpio_nr = gpio_nr
        # Base address for GPIO controller 0 (adjust if needed)
        self.gpio_base = 0x1E460000
        self.gpio_size = 0x1000
        self.mem_fd = -1
        self.gpio_map: int | None = None

    def __enter__(self) -> GPIOChip:
        try:
            # Open /dev/mem
            self.mem_fd = open_rest(b"/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC)
            if self.mem_fd < 0:
                raise OSError("Failed to open /dev/mem")

            # Map GPIO memory
            self.gpio_map = mmap(
                0,
                self.gpio_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.mem_fd,
                self.gpio_base,
            )

            if self.gpio_map == -1:
                raise OSError("Failed to map GPIO memory")
            return self

        except Exception as e:
            self._cleanup()
            raise e
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self._cleanup()
    
    def _cleanup(self) -> None:
        if self.gpio_map is not None and self.gpio_map != -1:
            munmap(self.gpio_map, self.gpio_size)
            self.gpio_map = None
        if self.mem_fd >= 0:
            close_rest(self.mem_fd)
            self.mem_fd = -1

    def get_value(self) -> int:
        """Read GPIO value"""
        if self.gpio_map is None:
            raise RuntimeError("GPIO not initialized")
        
        # Calculate register offset for GPIO data in
        # This is a simplified version - actual register layout depends on the hardware
        reg_offset = 0x134  # GPIO_DATAIN_0 register offset (adjust for your hardware)
        reg_addr = self.gpio_map + reg_offset
        
        # Read the register value
        reg_value = cast(reg_addr, POINTER(c_uint32)).contents.value
        
        # Check if the bit for our GPIO is set
        return (reg_value >> self.gpio_nr) & 0x1

class GPIOHandler:
    def __init__(self, gpio_nr: int) -> None:
        self.gpio_nr = gpio_nr
        self.gpio_chip: GPIOChip | None = None

    def __enter__(self) -> GPIOHandler:
        self.gpio_chip = GPIOChip(self.gpio_nr)
        self.gpio_chip.__enter__()
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        if self.gpio_chip is not None:
            self.gpio_chip.__exit__(exc_type, exc_val, exc_tb)
            self.gpio_chip = None
            
    def get_value(self) -> int:
        if self.gpio_chip is None:
            with GPIOChip(self.gpio_nr) as chip:
                return chip.get_value()
        return self.gpio_chip.get_value()

def export_gpio(gpio_nr: int) -> None:
    """Export a GPIO pin using sysfs"""
    try:
        with open("/sys/class/gpio/export", "w", encoding='utf-8') as f:
            f.write(str(gpio_nr))
    except OSError:
        pass  # GPIO might already be exported


def unexport_gpio(gpio_nr: int) -> None:
    """Unexport a GPIO pin using sysfs"""
    try:
        with open("/sys/class/gpio/unexport", "w", encoding='utf-8') as f:
            f.write(str(gpio_nr))
    except OSError:
        pass  # GPIO might not be exported


def set_gpio_direction(gpio_nr: int, direction: str) -> None:
    """Set GPIO direction ("in" or "out")"""
    gpio_path = f"/sys/class/gpio/gpio{gpio_nr}/direction"
    try:
        with open(gpio_path, "w", encoding='utf-8') as f:
            f.write(direction)
    except OSError as e:
        raise RuntimeError(f"Failed to set GPIO direction: {e}") from e
