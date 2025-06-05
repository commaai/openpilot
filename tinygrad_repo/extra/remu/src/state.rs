use std::ops::{Index, IndexMut};

pub trait Register {
    fn read64(&self, idx: usize) -> u64;
    fn write64(&mut self, idx: usize, addr: u64);
}
impl<T> Register for T where T: Index<usize, Output = u32> + IndexMut<usize> {
    fn read64(&self, idx: usize) -> u64 {
        let lsb = self[idx] as u64;
        let msb = self[idx + 1] as u64;
        (msb << 32) | lsb
    }

    fn write64(&mut self, idx: usize, value: u64) {
        self[idx] = (value & 0xffffffff) as u32;
        self[idx + 1] = ((value & (0xffffffff << 32)) >> 32) as u32;
    }
}

#[derive(Debug, Clone)]
pub struct VGPR {
    values: [[u32; 256]; 32],
    pub default_lane: Option<usize>,
}
impl Index<usize> for VGPR {
    type Output = u32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[self.default_lane.unwrap()][index]
    }
}
impl IndexMut<usize> for VGPR {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[self.default_lane.unwrap()][index]
    }
}
impl VGPR {
    pub fn new() -> Self {
        VGPR {
            values: [[0; 256]; 32],
            default_lane: None,
        }
    }
    pub fn get_lane(&self, lane: usize) -> [u32; 256] {
        *self.values.get(lane).unwrap()
    }
    pub fn get_lane_mut(&mut self, lane: usize) -> &mut [u32; 256] {
        self.values.get_mut(lane).unwrap()
    }
}

pub trait Value {
    fn mut_hi16(&mut self, val: u16);
    fn mut_lo16(&mut self, val: u16);
}
impl Value for u32 {
    fn mut_hi16(&mut self, val: u16) {
        *self = ((val as u32) << 16) | (*self as u16 as u32);
    }
    fn mut_lo16(&mut self, val: u16) {
        *self = ((((*self & (0xffff << 16)) >> 16) as u32) << 16) | val as u32;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WaveValue {
    pub value: u32,
    pub warp_size: usize,
    pub default_lane: Option<usize>,
    pub mutations: Option<[bool; 32]>,
}
impl WaveValue {
    pub fn new(value: u32, warp_size: usize) -> Self {
        Self {
            value,
            warp_size,
            default_lane: None,
            mutations: None,
        }
    }
    pub fn read(&self) -> bool {
        (self.value >> self.default_lane.unwrap()) & 1 == 1
    }
    pub fn set_lane(&mut self, value: bool) {
        if self.mutations.is_none() {
            self.mutations = Some([false; 32])
        }
        self.mutations.as_mut().unwrap()[self.default_lane.unwrap()] = value;
    }
    pub fn apply_muts(&mut self) {
        self.value = 0;
        for lane in 0..self.warp_size {
            if self.mutations.unwrap()[lane] {
                self.value |= 1 << lane;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct VecDataStore {
    pub data: Vec<u8>,
}

impl VecDataStore {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    pub fn write(&mut self, addr: usize, val: u32) {
        if addr + 4 >= self.data.len() {
            self.data.resize(self.data.len() + addr + 5, 0);
        }
        self.data[addr..addr + 4].iter_mut().enumerate().for_each(|(i, x)| {
            *x = val.to_le_bytes()[i];
        });
    }
    pub fn write64(&mut self, addr: usize, val: u64) {
        self.write(addr, (val & 0xffffffff) as u32);
        self.write(addr + 4, ((val & (0xffffffff << 32)) >> 32) as u32);
    }
    pub fn read(&self, addr: usize) -> u32 {
        let mut bytes: [u8; 4] = [0; 4];
        bytes.copy_from_slice(&self.data[addr + 0..addr + 4]);
        u32::from_le_bytes(bytes)
    }
    pub fn read64(&mut self, addr: usize) -> u64 {
        let lsb = self.read(addr);
        let msb = self.read(addr + 4);
        ((msb as u64) << 32) | lsb as u64
    }
}

#[cfg(test)]
mod test_state {
    use super::*;

    #[test]
    fn test_wave_value() {
        let mut val = WaveValue::new(0b11000000000000011111111111101110, 32);
        val.default_lane = Some(0);
        assert!(!val.read());
        val.default_lane = Some(31);
        assert!(val.read());
    }

    #[test]
    fn test_wave_value_small() {
        let mut val = WaveValue::new(0, 1);
        val.default_lane = Some(0);
        assert!(!val.read());
        assert_eq!(val.value, 0);
        val.set_lane(true);
        val.apply_muts();
        assert!(val.read());
        assert_eq!(val.value, 1);
    }

    #[test]
    fn test_wave_value_small_alt() {
        let mut val = WaveValue::new(0, 2);
        val.default_lane = Some(0);
        assert!(!val.read());
        assert_eq!(val.value, 0);
        val.set_lane(true);
        val.apply_muts();
        assert!(val.read());
        assert_eq!(val.value, 1);
    }

    #[test]
    fn test_wave_value_exec() {
        let warp_size = 32;
        let val = WaveValue::new(u32::MAX, warp_size);
        assert_eq!(val.value, u32::MAX);
        let warp_size = 3;
        let val = WaveValue::new((1 << warp_size) - 1, warp_size);
        assert_eq!(val.value, 7)
    }

    #[test]
    fn test_wave_value_toggle_one() {
        let warp_size = 2;
        let mut val = WaveValue::new(0b11, warp_size);
        // 0
        val.default_lane = Some(0);
        val.set_lane(false);
        // 1
        val.default_lane = Some(1);
        val.set_lane(true);
        val.apply_muts();
        assert_eq!(val.value, 2);
    }

    #[test]
    fn test_wave_value_mutate_small() {
        let mut val = WaveValue::new(0, 2);
        val.default_lane = Some(0);
        assert!(!val.read());
        assert_eq!(val.value, 0);
        val.set_lane(true);
        val.apply_muts();
        assert!(val.read());
        assert_eq!(val.value, 1);
    }

    #[test]
    fn test_wave_value_mutations() {
        let mut val = WaveValue::new(0b10001, 32);
        val.default_lane = Some(0);
        val.set_lane(false);
        assert!(val.mutations.unwrap().iter().all(|x| !x));
        val.default_lane = Some(1);
        val.set_lane(true);
        assert_eq!(val.value, 0b10001);
        assert_eq!(
            val.mutations,
            Some([
                false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false, false,
            ])
        );

        val.apply_muts();
        assert_eq!(val.value, 0b10);
    }

    #[test]
    fn test_write16() {
        let mut vgpr = VGPR::new();
        vgpr.default_lane = Some(0);
        vgpr[0] = 0b11100000000000001111111111111111;
        vgpr[0].mut_lo16(0b1011101111111110);
        assert_eq!(vgpr[0], 0b11100000000000001011101111111110);
    }

    #[test]
    fn test_write16hi() {
        let mut vgpr = VGPR::new();
        vgpr.default_lane = Some(0);
        vgpr[0] = 0b11100000000000001111111111111111;
        vgpr[0].mut_hi16(0b1011101111111110);
        assert_eq!(vgpr[0], 0b10111011111111101111111111111111);
    }

    #[test]
    fn test_vgpr() {
        let mut vgpr = VGPR::new();
        vgpr.default_lane = Some(0);
        vgpr[0] = 42;
        vgpr.default_lane = Some(10);
        vgpr[0] = 10;
        assert_eq!(vgpr.get_lane(0)[0], 42);
        assert_eq!(vgpr.get_lane(10)[0], 10);
    }
}
