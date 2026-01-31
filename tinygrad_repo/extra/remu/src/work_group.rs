use crate::helpers::{colored, DEBUG};
use crate::state::{Register, StateSnapshot, VecDataStore, WaveValue, VGPR};
use crate::thread::{Thread, END_PRG, SGPR_COUNT};
use std::collections::HashMap;

pub const WAVE_SIZE: usize = 32;

pub struct WorkGroup<'a> {
    dispatch_dim: u32,
    id: [u32; 3],
    lds: VecDataStore,
    kernel: &'a Vec<u32>,
    kernel_args: *const u64,
    launch_bounds: [u32; 3],
    wave_state: HashMap<usize, WaveState>,
}

#[derive(Debug, Clone)]
struct WaveState {
    scalar_reg: [u32; SGPR_COUNT],
    scc: u32,
    vcc: WaveValue,
    exec: WaveValue,
    vec_reg: VGPR,
    pc: usize,
    sds: HashMap<usize, VecDataStore>,
}

const SYNCS: [u32; 4] = [0xBF89FC07, 0xBC7C0000, 0xBF890007, 0xbFB60003];
const S_BARRIER: u32 = 0xBFBD0000;

/// Context for single-stepping through a wave - holds all mutable state
pub struct WaveContext {
    pub kernel: Vec<u32>,
    pub scalar_reg: [u32; SGPR_COUNT],
    pub scc: u32,
    pub pc: usize,
    pub vec_reg: VGPR,
    pub vcc: WaveValue,
    pub exec: WaveValue,
    pub lds: VecDataStore,
    pub sds: HashMap<usize, VecDataStore>,
    pub n_lanes: usize,
}

impl WaveContext {
    pub fn new(kernel: Vec<u32>, n_lanes: usize) -> Self {
        let active = (!0u32).wrapping_shr(32 - (n_lanes as u32));
        Self {
            kernel,
            scalar_reg: [0; SGPR_COUNT],
            scc: 0,
            pc: 0,
            vec_reg: VGPR::new(),
            vcc: WaveValue::new(0, n_lanes),
            exec: WaveValue::new(active, n_lanes),
            lds: VecDataStore::new(),
            sds: (0..=31).map(|i| (i, VecDataStore::new())).collect(),
            n_lanes,
        }
    }

    /// Execute a single instruction. Returns: 0=continue, -1=endpgm, -2=barrier, 1=done (pc past program), negative=error
    pub fn step(&mut self) -> i32 {
        if self.pc >= self.kernel.len() { return 1; }
        if self.kernel[self.pc] == END_PRG { return -1; }
        if self.kernel[self.pc] == S_BARRIER { self.pc += 1; return -2; }
        // Skip sync/nop instructions
        if SYNCS.contains(&self.kernel[self.pc]) || self.kernel[self.pc] >> 20 == 0xbf8 || self.kernel[self.pc] == 0x7E000000 {
            self.pc += 1;
            return 0;
        }

        let mut sgpr_co = None;
        for lane_id in 0..self.n_lanes {
            self.vec_reg.default_lane = Some(lane_id);
            self.vcc.default_lane = Some(lane_id);
            self.exec.default_lane = Some(lane_id);
            let mut thread = Thread {
                scalar_reg: &mut self.scalar_reg,
                scc: &mut self.scc,
                vec_reg: &mut self.vec_reg,
                vcc: &mut self.vcc,
                exec: &mut self.exec,
                lds: &mut self.lds,
                sds: &mut self.sds.get_mut(&lane_id).unwrap(),
                pc_offset: 0,
                stream: self.kernel[self.pc..].to_vec(),
                scalar: false,
                simm: None,
                warp_size: self.n_lanes,
                sgpr_co: &mut sgpr_co,
            };
            if let Err(e) = thread.interpret() { return e; }
            if thread.scalar {
                self.pc = ((self.pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                break;
            }
            if lane_id == self.n_lanes - 1 {
                self.pc = ((self.pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
            }
        }
        if self.vcc.mutations.is_some() { self.vcc.apply_muts(); self.vcc.mutations = None; }
        if self.exec.mutations.is_some() { self.exec.apply_muts(); self.exec.mutations = None; }
        if let Some((idx, mut wv)) = sgpr_co.take() { wv.apply_muts(); self.scalar_reg[idx] = wv.value; }
        0
    }

    pub fn get_snapshot(&self) -> StateSnapshot {
        let mut snap = StateSnapshot::new();
        snap.pc = self.pc as u32;
        snap.scc = self.scc;
        snap.vcc = self.vcc.value;
        snap.exec_mask = self.exec.value;
        snap.sgpr = self.scalar_reg;
        for lane in 0..32 { snap.vgpr[lane] = self.vec_reg.get_lane(lane); }
        snap
    }
}

impl<'a> WorkGroup<'a> {
    pub fn new(dispatch_dim: u32, id: [u32; 3], launch_bounds: [u32; 3], kernel: &'a Vec<u32>, kernel_args: *const u64) -> Self {
        Self { dispatch_dim, id, kernel, launch_bounds, kernel_args, lds: VecDataStore::new(), wave_state: HashMap::new() }
    }

    pub fn exec_waves(&mut self) -> Result<(), i32> {
        let mut threads = vec![];
        for z in 0..self.launch_bounds[2] {
            for y in 0..self.launch_bounds[1] {
                for x in 0..self.launch_bounds[0] {
                    threads.push([x, y, z])
                }
            }
        }
        let waves = threads.chunks(WAVE_SIZE).collect::<Vec<_>>();

        let mut sync = false;
        for (i, x) in self.kernel.iter().enumerate() {
            if i != 0 && *x == S_BARRIER {
                sync = true;
                break;
            }
        }

        for _ in 0..=(sync as usize) {
            for w in waves.iter().enumerate() {
                self.exec_wave(w)?
            }
        }
        Ok(())
    }

    fn exec_wave(&mut self, (wave_id, threads): (usize, &&[[u32; 3]])) -> Result<(), i32> {
        let (mut scalar_reg, mut scc, mut pc, mut vec_reg, mut vcc, mut exec, mut sds) = match self.wave_state.get(&wave_id) {
          None => {
            let mut scalar_reg = [0; SGPR_COUNT];
            scalar_reg.write64(0, self.kernel_args as u64);

            let [gx, gy, gz] = self.id;
            match self.dispatch_dim {
              3 => (scalar_reg[13], scalar_reg[14], scalar_reg[15]) = (gx, gy, gz),
              2 => (scalar_reg[14], scalar_reg[15]) = (gx, gy),
              _ => scalar_reg[15] = gx,
            }

            let mut vec_reg = VGPR::new();
            for (t, [x, y, z]) in threads.iter().enumerate() {
              vec_reg.get_lane_mut(t)[0] = match &self.launch_bounds {
                [_, 1, 1] => *x,
                _ => (z << 20) | (y << 10) | x,
              }
            }

            let vcc = WaveValue::new(0, threads.len());
            let active = (!0u32).wrapping_shr(32 - (threads.len() as u32));
            let exec = WaveValue::new(active, threads.len());

            let sds = (0..=31).map(|i| (i, VecDataStore::new())).collect();
            (scalar_reg, 0, 0, vec_reg, vcc, exec, sds)
          }

          Some(val) => {
            let val = val.clone();
            (val.scalar_reg, val.scc, val.pc, val.vec_reg, val.vcc, val.exec, val.sds)
          }
        };

        loop {
            if self.kernel[pc] == END_PRG {
                break Ok(());
            }
            if self.kernel[pc] == S_BARRIER && self.wave_state.get(&wave_id).is_none() {
                self.wave_state.insert(wave_id, WaveState { scalar_reg, scc, vec_reg, vcc, exec, pc, sds });
                break Ok(());
            }
            if self.kernel[pc] == S_BARRIER || SYNCS.contains(&self.kernel[pc]) || self.kernel[pc] >> 20 == 0xbf8 || self.kernel[pc] == 0x7E000000 {
                pc += 1;
                continue;
            }

            let mut sgpr_co = None;
            for (lane_id, [x, y, z]) in threads.iter().enumerate() {
                vec_reg.default_lane = Some(lane_id);
                vcc.default_lane = Some(lane_id);
                exec.default_lane = Some(lane_id);
                if *DEBUG {
                    let lane = format!("{:<2} {:08X} ", lane_id, self.kernel[pc]);
                    let state = match exec.read() {
                        true => "green",
                        false => "gray",
                    };
                    let [id0, id1, id2] = self.id;
                    print!("[{id0:<3} {id1:<3} {id2:<3}] [{x:<3} {y:<3} {z:<3}] {}", colored(&lane, state));
                }
                let mut thread = Thread {
                    scalar_reg: &mut scalar_reg,
                    scc: &mut scc,
                    vec_reg: &mut vec_reg,
                    vcc: &mut vcc,
                    exec: &mut exec,
                    lds: &mut self.lds,
                    sds: &mut sds.get_mut(&lane_id).unwrap(),
                    pc_offset: 0,
                    stream: self.kernel[pc..self.kernel.len()].to_vec(),
                    scalar: false,
                    simm: None,
                    warp_size: threads.len(),
                    sgpr_co: &mut sgpr_co,
                };
                thread.interpret()?;
                if *DEBUG {
                    println!();
                }
                if thread.scalar {
                    pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                    break;
                }
                if lane_id == threads.len() - 1 {
                    pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                }
            }

            if vcc.mutations.is_some() {
                vcc.apply_muts();
                vcc.mutations = None;
            }
            if exec.mutations.is_some() {
                exec.apply_muts();
                exec.mutations = None;
            }
            if let Some((idx, mut wv)) = sgpr_co.take() {
                wv.apply_muts();
                scalar_reg[idx] = wv.value;
            }
        }
    }
}

#[cfg(test)]
mod test_workgroup {
    use super::*;

    // TODO: make this generic by adding the assembler
    fn global_store_sgpr(addr: u64, instructions: Vec<u32>, src: u32) -> Vec<u32> {
        [
            instructions,
            vec![
                0x7E020200 + src,
                0x7E0402FF,
                addr as u32,
                0x7E0602FF,
                (addr >> 32) as u32,
                0xDC6A0000,
                0x007C0102,
            ],
            vec![END_PRG],
        ]
        .concat()
    }

    #[test]
    fn test_wave_value_state_vcc() {
        let mut ret: u32 = 0;
        let kernel = vec![
            0xBEEA00FF,
            0b11111111111111111111111111111111, // initial vcc state
            0x7E140282,
            0x7C94010A, // cmp blockDim.x == 2
        ];
        let addr = (&mut ret as *mut u32) as u64;
        let kernel = global_store_sgpr(addr, kernel, 106);
        let mut wg = WorkGroup::new(1, [0, 0, 0], [3, 1, 1], &kernel, [addr].as_ptr());
        wg.exec_waves().unwrap();
        assert_eq!(ret, 0b100);
    }

    #[test]
    fn test_wave_value_state_exec() {
        let mut ret: u32 = 0;
        let kernel = vec![
            0xBEFE00FF,
            0b11111111111111111111111111111111,
            0x7E140282,
            0x7D9C010A, // cmpx blockDim.x <= 2
        ];
        let addr = (&mut ret as *mut u32) as u64;
        let kernel = global_store_sgpr(addr, kernel, 126);
        let mut wg = WorkGroup::new(1, [0, 0, 0], [4, 1, 1], &kernel, [addr].as_ptr());
        wg.exec_waves().unwrap();
        assert_eq!(ret, 0b0111);
    }

    #[test]
    fn test_wave_value_sgpr_co() {
        let mut ret: u32 = 0;
        let kernel = vec![0xBE8D00FF, 0x7FFFFFFF, 0x7E1402FF, u32::MAX, 0xD700000A, 0x0002010A];
        let addr = (&mut ret as *mut u32) as u64;
        let kernel = global_store_sgpr(addr, kernel, 0);
        let mut wg = WorkGroup::new(1, [0, 0, 0], [5, 1, 1], &kernel, [addr].as_ptr());
        wg.exec_waves().unwrap();
        assert_eq!(ret, 0b11110);
    }
}
