use crate::helpers::{extract_mantissa, f16_hi, f16_lo, ldexp, nth, sign_ext, IEEEClass, VOPModifier};
use crate::helpers::DEBUG;
use crate::state::{Register, Value, VecDataStore, WaveValue, VGPR};
use crate::todo_instr;
use half::{bf16, f16};
use crate::rdna3::{Instruction, decode};
use num_traits::Float;

pub const SGPR_COUNT: usize = 128;
pub const VCC: usize = 106;
pub const EXEC: usize = 126;
pub const NULL_SRC: usize = 124;
pub const SGPR_SRC: usize = 105;

const VGPR_COUNT: usize = 256;
const SIMM_SRC: usize = 255;

pub const END_PRG: u32 = 0xbfb00000;

pub struct Thread<'a> {
    pub scalar_reg: &'a mut [u32; SGPR_COUNT],
    pub scc: &'a mut u32, // SCC is physically an sgpr, unclear which one

    pub vec_reg: &'a mut VGPR,
    pub vcc: &'a mut WaveValue,
    pub exec: &'a mut WaveValue,

    pub lds: &'a mut VecDataStore,
    pub sds: &'a mut VecDataStore,

    pub pc_offset: usize,
    pub stream: Vec<u32>,
    pub simm: Option<u32>,
    pub sgpr_co: &'a mut Option<(usize, WaveValue)>,
    pub warp_size: usize,
    pub scalar: bool,
}

impl<'a> Thread<'a> {
    pub fn interpret(&mut self) -> Result<(), i32> {
        let instruction = self.stream[self.pc_offset];
        let decoded = decode(self.stream[self.pc_offset], self.stream.get(self.pc_offset+1));
        if *DEBUG {
            print!("{:?}", decoded);
        }
        if let Instruction::SMEM { sbase, sdata, op, offset, soffset, .. } = decoded {
            let _ = self.u64_instr();
            let soffset: u32 = self.val(soffset as usize);

            // TODO: refactor vcc_lo to store in scalar register 106
            let base_addr = match sbase as usize {
                VCC => ((self.scalar_reg[107] as u64) << 32) | self.vcc.value as u64,
                s => self.scalar_reg.read64(s),
            };
            let addr = (base_addr as i64 + offset as i64 + soffset as i64) as u64;

            match op {
                0..=4 => (0..2_usize.pow(op as u32)).for_each(|i| {
                    let ret = unsafe { *((addr + (4 * i as u64)) as *const u32) };
                    self.write_to_sdst(sdata as usize + i, ret);
                }),
                _ => todo_instr!(instruction)?,
            };
            self.scalar = true;
        }
        else if let Instruction::SOP1 { ssrc0, op, sdst } = decoded {
            let src = ssrc0 as usize;
            let sdst = sdst as usize;

            match op {
                1 => {
                    let s0 = self.val(src);
                    let ret = match op {
                        1 => s0,
                        _ => todo_instr!(instruction)?,
                    };
                    self.scalar_reg.write64(sdst as usize, ret);
                }
                _ => {
                    let s0 = self.val(src);
                    let ret = match op {
                        0 => s0,
                        10 => self.clz_i32_u32(s0),
                        12 => self.cls_i32(s0),
                        4 => s0.reverse_bits(),
                        14 => s0 as i8 as i32 as u32,
                        15 => s0 as i16 as i32 as u32,
                        16 | 18 => {
                            let sdst: u32 = self.val(sdst as usize);
                            if op == 16 {
                                sdst & !(1 << (s0 & 0x1f))
                            } else {
                                sdst | (1 << (s0 & 0x1f))
                            }
                        }
                        21 => {
                            let s0 = s0 as i32;
                            let ret = s0.abs();
                            *self.scc = (ret != 0) as u32;
                            ret as u32
                        }
                        30 => {
                            let ret = !s0;
                            *self.scc = (ret != 0) as u32;
                            ret
                        }
                        32 | 34 | 48 => {
                            let saveexec = self.exec.value;
                            self.exec.value = match op {
                                32 => s0 & saveexec,
                                34 => s0 | saveexec,
                                48 => s0 & !saveexec,
                                _ => todo_instr!(instruction)?,
                            };
                            *self.scc = (self.exec.value != 0) as u32;
                            saveexec
                        }
                        _ => todo_instr!(instruction)?,
                    };

                    self.write_to_sdst(sdst, ret);
                }
            };
            self.scalar = true;
        }
        else if let Instruction::SOPC { ssrc0, ssrc1, op } = decoded {
            let s0 = ssrc0 as usize;
            let s1 = ssrc1 as usize;

            fn scmp<T>(s0: T, s1: T, offset: u8, op: u8) -> bool
            where
                T: PartialOrd + PartialEq,
            {
                match op - offset {
                    0 => s0 == s1,
                    1 => s0 != s1,
                    2 => s0 > s1,
                    3 => s0 >= s1,
                    4 => s0 < s1,
                    _ => s0 <= s1,
                }
            }
            *self.scc = match op {
                0..=5 => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.val(s1));
                    scmp(s0 as i32, s1 as i32, 0, op)
                }
                6..=11 => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.val(s1));
                    scmp(s0, s1, 6, op)
                }
                12 => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.val(s1));
                    s0 & (1 << (s1 & 0x1F)) == 0
                }
                13 => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.val(s1));
                    s0 & (1 << (s1 & 0x1F)) == 1
                }
                16 | 17 => {
                    let (s0, s1): (u64, u64) = (self.val(s0), self.val(s1));
                    scmp(s0, s1, 16, op)
                }
                _ => todo_instr!(instruction)?,
            } as u32;
            self.scalar = true;
        }
        else if let Instruction::SOPP { simm16, op } = decoded {

            match op {
                32..=42 => {
                    let should_jump = match op {
                        32 => true,
                        33 => *self.scc == 0,
                        34 => *self.scc == 1,
                        35 => self.vcc.value == 0,
                        36 => self.vcc.value != 0,
                        37 => self.exec.value == 0,
                        38 => self.exec.value != 0,
                        _ => todo_instr!(instruction)?,
                    };
                    if should_jump {
                        self.pc_offset = (self.pc_offset as i64 + simm16 as i64) as usize;
                    }
                }
                _ => todo_instr!(instruction)?,
            };
            self.scalar = true;
        }
        else if let Instruction::SOPK { simm16, sdst, op } = decoded {
            let simm = simm16 as u16;
            let sdst = sdst as usize;
            let s0: u32 = self.val(sdst);

            match op {
                0 => self.write_to_sdst(sdst, simm as i16 as i32 as u32),
                3..=8 => {
                    let s1 = simm as i16 as i64;
                    let s0 = s0 as i32 as i64;
                    *self.scc = match op {
                        3 => s0 == s1,
                        4 => s0 != s1,
                        5 => s0 > s1,
                        7 => s0 < s1,
                        _ => todo_instr!(instruction)?,
                    } as u32
                }
                9..=14 => {
                    let s1 = simm as u16 as u32;
                    *self.scc = match op {
                        9 => s0 == s1,
                        10 => s0 != s1,
                        11 => s0 > s1,
                        12 => s0 >= s1,
                        13 => s0 < s1,
                        14 => s0 <= s1,
                        _ => todo_instr!(instruction)?,
                    } as u32
                }
                15 => {
                    let temp = s0 as i32;
                    let simm16 = simm as i16;
                    let dest = (temp as i64 + simm16 as i64) as i32;
                    self.write_to_sdst(sdst, dest as u32);
                    let temp_sign = ((temp >> 31) & 1) as u32;
                    let simm_sign = ((simm16 >> 15) & 1) as u32;
                    let dest_sign = ((dest >> 31) & 1) as u32;
                    *self.scc = ((temp_sign == simm_sign) && (temp_sign != dest_sign)) as u32;
                }
                16 => {
                    let simm16 = simm as i16;
                    let ret = (s0 as i32 * simm16 as i32) as u32;
                    self.write_to_sdst(sdst, ret);
                }
                _ => todo_instr!(instruction)?,
            };
            self.scalar = true;
        }
        else if let Instruction::SOP2 { ssrc0, ssrc1, sdst, op } = decoded {
            let s0 = ssrc0 as usize;
            let s1 = ssrc1 as usize;
            let sdst = sdst as usize;

            match op {
                23 | 25 | 27 => {
                    let (s0, s1): (u64, u64) = (self.val(s0), self.val(s1));
                    let ret = match op {
                        23 => s0 & s1,
                        25 => s0 | s1,
                        27 => s0 ^ s1,
                        _ => todo_instr!(instruction)?,
                    };
                    self.scalar_reg.write64(sdst as usize, ret);
                    *self.scc = (ret != 0) as u32;
                }
                9 | 13 | 11 | 40 | 41 => {
                    let (s0, s1): (u64, u32) = (self.val(s0), self.val(s1));
                    let ret = match op {
                        9 => {
                            let ret = s0 << (s1 & 0x3f);
                            (ret, Some(ret != 0))
                        }
                        11 => {
                            let ret = s0 >> (s1 & 0x3f);
                            (ret as u64, Some(ret != 0))
                        }
                        13 => {
                            let ret = (s0 as i64) >> (s1 & 0x3f);
                            (ret as u64, Some(ret != 0))
                        }
                        40 => {
                            let ret = (s0 >> (s1 & 0x3f)) & ((1 << ((s1 >> 16) & 0x7f)) - 1);
                            (ret as u64, Some(ret != 0))
                        }
                        41 => {
                            let s0 = s0 as i64;
                            let mut ret = (s0 >> (s1 & 0x3f)) & ((1 << ((s1 >> 16) & 0x7f)) - 1);
                            let shift = 64 - ((s1 >> 16) & 0x7f);
                            ret = (ret << shift) >> shift;
                            (ret as u64, Some(ret != 0))
                        }
                        _ => todo_instr!(instruction)?,
                    };
                    self.scalar_reg.write64(sdst as usize, ret.0);
                    if let Some(val) = ret.1 {
                        *self.scc = val as u32
                    }
                }
                _ => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.val(s1));
                    let ret = match op {
                        0 | 4 => {
                            let (s0, s1) = (s0 as u64, s1 as u64);
                            let ret = match op {
                                0 => s0 + s1,
                                4 => s0 + s1 + *self.scc as u64,
                                _ => todo_instr!(instruction)?,
                            };
                            (ret as u32, Some(ret >= 0x100000000))
                        }
                        1 => (s0 - s1, Some(s1 > s0)),
                        5 => (s0 - s1 - *self.scc, Some((s1 as u64 + *self.scc as u64) > s0 as u64)),
                        2 | 3 => {
                            let s0 = s0 as i32 as i64;
                            let s1 = s1 as i32 as i64;
                            let ret = match op {
                                2 => s0 + s1,
                                3 => s0 - s1,
                                _ => todo_instr!(instruction)?,
                            };
                            let overflow = (nth(s0 as u32, 31) == nth(s1 as u32, 31)) && (nth(s0 as u32, 31) != nth(ret as u32, 31));

                            (ret as i32 as u32, Some(overflow))
                        }
                        (8..=17) => {
                            let s1 = s1 & 0x1f;
                            let ret = match op {
                                8 => s0 << s1,
                                10 => s0 >> s1,
                                12 => ((s0 as i32) >> (s1 as i32)) as u32,
                                _ => todo_instr!(instruction)?,
                            };
                            (ret, Some(ret != 0))
                        }
                        (18..=21) => {
                            let scc = match op {
                                18 => (s0 as i32) < (s1 as i32),
                                19 => s0 < s1,
                                20 => (s0 as i32) > (s1 as i32),
                                21 => s0 > s1,
                                _ => todo_instr!(instruction)?,
                            };
                            let ret = match scc {
                                true => s0,
                                false => s1,
                            };
                            (ret, Some(scc))
                        }
                        (22..=26) | 34 | 36 => {
                            let ret = match op {
                                22 => s0 & s1,
                                24 => s0 | s1,
                                26 => s0 ^ s1,
                                34 => s0 & !s1,
                                36 => s0 | !s1,
                                _ => todo_instr!(instruction)?,
                            };
                            (ret, Some(ret != 0))
                        }
                        38 => {
                            let ret = (s0 >> (s1 & 0x1f)) & ((1 << ((s1 >> 16) & 0x7f)) - 1);
                            (ret, Some(ret != 0))
                        }
                        39 => {
                            let s0 = s0 as i32;
                            let mut ret = (s0 >> (s1 & 0x1f)) & ((1 << ((s1 >> 16) & 0x1f)) - 1);
                            let shift = 32 - ((s1 >> 16) & 0x7f);
                            ret = (ret << shift) >> shift;
                            (ret as u32, Some(ret != 0))
                        }
                        44 => (((s0 as i32) * (s1 as i32)) as u32, None),
                        45 => (((s0 as u64) * (s1 as u64) >> 32) as u32, None),
                        46 => ((((s0 as i32 as i64 * s1 as i32 as i64) as u64) >> 32u64) as i32 as u32, None),
                        48 => match *self.scc != 0 {
                            true => (s0, None),
                            false => (s1, None),
                        },
                        50..=53 => {
                            let (s0, s1) = match op {
                                50 => (s0 as u16, s1 as u16),
                                51 => (s0 as u16, (s1 >> 16) as u16),
                                52 => ((s0 >> 16) as u16, (s1 >> 16) as u16),
                                _ => ((s0 >> 16) as u16, s1 as u16),
                            };
                            (((s1 as u32) << 16) | (s0 as u32), None)
                        }
                        _ => todo_instr!(instruction)?,
                    };

                    self.write_to_sdst(sdst, ret.0);
                    if let Some(val) = ret.1 {
                        *self.scc = val as u32
                    }
                }
            };
            self.scalar = true;
        }
        // vopp
        else if instruction >> 24 == 0b11001100 {
            let instr = self.u64_instr();
            let vdst = (instr & 0xff) as usize;
            let clmp = (instr >> 15) & 0x1;
            assert_eq!([clmp], [0]);
            let op = (instr >> 16) & 0x7f;

            let mut src = |x: usize| -> (u16, u16, u32) {
                let val: u32 = self.val(x);
                match x {
                    255 => {
                        let val_lo: u16 = self.val(x);
                        (val_lo, val_lo, val)
                    }
                    (240..=247) => {
                        let val_lo: u16 = self.val(x);
                        (val_lo, f16::from_bits(0).to_bits(), val)
                    }
                    _ => ((val & 0xffff) as u16, ((val >> 16) & 0xffff) as u16, val),
                }
            };

            let s = [32, 41, 50].iter().map(|x| ((instr >> x) & 0x1ff) as usize).collect::<Vec<_>>();
            let src_parts = s.iter().map(|x| src(*x)).collect::<Vec<_>>();

            let b = |i: usize| (instr >> i) & 0x1 != 0;
            let neg_hi = ((instr >> 8) & 0x7) as usize;
            let neg = ((instr >> 61) & 0x7) as usize;
            let opsel = [b(11), b(12), b(13)];
            let opsel_hi = [b(59), b(60), b(14)];

            match op {
                0..=18 => {
                    let fxn = |x, y, z| -> u16 {
                        match op {
                            0 => x * y + z,
                            1 => x * y,
                            2 => x + y,
                            3 => x - y,
                            4 => y << (x & 0xf),
                            5 => y >> (x & 0xf),
                            6 => ((y as i16) >> ((x as i16) & 0xf)) as u16,
                            7 => i16::max(x as i16, y as i16) as u16,
                            8 => i16::min(x as i16, y as i16) as u16,
                            9 => x * y + z,
                            10 => x + y,
                            11 => x - y,
                            12 => u16::max(x, y),
                            13 => u16::min(x, y),
                            _ => {
                                let (x, y, z) = (f16::from_bits(x), f16::from_bits(y), f16::from_bits(z));
                                let ret = match op {
                                    14 => f16::mul_add(x, y, z),
                                    15 => x + y,
                                    16 => x * y,
                                    17 => f16::min(x, y),
                                    18 => f16::max(x, y),
                                    _ => unreachable!("op should be in range 0..=18, got {op}"),
                                };
                                ret.to_bits()
                            }
                        }
                    };
                    let src = |opsel: [bool; 3]| {
                        opsel
                            .iter()
                            .enumerate()
                            .map(|(i, sel)| {
                                if (14..=19).contains(&op) {
                                    let half = |x, n| f16::from_bits(x).negate(i, n).to_bits();
                                    match sel {
                                        true => half(src_parts[i].1, neg),
                                        false => half(src_parts[i].0, neg_hi),
                                    }
                                } else {
                                    match sel {
                                        true => src_parts[i].1,
                                        false => src_parts[i].0,
                                    }
                                }
                            })
                            .collect::<Vec<u16>>()
                    };
                    let (src_hi, src_lo) = (src(opsel_hi), src(opsel));
                    let ret = ((fxn(src_hi[0], src_hi[1], src_hi[2]) as u32) << 16) | (fxn(src_lo[0], src_lo[1], src_lo[2]) as u32);

                    if self.exec.read() {
                        self.vec_reg[vdst] = ret;
                    }
                }
                32..=34 => {
                    let src: Vec<f32> = src_parts
                        .iter()
                        .enumerate()
                        .map(|(i, (lo, hi, full))| {
                            if !opsel_hi[i] {
                                f32::from_bits(*full).absolute(i, neg_hi)
                            } else if opsel[i] {
                                f32::from(f16::from_bits(*hi)).absolute(i, neg_hi)
                            } else {
                                f32::from(f16::from_bits(*lo)).absolute(i, neg_hi)
                            }
                        })
                        .collect();
                    let ret = match op {
                        32 => f32::mul_add(src[0], src[1], src[2]).to_bits(),
                        33 | 34 => {
                            let ret = f16::from_f32(f32::mul_add(src[0], src[1], src[2])).to_bits();
                            match op {
                                33 => (self.vec_reg[vdst] & 0xffff0000) | (ret as u32),
                                34 => (self.vec_reg[vdst] & 0x0000ffff) | ((ret as u32) << 16),
                                _ => todo_instr!(instruction)?,
                            }
                        }
                        _ => todo_instr!(instruction)?,
                    };
                    if self.exec.read() {
                        self.vec_reg[vdst] = ret;
                    }
                }
                64..=69 => {
                    match op {
                        64 => {
                            let a = self.wmma_b16_16x16(s[0]).map(|v| f16::from_bits(v).to_f32());
                            let b = self.wmma_b16_16x16(s[1]).map(|v| f16::from_bits(v).to_f32());
                            let c = self.wmma_b32_16x16(s[2]).map(|v| f32::from_bits(v));
                            let ret = wmma(a.collect(), b.collect(), c.collect());
                            for (i, val) in ret.into_iter().enumerate() {
                                let lane = i % 32;
                                self.vec_reg.get_lane_mut(lane)[(i / 32) + vdst] = val.to_bits();
                            }
                        }
                        65 => {
                            let a = self.wmma_b16_16x16(s[0]).map(|v| bf16::from_bits(v).to_f32());
                            let b = self.wmma_b16_16x16(s[1]).map(|v| bf16::from_bits(v).to_f32());
                            let c = self.wmma_b32_16x16(s[2]).map(|v| f32::from_bits(v));
                            let ret = wmma(a.collect(), b.collect(), c.collect());
                            for (i, val) in ret.into_iter().enumerate() {
                                let register = (i / 32) + vdst;
                                let lane = i % 32;
                                self.vec_reg.get_lane_mut(lane)[register] = val.to_bits()
                            }
                        }
                        66 => {
                            let a = self.wmma_b16_16x16(s[0]).map(|v| f16::from_bits(v));
                            let b = self.wmma_b16_16x16(s[1]).map(|v| f16::from_bits(v));
                            let c = self.wmma_b32_16x16(s[2]).map(|v| f16::from_bits(v as u16));
                            let ret = wmma(a.collect(), b.collect(), c.collect());
                            for (i, val) in ret.into_iter().enumerate() {
                                let register = (i / 32) + vdst;
                                let lane = i % 32;
                                self.vec_reg.get_lane_mut(lane)[register].mut_lo16(val.to_bits());
                            }
                        }
                        _ => todo_instr!(instruction)?,
                    };
                    self.scalar = true;
                }
                _ => todo_instr!(instruction)?,
            }
        }
        else if let Instruction::VOP1 { src, op, vdst } = decoded {
            let s0 = src as usize;
            let vdst = vdst as usize;

            match op {
                3 | 15 | 21 | 23 | 25 | 26 | 60 | 61 | 47 | 49 => {
                    let s0: u64 = self.val(s0);
                    match op {
                        3 | 15 | 21 | 23 | 25 | 26 | 60 | 61 | 47 | 49 => {
                            let s0 = f64::from_bits(s0);
                            match op {
                                23 | 25 | 26 | 61 | 47 | 49 => {
                                    let ret = match op {
                                        23 => f64::trunc(s0),
                                        25 => {
                                            let mut temp = f64::floor(s0 + 0.5);
                                            if f64::floor(s0) % 2.0 != 0.0 && f64::fract(s0) == 0.5 {
                                                temp -= 1.0;
                                            }
                                            temp
                                        }
                                        26 => f64::floor(s0),
                                        47 => 1.0 / s0,
                                        49 => 1.0 / f64::sqrt(s0),
                                        61 => extract_mantissa(s0),
                                        _ => todo_instr!(instruction)?,
                                    };
                                    if self.exec.read() {
                                        self.vec_reg.write64(vdst, ret.to_bits())
                                    }
                                }
                                _ => {
                                    let ret = match op {
                                        3 => s0 as i32 as u32,
                                        15 => (s0 as f32).to_bits(),
                                        21 => s0 as u32,
                                        60 => match (s0 == f64::INFINITY) || (s0 == f64::NEG_INFINITY) || s0.is_nan() {
                                            true => 0,
                                            false => (s0.exponent() as i32 - 1023 + 1) as u32,
                                        },
                                        _ => todo_instr!(instruction)?,
                                    };
                                    if self.exec.read() {
                                        self.vec_reg[vdst] = ret;
                                    }
                                }
                            }
                        }
                        _ => todo_instr!(instruction)?,
                    }
                }
                84..=97 => {
                    let s0 = f16::from_bits(self.val(s0));
                    let ret = match op {
                        84 => f16::recip(s0),
                        85 => f16::sqrt(s0),
                        87 => f16::log2(s0),
                        88 => f16::exp2(s0),
                        _ => todo_instr!(instruction)?,
                    };
                    if self.exec.read() {
                        self.vec_reg[vdst] = ret.to_bits() as u32;
                    }
                }
                _ => {
                    let s0: u32 = self.val(s0);
                    match op {
                        4 | 16 | 22 => {
                            let ret = match op {
                                4 => (s0 as i32 as f64).to_bits(),
                                22 => (s0 as f64).to_bits(),
                                16 => (f32::from_bits(s0) as f64).to_bits(),
                                _ => todo_instr!(instruction)?,
                            };
                            if self.exec.read() {
                                self.vec_reg.write64(vdst, ret)
                            }
                        }
                        2 => {
                            let idx = self.exec.value.trailing_zeros() as usize;
                            self.scalar_reg[vdst] = self.vec_reg.get_lane(idx)[(instruction & 0x1ff) as usize - VGPR_COUNT];
                        }
                        _ => {
                            let ret = match op {
                                1 => s0,
                                5 => (s0 as i32 as f32).to_bits(),
                                6 => (s0 as f32).to_bits(),
                                7 => f32::from_bits(s0) as u32,
                                8 => f32::from_bits(s0) as i32 as u32,
                                10 => f16::from_f32(f32::from_bits(s0)).to_bits() as u32,
                                11 => f32::from(f16::from_bits(s0 as u16)).to_bits(),
                                17 => ((s0 & 0xff) as f32).to_bits(),
                                18 => (((s0 >> 8) & 0xff) as f32).to_bits(),
                                19 => (((s0 >> 16) & 0xff) as f32).to_bits(),
                                20 => (((s0 >> 24) & 0xff) as f32).to_bits(),
                                56 => s0.reverse_bits(),
                                57 => self.clz_i32_u32(s0),
                                33..=51 => {
                                    let s0 = f32::from_bits(s0);
                                    match op {
                                        33 => s0.trunc(),
                                        34 => {
                                            let mut d0 = s0.trunc();
                                            if s0 > 0.0 && s0 != d0 {
                                                d0 += 1.0;
                                            }
                                            d0
                                        }
                                        35 => {
                                            let mut temp = f32::floor(s0 + 0.5);
                                            if f32::floor(s0) % 2.0 != 0.0 && f32::fract(s0) == 0.5 {
                                                temp -= 1.0;
                                            }
                                            temp
                                        }
                                        36 => {
                                            let mut d0 = s0.trunc();
                                            if s0 < 0.0 && s0 != d0 {
                                                d0 -= 1.0;
                                            }
                                            d0
                                        }
                                        37 => f32::exp2(s0),
                                        39 => f32::log2(s0),
                                        42 => 1.0 / s0,
                                        43 => 1.0 / s0,
                                        51 => f32::sqrt(s0),
                                        _ => todo_instr!(instruction)?,
                                    }
                                    .to_bits()
                                }
                                55 => !s0,
                                59 => self.cls_i32(s0),
                                80 => f16::from_f32(s0 as u16 as f32).to_bits() as u32,
                                81 => f16::from_f32(s0 as i16 as f32).to_bits() as u32,
                                82 => f32::from(f16::from_bits(s0 as u16)) as u32,
                                83 => f32::from(f16::from_bits(s0 as u16)) as i16 as u32,
                                _ => todo_instr!(instruction)?,
                            };
                            if self.exec.read() {
                                self.vec_reg[vdst] = ret;
                            }
                        }
                    }
                }
            }
        }
        // vopd
        else if instruction >> 26 == 0b110010 {
            let instr = self.u64_instr();
            let sx = instr & 0x1ff;
            let vx = (instr >> 9) & 0xff;
            let srcx0 = self.val(sx as usize);
            let vsrcx1 = self.vec_reg[(vx) as usize] as u32;
            let opy = (instr >> 17) & 0x1f;
            let sy = (instr >> 32) & 0x1ff;
            let vy = (instr >> 41) & 0xff;
            let opx = (instr >> 22) & 0xf;
            let srcy0 = match sy {
                255 => match sx {
                    255 => srcx0,
                    _ => self.val(sy as usize),
                },
                _ => self.val(sy as usize),
            };
            let vsrcy1 = self.vec_reg[(vy) as usize];

            let vdstx = ((instr >> 56) & 0xff) as usize;
            // LSB is the opposite of VDSTX[0]
            let vdsty = (((instr >> 49) & 0x7f) << 1 | ((vdstx as u64 & 1) ^ 1)) as usize;

            for (op, s0, s1, dst) in ([(opx, srcx0, vsrcx1, vdstx), (opy, srcy0, vsrcy1, vdsty)]).iter() {
                let ret = match *op {
                    0 | 1 | 2 | 3 | 4 | 5 | 6 | 10 | 11 => {
                        let s0 = f32::from_bits(*s0 as u32);
                        let s1 = f32::from_bits(*s1 as u32);
                        match *op {
                            0 => f32::mul_add(s0, s1, f32::from_bits(self.vec_reg[*dst])),
                            1 => f32::mul_add(s0, s1, f32::from_bits(self.val(SIMM_SRC))),
                            2 => f32::mul_add(s0, f32::from_bits(self.val(SIMM_SRC)), s1),
                            3 => s0 * s1,
                            4 => s0 + s1,
                            5 => s0 - s1,
                            6 => s1 - s0,
                            10 => f32::max(s0, s1),
                            11 => f32::min(s0, s1),
                            _ => todo_instr!(instruction)?,
                        }
                        .to_bits()
                    }
                    8 => *s0,
                    9 => match self.vcc.read() {
                        true => *s1,
                        false => *s0,
                    },
                    16 => s0 + s1,
                    17 => s1 << s0,
                    18 => s0 & s1,
                    _ => todo_instr!(instruction)?,
                };
                if self.exec.read() {
                    self.vec_reg[*dst] = ret;
                };
            }
        }
        // vopc
        else if let Instruction::VOPC { vsrc, src, op } = decoded {
            let s0 = src as usize;
            let s1 = vsrc as usize;
            let op = op as u32;

            let dest_offset = if op >= 128 { 128 } else { 0 };
            let ret = match op {
                (0..=15) | 125 | (128..=143) => {
                    let s0 = f16::from_bits(self.val(s0));
                    let s1 = f16::from_bits(self.vec_reg[s1] as u16);
                    match op {
                        125 => self.cmp_class_f16(s0, s1.to_bits()),
                        _ => self.cmpf(s0, s1, op - dest_offset),
                    }
                }
                (16..=31) | 126 | (144..=159) => {
                    let s0 = f32::from_bits(self.val(s0));
                    let s1 = f32::from_bits(self.vec_reg[s1]);
                    match op {
                        126 => self.cmp_class_f32(s0, s1.to_bits()),
                        _ => self.cmpf(s0, s1, op - 16 - dest_offset),
                    }
                }
                (32..=47) | 127 | (160..=174) => {
                    let s0 = self.val(s0);
                    match op {
                        127 => {
                            let s1 = self.val(s1);
                            self.cmp_class_f64(s0, s1)
                        }
                        _ => {
                            let s1 = f64::from_bits(self.vec_reg.read64(s1));
                            self.cmpf(s0, s1, op - 32 - dest_offset)
                        }
                    }
                }
                (49..=54) | (177..=182) => {
                    let (s0, s1): (u16, u16) = (self.val(s0), self.vec_reg[s1] as u16);
                    self.cmpi(s0 as i16, s1 as i16, op - 48 - dest_offset)
                }
                (57..=62) | (185..=190) => {
                    let (s0, s1): (u16, u16) = (self.val(s0), self.vec_reg[s1] as u16);
                    self.cmpi(s0, s1, op - 56 - dest_offset)
                }
                (64..=71) | (192..=199) => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.vec_reg[s1]);
                    self.cmpi(s0 as i32, s1 as i32, op - 64 - dest_offset)
                }
                (72..=79) | (200..=207) => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.vec_reg[s1]);
                    self.cmpi(s0, s1, op - 72 - dest_offset)
                }
                (80..=87) | (208..=215) => {
                    let (s0, s1): (u64, u64) = (self.val(s0), self.vec_reg.read64(s1));
                    self.cmpi(s0 as i64, s1 as i64, op - 80 - dest_offset)
                }
                (88..=95) | (216..=223) => {
                    let (s0, s1): (u64, u64) = (self.val(s0), self.vec_reg.read64(s1));
                    self.cmpi(s0, s1, op - 88 - dest_offset)
                }
                _ => todo_instr!(instruction)?,
            };

            if self.exec.read() {
                match op >= 128 {
                    true => self.exec.set_lane(ret),
                    false => self.vcc.set_lane(ret),
                };
            }
        }
        else if let Instruction::VOP2 { vsrc, src, vdst, op } = decoded {
            let s0 = src as usize;
            let s1 = self.vec_reg[vsrc as usize];
            let vdst = vdst as usize;

            match op {
                (50..=60) => {
                    let (s0, s1) = (f16::from_bits(self.val(s0)), f16::from_bits(s1 as u16));
                    let ret = match op {
                        50 => s0 + s1,
                        51 => s0 - s1,
                        53 => s0 * s1,
                        54 => f16::mul_add(s0, s1, f16::from_bits(self.vec_reg[vdst] as u16)),
                        55 => f16::mul_add(s0, f16::from_bits(self.val(SIMM_SRC)), s1),
                        56 => f16::mul_add(s0, s1, f16::from_bits(self.val(SIMM_SRC))),
                        57 => f16::max(s0, s1),
                        58 => f16::min(s0, s1),
                        _ => todo_instr!(instruction)?,
                    };
                    if self.exec.read() {
                        self.vec_reg[vdst] = ret.to_bits() as u32;
                    }
                }
                _ => {
                    let s0 = self.val(s0);
                    let ret = match op {
                        1 => match self.vcc.read() {
                            true => s1,
                            false => s0,
                        },
                        2 => {
                            let mut acc = f32::from_bits(self.vec_reg[vdst]);
                            acc += f32::from(f16_lo(s0)) * f32::from(f16_lo(s1));
                            acc += f32::from(f16_hi(s0)) * f32::from(f16_hi(s1));
                            acc.to_bits()
                        }

                        3 | 4 | 5 | 8 | 15 | 16 | 43 | 44 | 45 => {
                            let (s0, s1) = (f32::from_bits(s0), f32::from_bits(s1));
                            match op {
                                3 => s0 + s1,
                                4 => s0 - s1,
                                5 => s1 - s0,
                                8 => s0 * s1,
                                15 => f32::min(s0, s1),
                                16 => f32::max(s0, s1),
                                43 => f32::mul_add(s0, s1, f32::from_bits(self.vec_reg[vdst])),
                                44 => f32::mul_add(s0, f32::from_bits(self.val(SIMM_SRC)), s1),
                                45 => f32::mul_add(s0, s1, f32::from_bits(self.val(SIMM_SRC))),
                                _ => todo_instr!(instruction)?,
                            }
                            .to_bits()
                        }
                        9 => {
                            let s0 = sign_ext((s0 & 0xffffff) as u64, 24) as i32;
                            let s1 = sign_ext((s1 & 0xffffff) as u64, 24) as i32;
                            (s0 * s1) as u32
                        }
                        17 | 18 | 26 => {
                            let (s0, s1) = (s0 as i32, s1 as i32);
                            (match op {
                                17 => i32::min(s0, s1),
                                18 => i32::max(s0, s1),
                                26 => s1 >> s0,
                                _ => todo_instr!(instruction)?,
                            }) as u32
                        }
                        32 => {
                            let temp = s0 as u64 + s1 as u64 + self.vcc.read() as u64;
                            self.vcc.set_lane(temp >= 0x100000000);
                            temp as u32
                        }
                        33 | 34 => {
                            let temp = match op {
                                33 => s0 - s1 - self.vcc.read() as u32,
                                34 => s1 - s0 - self.vcc.read() as u32,
                                _ => todo_instr!(instruction)?,
                            };
                            self.vcc.set_lane((s1 as u64 + self.vcc.read() as u64) > s0 as u64);
                            temp
                        }
                        11 => s0 * s1,
                        19 => u32::min(s0, s1),
                        20 => u32::max(s0, s1),
                        24 => s1 << s0,
                        25 => s1 >> s0,
                        27 => s0 & s1,
                        28 => s0 | s1,
                        29 => s0 ^ s1,
                        37 => s0 + s1,
                        38 => s0 - s1,
                        39 => s1 - s0,
                        _ => todo_instr!(instruction)?,
                    };
                    if self.exec.read() {
                        self.vec_reg[vdst] = ret;
                    }
                }
            };
        }
        // vop3
        else if instruction >> 26 == 0b110101 {
            let instr = self.u64_instr();

            let op = ((instr >> 16) & 0x3ff) as u32;
            match op {
                764 | 765 | 288 | 289 | 290 | 766 | 768 | 769 => {
                    let vdst = (instr & 0xff) as usize;
                    let sdst = ((instr >> 8) & 0x7f) as usize;
                    let f = |i: u32| -> usize { ((instr >> i) & 0x1ff) as usize };
                    let (s0, s1, s2) = (f(32), f(41), f(50));
                    let mut carry_in = WaveValue::new(self.val(s2), self.warp_size);
                    carry_in.default_lane = self.vcc.default_lane;
                    let omod = (instr >> 59) & 0x3;
                    let _neg = (instr >> 61) & 0x7;
                    let clmp = (instr >> 15) & 0x1;
                    assert_eq!(omod, 0);
                    assert_eq!(clmp, 0);

                    let vcc = match op {
                        766 => {
                            let (s0, s1, s2): (u32, u32, u64) = (self.val(s0), self.val(s1), self.val(s2));
                            let (mul_result, overflow_mul) = (s0 as u64).overflowing_mul(s1 as u64);
                            let (ret, overflow_add) = mul_result.overflowing_add(s2);
                            let overflowed = overflow_mul || overflow_add;
                            if self.exec.read() {
                                self.vec_reg.write64(vdst, ret);
                            }
                            overflowed
                        }
                        765 => {
                            assert!(f64::from_bits(self.val(s2)).exponent() <= 1076);
                            let ret = ldexp(self.val(s0), 128);
                            if self.exec.read() {
                                self.vec_reg.write64(vdst, ret.to_bits());
                            }
                            false
                        }
                        _ => {
                            let (s0, s1, _s2): (u32, u32, u32) = (self.val(s0), self.val(s1), self.val(s2));
                            let (ret, vcc) = match op {
                                288 => {
                                    let ret = s0 as u64 + s1 as u64 + carry_in.read() as u64;
                                    (ret as u32, ret >= 0x100000000)
                                }
                                289 => {
                                    let ret = (s0 as u64).wrapping_sub(s1 as u64).wrapping_sub(carry_in.read() as u64);
                                    (ret as u32, s1 as u64 + (carry_in.read() as u64) > s0 as u64)
                                }
                                290 => {
                                    let ret = (s1 as u64).wrapping_sub(s0 as u64).wrapping_sub(carry_in.read() as u64);
                                    (ret as u32, s1 as u64 + (carry_in.read() as u64) > s0 as u64)
                                }
                                764 => (0, false), // NOTE: div scaling isn't required
                                768 => {
                                    let ret = s0 as u64 + s1 as u64;
                                    (ret as u32, ret >= 0x100000000)
                                }
                                769 => {
                                    let ret = s0.wrapping_sub(s1);
                                    (ret as u32, s1 > s0)
                                }
                                _ => todo_instr!(instruction)?,
                            };
                            if self.exec.read() {
                                self.vec_reg[vdst] = ret;
                            }
                            vcc
                        }
                    };

                    match sdst {
                        VCC => self.vcc.set_lane(vcc),
                        NULL_SRC => {}
                        _ => self.set_sgpr_co(sdst, vcc),
                    }
                }
                _ => {
                    let vdst = (instr & 0xff) as usize;
                    let abs = ((instr >> 8) & 0x7) as usize;
                    let opsel = ((instr >> 11) & 0xf) as usize;
                    let cm = (instr >> 15) & 0x1;

                    let s = |n: usize| ((instr >> n) & 0x1ff) as usize;
                    let src = (s(32), s(41), s(50));

                    let omod = (instr >> 59) & 0x3;
                    let neg = ((instr >> 61) & 0x7) as usize;
                    assert_eq!(omod, 0);
                    assert_eq!(cm, 0);
                    assert_eq!(opsel, 0);

                    match op {
                        // VOPC using VOP3 encoding
                        0..=255 => {
                            let dest_offset = if op >= 128 { 128 } else { 0 };
                            let ret = match op {
                                (0..=15) | 125 | (128..=143) => {
                                    let (s0, s1) = (self.val(src.0), self.val(src.1));
                                    let s0 = f16::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f16::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    match op {
                                        125 => self.cmp_class_f16(s0, s1.to_bits()),
                                        _ => self.cmpf(s0, s1, op - dest_offset),
                                    }
                                }
                                (16..=31) | 126 | (144..=159) => {
                                    let (s0, s1) = (self.val(src.0), self.val(src.1));
                                    let s0 = f32::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f32::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    match op {
                                        126 => self.cmp_class_f32(s0, s1.to_bits()),
                                        _ => self.cmpf(s0, s1, op - 16 - dest_offset),
                                    }
                                }
                                (32..=47) | 127 | (160..=174) => {
                                    let s0: f64 = self.val(src.0);
                                    let s0 = s0.negate(0, neg).absolute(0, abs);
                                    match op {
                                        127 => {
                                            let s1 = self.val(src.1);
                                            self.cmp_class_f64(s0, s1)
                                        }
                                        _ => {
                                            let s1 = self.val(src.1);
                                            let s1 = f64::from_bits(s1).negate(1, neg).absolute(1, abs);
                                            self.cmpf(s0, s1, op - 32 - dest_offset)
                                        }
                                    }
                                }
                                (49..=54) | (177..=182) => {
                                    let (s0, s1): (u16, u16) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0 as i16, s1 as i16, op - 48 - dest_offset)
                                }
                                (57..=62) | (185..=190) => {
                                    let (s0, s1): (u16, u16) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0, s1, op - 56 - dest_offset)
                                }
                                (64..=71) | (192..=199) => {
                                    let (s0, s1): (u32, u32) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0 as i32, s1 as i32, op - 64 - dest_offset)
                                }
                                (72..=79) | (200..=207) => {
                                    let (s0, s1): (u32, u32) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0, s1, op - 72 - dest_offset)
                                }
                                (80..=87) | (208..=215) => {
                                    let (s0, s1): (u64, u64) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0 as i64, s1 as i64, op - 80 - dest_offset)
                                }
                                (88..=95) | (216..=223) => {
                                    let (s0, s1): (u64, u64) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0, s1, op - 88 - dest_offset)
                                }
                                _ => todo_instr!(instruction)?,
                            };

                            if self.exec.read() {
                                match vdst {
                                    0..=SGPR_SRC | 107 => self.set_sgpr_co(vdst, ret),
                                    VCC => self.vcc.set_lane(ret),
                                    EXEC => self.exec.set_lane(ret),
                                    _ => todo_instr!(instruction)?,
                                }
                            }
                        }
                        828..=830 => {
                            let (s0, s1, _s2): (u32, u64, u64) = (self.val(src.0), self.val(src.1), self.val(src.2));
                            let shift = s0 & 0x3f;
                            let ret = match op {
                                828 => s1 << shift,
                                829 => s1 >> shift,
                                830 => ((s1 as i64) >> shift) as u64,
                                _ => todo_instr!(instruction)?,
                            };
                            if self.exec.read() {
                                self.vec_reg.write64(vdst, ret)
                            }
                        }
                        407 | 532 | 552 | 568 | (807..=811) => {
                            let (_s0, _s1, _s2): (f64, f64, f64) = (self.val(src.0), self.val(src.1), self.val(src.2));
                            let s0 = _s0.negate(0, neg).absolute(0, abs);
                            let s1 = _s1.negate(1, neg).absolute(1, abs);
                            let s2 = _s2.negate(2, neg).absolute(2, abs);
                            let ret = match op {
                                407 => f64::trunc(s0),
                                532 => f64::mul_add(s0, s1, s2),
                                552 => {
                                    assert!(s0.is_normal());
                                    s0
                                }
                                807 => s0 + s1,
                                808 => s0 * s1,
                                809 => f64::min(s0, s1),
                                810 => f64::max(s0, s1),
                                811 => {
                                    let s1: u32 = self.val(src.1);
                                    s0 * 2f64.powi(s1 as i32)
                                }
                                568 => {
                                    assert!(!self.vcc.read());
                                    f64::mul_add(s0, s1, s2)
                                }
                                _ => todo_instr!(instruction)?,
                            }
                            .to_bits();
                            if self.exec.read() {
                                self.vec_reg.write64(vdst, ret)
                            }
                        }
                        306 | 309 | 310 | 313 | 596 | 584 | 585 | 588 => {
                            let (s0, s1, s2) = (self.val(src.0), self.val(src.1), self.val(src.2));
                            let s0 = f16::from_bits(s0).negate(0, neg).absolute(0, abs);
                            let s1 = f16::from_bits(s1).negate(1, neg).absolute(1, abs);
                            let s2 = f16::from_bits(s2).negate(1, neg).absolute(1, abs);
                            let ret = match op {
                                309 => s0 * s1,
                                310 => f16::mul_add(s0, s1, f16::from_bits(self.vec_reg[vdst] as u16)),
                                306 => s0 + s1,
                                584 => f16::mul_add(s0, s1, s2),
                                585 => f16::min(f16::min(s0, s1), s2),
                                588 => f16::max(f16::max(s0, s1), s2),
                                596 => s2 / s1,
                                313 => f16::max(s0, s1),
                                314 => f16::min(s0, s1),
                                _ => todo_instr!(instruction)?,
                            }
                            .to_bits();
                            if self.exec.read() {
                                self.vec_reg[vdst] = ret as u32;
                            }
                        }
                        394 => {
                            let s0 = f32::from_bits(self.val(src.0)).negate(0, neg).absolute(0, abs);
                            if self.exec.read() {
                                self.vec_reg[vdst].mut_lo16(f16::from_f32(s0).to_bits());
                            }
                        }
                        467 => {
                            let s0 = f16::from_bits(self.val(src.0)).negate(0, neg).absolute(0, abs);
                            if self.exec.read() {
                                self.vec_reg[vdst] = s0.to_f32() as i16 as u32;
                            }
                        }
                        395 => {
                            let s0 = f16::from_bits(self.val(src.0)).negate(0, neg).absolute(0, abs);
                            if self.exec.read() {
                                self.vec_reg[vdst] = f32::from(s0).to_bits();
                            }
                        }
                        399 => {
                            let s0: f64 = self.val(src.0);
                            let s0 = s0.negate(0, neg).absolute(0, abs);
                            if self.exec.read() {
                                self.vec_reg[vdst] = (s0 as f32).to_bits();
                            }
                        }
                        785 => {
                            let (s0, s1) = (self.val(src.0), self.val(src.1));
                            if self.exec.read() {
                                self.vec_reg[vdst] = (f16::from_bits(s1).to_bits() as u32) << 16 | f16::from_bits(s0).to_bits() as u32;
                            }
                        }
                        _ => {
                            let (s0, s1, s2) = (self.val(src.0), self.val(src.1), self.val(src.2));
                            match op {
                                865 => {
                                    if self.exec.read() {
                                        self.vec_reg.get_lane_mut(s1 as usize)[vdst] = s0;
                                    }
                                    return Ok(());
                                }
                                864 => {
                                    let val = self.vec_reg.get_lane(s1 as usize)[src.0 - VGPR_COUNT];
                                    self.write_to_sdst(vdst, val);
                                    return Ok(());
                                }
                                826 => {
                                    if self.exec.read() {
                                        self.vec_reg[vdst].mut_lo16(((s1 as i16) >> (s0 & 0xf)) as u16);
                                    }
                                    return Ok(());
                                }
                                587 | 577 | 590 | 771 | 772 | 773 | 777 | 779 | 824 | 825 => {
                                    let (s0, s1, s2) = (s0 as u16, s1 as u16, s2 as u16);
                                    let ret = match op {
                                        587 => u16::min(u16::min(s0, s1), s2),
                                        590 => u16::max(u16::max(s0, s1), s2),
                                        577 => s0 * s1 + s2,
                                        771 => s0 + s1,
                                        772 => s0 - s1,
                                        773 => s0 * s1,
                                        777 => u16::max(s0, s1),
                                        779 => u16::min(s0, s1),
                                        824 => s1 << s0,
                                        825 => s1 >> s0,
                                        _ => todo_instr!(instruction)?,
                                    };
                                    if self.exec.read() {
                                        self.vec_reg[vdst].mut_lo16(ret);
                                    }
                                    return Ok(());
                                }
                                586 | 589 | 778 | 780 | 781 | 782 => {
                                    let (s0, s1, s2) = (s0 as i16, s1 as i16, s2 as i16);
                                    let ret = match op {
                                        586 => i16::min(i16::min(s0, s1), s2),
                                        589 => i16::max(i16::max(s0, s1), s2),
                                        778 => i16::max(s0, s1),
                                        780 => i16::min(s0, s1),
                                        781 => s0 + s1,
                                        782 => s0 - s1,
                                        _ => todo_instr!(instruction)?,
                                    };
                                    if self.exec.read() {
                                        self.vec_reg[vdst].mut_lo16(ret as u16);
                                    }
                                    return Ok(());
                                }
                                _ => {}
                            }

                            let ret = match op {
                                257 | 259 | 299 | 260 | 261 | 264 | 272 | 392 | 426 | 531 | 537 | 540 | 551 | 567 | 796 => {
                                    let s0 = f32::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f32::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    let s2 = f32::from_bits(s2).negate(2, neg).absolute(2, abs);
                                    match op {
                                        259 => s0 + s1,
                                        260 => s0 - s1,
                                        261 => s1 - s0,
                                        264 => s0 * s1,
                                        272 => f32::max(s0, s1),
                                        299 => f32::mul_add(s0, s1, f32::from_bits(self.vec_reg[vdst])),
                                        426 => s0.recip(),
                                        531 => f32::mul_add(s0, s1, s2),
                                        537 => f32::min(f32::min(s0, s1), s2),
                                        540 => f32::max(f32::max(s0, s1), s2),
                                        551 => s2 / s1,
                                        567 => {
                                            let ret = f32::mul_add(s0, s1, s2);
                                            match self.vcc.read() {
                                                true => 2.0_f32.powi(32) * ret,
                                                false => ret,
                                            }
                                        }
                                        796 => s0 * 2f32.powi(s1.to_bits() as i32),
                                        // cnd_mask isn't a float only ALU but supports neg
                                        257 => {
                                            let mut cond = WaveValue::new(s2.to_bits(), self.warp_size);
                                            cond.default_lane = self.vcc.default_lane;
                                            match cond.read() {
                                                true => s1,
                                                false => s0,
                                            }
                                        }
                                        392 => f32::from_bits(s0 as i32 as u32),
                                        _ => todo_instr!(instruction)?,
                                    }
                                    .to_bits()
                                }
                                _ => {
                                    if neg != 0 {
                                        todo_instr!(instruction)?
                                    }
                                    match op {
                                        529 => {
                                            let s0 = s0 as i32;
                                            let shift = 32 - (s2 & 0x1f);
                                            let mask: i32 = 1 << (s2 & 0x1f);
                                            let ret = (s0 >> (s1 & 0x1f)) & (mask.wrapping_sub(1));
                                            ((ret << shift) >> shift) as u32
                                        }
                                        522 | 541 | 538 | 544 | 814 => {
                                            let (s0, s1, s2) = (s0 as i32, s1 as i32, s2 as i32);

                                            (match op {
                                                522 => {
                                                    let s0 = sign_ext((s0 & 0xffffff) as u64, 24) as i32;
                                                    let s1 = sign_ext((s1 & 0xffffff) as u64, 24) as i32;
                                                    s0 * s1 + s2
                                                }
                                                538 => i32::min(i32::min(s0, s1), s2),
                                                541 => i32::max(i32::max(s0, s1), s2),
                                                544 => {
                                                    if (i32::max(i32::max(s0, s1), s2)) == s0 {
                                                        i32::max(s1, s2)
                                                    } else if (i32::max(i32::max(s0, s1), s2)) == s1 {
                                                        i32::max(s0, s2)
                                                    } else {
                                                        i32::max(s0, s1)
                                                    }
                                                }
                                                814 => ((s0 as i64) * (s1 as i64) >> 32) as i32,
                                                _ => todo_instr!(instruction)?,
                                            }) as u32
                                        }
                                        275 => u32::min(s0, s1),
                                        276 => u32::max(s0, s1),
                                        280 => s1 << s0,
                                        281 => s1 >> s0,
                                        283 => s0 & s1,
                                        284 => s0 | s1,
                                        285 => s0 ^ s1,
                                        286 => !(s0 ^ s1),
                                        523 => s0 * s1 + s2, // TODO 24 bit trunc
                                        528 => (s0 >> s1) & ((1 << s2) - 1),
                                        530 => (s0 & s1) | (!s0 & s2),
                                        534 => {
                                            let val = ((s0 as u64) << 32) | (s1 as u64);
                                            let shift = (s2 & 0x1F) as u64;
                                            ((val >> shift) & 0xffffffff) as u32
                                        }
                                        542 => u32::max(u32::max(s0, s1), s2),
                                        576 => s0 ^ s1 ^ s2,
                                        580 => {
                                            fn byte_permute(data: u64, sel: u32) -> u8 {
                                                let bytes = data.to_ne_bytes();
                                                match sel {
                                                    13..=u32::MAX => 0xff,
                                                    12 => 0x00,
                                                    11 => ((bytes[7] & 0x80) != 0) as u8 * 0xff,
                                                    10 => ((bytes[5] & 0x80) != 0) as u8 * 0xff,
                                                    9 => ((bytes[3] & 0x80) != 0) as u8 * 0xff,
                                                    8 => ((bytes[1] & 0x80) != 0) as u8 * 0xff,
                                                    _ => bytes[sel as usize],
                                                }
                                            }
                                            let combined = ((s0 as u64) << 32) | s1 as u64;
                                            let d0 = ((byte_permute(combined, s2 >> 24) as u32) << 24)
                                                | ((byte_permute(combined, (s2 >> 16) & 0xFF) as u32) << 16)
                                                | ((byte_permute(combined, (s2 >> 8) & 0xFF) as u32) << 8)
                                                | (byte_permute(combined, s2 & 0xFF) as u32);
                                            d0
                                        }
                                        581 => (s0 ^ s1) + s2,
                                        582 => (s0 << s1) + s2,
                                        583 => (s0 + s1) << s2,
                                        597 => s0 + s1 + s2,
                                        598 => (s0 << s1) | s2,
                                        599 => (s0 & s1) | s2,
                                        600 => s0 | s1 | s2,
                                        798 => {
                                            let mut ret = s1;
                                            (0..=31).into_iter().for_each(|i| ret += nth(s0, i));
                                            ret
                                        }
                                        812 => s0 * s1,
                                        813 => ((s0 as u64) * (s1 as u64) >> 32) as u32,
                                        _ => todo_instr!(instruction)?,
                                    }
                                }
                            };
                            if self.exec.read() {
                                self.vec_reg[vdst] = ret;
                            }
                        }
                    };
                }
            }
        } else if let Instruction::DS { op, gds, addr, data0, offset0, data1, offset1, vdst } = decoded {
            let _ = self.u64_instr();
            if gds {
                return todo_instr!(instruction)?;
            }
            if !self.exec.read() {
                return Ok(());
            }

            let [data0, data1, vdst] = [data0 as usize, data1 as usize, vdst as usize];
            let lds_base = self.vec_reg[addr as usize];
            let single_addr = || (lds_base + u16::from_le_bytes([offset0, offset1]) as u32) as usize;
            let double_addr = |adj: u32| {
                let addr0 = lds_base + offset0 as u32 * adj;
                let addr1 = lds_base + offset1 as u32 * adj;
                (addr0 as usize, addr1 as usize)
            };

            match op {
                // load
                54 | 118 | 254 | 255 => {
                    let dwords = match op {
                        255 => 4,
                        254 => 3,
                        118 => 2,
                        _ => 1,
                    };
                    (0..dwords).for_each(|i| {
                        self.vec_reg[vdst + i] = self.lds.read(single_addr() + 4 * i);
                    });
                }
                58 => self.vec_reg[vdst] = self.lds.read(single_addr()) as u8 as u32,
                60 => self.vec_reg[vdst] = self.lds.read(single_addr()) as u16 as u32,
                55 => {
                    let (addr0, addr1) = double_addr(4);
                    self.vec_reg[vdst] = self.lds.read(addr0);
                    self.vec_reg[vdst + 1] = self.lds.read(addr1);
                }
                119 => {
                    let (addr0, addr1) = double_addr(8);
                    self.vec_reg.write64(vdst, self.lds.read64(addr0));
                    self.vec_reg.write64(vdst + 2, self.lds.read64(addr1));
                }
                // store
                13 | 77 | 222 | 223 => {
                    let dwords = match op {
                        223 => 4,
                        222 => 3,
                        77 => 2,
                        _ => 1,
                    };
                    (0..dwords).for_each(|i| {
                        self.lds.write(single_addr() + 4 * i, self.vec_reg[data0 + i]);
                    })
                }
                30 => {
                    let addr = single_addr();
                    if addr + 1 >= self.lds.data.len() {
                        self.lds.data.resize(self.lds.data.len() + addr + 2, 0);
                    }
                    self.lds.data[addr..addr + 1].iter_mut().enumerate().for_each(|(i, x)| {
                        *x = (self.vec_reg[data0] as u8).to_le_bytes()[i];
                    });
                }
                31 | 161 => {
                    let addr = single_addr();
                    if addr + 2 >= self.lds.data.len() {
                        self.lds.data.resize(self.lds.data.len() + addr + 3, 0);
                    }
                    let b32 = self.vec_reg[data0];
                    self.lds.data[addr..addr + 2].iter_mut().enumerate().for_each(|(i, x)| {
                        *x = (if op == 31 { b32 as u16 } else { ((b32 >> 16) & 0xffff) as u16 }).to_le_bytes()[i];
                    });
                }
                14 => {
                    let (addr0, addr1) = double_addr(4);
                    self.lds.write(addr0, self.vec_reg[data0]);
                    self.lds.write(addr1, self.vec_reg[data1]);
                }
                78 => {
                    let (addr0, addr1) = double_addr(8);
                    self.lds.write64(addr0, self.vec_reg.read64(data0));
                    self.lds.write64(addr1, self.vec_reg.read64(data1));
                }
                _ => todo_instr!(instruction)?,
            }
        }
        // global
        // flat
        else if instruction >> 26 == 0b110111 {
            let instr = self.u64_instr();
            if !self.exec.read() {
                return Ok(());
            }
            let offset = sign_ext(instr & 0x1fff, 13);
            let seg = (instr >> 16) & 0x3;
            let op = ((instr >> 18) & 0x7f) as usize;
            let addr = ((instr >> 32) & 0xff) as usize;
            let data = ((instr >> 40) & 0xff) as usize;
            let saddr = ((instr >> 48) & 0x7f) as usize;
            let vdst = ((instr >> 56) & 0xff) as usize;

            let saddr_val: u32 = self.val(saddr);
            let saddr_off = saddr_val == 0x7F || saddr == NULL_SRC;

            match seg {
                1 => {
                    let sve = ((instr >> 50) & 0x1) != 0;

                    let addr = match (sve, saddr_off) {
                        (true, true) => offset as u64 as usize,
                        (false, false) => saddr_val as usize,
                        _ => todo_instr!(instruction)?,
                    };
                    match op {
                        // load
                        20..=23 => (0..op - 19).for_each(|i| {
                            self.vec_reg[vdst + i] = self.sds.read(addr + 4 * i);
                        }),
                        // store
                        26..=29 => (0..op - 25).for_each(|i| {
                            self.sds.write(addr + 4 * i, self.vec_reg[data + i]);
                        }),
                        _ => todo_instr!(instruction)?,
                    }
                }
                2 => {
                    let addr = match saddr_off {
                        true => self.vec_reg.read64(addr) as i64 + (offset as i64),
                        false => {
                            let scalar_addr = self.scalar_reg.read64(saddr);
                            let vgpr_offset = self.vec_reg[addr];
                            scalar_addr as i64 + vgpr_offset as i64 + offset
                        }
                    } as u64;

                    unsafe {
                        match op {
                            // load
                            16 => self.vec_reg[vdst] = *(addr as *const u8) as u32,
                            17 => self.vec_reg[vdst] = *(addr as *const i8) as u32,
                            18 => self.vec_reg[vdst] = *(addr as *const u16) as u32,
                            19 => self.vec_reg[vdst] = *(addr as *const i16) as u32,

                            20..=23 => (0..op - 19).for_each(|i| {
                                self.vec_reg[vdst + i] = *((addr + 4 * i as u64) as *const u32);
                            }),
                            32 => self.vec_reg[vdst].mut_lo16(*(addr as *const u16)),
                            35 => self.vec_reg[vdst].mut_hi16(*(addr as *const u16)),
                            // store
                            24 => *(addr as *mut u8) = self.vec_reg[data] as u8,
                            25 => *(addr as *mut u16) = self.vec_reg[data] as u16,
                            26..=29 => (0..op - 25).for_each(|i| {
                                *((addr + 4 * i as u64) as u64 as *mut u32) = self.vec_reg[data + i];
                            }),
                            37 => *(addr as *mut u16) = ((self.vec_reg[data] >> 16) & 0xffff) as u16,
                            _ => todo_instr!(instruction)?,
                        };
                    }
                }
                _ => todo_instr!(instruction)?,
            };
        }
        // mubuf
        else if instruction >> 26 == 0b111000 {
            let instr = self.u64_instr();
            let op = ((instr >> 18) & 0x7f) as usize;
            match op {
                43 => {} // NOTE: remu doesn't have an l0 cache, it just has the software managed lds
                _ => todo_instr!(instruction)?,
            };
        } else {
            todo_instr!(instruction)?;
        }
        Ok(())
    }

    fn cmpf<T>(&self, s0: T, s1: T, offset: u32) -> bool
    where
        T: Float + std::fmt::Display,
    {
        return match offset {
            0 => true,
            1 => s0 < s1,
            2 => s0 == s1,
            3 => s0 <= s1,
            4 => s0 > s1,
            5 => s0 != s1,
            6 => s0 >= s1,
            7 => (!s0.is_nan()) && (!s1.is_nan()),
            8 => s0.is_nan() || s1.is_nan(),
            9 => !(s0 >= s1),
            10 => !(s0 != s1),
            11 => !(s0 > s1),
            12 => !(s0 <= s1),
            13 => !(s0 == s1),
            14 => !(s0 < s1),
            15 => true,
            _ => panic!("invalid offset for float compare {offset}"),
        };
    }
    fn cmp_class_f64(&self, s0: f64, s1: u32) -> bool {
        let offset = match s0 {
            _ if s0.is_nan() => 1,
            _ if s0.is_infinite() => match s0.signum() == -1.0 {
                true => 2,
                false => 9,
            },
            _ if s0.exponent() > 0 => match s0.signum() == -1.0 {
                true => 3,
                false => 8,
            },
            _ if s0.abs() > 0.0 => match s0.signum() == -1.0 {
                true => 4,
                false => 7,
            },
            _ => match s0.signum() == -1.0 {
                true => 5,
                false => 6,
            },
        };
        ((s1 >> offset) & 1) != 0
    }
    fn cmp_class_f32(&self, s0: f32, s1: u32) -> bool {
        let offset = match s0 {
            _ if (s0 as f64).is_nan() => 1,
            _ if s0.exponent() == 255 => match s0.signum() == -1.0 {
                true => 2,
                false => 9,
            },
            _ if s0.exponent() > 0 => match s0.signum() == -1.0 {
                true => 3,
                false => 8,
            },
            _ if s0.abs() as f64 > 0.0 => match s0.signum() == -1.0 {
                true => 4,
                false => 7,
            },
            _ => match s0.signum() == -1.0 {
                true => 5,
                false => 6,
            },
        };
        ((s1 >> offset) & 1) != 0
    }
    fn cmp_class_f16(&self, s0: f16, s1: u16) -> bool {
        let offset = match s0 {
            _ if (f64::from(s0)).is_nan() => 1,
            _ if s0.exponent() == 31 => match s0.signum() == f16::NEG_ONE {
                true => 2,
                false => 9,
            },
            _ if s0.exponent() > 0 => match s0.signum() == f16::NEG_ONE {
                true => 3,
                false => 8,
            },
            _ if f64::from(s0.abs()) > 0.0 => match s0.signum() == f16::NEG_ONE {
                true => 4,
                false => 7,
            },
            _ => match s0.signum() == f16::NEG_ONE {
                true => 5,
                false => 6,
            },
        };
        ((s1 >> offset) & 1) != 0
    }
    fn cmpi<T>(&self, s0: T, s1: T, offset: u32) -> bool
    where
        T: PartialOrd + PartialEq,
    {
        return match offset {
            0 => false,
            1 => s0 < s1,
            2 => s0 == s1,
            3 => s0 <= s1,
            4 => s0 > s1,
            5 => s0 != s1,
            6 => s0 >= s1,
            7 => true,
            _ => panic!("invalid offset for integer compare {offset}"),
        };
    }
    fn cls_i32(&self, s0: u32) -> u32 {
        let mut ret: i32 = -1;
        let s0 = s0 as i32;
        for i in (1..=31).into_iter() {
            if s0 >> (31 - i as u32) != s0 >> 31 {
                ret = i;
                break;
            }
        }
        ret as u32
    }
    fn clz_i32_u32(&self, s0: u32) -> u32 {
        let mut ret: i32 = -1;
        for i in (0..=31).into_iter() {
            if s0 >> (31 - i as u32) == 1 {
                ret = i;
                break;
            }
        }
        ret as u32
    }

    /* ALU utils */
    fn _common_srcs(&mut self, code: usize) -> u32 {
        match code {
            VCC => self.vcc.value,
            107 => self.scalar_reg[code as usize],
            EXEC => self.exec.value,
            NULL_SRC | 128 => 0,
            253 => *self.scc as u32,
            255 => match self.simm {
                None => {
                    let val = self.stream[self.pc_offset + 1];
                    self.simm = Some(val);
                    self.pc_offset += 1;
                    val
                }
                Some(val) => val,
            },
            _ => todo!("resolve_src={code}"),
        }
    }
    fn write_to_sdst(&mut self, sdst_bf: usize, val: u32) {
        match sdst_bf {
            // NOTE: remu is only wave32, vcc_hi is treated as a regular SGPR
            0..=SGPR_SRC | 107 => self.scalar_reg[sdst_bf] = val,
            VCC => self.vcc.value = val,
            126 => self.exec.value = val,
            _ => todo!("write to sdst {}", sdst_bf),
        }
    }
    fn set_sgpr_co(&mut self, idx: usize, val: bool) {
        let mut wv = self.sgpr_co.map(|(_, wv)| wv).unwrap_or_else(|| WaveValue::new(0, self.warp_size));
        wv.default_lane = self.vcc.default_lane;
        wv.set_lane(val);
        *self.sgpr_co = Some((idx, wv));
    }

    fn u64_instr(&mut self) -> u64 {
        let msb = self.stream[self.pc_offset + 1] as u64;
        let instr = msb << 32 | self.stream[self.pc_offset] as u64;
        self.pc_offset += 1;
        return instr;
    }

    fn wmma_b16_16x16(&'a self, vsrc: usize) -> impl Iterator<Item = u16> + 'a {
        (0..16).flat_map(move |i| {
            let lane = self.vec_reg.get_lane(i);
            (vsrc..=vsrc + 7).flat_map(move |j| {
                let val = lane[j - VGPR_COUNT];
                [(val & 0xffff) as u16, (val >> 16) as u16]
            })
        })
    }

    fn wmma_b32_16x16(&'a self, vsrc: usize) -> impl Iterator<Item = u32> + 'a {
        (0..256).map(move |i| self.vec_reg.get_lane(i % 32)[(i / 32) + vsrc - VGPR_COUNT])
    }
}

fn wmma<T: Float>(a: Vec<T>, b: Vec<T>, c: Vec<T>) -> [T; 256] {
    let mut ret = [T::zero(); 256];
    for row in 0..16 {
        for col in 0..16 {
            let mut sum = T::zero();
            for k in 0..16 {
                let a_val = a[row * 16 + k];
                let b_val = b[col * 16 + k];
                sum = sum + (a_val * b_val);
            }
            let c_val = c[row * 16 + col];
            ret[row * 16 + col] = sum + c_val;
        }
    }
    ret
}

pub trait ALUSrc<T> {
    fn val(&mut self, code: usize) -> T;
}
impl ALUSrc<u16> for Thread<'_> {
    fn val(&mut self, code: usize) -> u16 {
        match code {
            0..=SGPR_SRC => self.scalar_reg[code] as u16,
            VGPR_COUNT..=511 => self.vec_reg[code - VGPR_COUNT] as u16,
            129..=192 => (code - 128) as u16,
            193..=208 => ((code - 192) as i16 * -1) as u16,
            240..=247 => f16::from_f32(
                [
                    (240, 0.5_f32),
                    (241, -0.5_f32),
                    (242, 1_f32),
                    (243, -1.0_f32),
                    (244, 2.0_f32),
                    (245, -2.0_f32),
                    (246, 4.0_f32),
                    (247, -4.0_f32),
                ]
                .iter()
                .find(|x| x.0 == code)
                .unwrap()
                .1,
            )
            .to_bits(),
            _ => self._common_srcs(code) as u16,
        }
    }
}
impl ALUSrc<u32> for Thread<'_> {
    fn val(&mut self, code: usize) -> u32 {
        match code {
            0..=SGPR_SRC => self.scalar_reg[code],
            VGPR_COUNT..=511 => self.vec_reg[code - VGPR_COUNT],
            129..=192 => (code - 128) as u32,
            193..=208 => ((code - 192) as i32 * -1) as u32,
            240..=247 => [
                (240, 0.5_f32),
                (241, -0.5_f32),
                (242, 1_f32),
                (243, -1.0_f32),
                (244, 2.0_f32),
                (245, -2.0_f32),
                (246, 4.0_f32),
                (247, -4.0_f32),
            ]
            .iter()
            .find(|x| x.0 == code)
            .unwrap()
            .1
            .to_bits(),
            _ => self._common_srcs(code),
        }
    }
}
impl ALUSrc<u64> for Thread<'_> {
    fn val(&mut self, code: usize) -> u64 {
        match code {
            0..=SGPR_SRC => self.scalar_reg.read64(code),
            VGPR_COUNT..=511 => self.vec_reg.read64(code - VGPR_COUNT),
            129..=192 => (code - 128) as u64,
            193..=208 => ((code - 192) as i64 * -1) as u64,
            240..=247 => [
                (240, 0.5_f64),
                (241, -0.5_f64),
                (242, 1_f64),
                (243, -1.0_f64),
                (244, 2.0_f64),
                (245, -2.0_f64),
                (246, 4.0_f64),
                (247, -4.0_f64),
            ]
            .iter()
            .find(|x| x.0 == code)
            .unwrap()
            .1
            .to_bits(),
            _ => self._common_srcs(code) as u64,
        }
    }
}
impl ALUSrc<f64> for Thread<'_> {
    fn val(&mut self, code: usize) -> f64 {
        let uret: u64 = self.val(code);
        match code {
            SIMM_SRC => f64::from_bits(uret << 32),
            _ => f64::from_bits(uret),
        }
    }
}

#[cfg(test)]
mod test_alu_utils {
    use super::*;

    #[test]
    fn test_write_to_sdst_sgpr() {
        let mut thread = _helper_test_thread();
        thread.write_to_sdst(10, 200);
        assert_eq!(thread.scalar_reg[10], 200);
    }

    #[test]
    fn test_write_to_sdst_vcc_val() {
        let mut thread = _helper_test_thread();
        let val = 0b1011101011011011111011101111;
        thread.write_to_sdst(VCC, val);
        assert_eq!(thread.vcc.value, 195935983);
    }

    #[test]
    fn test_clz_i32_u32() {
        let thread = _helper_test_thread();
        assert_eq!(thread.clz_i32_u32(0x00000000), 0xffffffff);
        assert_eq!(thread.clz_i32_u32(0x0000cccc), 16);
        assert_eq!(thread.clz_i32_u32(0xffff3333), 0);
        assert_eq!(thread.clz_i32_u32(0x7fffffff), 1);
        assert_eq!(thread.clz_i32_u32(0x80000000), 0);
        assert_eq!(thread.clz_i32_u32(0xffffffff), 0);
    }

    #[test]
    fn test_cls_i32() {
        let thread = _helper_test_thread();
        assert_eq!(thread.cls_i32(0x00000000), 0xffffffff);
        assert_eq!(thread.cls_i32(0x0000cccc), 16);
        assert_eq!(thread.cls_i32(0xffff3333), 16);
        assert_eq!(thread.cls_i32(0x7fffffff), 1);
        assert_eq!(thread.cls_i32(0x80000000), 1);
    }

    #[test]
    fn test_sgpr_co_init() {
        let mut thread = _helper_test_thread();
        thread.vcc.default_lane = Some(0);
        thread.set_sgpr_co(10, true);
        thread.vcc.default_lane = Some(1);
        assert_eq!(thread.sgpr_co.unwrap().0, 10);
        assert_eq!(thread.sgpr_co.unwrap().1.mutations.unwrap()[0], true);
        thread.set_sgpr_co(10, true);
        assert_eq!(thread.sgpr_co.unwrap().0, 10);
        assert_eq!(thread.sgpr_co.unwrap().1.mutations.unwrap()[1], true);
        assert_eq!(thread.sgpr_co.unwrap().1.mutations.unwrap()[0], true);
    }
}

#[cfg(test)]
mod test_smem {
    use super::*;

    #[test]
    fn test_s_load_b32_simple() {
        let mut thread = _helper_test_thread();
        let mut buf = vec![0u8; 4];
        let a: u32 = 0xDEADBEEF;
        unsafe {
            *(buf.as_mut_ptr() as *mut u32) = a;
        }
        let base_addr = buf.as_ptr() as u64;
        thread.scalar_reg.write64(0, base_addr);
        r(&vec![0xF4000040, 0xF8000000, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[1], a);
        std::mem::forget(buf);
    }

    #[test]
    fn test_s_load_b32_vcc() {
        let mut thread = _helper_test_thread();
        let mut buf = vec![0u8; 4];
        let a: u32 = 0xDEADBEEF;
        unsafe {
            *(buf.as_mut_ptr() as *mut u32) = a;
        }
        let base_addr = buf.as_ptr() as u64;
        thread.scalar_reg.write64(0, base_addr);
        r(&vec![0xF4001A80, 0xF8000000, END_PRG], &mut thread);
        assert_eq!(thread.vcc.value, a);
        std::mem::forget(buf);
    }

    #[test]
    fn test_s_load_b32_vcc_addr() {
        let mut thread = _helper_test_thread();
        let mut buf = vec![0u8; 4];
        let a: u32 = 0xDEADBEEF;
        unsafe {
            *(buf.as_mut_ptr() as *mut u32) = a;
        }
        let addr = buf.as_ptr() as u64;
        // NOTE: vcc is an alias for s[106:107]
        thread.scalar_reg.write64(VCC, addr);
        // TODO: vcc_lo should just read from s106
        thread.vcc.value = (addr & 0xffffffff) as u32;
        r(&vec![0xF4000035, 0xF8000000, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[0], a);
        std::mem::forget(buf);
    }
}

#[cfg(test)]
mod test_sop1 {
    use super::*;

    #[test]
    fn test_s_brev_b32() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[5] = 8;
        r(&vec![0xBE850405, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[5], 268435456);
    }

    #[test]
    fn test_s_mov_b64() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg.write64(16, 5236523008);
        r(&vec![0xBE880110, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg.read64(8), 5236523008);
        assert_eq!(thread.scalar, true);
    }

    #[test]
    fn test_mov_exec() {
        let mut thread = _helper_test_thread();
        thread.exec.value = 0b11111111110111111110111111111111;
        r(&vec![0xBE80007E, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[0], 0b11111111110111111110111111111111);
    }

    #[test]
    fn test_s_mov_b32() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[15] = 42;
        r(&vec![0xbe82000f, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[2], 42);
    }

    #[test]
    fn test_s_bitset0_b32() {
        [
            [
                0b11111111111111111111111111111111,
                0b00000000000000000000000000000001,
                0b11111111111111111111111111111101,
            ],
            [
                0b11111111111111111111111111111111,
                0b00000000000000000000000000000010,
                0b11111111111111111111111111111011,
            ],
        ]
        .iter()
        .for_each(|[a, b, ret]| {
            let mut thread = _helper_test_thread();
            thread.scalar_reg[20] = *a;
            thread.scalar_reg[10] = *b;
            r(&vec![0xBE94100A, END_PRG], &mut thread);
            assert_eq!(thread.scalar_reg[20], *ret);
        });
    }

    #[test]
    fn test_s_bitset1_b32() {
        [
            [
                0b00000000000000000000000000000000,
                0b00000000000000000000000000000001,
                0b00000000000000000000000000000010,
            ],
            [
                0b00000000000000000000000000000000,
                0b00000000000000000000000000000010,
                0b00000000000000000000000000000100,
            ],
        ]
        .iter()
        .for_each(|[a, b, ret]| {
            let mut thread = _helper_test_thread();
            thread.scalar_reg[20] = *a;
            thread.scalar_reg[10] = *b;
            r(&vec![0xbe94120a, END_PRG], &mut thread);
            assert_eq!(thread.scalar_reg[20], *ret);
        });
    }

    #[test]
    fn test_s_not_b32() {
        [[0, 4294967295, 1], [1, 4294967294, 1], [u32::MAX, 0, 0]]
            .iter()
            .for_each(|[a, ret, scc]| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[10] = *a;
                r(&vec![0xBE8A1E0A, END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[10], *ret);
                assert_eq!(*thread.scc, *scc);
            });
    }
}

#[cfg(test)]
mod test_sopk {
    use super::*;

    #[test]
    fn test_cmp_zero_extend() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[20] = 0xcd14;
        r(&vec![0xB494CD14, END_PRG], &mut thread);
        assert_eq!(*thread.scc, 1);

        r(&vec![0xB194CD14, END_PRG], &mut thread);
        assert_eq!(*thread.scc, 0);
    }

    #[test]
    fn test_cmp_sign_extend() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[6] = 0x2db4;
        r(&vec![0xB1862DB4, END_PRG], &mut thread);
        assert_eq!(*thread.scc, 1);

        r(&vec![0xB1862DB4, END_PRG], &mut thread);
        assert_eq!(*thread.scc, 1);
    }
}

#[cfg(test)]
mod test_sop2 {
    use super::*;

    #[test]
    fn test_xor_exec() {
        let mut thread = _helper_test_thread();
        thread.exec.value = 0b10010010010010010010010010010010;
        thread.scalar_reg[2] = 0b11111111111111111111111111111111;
        r(&vec![0x8D02027E, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[2], 1840700269);
    }

    #[test]
    fn test_s_add_u32() {
        [[10, 20, 30, 0], [u32::MAX, 10, 9, 1], [u32::MAX, 0, u32::MAX, 0]]
            .iter()
            .for_each(|[a, b, expected, scc]| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[2] = *a;
                thread.scalar_reg[6] = *b;
                r(&vec![0x80060206, END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[6], *expected);
                assert_eq!(*thread.scc, *scc);
            });
    }

    #[test]
    fn test_s_addc_u32() {
        [[10, 20, 31, 1, 0], [10, 20, 30, 0, 0], [u32::MAX, 10, 10, 1, 1]]
            .iter()
            .for_each(|[a, b, expected, scc_before, scc_after]| {
                let mut thread = _helper_test_thread();
                *thread.scc = *scc_before;
                thread.scalar_reg[7] = *a;
                thread.scalar_reg[3] = *b;
                r(&vec![0x82070307, END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[7], *expected);
                assert_eq!(*thread.scc, *scc_after);
            });
    }

    #[test]
    fn test_s_add_i32() {
        [[-10, 20, 10, 0], [i32::MAX, 1, -2147483648, 1]]
            .iter()
            .for_each(|[a, b, expected, scc]| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[14] = *a as u32;
                thread.scalar_reg[10] = *b as u32;
                r(&vec![0x81060E0A, END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[6], *expected as u32);
                assert_eq!(*thread.scc, *scc as u32);
            });
    }

    #[test]
    fn test_s_sub_i32() {
        [[-10, 20, -30, 0], [i32::MAX, -1, -2147483648, 1]]
            .iter()
            .for_each(|[a, b, expected, scc]| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[13] = *a as u32;
                thread.scalar_reg[8] = *b as u32;
                r(&vec![0x818C080D, END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[12], *expected as u32);
                assert_eq!(*thread.scc, *scc as u32);
            });
    }

    #[test]
    fn test_s_lshl_b32() {
        [[20, 40, 1], [0, 0, 0]].iter().for_each(|[a, expected, scc]| {
            let mut thread = _helper_test_thread();
            thread.scalar_reg[15] = *a as u32;
            r(&vec![0x8408810F, END_PRG], &mut thread);
            assert_eq!(thread.scalar_reg[8], *expected as u32);
            assert_eq!(*thread.scc, *scc as u32);
        });
    }

    #[test]
    fn test_s_lshl_b64() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg.write64(2, u64::MAX - 30);
        r(&vec![0x84828202, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[2], 4294967172);
        assert_eq!(thread.scalar_reg[3], 4294967295);
        assert_eq!(*thread.scc, 1);
    }

    #[test]
    fn test_s_ashr_i32() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[2] = 36855;
        r(&vec![0x86039F02, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[3], 0);
        assert_eq!(*thread.scc, 0);
    }

    #[test]
    fn test_source_vcc() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[10] = 0x55;
        thread.vcc.value = 29;
        r(&vec![0x8B140A6A, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[20], 21);
    }

    #[test]
    fn test_s_min_i32() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[2] = -42i32 as u32;
        thread.scalar_reg[3] = -92i32 as u32;
        r(&vec![0x89020203, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[2], -92i32 as u32);
        assert_eq!(*thread.scc, 1);
    }

    #[test]
    fn test_s_mul_hi_u32() {
        [[u32::MAX, 10, 9], [u32::MAX / 2, 4, 1]].iter().for_each(|[a, b, expected]| {
            let mut thread = _helper_test_thread();
            thread.scalar_reg[0] = *a;
            thread.scalar_reg[8] = *b;
            r(&vec![0x96810800, END_PRG], &mut thread);
            assert_eq!(thread.scalar_reg[1], *expected);
        });
    }

    #[test]
    fn test_s_mul_hi_i32() {
        [[(u64::MAX) as i32, (u64::MAX / 2) as i32, 0], [2, -2, -1]]
            .iter()
            .for_each(|[a, b, expected]| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[0] = *a as u32;
                thread.scalar_reg[8] = *b as u32;
                r(&vec![0x97010800, END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[1], *expected as u32);
            });
    }

    #[test]
    fn test_s_mul_i32() {
        [[40, 2, 80], [-10, -10, 100]].iter().for_each(|[a, b, expected]| {
            let mut thread = _helper_test_thread();
            thread.scalar_reg[0] = *a as u32;
            thread.scalar_reg[6] = *b as u32;
            r(&vec![0x96000600, END_PRG], &mut thread);
            assert_eq!(thread.scalar_reg[0], *expected as u32);
        });
    }

    #[test]
    fn test_s_bfe_u64() {
        [[2, 4, 2, 0], [800, 400, 32, 0], [-10i32 as u32, 3, 246, 0], [u32::MAX, u32::MAX, 255, 0]]
            .iter()
            .for_each(|[a_lo, a_hi, ret_lo, ret_hi]| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[6] = *a_lo;
                thread.scalar_reg[7] = *a_hi;
                r(&vec![0x940cff06, 524288, END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[12], *ret_lo);
                assert_eq!(thread.scalar_reg[13], *ret_hi);
            });
    }

    #[test]
    fn test_s_bfe_i64() {
        [[131073, 0, 1, 0, 0x100000], [-2, 0, -2, -1, 524288], [2, 0, 2, 0, 524288]]
            .iter()
            .for_each(|[a_lo, a_hi, ret_lo, ret_hi, shift]| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[6] = *a_lo as u32;
                thread.scalar_reg[7] = *a_hi as u32;
                r(&vec![0x948cff06, *shift as u32, END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[12], *ret_lo as u32);
                assert_eq!(thread.scalar_reg[13], *ret_hi as u32);
            });
    }

    #[test]
    fn test_s_bfe_u32() {
        [
            [67305985, 2],
            [0b100000000110111111100000001, 0b1111111],
            [0b100000000100000000000000001, 0b0],
            [0b100000000111000000000000001, 0b10000000],
            [0b100000000111111111100000001, 0b11111111],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut thread = _helper_test_thread();
            thread.scalar_reg[0] = *a;
            r(&vec![0x9303FF00, 0x00080008, END_PRG], &mut thread);
            assert_eq!(thread.scalar_reg[3], *ret);
        });
    }

    #[test]
    fn test_s_pack_xx_b32_b16() {
        let mut thread = _helper_test_thread();
        // ll
        thread.scalar_reg[0] = 0x12345678;
        r(&vec![0x9903ff00, 0x9ABCDEF0, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[3], 0xdef05678);
        // lh
        r(&vec![0x9983ff00, 0x9ABCDEF0, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[3], 0x9abc5678);
        // hh
        r(&vec![0x9a03ff00, 0x9ABCDEF0, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[3], 2596016692);
        // hl
        r(&vec![0x9a83ff00, 0x9ABCDEF0, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[3], 3740275252);
    }
}

#[cfg(test)]
mod test_sopc {
    use super::*;

    #[test]
    fn test_s_bitcmp0_b32() {
        [[0b00, 0b1, 0], [0b01, 0b1, 1], [0b10, 0b1, 1], [0b10000000, 0b1, 0]]
            .iter()
            .for_each(|[s0, s1, scc]| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[3] = *s0;
                thread.scalar_reg[4] = *s1;
                r(&vec![0xBF0C0304, END_PRG], &mut thread);
                assert_eq!(*thread.scc, *scc);
            })
    }
}

#[cfg(test)]
mod test_vopd {
    use super::*;

    #[test]
    fn test_inline_const_vopx_only() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = f32::to_bits(0.5);
        let constant = f32::from_bits(0x39a8b099);
        thread.vec_reg[1] = 10;
        r(&vec![0xC8D000FF, 0x00000080, 0x39A8B099, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[0]), 0.5 * constant);
        assert_eq!(thread.vec_reg[1], 0);
    }

    #[test]
    fn test_inline_const_vopy_only() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 10;
        thread.vec_reg[1] = 10;
        r(&vec![0xCA100080, 0x000000FF, 0x3E15F480, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 0);
        assert_eq!(thread.vec_reg[1], 0x3e15f480);

        let mut thread = _helper_test_thread();
        thread.vec_reg[18] = f32::to_bits(2.0);
        thread.vec_reg[32] = f32::to_bits(4.0);
        thread.vec_reg[7] = 10;
        r(&vec![0xC9204112, 0x00060EFF, 0x0000006E, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[0]), 2.0f32 + 4.0f32);
        assert_eq!(thread.vec_reg[7], 120);
    }

    #[test]
    fn test_inline_const_shared() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[2] = f32::to_bits(2.0);
        thread.vec_reg[3] = f32::to_bits(4.0);
        let constant = f32::from_bits(0x3e800000);
        r(&vec![0xC8C604FF, 0x020206FF, 0x3E800000, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[2]), 2.0 * constant);
        assert_eq!(f32::from_bits(thread.vec_reg[3]), 4.0 * constant);
    }

    #[test]
    fn test_simm_op_shared_1() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[23] = f32::to_bits(4.0);
        thread.vec_reg[12] = f32::to_bits(2.0);

        thread.vec_reg[13] = f32::to_bits(10.0);
        thread.vec_reg[24] = f32::to_bits(3.0);

        let simm = f32::from_bits(0x3e000000);
        r(&vec![0xC8841917, 0x0C0C1B18, 0x3E000000, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[12]), 4.0 * simm + 2.0);
        assert_eq!(f32::from_bits(thread.vec_reg[13]), 3.0 * simm + 10.0);
    }

    #[test]
    fn test_simm_op_shared_2() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[29] = f32::to_bits(4.0);
        thread.vec_reg[10] = f32::to_bits(2.0);

        thread.vec_reg[11] = f32::to_bits(10.0);
        thread.vec_reg[26] = f32::to_bits(6.5);

        let simm = 0.125;
        r(&vec![0xC880151D, 0x0A0A34FF, 0x3E000000, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[10]), 4.0 * simm + 2.0);
        assert_eq!(f32::from_bits(thread.vec_reg[11]), simm * 6.5 + 10.0);
    }

    #[test]
    fn test_add_mov() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = f32::to_bits(10.5);
        r(&vec![0xC9100300, 0x00000080, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[0]), 10.5);
        assert_eq!(thread.vec_reg[1], 0);
    }

    #[test]
    fn test_max_add() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = f32::to_bits(5.0);
        thread.vec_reg[3] = f32::to_bits(2.0);
        thread.vec_reg[1] = f32::to_bits(2.0);
        r(&vec![0xCA880280, 0x01000700, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[0]), 7.0);
        assert_eq!(f32::from_bits(thread.vec_reg[1]), 2.0);
    }
}
#[cfg(test)]
mod test_vop1 {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn test_v_cvt_f32_f64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.write64(0, 2.0f64.to_bits());
        r(&vec![0xD58F0101, 0x00000100, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[1]), 2.0);
        thread.vec_reg.write64(0, (-2.0f64).to_bits());
        r(&vec![0xD58F0101, 0x00000100, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[1]), 2.0);
    }

    #[test]
    fn test_v_mov_b32_srrc_const0() {
        let mut thread = _helper_test_thread();
        r(&vec![0x7e000280, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 0);
        r(&vec![0x7e020280, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[1], 0);
        r(&vec![0x7e040280, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 0);
    }

    #[test]
    fn test_v_mov_b32_srrc_register() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[6] = 31;
        r(&vec![0x7e020206, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[1], 31);
    }

    fn helper_test_fexp(val: f32) -> f32 {
        let mut thread = _helper_test_thread();
        thread.vec_reg[6] = val.to_bits();
        r(&vec![0x7E0C4B06, END_PRG], &mut thread);
        f32::from_bits(thread.vec_reg[6])
    }

    #[test]
    fn test_fexp_1ulp() {
        let test_values = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        for &val in test_values.iter() {
            let expected = (2.0_f32).powf(val);
            assert!((helper_test_fexp(val) - expected).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn test_fexp_flush_denormals() {
        assert_eq!(helper_test_fexp(f32::from_bits(0xff800000)), 0.0);
        assert_eq!(helper_test_fexp(f32::from_bits(0x80000000)), 1.0);
        assert_eq!(helper_test_fexp(f32::from_bits(0x7f800000)), f32::from_bits(0x7f800000));
    }

    #[test]
    fn test_cast_f32_i32() {
        let mut thread = _helper_test_thread();
        [(10.42, 10i32), (-20.08, -20i32)].iter().for_each(|(src, expected)| {
            thread.scalar_reg[2] = f32::to_bits(*src);
            r(&vec![0x7E001002, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[0] as i32, *expected);
        })
    }

    #[test]
    fn test_cast_f32_u32() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[4] = 2;
        r(&vec![0x7E000C04, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 1073741824);
    }

    #[test]
    fn test_cast_u32_f32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 1325400062;
        r(&vec![0x7E000F00, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 2147483392);
    }

    #[test]
    fn test_cast_i32_f32() {
        let mut thread = _helper_test_thread();
        [(10.0, 10i32), (-20.0, -20i32)].iter().for_each(|(expected, src)| {
            thread.vec_reg[0] = *src as u32;
            r(&vec![0x7E000B00, END_PRG], &mut thread);
            assert_eq!(f32::from_bits(thread.vec_reg[0]), *expected);
        })
    }

    #[test]
    fn test_v_readfirstlane_b32_basic() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 2147483392;
        r(&vec![0x7E060500, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[3], 2147483392);
    }

    #[test]
    fn test_v_readfirstlane_b32_fancy() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.get_lane_mut(0)[13] = 44;
        thread.vec_reg.get_lane_mut(1)[13] = 22;
        thread.exec.value = 0b00000000000000000000000000000010;
        thread.exec.default_lane = Some(2);
        r(&vec![0x7E1A050D, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[13], 22);

        thread.exec.value = 0b00000000000000000000000000000000;
        thread.exec.default_lane = Some(1);
        r(&vec![0x7E1A050D, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[13], 44);

        thread.exec.value = 0b10000000000000000000000000000000;
        thread.vec_reg.get_lane_mut(31)[13] = 88;
        thread.exec.default_lane = Some(1);
        r(&vec![0x7E1A050D, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[13], 88);
    }

    #[test]
    fn test_v_cls_i32() {
        fn t(val: u32) -> u32 {
            let mut thread = _helper_test_thread();
            thread.vec_reg[2] = val;
            r(&vec![0x7E087702, END_PRG], &mut thread);
            return thread.vec_reg[4];
        }

        assert_eq!(t(0x00000000), 0xffffffff);
        assert_eq!(t(0x40000000), 1);
        assert_eq!(t(0x80000000), 1);
        assert_eq!(t(0x0fffffff), 4);
        assert_eq!(t(0xffff0000), 16);
        assert_eq!(t(0xfffffffe), 31);
    }

    #[test]
    fn test_v_rndne_f32() {
        [
            [1.2344, 1.0],
            [2.3, 2.0], // [0.5f32, 0.0f32],
            [0.51, 1.0],
            [f32::from_bits(1186963295), f32::from_bits(1186963456)],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut thread = _helper_test_thread();
            thread.vec_reg[0] = f32::to_bits(*a);
            r(&vec![0x7E024700, END_PRG], &mut thread);
            assert_eq!(f32::from_bits(thread.vec_reg[1]), *ret);
        })
    }

    #[test]
    fn test_v_rndne_f64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 0x652b82fe;
        thread.vec_reg[1] = 0x40071547;
        r(&vec![0x7E043300, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 0);
        assert_eq!(thread.vec_reg[3], 1074266112);
    }

    #[test]
    fn test_v_cvt_i32_f64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[2] = 0;
        thread.vec_reg[3] = 0x40080000;
        r(&vec![0x7E080702, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 3);
    }

    #[test]
    fn test_v_frexp_mant_f64() {
        [[2.0, 0.5], [1.0, 0.5], [0.54, 0.54], [f64::NAN, f64::NAN]]
            .iter()
            .for_each(|[x, expected]| {
                let mut thread = _helper_test_thread();
                thread.vec_reg.write64(0, f64::to_bits(*x));
                r(&vec![0x7E047B00, END_PRG], &mut thread);
                let ret = f64::from_bits(thread.vec_reg.read64(2));
                if ret.is_nan() {
                    assert!(ret.is_nan() && expected.is_nan());
                } else {
                    assert_eq!(f64::from_bits(thread.vec_reg.read64(2)), *expected)
                }
            })
    }

    #[test]
    fn test_v_rcp_f64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 0;
        thread.vec_reg[1] = 1073741824;
        r(&vec![0x7E045F00, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 0);
        assert_eq!(thread.vec_reg[3], 1071644672);
    }

    #[test]
    fn test_v_frexp_exp_i32_f64() {
        [(3573412790272.0, 42), (69.0, 7), (2.0, 2), (f64::NEG_INFINITY, 0)]
            .iter()
            .for_each(|(x, ret)| {
                let mut thread = _helper_test_thread();
                thread.vec_reg.write64(0, f64::to_bits(*x));
                r(&vec![0x7E047900, END_PRG], &mut thread);
                assert_eq!(thread.vec_reg[2], *ret);
            })
    }

    #[test]
    fn test_v_rsq_f64() {
        [(2.0, 0.707)].iter().for_each(|(x, ret)| {
            let mut thread = _helper_test_thread();
            thread.vec_reg.write64(0, f64::to_bits(*x));
            println!("{} {}", thread.vec_reg[0], thread.vec_reg[1]);
            r(&vec![0x7E046300, END_PRG], &mut thread);
            assert!(approx_eq!(f64, f64::from_bits(thread.vec_reg.read64(2)), *ret, (0.01, 2)));
        })
    }
}

#[cfg(test)]
mod test_vopc {
    use super::*;

    #[test]
    fn test_v_cmp_gt_i32() {
        let mut thread = _helper_test_thread();

        thread.vec_reg[1] = (4_i32 * -1) as u32;
        r(&vec![0x7c8802c1, END_PRG], &mut thread);
        assert_eq!(thread.vcc.read(), true);

        thread.vec_reg[1] = 4;
        r(&vec![0x7c8802c1, END_PRG], &mut thread);
        assert_eq!(thread.vcc.read(), false);
    }

    #[test]
    fn test_v_cmpx_nlt_f32() {
        let mut thread = _helper_test_thread();
        thread.exec.value = 0b010011;
        thread.vec_reg[0] = f32::to_bits(0.9);
        thread.vec_reg[3] = f32::to_bits(0.4);
        r(&vec![0x7D3C0700, END_PRG], &mut thread);
        assert_eq!(thread.exec.read(), true);
    }

    #[test]
    fn test_v_cmpx_gt_i32_e32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[3] = 100;
        r(&vec![0x7D8806FF, 0x00000041, END_PRG], &mut thread);
        assert_eq!(thread.exec.read(), false);

        thread.vec_reg[3] = -20i32 as u32;
        r(&vec![0x7D8806FF, 0x00000041, END_PRG], &mut thread);
        assert_eq!(thread.exec.read(), true);
    }

    #[test]
    fn test_cmp_class_f32() {
        let thread = _helper_test_thread();
        assert!(!thread.cmp_class_f32(f32::NAN, 0b00001));
        assert!(thread.cmp_class_f32(f32::NAN, 0b00010));

        assert!(thread.cmp_class_f32(f32::INFINITY, 0b00000000000000000000001000000000));
        assert!(!thread.cmp_class_f32(f32::INFINITY, 0b00000000000000000000000000000010));

        assert!(thread.cmp_class_f32(f32::NEG_INFINITY, 0b00000000000000000000000000000100));
        assert!(!thread.cmp_class_f32(f32::NEG_INFINITY, 0b00000000000000000000010000000000));

        assert!(!thread.cmp_class_f32(0.752, 0b00000000000000000000000000000000));
        assert!(thread.cmp_class_f32(0.752, 0b00000000000000000000000100000000));

        assert!(!thread.cmp_class_f32(-0.752, 0b00000000000000000000010000000000));
        assert!(thread.cmp_class_f32(-0.752, 0b00000000000000000000010000001000));

        assert!(!thread.cmp_class_f32(1.0e-42, 0b11111111111111111111111101111111));
        assert!(thread.cmp_class_f32(1.0e-42, 0b00000000000000000000000010000000));

        assert!(thread.cmp_class_f32(-1.0e-42, 0b00000000000000000000000000010000));
        assert!(!thread.cmp_class_f32(-1.0e-42, 0b11111111111111111111111111101111));

        assert!(thread.cmp_class_f32(-0.0, 0b00000000000000000000000000100000));
        assert!(thread.cmp_class_f32(0.0, 0b00000000000000000000000001000000));
    }

    #[test]
    fn test_cmp_class_f64() {
        let thread = _helper_test_thread();

        assert!(!thread.cmp_class_f64(f64::NAN, 0b00001));
        assert!(thread.cmp_class_f64(f64::NAN, 0b00010));

        assert!(thread.cmp_class_f64(f64::INFINITY, 0b00000000000000000000001000000000));
        assert!(!thread.cmp_class_f64(f64::INFINITY, 0b00000000000000000000000000000010));

        assert!(thread.cmp_class_f64(f64::NEG_INFINITY, 0b00000000000000000000000000000100));
        assert!(!thread.cmp_class_f64(f64::NEG_INFINITY, 0b00000000000000000000010000000000));

        assert!(!thread.cmp_class_f64(0.752, 0b00000000000000000000000000000000));
        assert!(thread.cmp_class_f64(0.752, 0b00000000000000000000000100000000));

        assert!(!thread.cmp_class_f64(-1.0e-42, 0b00000000000000000000000000010000));
        assert!(thread.cmp_class_f64(-1.0e-42, 0b11111111111111111111111111101111));

        assert!(thread.cmp_class_f64(-0.0, 0b00000000000000000000000000100000));
        assert!(thread.cmp_class_f64(0.0, 0b00000000000000000000000001000000));
    }
}
#[cfg(test)]
mod test_vop2 {
    use super::*;

    #[test]
    fn test_v_add_f32_e32() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[2] = f32::to_bits(42.0);
        thread.vec_reg[0] = f32::to_bits(1.0);
        r(&vec![0x06000002, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[0]), 43.0);
    }

    #[test]
    fn test_v_and_b32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[10] = 15;
        r(&vec![0x36141482, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[10], 2);
    }

    #[test]
    fn test_v_mul_f32_e32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[2] = f32::to_bits(21.0);
        thread.vec_reg[4] = f32::to_bits(2.0);
        r(&vec![0x10060504, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[3]), 42.0);
    }

    #[test]
    fn test_v_ashrrev_i32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 4294967295;
        r(&vec![0x3402009F, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[1] as i32, -1);
    }

    #[test]
    fn test_v_mul_i32_i24() {
        [
            [18, 0x64, 1800],
            [0b10000000000000000000000000, 0b1, 0],
            [0b100000000000000000000000, 0b1, 0b11111111100000000000000000000000],
        ]
        .iter()
        .for_each(|[a, b, ret]| {
            let mut thread = _helper_test_thread();
            thread.vec_reg[1] = *a;
            r(&vec![0x124E02FF, *b, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[39], *ret);
        });
    }

    #[test]
    fn test_v_add_nc_u32_const() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[18] = 7;
        r(&vec![0x4A3024B8, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[24], 63);
    }

    #[test]
    fn test_v_add_nc_u32_sint() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[14] = 7;
        thread.vec_reg[6] = 4294967279;
        r(&vec![0x4A0C1D06, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[6], 4294967286);
    }
}

#[cfg(test)]
mod test_vopsd {
    use super::*;

    #[test]
    fn test_v_add_co_u32_scalar_co_zero() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[10] = 0;
        thread.vcc.default_lane = Some(1);
        thread.vec_reg.default_lane = Some(1);
        thread.vec_reg[10] = u32::MAX;
        thread.vec_reg[20] = 20;
        r(&vec![0xD7000A0A, 0x0002290A, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[10], 19);
        assert_eq!(thread.scalar_reg[10], 2);
    }

    #[test]
    fn test_v_add_co_u32_scalar_co_override() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[10] = 0b11111111111111111111111111111111;
        thread.vcc.default_lane = Some(2);
        thread.vec_reg.default_lane = Some(2);
        thread.vec_reg[10] = u32::MAX;
        thread.vec_reg[20] = 20;
        r(&vec![0xD7000A0A, 0x0002290A, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[10], 19);
        // NOTE: the co mask only writes to the bit that it needs to write, then at the _wave_
        // level, the final result accumulates
        assert_eq!(thread.scalar_reg[10], 0b100);
    }

    #[test]
    fn test_v_add_co_ci_u32() {
        [[0, 0, 0b0], [1, -1i32 as usize, 0b10]].iter().for_each(|[lane_id, result, carry_out]| {
            let mut thread = _helper_test_thread();
            thread.vcc.default_lane = Some(*lane_id);
            thread.vec_reg.default_lane = Some(*lane_id);
            thread.scalar_reg[20] = 0b10;
            thread.vec_reg[1] = 2;
            thread.vec_reg[2] = 2;
            r(&vec![0xD5211401, 0x00520501, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[1], *result as u32);
            assert_eq!(thread.scalar_reg[20], *carry_out as u32);
        })
    }

    #[test]
    fn test_v_sub_co_ci_u32() {
        [[3, 2, 0b1000], [2, 0, 0b100]].iter().for_each(|[lane_id, result, carry_out]| {
            let mut thread = _helper_test_thread();
            thread.vcc.default_lane = Some(*lane_id);
            thread.vec_reg.default_lane = Some(*lane_id);
            thread.scalar_reg[20] = 0b1010;
            thread.vec_reg[1] = *lane_id as u32;
            thread.vec_reg[2] = u32::MAX - 1;
            r(&vec![0xD5201401, 0x00520501, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[1], *result as u32);
            assert_eq!(thread.scalar_reg[20], *carry_out as u32);
        })
    }

    #[test]
    fn test_v_mad_u64_u32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.write64(3, u64::MAX - 3);
        thread.scalar_reg[13] = 3;
        thread.scalar_reg[10] = 1;
        r(&vec![0xD6FE0D06, 0x040C140D, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg.read64(6), u64::MAX);
        assert_eq!(thread.scalar_reg[13], 0);

        thread.vec_reg.write64(3, u64::MAX - 3);
        thread.scalar_reg[13] = 4;
        thread.scalar_reg[10] = 1;
        r(&vec![0xD6FE0D06, 0x040C140D, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[6], 0);
        assert_eq!(thread.vec_reg[7], 0);
        assert_eq!(thread.scalar_reg[13], 1);
    }

    #[test]
    fn test_v_add_co_u32() {
        let mut thread = _helper_test_thread();
        thread.vcc.default_lane = Some(1);
        thread.vec_reg[2] = u32::MAX;
        thread.vec_reg[3] = 3;
        r(&vec![0xD7000D02, 0x00020503, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 2);
        assert_eq!(thread.scalar_reg[13], 0b10);
    }

    #[test]
    fn test_v_sub_co_u32() {
        [[69, 0, 69, 0], [100, 200, 4294967196, 1]].iter().for_each(|[a, b, ret, scc]| {
            let mut thread = _helper_test_thread();
            thread.vec_reg[4] = *a;
            thread.vec_reg[15] = *b;
            r(&vec![0xD7016A04, 0x00021F04, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[4], *ret);
            assert_eq!(thread.vcc.read(), *scc != 0);
        })
    }

    #[test]
    fn test_return_value_exec_zero() {
        let mut thread = _helper_test_thread();
        thread.exec.value = 0b11111111111111111111111111111101;
        thread.vcc.default_lane = Some(1);
        thread.exec.default_lane = Some(1);
        thread.vec_reg[2] = u32::MAX;
        thread.vec_reg[3] = 3;
        r(&vec![0xD7000D02, 0x00020503, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], u32::MAX);
        assert_eq!(thread.scalar_reg[13], 0b10);
    }

    #[test]
    fn test_v_div_scale_f64() {
        let mut thread = _helper_test_thread();
        let v = -0.41614683654714246;
        thread.vec_reg.write64(0, f64::to_bits(v));
        thread.vec_reg.write64(2, f64::to_bits(v));
        thread.vec_reg.write64(4, f64::to_bits(0.909));
        r(&vec![0xD6FD7C06, 0x04120500, END_PRG], &mut thread);
        thread.vec_reg[6] = 1465086470;
        thread.vec_reg[7] = 3218776614;
        let ret = f64::from_bits(thread.vec_reg.read64(6));
        assert_eq!(ret, v);
    }
}

#[cfg(test)]
mod test_vop3 {
    use super::*;
    use float_cmp::approx_eq;

    fn helper_test_vop3(op: u32, a: f32, b: f32) -> f32 {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[0] = f32::to_bits(a);
        thread.scalar_reg[6] = f32::to_bits(b);
        r(&vec![op, 0x00000006, END_PRG], &mut thread);
        return f32::from_bits(thread.vec_reg[0]);
    }

    #[test]
    fn test_v_add_f32() {
        assert_eq!(helper_test_vop3(0xd5030000, 0.4, 0.2), 0.6);
    }

    #[test]
    fn test_v_mul_f16() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[1].mut_lo16(f16::from_f32(2.0).to_bits());
        thread.vec_reg[2].mut_lo16(f16::from_f32(4.0).to_bits());
        r(&vec![0xD5350000, 0x00020501, END_PRG], &mut thread);
        assert_eq!(f16::from_bits(thread.vec_reg[0] as u16), f16::from_f32(8.0));
    }

    #[test]
    fn test_v_max_f32() {
        assert_eq!(helper_test_vop3(0xd5100000, 0.4, 0.2), 0.4);
        assert_eq!(helper_test_vop3(0xd5100000, 0.2, 0.8), 0.8);
    }

    #[test]
    fn test_v_mul_f32() {
        assert_eq!(helper_test_vop3(0xd5080000, 0.4, 0.2), 0.4 * 0.2);
    }

    #[test]
    fn test_signed_src() {
        // v0, max(s2, s2)
        let mut thread = _helper_test_thread();
        thread.scalar_reg[2] = f32::to_bits(0.5);
        r(&vec![0xd5100000, 0x00000402, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[0]), 0.5);

        // v1, max(-s2, -s2)
        let mut thread = _helper_test_thread();
        thread.scalar_reg[2] = f32::to_bits(0.5);
        r(&vec![0xd5100001, 0x60000402, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[1]), -0.5);
    }

    #[test]
    fn test_cnd_mask_cond_src_sgpr() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[3] = 0b001;
        r(&vec![0xD5010000, 0x000D0280, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 1);

        thread.scalar_reg[3] = 0b00;
        r(&vec![0xD5010000, 0x000D0280, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 0);
    }

    #[test]
    fn test_cnd_mask_cond_src_vcclo() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[2] = 20;
        thread.vec_reg[0] = 100;
        r(&vec![0xD5010002, 0x41AA0102, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 20);
    }

    #[test]
    fn test_cnd_mask_float_const() {
        let mut thread = _helper_test_thread();
        thread.vcc.value = 0b00000010;
        thread.vcc.default_lane = Some(0);
        r(&vec![0xD5010003, 0x01A9E480, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[3], 0);

        thread.vcc.value = 0b00000010;
        thread.vcc.default_lane = Some(1);
        r(&vec![0xD5010003, 0x01A9E480, END_PRG], &mut thread);
        assert_eq!(f32::from_bits(thread.vec_reg[3]), 1.0);
    }

    #[test]
    fn test_v_cndmask_b32_e64_neg() {
        [[0.0f32, -0.0], [-0.0f32, 0.0], [1.0f32, -1.0], [-1.0f32, 1.0]].iter().for_each(|[input, ret]| {
            let mut thread = _helper_test_thread();
            thread.scalar_reg[0] = false as u32;
            thread.vec_reg[3] = input.to_bits();
            r(&vec![0xD5010003, 0x2001FF03, 0x80000000, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[3], ret.to_bits());
        });
    }

    #[test]
    fn test_v_mul_hi_i32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[2] = -2i32 as u32;
        r(&vec![0xD72E0003, 0x000204FF, 0x2E8BA2E9, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[3] as i32, -1);

        thread.vec_reg[2] = 2;
        r(&vec![0xD72E0003, 0x000204FF, 0x2E8BA2E9, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[3], 0);
    }

    #[test]
    fn test_v_writelane_b32() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[8] = 25056;
        r(&vec![0xD7610004, 0x00010008, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg.get_lane(0)[4], 25056);

        thread.scalar_reg[9] = 25056;
        r(&vec![0xD7610004, 0x00010209, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg.get_lane(1)[4], 25056);
    }

    #[test]
    fn test_v_readlane_b32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.get_lane_mut(15)[4] = 0b1111;
        r(&vec![0xD760006A, 0x00011F04, END_PRG], &mut thread);
        assert_eq!(thread.vcc.read(), true);
    }

    #[test]
    fn test_v_lshlrev_b64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.write64(2, 100);
        thread.vec_reg[4] = 2;
        r(&vec![0xD73C0002, 0x00020504, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg.read64(2), 400);
    }

    #[test]
    fn test_v_lshrrev_b64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.write64(2, 100);
        thread.vec_reg[4] = 2;
        r(&vec![0xd73d0002, 0x00020504, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg.read64(2), 25);
    }

    #[test]
    fn test_v_add_f64_neg_modifier() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 0x652b82fe;
        thread.vec_reg[1] = 0x40071547;
        thread.vec_reg[2] = 0;
        thread.vec_reg[3] = 0x40080000;
        r(&vec![0xD7270004, 0x40020500, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 1519362112);
        assert_eq!(thread.vec_reg[5], 3216856851);
    }

    #[test]
    fn test_v_cvt_f32_f16_abs_modifier() {
        [[0.4, 0.4], [-0.4, 0.4]].iter().for_each(|[a, ret]| {
            let mut thread = _helper_test_thread();
            thread.vec_reg[1] = f16::from_f32_const(*a).to_bits() as u32;
            r(&vec![0xD58B0102, 0x00000101, END_PRG], &mut thread);
            assert!(approx_eq!(f32, f32::from_bits(thread.vec_reg[2]), *ret, (0.01, 2)));
        });
    }

    #[test]
    fn test_v_alignbit_b32() {
        let mut thread = _helper_test_thread();
        thread.scalar_reg[4] = 5340353;
        thread.scalar_reg[10] = 3072795146;
        thread.vec_reg[0] = 8;
        r(&vec![0xD6160001, 0x04001404, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[1], 3250005794);
    }

    #[test]
    fn test_v_bfe_i32() {
        [
            [0b00000000000000000000000000000001, -1],
            [0b00000000000000000000000000000000, 0],
            [0b00000000000000000000000000000010, 0],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut thread = _helper_test_thread();
            thread.vec_reg[2] = *a as u32;
            r(&vec![0xD6110005, 0x02050102, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[5] as i32, *ret);
        });

        [
            [0b00000000000000000000000000000010, -2],
            [0b00000000000000000000000000000001, 1],
            [0b00000000000000000000000000000100, 0],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut thread = _helper_test_thread();
            thread.vec_reg[2] = *a as u32;
            r(&vec![0xD6110005, 0x02090102, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[5] as i32, *ret);
        });

        [
            [0b00100000000000000000000000000000, 0b100000000000000000000000000000],
            [0b00000000000000001000000000000000, 0b1000000000000000],
            [-1, -1],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut thread = _helper_test_thread();
            thread.vec_reg[2] = *a as u32;
            r(&vec![0xD6110005, 0x03050102, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[5] as i32, *ret);
        });
    }

    #[test]
    fn test_v_ashrrev_i16() {
        let mut thread = _helper_test_thread();
        [
            [0b10000000000000000000000000000000, 0],
            [0b10000000000000000000000000000111, 3],
            [0b0000000000000000, 0],
            [0b1000000000000000, 0b1100000000000000],
            [0b0100000000000000, 0b0010000000000000],
            [0b0010000000000000, 0b0001000000000000],
            [0b1010000000000000, 0b1101000000000000],
            [0b1110000000000000, 0b1111000000000000],
            [0b0110000000000000, 0b0011000000000000],
        ]
        .iter()
        .for_each(|[a, ret]| {
            thread.vec_reg[2] = *a;
            thread.scalar_reg[1] = 1;
            r(&vec![0xd73a0005, 0b11000001100000010000000001, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[5], *ret);
        });

        [
            [0b1000000000000000, 0b1111, 0b1111111111111111],
            [0b1000000000000000, 0b11111, 0b1111111111111111],
            [0b1000000000000000, 0b0111, 0b1111111100000000],
        ]
        .iter()
        .for_each(|[a, shift, ret]| {
            thread.vec_reg[2] = *a;
            thread.scalar_reg[1] = *shift;
            r(&vec![0xd73a0005, 0b11000001100000010000000001, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[5], *ret);
        });

        thread.vec_reg[5] = 0b11100000000000001111111111111111;
        thread.vec_reg[2] = 0b0100000000000000;
        thread.scalar_reg[1] = 1;
        r(&vec![0xd73a0005, 0b11000001100000010000000001, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[5], 0b11100000000000000010000000000000);
    }

    #[test]
    fn test_v_add_nc_u16() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[5] = 10;
        thread.vec_reg[8] = 20;
        r(&vec![0xD7030005, 0x00021105, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[5], 30);
    }

    #[test]
    fn test_v_mul_lo_u16() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[5] = 2;
        thread.vec_reg[15] = 0;
        r(&vec![0xD705000F, 0x00010B05, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[15], 10);

        thread.vec_reg[5] = 2;
        thread.vec_reg[15] = 0b10000000000000000000000000000000;
        r(&vec![0xD705000F, 0x00010B05, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[15], 0b10000000000000000000000000000000 + 10);
    }

    #[test]
    fn test_v_cmp_gt_u16() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[1] = 52431;
        thread.scalar_reg[5] = 0;
        r(&vec![0xD43C0005, 0x000202FF, 0x00003334, END_PRG], &mut thread);
        assert_eq!(thread.scalar_reg[5], 0);
    }

    #[test]
    fn test_v_cmp_ngt_f32_abs() {
        [(0.5f32, 0.5f32, 1), (-0.5, 0.5, 1), (0.1, 0.2, 0), (-0.1, 0.2, 0)]
            .iter()
            .for_each(|(x, y, ret)| {
                let mut thread = _helper_test_thread();
                thread.scalar_reg[2] = x.to_bits();
                r(&vec![0xD41B0203, 0x000004FF, y.to_bits(), END_PRG], &mut thread);
                assert_eq!(thread.scalar_reg[3], *ret);
            })
    }
    #[test]
    fn test_fma() {
        fn v_fma_f32(a: u32, b: u32, c: u32, ret: u32) {
            let mut thread = _helper_test_thread();
            thread.vec_reg[1] = b;
            thread.scalar_reg[3] = c;
            r(&vec![0xD6130000, 0x000E02FF, a, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[0], ret);
        }
        fn v_fmac_f32(a: u32, b: u32, c: u32, ret: u32) {
            let mut thread = _helper_test_thread();
            thread.scalar_reg[1] = a;
            thread.scalar_reg[2] = b;
            thread.vec_reg[0] = c;
            r(&vec![0xd52b0000, 0x401, END_PRG], &mut thread);
            assert_eq!(thread.vec_reg[0], ret);
        }
        [[0xbfc90fda, 1186963456, 1192656896, 3204127872]].iter().for_each(|[a, b, c, ret]| {
            v_fma_f32(*a, *b, *c, *ret);
            v_fmac_f32(*a, *b, *c, *ret);
        })
    }

    #[test]
    fn test_v_perm_b32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[1] = 15944;
        thread.vec_reg[0] = 84148480;
        r(&vec![0xD644000F, 0x03FE0101, 0x05040100, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[15], 1044906240);
    }

    #[test]
    fn test_v_mul_f64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 0x5a8fa040;
        thread.vec_reg[1] = 0xbfbd5713;
        thread.vec_reg[2] = 0x3b39803f;
        thread.vec_reg[3] = 0x3c7abc9e;
        r(&vec![0xD7280004, 0x00020500, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 1602589062);
        assert_eq!(thread.vec_reg[5], 3158868912);
    }

    #[test]
    fn test_v_fma_f64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 0x5a8fa040;
        thread.vec_reg[1] = 0xbfbd5713;
        thread.vec_reg[2] = 0xfefa39ef;
        thread.vec_reg[3] = 0x3fe62e42;
        thread.vec_reg[4] = 0x5f859186;
        thread.vec_reg[5] = 0xbc4883b0;
        r(&vec![0xD6140006, 0x04120500, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[6], 3883232879);
        assert_eq!(thread.vec_reg[7], 3216266823);
    }

    #[test]
    fn test_v_fma_f64_const() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[0] = 0xf690ecbf;
        thread.vec_reg[1] = 0x3fdf2b4f;
        thread.vec_reg[2] = 0xe7756e6f;
        thread.vec_reg[3] = 0xbfb45647;
        r(&vec![0xD6140004, 0x03CA0500, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 962012421);
        assert_eq!(thread.vec_reg[5], 1072612110);
    }

    #[test]
    fn test_v_ldexp_f64() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.write64(0, f64::to_bits(5.0));
        thread.vec_reg[2] = 3;
        thread.vec_reg[3] = 3;
        r(&vec![0xD72B0000, 0x00020500, END_PRG], &mut thread);
        let val = f64::from_bits(thread.vec_reg.read64(0));
        assert_eq!(val, 40.0);
    }

    #[test]
    fn test_simm_resolve_int_in_double_op() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.write64(0, 3.0f64.to_bits());
        let simm = 0xFFFFFFE0;
        r(&vec![0xD72B0002, 0x0001FF00, simm, END_PRG], &mut thread);
        assert_eq!(f64::from_bits(thread.vec_reg.read64(2)), 3.0 * 2.0.powi(-32));
    }

    #[test]
    fn test_simm_resolve_double_in_double_op() {
        let mut thread = _helper_test_thread();
        thread.vec_reg.write64(0, 2.0f64.to_bits());
        let simm = 0x40080000;
        r(&vec![0xD7280000, 0x000200FF, simm, END_PRG], &mut thread);
        assert_eq!(f64::from_bits(thread.vec_reg.read64(0)), 6.0);
    }
}

#[cfg(test)]
mod test_vopp {
    use super::*;

    #[test]
    fn test_v_fma_mix_f32() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[2] = 1065353216;
        thread.scalar_reg[2] = 3217620992;
        thread.vec_reg[1] = 15360;
        r(&vec![0xCC204403, 0x04040502, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[3], 3205627904);

        thread.vec_reg[2] = 1065353216;
        thread.scalar_reg[2] = 3217620992;
        thread.vec_reg[1] = 48128;
        r(&vec![0xCC204403, 0x04040502, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[3], 3205627904);
    }

    #[test]
    fn test_packed_opsel_000_op_000() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[1] = 1;
        thread.vec_reg[2] = 2;
        thread.vec_reg[3] = 3;
        r(&vec![0xCC090004, 0x040E0501, 0xBFB00000, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b1010000000000000101);
    }

    #[test]
    fn test_packed_opsel_001_op_100() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[1] = 1;
        thread.vec_reg[2] = 2;
        thread.vec_reg[3] = 3;
        r(&vec![0xCC092004, 0x0C0E0501, 0xBFB00000, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b110000000000000010);
    }

    #[test]
    fn test_packed_inline_const_int() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[1] = 1;
        thread.vec_reg[2] = 2;
        thread.vec_reg[3] = 3;

        r(&vec![0xCC090004, 0x020E0501, 0xBFB00000, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b1010000000000000101);

        r(&vec![0xCC090804, 0x0A0E0501, 0xBFB00000, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b110000000000000011);

        r(&vec![0xCC096004, 0x020E0501, 0xBFB00000, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b100000000000000010);

        r(&vec![0xCC090004, 0x03FE0501, 0x00000080, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 8519810);
    }

    #[test]
    fn test_pk_fma_f16_inline_const() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[2] = 0x393a35f6;
        thread.vec_reg[3] = 0x2800;

        r(&vec![0xCC0E0004, 0x03FE0702, 0x0000A400, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 2618596372);

        r(&vec![0xCC0E0004, 0x0BFE0702, 0x0000A400, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 485006356);

        r(&vec![0xCC0E0004, 0x1BFE0702, 0x0000A400, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 2751503380);

        r(&vec![0xCC0E0804, 0x03FE0702, 0x0000A400, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 2618563816);

        r(&vec![0xCC0E1804, 0x03FE0702, 0x0000A400, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 2618598400);
    }

    #[test]
    fn test_v_fma_mixhilo_f16() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[11] = 1065353216;
        thread.vec_reg[7] = 3047825943;
        thread.vec_reg[16] = 3047825943;

        thread.vec_reg[14] = 0b10101010101010101111111111111111;
        r(&vec![0xCC21000E, 0x04420F0B, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[14], 0b10101010101010101000000000101011);

        thread.vec_reg[14] = 0b10101010101010101111111111111111;
        r(&vec![0xCC22000E, 0x04420F0B, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[14], 0b10000000001010111111111111111111);
    }

    #[test]
    fn test_v_pk_lshlrev_b16() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[3] = 0b1010101011101101;

        r(&vec![0xCC044004, 0x0002068E, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b1000000000000000100000000000000);

        r(&vec![0xCC044004, 0x1002068E, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b100000000000000);

        r(&vec![0xCC044004, 0x100206FF, 0x00010002, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b1010101110110100);
        r(&vec![0xCC044004, 0x100206FF, 0x05012002, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b1010101110110100);

        r(&vec![0xCC044004, 0x100206FF, 0x0503E00F, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b1000000000000000);
        r(&vec![0xCC044004, 0x100206FF, 0x0503E007, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b111011010000000);
        r(&vec![0xCC044004, 0x100206FF, 0x0503E01F, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[4], 0b1000000000000000);
    }

    #[test]
    fn test_pk_fma_with_neg() {
        let mut thread = _helper_test_thread();
        let a1 = f16::from_f32(1.0);
        let b1 = f16::from_f32(2.0);
        let c1 = f16::from_f32(3.0);

        let a2 = f16::from_f32(4.0);
        let b2 = f16::from_f32(5.0);
        let c2 = f16::from_f32(6.0);

        thread.vec_reg[0] = (a1.to_bits() as u32) << 16 | (a2.to_bits() as u32);
        thread.vec_reg[9] = (b1.to_bits() as u32) << 16 | (b2.to_bits() as u32);
        thread.vec_reg[10] = (c1.to_bits() as u32) << 16 | (c2.to_bits() as u32);

        r(&vec![0xCC0E3805, 0x042A1300, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[5], 1317029120);

        r(&vec![0xCC0E3805, 0x242A1300, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[5], 1317026816);

        r(&vec![0xCC0E3B05, 0x042A1300, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[5], 1317029120);

        r(&vec![0xCC0E3905, 0x042A1300, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[5], 3405792512);
    }

    #[test]
    fn test_pk_add_f16_with_float_const() {
        let mut thread = _helper_test_thread();
        let a1 = f16::from_f32(5.0);
        let a2 = f16::from_f32(10.0);

        thread.vec_reg[1] = (a1.to_bits() as u32) << 16 | (a2.to_bits() as u32);
        r(&vec![0xCC0F4002, 0x0001E501, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 1233144192);

        r(&vec![0xCC0F5002, 0x0001E501, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 1233144064);

        r(&vec![0xCC0F5002, 0x1001E501, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 1224755456);

        r(&vec![0xCC0F5802, 0x1801E501, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[2], 1157645568);
    }
}

#[cfg(test)]
mod test_flat {
    use super::*;
    use std::alloc::{alloc, handle_alloc_error, Layout};

    #[test]
    fn test_scratch_swap_values() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[13] = 42;
        thread.vec_reg[14] = 10;
        r(
            &vec![
                0xDC690096, 0x007C0D00, 0xDC69001E, 0x007C0E00, 0xDC51001E, 0x0D7C0000, 0xDC510096, 0x0E7C0000, END_PRG,
            ],
            &mut thread,
        );
        assert_eq!(thread.vec_reg[13], 10);
        assert_eq!(thread.vec_reg[14], 42);
    }

    #[test]
    fn test_scratch_load_dword_offset() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[14] = 14;
        thread.vec_reg[15] = 23;
        r(&vec![0xDC6D000A, 0x007C0E00, 0xDC51000A, 0x0E7C0000, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[14], 14);

        r(&vec![0xDC6D000A, 0x007C0E00, 0xDC51000E, 0x0E7C0000, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[14], 23);
    }

    #[test]
    fn test_global_load_d16_hi_b16() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[13] = 0b10101011101101001111111111111111;
        unsafe {
            let layout = Layout::new::<u16>();
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout)
            }
            *(ptr as *mut u16) = 42;
            thread.vec_reg.write64(10, ptr as u64);
        }
        r(&vec![0xDC8E0000, 0x0D7C000A, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[13], 0b00000000001010101111111111111111);
    }
}

#[cfg(test)]
mod test_lds {
    use super::*;
    #[test]
    fn test_ds_load_offset() {
        let mut thread = _helper_test_thread();
        thread.lds.write(256, 69);
        thread.vec_reg[9] = 0;
        r(&vec![0xD8D80100, 0x01000009, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[1], 69);

        thread.lds.write(800, 69);
        thread.vec_reg[9] = 0;
        r(&vec![0xD8D80320, 0x01000009, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[1], 69);

        thread.lds.write(3, 69);
        thread.vec_reg[9] = 0;
        r(&vec![0xD8D80003, 0x01000009, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[1], 69);
    }

    #[test]
    fn test_ds_load_dwords() {
        let mut thread = _helper_test_thread();
        thread.lds.write(0, 100);
        thread.lds.write(4, 200);
        thread.vec_reg[9] = 0;
        r(&vec![0xD9D80000, 0x00000009, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg.read64(0), 858993459300);

        thread.lds.write(0, 1);
        thread.lds.write(4, 2);
        thread.lds.write(8, 3);
        thread.lds.write(12, 4);
        thread.vec_reg[9] = 0;
        r(&vec![0xDBFC0000, 0x00000009, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 1);
        assert_eq!(thread.vec_reg[1], 2);
        assert_eq!(thread.vec_reg[2], 3);
        assert_eq!(thread.vec_reg[3], 4);
    }

    #[test]
    fn test_ds_load_u8() {
        let mut thread = _helper_test_thread();
        thread.lds.write(0, 17);
        thread.vec_reg[0] = 0;
        r(&vec![0xD8E80000, 0x00000100, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 17);

        thread.lds.write(0, 264);
        thread.vec_reg[0] = 0;
        r(&vec![0xD8E80000, 0x00000100, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 8);

        thread.lds.write(8, 23);
        thread.vec_reg[0] = 0;
        r(&vec![0xD8E80008, 0x00000100, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 23);

        thread.lds.write(16, 29);
        thread.vec_reg[0] = 0;
        r(&vec![0xD8E80010, 0x00000100, END_PRG], &mut thread);
        assert_eq!(thread.vec_reg[0], 29);
    }

    #[test]
    fn test_ds_store_dwords() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[9] = 69;
        thread.vec_reg[0] = 0;
        r(&vec![0xD83403E8, 0x00000900, END_PRG], &mut thread);
        assert_eq!(thread.lds.read(1000), 69);
    }

    #[test]
    fn test_ds_store_half() {
        let mut thread = _helper_test_thread();
        thread.vec_reg[9].mut_lo16(f16::from_f32(1.2).to_bits());
        thread.vec_reg[9].mut_hi16(f16::from_f32(4.3).to_bits());
        thread.vec_reg[0] = 0;
        thread.vec_reg[1] = 2;
        r(&vec![0xDA840000, 0x00000900, 0xD87C0000, 0x00000901, END_PRG], &mut thread);
        assert_eq!(thread.lds.read(0) as u16, f16::from_f32(4.3).to_bits());
        assert_eq!(thread.lds.read(2) as u16, f16::from_f32(1.2).to_bits());
    }
}
#[allow(dead_code)]
fn r(prg: &Vec<u32>, thread: &mut Thread) {
    let mut pc = 0;
    let instructions = prg.to_vec();
    thread.pc_offset = 0;
    if thread.exec.value == 0 {
        thread.exec.value = u32::MAX;
    }

    loop {
        if instructions[pc] == END_PRG {
            break;
        }
        if instructions[pc] == 0xbfb60003 || instructions[pc] >> 20 == 0xbf8 {
            pc += 1;
            continue;
        }
        thread.pc_offset = 0;
        thread.stream = instructions[pc..instructions.len()].to_vec();
        thread.interpret().unwrap();
        thread.simm = None;
        if thread.vcc.mutations.is_some() {
            thread.vcc.apply_muts();
            thread.vcc.mutations = None;
        }
        if thread.exec.mutations.is_some() {
            thread.exec.apply_muts();
            thread.exec.mutations = None;
        }
        if let Some((idx, mut wv)) = thread.sgpr_co {
            wv.apply_muts();
            thread.scalar_reg[*idx] = wv.value;
        }
        if *DEBUG {
            println!()
        }
        pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
    }
}
fn _helper_test_thread() -> Thread<'static> {
    let static_lds: &'static mut VecDataStore = Box::leak(Box::new(VecDataStore::new()));
    let static_sgpr: &'static mut [u32; SGPR_COUNT] = Box::leak(Box::new([0; SGPR_COUNT]));
    let static_vgpr: &'static mut VGPR = Box::leak(Box::new(VGPR::new()));
    let static_scc: &'static mut u32 = Box::leak(Box::new(0));
    let static_exec: &'static mut WaveValue = Box::leak(Box::new(WaveValue::new(u32::MAX, 32)));
    let static_vcc: &'static mut WaveValue = Box::leak(Box::new(WaveValue::new(0, 32)));
    let static_sds: &'static mut VecDataStore = Box::leak(Box::new(VecDataStore::new()));
    let static_co: &'static mut Option<(usize, WaveValue)> = Box::leak(Box::new(None));

    let thread = Thread {
        scalar_reg: static_sgpr,
        vec_reg: static_vgpr,
        scc: static_scc,
        vcc: static_vcc,
        exec: static_exec,
        lds: static_lds,
        sds: static_sds,
        simm: None,
        pc_offset: 0,
        stream: vec![],
        sgpr_co: static_co,
        warp_size: 32,
        scalar: false,
    };
    thread.vec_reg.default_lane = Some(0);
    thread.vcc.default_lane = Some(0);
    thread.exec.default_lane = Some(0);
    return thread;
}
