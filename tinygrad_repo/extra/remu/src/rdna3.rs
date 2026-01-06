use crate::helpers::{bits, sign_ext};

#[derive(Debug, PartialEq)]
pub enum Instruction {
    SOP2 { op: u8, ssrc0: u8, ssrc1: u8, sdst: u8 },
    SOP1 { op: u8, ssrc0: u8, sdst: u8 },
    SOPK { op: u8, simm16: i16, sdst: u8 },
    SOPP { op: u8, simm16: i16 },
    SOPC { op: u8, ssrc0: u8, ssrc1: u8 },

    SMEM { op: u8, sdata: u8, sbase: u8, offset: i32, soffset: u8, glc: bool, dlc: bool },

    VOP1 { op: u8, vdst: u8, src: u16 },
    VOP2 { op: u8, vdst: u8, vsrc: u8, src: u16 },
    VOPC { op: u8, vsrc: u8, src: u16 },
    VOP3 { op: u32, opsel: u8, cm: bool, abs: u8, vdst: u8, neg: u8, omod: u8, src2: u16, src1: u16, src0: u16 },
    VOP3SD { op: u32, cm: bool, sdst: u8, vdst: u8, neg: u8, omod: u8, src2: u16, src1: u16, src0: u16 },
    VOP3P { op: u8, vdst: u8, neg_hi: u8, opsel: u8, opsel_hi: u8, opsel_hi2: bool, cm: bool, src2: u16, src1: u16, src0: u16, neg: u8 },
    VOPD { opx: u8, opy: u8, vdstx: u8, vdsty: u8, vsrcx1: u8, vsrcy1: u8, srcx0: u16, srcy0: u16 },

    DS { op: u8, gds: bool, offset1: u8, offset0: u8, vdst: u8, data1: u8, data0: u8, addr: u8 },

    FLAT { op: u8, offset: u16, dlc: bool, glc: bool, slc: bool, seg: u8, addr: u8, data: u8, saddr: u8, sve: bool, vdst: u8 }
}

const VOP3SD_OPS: [u32; 7] = [764, 765, 766, 767, 768, 769, 770];

pub fn decode(word:u32, word1:Option<&u32>) -> Instruction {
    match bits(word, 31, 30) {
        0b11 => {
            let word = (*word1.unwrap() as u64) << 32 | (word as u64);
            match bits(word, 29, 26) {
                0b1101 => {
                    let sbase = (bits(word, 5, 0) as u8) << 1;
                    let sdata = bits(word, 12, 6) as u8;
                    let dlc = bits(word, 13, 13) != 0;
                    let glc = bits(word, 14, 14) != 0;
                    let op = bits(word, 25, 18) as u8;
                    let offset = sign_ext(bits(word, 52, 32), 21) as i32;
                    let soffset = bits(word, 63, 57) as u8;
                    Instruction::SMEM { sbase, sdata, dlc, glc, op, offset, soffset }
                }
                0b0101 => {
                    let op = bits(word, 25, 16) as u32;
                    let vdst = bits(word, 7, 0) as u8;
                    let cm = bits(word, 15, 15) != 0;
                    let src0 = bits(word, 40, 32) as u16;
                    let src1 = bits(word, 49, 41) as u16;
                    let src2 = bits(word, 58, 50) as u16;
                    let omod = bits(word, 60, 59) as u8;
                    let neg = bits(word, 63, 61) as u8;
                     if VOP3SD_OPS.contains(&op) {
                         let sdst = bits(word, 14, 8) as u8;
                         Instruction::VOP3SD { op, vdst, sdst, cm, src0, src1, src2, omod, neg }
                     } else {
                        let abs = bits(word, 10, 8) as u8;
                        let opsel = bits(word, 14, 11) as u8;
                        Instruction::VOP3 { opsel, cm, abs, vdst, neg, omod, src2, src1, src0, op }
                     }
                }
                0b0011 => {
                    let op = bits(word, 22, 16) as u8;
                    let vdst = bits(word, 7, 0) as u8;
                    let neg_hi = bits(word, 10, 8) as u8;
                    let opsel = bits(word, 13, 11) as u8;
                    let opsel_hi2 = bits(word, 14, 14) != 0;
                    let cm = bits(word, 15, 15) != 0;
                    let src0 = bits(word, 40, 32) as u16;
                    let src1 = bits(word, 49, 41) as u16;
                    let src2 = bits(word, 58, 50) as u16;
                    let opsel_hi = bits(word, 60, 59) as u8;
                    let neg = bits(word, 63, 61) as u8;
                    Instruction::VOP3P { op, vdst, neg_hi, opsel, opsel_hi, opsel_hi2, cm, src0, src1, src2, neg }
                }
                0b0110 => {
                    let offset0 = bits(word, 7, 0) as u8;
                    let offset1 = bits(word, 15, 8) as u8;
                    let gds = bits(word, 17, 17) != 0;
                    let op = bits(word, 25, 18) as u8;
                    let addr = bits(word, 39, 32) as u8;
                    let data0 = bits(word, 47, 40) as u8;
                    let data1 = bits(word, 55, 48) as u8;
                    let vdst = bits(word, 63, 56) as u8;
                    Instruction::DS { op, gds, offset1, offset0, vdst, data1, data0, addr }
                }
                0b0111 => {
                    let offset = bits(word, 12, 0) as u16;
                    let dlc = bits(word, 13, 13) != 0;
                    let glc = bits(word, 14, 14) != 0;
                    let slc = bits(word, 15, 15) != 0;
                    let seg = bits(word, 17, 16) as u8;
                    let op = bits(word, 24, 18) as u8;
                    let addr = bits(word, 39, 32) as u8;
                    let data = bits(word, 47, 40) as u8;
                    let saddr = bits(word, 54, 48) as u8;
                    let sve = bits(word, 55, 55) != 0;
                    let vdst = bits(word, 63, 56) as u8;
                    Instruction::FLAT { offset, dlc, glc, slc, seg, op, addr, data, saddr, sve, vdst }
                },
                0b0010 => {
                    let srcx0 = bits(word, 8, 0) as u16;
                    let vsrcx1 = bits(word, 16, 9) as u8;
                    let opy = bits(word, 21, 17) as u8;
                    let opx = bits(word, 25, 22) as u8;
                    let srcy0 = bits(word, 40, 32) as u16;
                    let vsrcy1 = bits(word, 48, 41) as u8;
                    let vdsty = bits(word, 55, 49) as u8;
                    let vdstx = bits(word, 63, 56) as u8;
                    Instruction::VOPD { opx, opy, vdstx, vdsty, vsrcx1, vsrcy1, srcx0, srcy0 }
                }
                _ => todo!(),
            }
        }
        0b10 => {
            let ssrc0 = bits(word, 7, 0) as u8;
            let ssrc1 = bits(word, 15, 8) as u8;
            let simm16 = word as i16;
            let sdst = bits(word, 22, 16) as u8;
            match bits(word, 29, 23) {
                0b1111101 => Instruction::SOP1 { ssrc0, sdst, op: bits(word, 15, 8) as u8 },
                0b1111110 => Instruction::SOPC { ssrc0, ssrc1, op: bits(word, 22, 16) as u8 },
                0b1111111 => Instruction::SOPP { simm16, op: bits(word, 22, 16) as u8 },
                _ => {
                    match bits(word, 29, 28) {
                        0b11 => Instruction::SOPK { simm16, sdst, op: bits(word, 27, 23) as u8 },
                        _ => Instruction::SOP2 { ssrc0, ssrc1, sdst, op: bits(word, 29, 23) as u8 }
                    }
                }
            }
        }
        _ => {
            let vdst = bits(word, 24, 17) as u8;
            let src = bits(word, 8, 0) as u16;
            let vsrc = bits(word, 16, 9) as u8;
            match bits(word, 30, 25) {
                0b111110 => Instruction::VOPC { vsrc, src, op: bits(word, 24, 17) as u8 },
                0b111111 => Instruction::VOP1 { vdst, src, op: vsrc },
                _ => Instruction::VOP2 { vdst, vsrc, src, op: bits(word, 30, 25) as u8 },
            }
        },
    }
}

#[cfg(test)]
mod test_rdna3 {
    use super::*;

    use std::process::{Stdio, Command};
    use std::io::{Result, Write};

    const LLVM_ARGS: &[&str; 3] = &["--arch=amdgcn", "--mcpu=gfx1100", "--triple=amdgcn-amd-amdhsa"];
    const OFFSET_PRG: usize = 16;
    const NULL: u8 = 124;

    fn llvm_assemble(asm: &str) -> Result<Vec<u8>> {
        let mut proc = Command::new("llvm-mc").args(LLVM_ARGS).args(["-filetype=obj", "-o", "-"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        proc.stdin.as_mut().unwrap().write_all(asm.as_bytes())?;
        let out = proc.wait_with_output()?;
        match out.status.success() {
            true => Ok(out.stdout),
            false => Err(std::io::Error::new(std::io::ErrorKind::Other, "llvm-mc err")),
        }
    }

    fn llvm_disassemble(code: &Vec<u8>) -> Result<String> {
        let mut proc = Command::new("llvm-objdump").args(LLVM_ARGS).args(["--disassemble", "-"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        proc.stdin.as_mut().unwrap().write_all(code)?;
        let out = proc.wait_with_output()?;
        match out.status.success() {
            true => Ok(String::from_utf8(out.stdout).unwrap()),
            false => Err(std::io::Error::new(std::io::ErrorKind::Other, "llvm-objdump err")),
        }
    }

    fn test_decode(asm: &str) -> Instruction {
        let lib = llvm_assemble(asm).unwrap();
        println!("{}", llvm_disassemble(&lib).unwrap());
        let stream: Vec<u32> = lib.chunks_exact(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).skip(OFFSET_PRG).collect();
        decode(stream[0], stream.get(1))
    }

    #[test]
    fn test_decode_smem() {
        assert_eq!(test_decode("s_load_b128 s[4:7], s[0:1], null"), Instruction::SMEM { op: 2, sdata: 4, sbase: 0, offset: 0, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s10, s[0:1], 0xc"), Instruction::SMEM { op: 0, sdata: 10, sbase: 0, offset: 0xc, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], s6"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: 6, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], glc dlc"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: NULL, glc: true, dlc: true });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], glc"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: NULL, glc: true, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], -20"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: -20, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], -1048576"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: -1048576, soffset: NULL, glc: false, dlc: false });
    }

    #[test]
    fn test_decode_salu() {
        assert_eq!(test_decode("s_add_u32 s1 s2 s3"), Instruction::SOP2 { op: 0, ssrc0: 2, ssrc1: 3, sdst: 1 });
        assert_eq!(test_decode("s_add_u32 vcc_hi exec_lo vcc_lo"), Instruction::SOP2 { op: 0, ssrc0: 126, ssrc1: 106, sdst: 107 });
        assert_eq!(test_decode("s_mov_b32 s1 -0.5"), Instruction::SOP1 { op: 0, ssrc0: 241, sdst: 1 });
        assert_eq!(test_decode("s_cmpk_eq_i32 s0 -30"), Instruction::SOPK { op: 3, sdst: 0, simm16: -30 });
        assert_eq!(test_decode("s_cmpk_eq_u32 s0 65535"), Instruction::SOPK { op: 9, sdst: 0, simm16: -1 });
        assert_eq!(test_decode("s_cmp_ge_i32 s1 s2"), Instruction::SOPC { op: 3, ssrc0: 1, ssrc1: 2 });
    }

    #[test]
    fn test_decode_valu_e32() {
        assert_eq!(test_decode("v_mov_b32 v0, v0"), Instruction::VOP1 { op: 1, vdst: 0, src: 256 });
        assert_eq!(test_decode("v_mov_b32 v0, s0"), Instruction::VOP1 { op: 1, vdst: 0, src: 0 });
        assert_eq!(test_decode("v_cmp_t_f32 v1, v0"), Instruction::VOPC { op: 31, vsrc: 0, src: 257 });
    }

    #[test]
    fn test_decode_valu_e64() {
        assert_eq!(test_decode("v_log_f32_e64 v2, |v0|"), Instruction::VOP3 { op: 423, vdst: 2, src0: 256, src1: 0, src2: 0, abs: 0b001, neg: 0, opsel: 0, omod: 0, cm: false });
        assert_eq!(test_decode("v_div_scale_f32 v2, s1, v0, v1, v2"), Instruction::VOP3SD { op: 764, cm: false, vdst: 2, sdst: 1, src0: 256, src1: 257, src2: 258, omod: 0, neg: 0 });
        assert_eq!(test_decode("v_pk_add_i16 v1, v0, v2"), Instruction::VOP3P { op: 2, vdst: 1, neg_hi: 0, opsel: 0, opsel_hi: 3, opsel_hi2: true, cm: false, src2: 0, src1: 258, src0: 256, neg: 0 });
    }

    #[test]
    fn test_decode_ds() {
        assert_eq!(test_decode("ds_add_u32 v2, v4 offset:16"), Instruction::DS { op: 0, gds: false, offset1: 0, offset0: 0x10, vdst: 0, data1: 0, data0: 4, addr: 2 });
        assert_eq!(test_decode("ds_store_b32 v0, v1, offset: 0x04 gds"), Instruction::DS { op: 13, gds: true, offset1: 0, offset0: 0x04, vdst: 0, data1: 0, data0: 1, addr: 0 });
        assert_eq!(test_decode("ds_load_u8 v1, v0 offset:16"), Instruction::DS { op: 58, gds: false, offset1: 0, offset0: 16, vdst: 1, data1: 0, data0: 0, addr: 0 });
    }
}
