use half::f16;
use num_traits::{float::FloatCore, PrimInt, Unsigned};

pub fn bits<T>(word: T, hi: usize, lo: usize) -> T where T: PrimInt + Unsigned {
    assert!(hi >= lo);
    let width = hi - lo + 1;
    (word >> lo) & ((T::one() << width) - T::one())
}

pub fn nth(val: u32, pos: usize) -> u32 {
    (val >> (31 - pos as u32)) & 1
}
pub fn f16_lo(val: u32) -> f16 {
    f16::from_bits((val & 0xffff) as u16)
}
pub fn f16_hi(val: u32) -> f16 {
    f16::from_bits(((val >> 16) & 0xffff) as u16)
}

pub fn sign_ext(num: u64, bits: usize) -> i64 {
    let mut value = num;
    let is_negative = (value >> (bits - 1)) & 1 != 0;
    if is_negative {
        value |= !0 << bits;
    }
    value as i64
}

pub trait IEEEClass<T> {
    fn exponent(&self) -> T;
}
impl IEEEClass<u32> for f32 {
    fn exponent(&self) -> u32 {
        (self.to_bits() & 0b01111111100000000000000000000000) >> 23
    }
}
impl IEEEClass<u16> for f16 {
    fn exponent(&self) -> u16 {
        (self.to_bits() & 0b0111110000000000) >> 10
    }
}
impl IEEEClass<u64> for f64 {
    fn exponent(&self) -> u64 {
        (self.to_bits() & 0b0111111111110000000000000000000000000000000000000000000000000000) >> 52
    }
}

pub trait VOPModifier<T> {
    fn negate(&self, pos: usize, modifier: usize) -> T;
    fn absolute(&self, pos: usize, modifier: usize) -> T;
}
impl<T> VOPModifier<T> for T
where
    T: FloatCore,
{
    fn negate(&self, pos: usize, modifier: usize) -> T {
        match (modifier >> pos) & 1 {
            1 => match self.is_zero() {
                true => T::zero(),
                false => -*self,
            },
            _ => *self,
        }
    }
    fn absolute(&self, pos: usize, modifier: usize) -> T {
        match (modifier >> pos) & 1 {
            1 => self.abs(),
            _ => *self,
        }
    }
}

pub fn extract_mantissa(x: f64) -> f64 {
    if x.is_infinite() || x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let mantissa_mask: u64 = 0x000FFFFFFFFFFFFF;
    let bias: u64 = 1023;
    let normalized_mantissa_bits = (bits & mantissa_mask) | ((bias - 1) << 52);
    return f64::from_bits(normalized_mantissa_bits);
}
pub fn ldexp(x: f64, exp: i32) -> f64 {
    x * 2f64.powi(exp)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_extract_mantissa() {
        assert_eq!(extract_mantissa(2.0f64), 0.5);
    }

    #[test]
    fn test_normal_exponent() {
        assert_eq!(2.5f32.exponent(), 128);
        assert_eq!(1.17549435e-38f32.exponent(), 1);
        assert_eq!(f32::INFINITY.exponent(), 255);
        assert_eq!(f32::NEG_INFINITY.exponent(), 255);
    }

    #[test]
    fn test_denormal_exponent() {
        assert_eq!(1.0e-40f32.exponent(), 0);
        assert_eq!(1.0e-42f32.exponent(), 0);
        assert_eq!(1.0e-44f32.exponent(), 0);
        assert_eq!((1.17549435e-38f32 / 2.0).exponent(), 0);
    }

    #[test]
    fn test_normal_exponent_f16() {
        assert_eq!(f16::from_f32(3.14f32).exponent(), 16);
        assert_eq!(f16::NEG_INFINITY.exponent(), 31);
        assert_eq!(f16::INFINITY.exponent(), 31);
    }

    #[test]
    fn test_neg() {
        assert_eq!(0.3_f32.negate(0, 0b001), -0.3_f32);
        assert_eq!(0.3_f32.negate(1, 0b010), -0.3_f32);
        assert_eq!(0.3_f32.negate(2, 0b100), -0.3_f32);
        assert_eq!(0.3_f32.negate(0, 0b110), 0.3_f32);
        assert_eq!(0.3_f32.negate(1, 0b010), -0.3_f32);
        assert_eq!(0.0_f32.negate(0, 0b001).to_bits(), 0);
        assert_eq!((-0.0_f32).negate(0, 0b001).to_bits(), 0);
    }

    #[test]
    fn test_sign_ext() {
        assert_eq!(sign_ext(0b000000000000000101000, 21), 40);
        assert_eq!(sign_ext(0b111111111111111011000, 21), -40);
        assert_eq!(sign_ext(0b000000000000000000000, 21), 0);
        assert_eq!(sign_ext(0b111111111111111111111, 21), -1);
        assert_eq!(sign_ext(0b111000000000000000000, 21), -262144);
        assert_eq!(sign_ext(0b000111111111111111111, 21), 262143);
        assert_eq!(sign_ext(7608, 13), -584);
    }
}

use std::sync::LazyLock;
pub static DEBUG: LazyLock<bool> = LazyLock::new(|| std::env::var("DEBUG").map(|v| v.parse::<usize>().unwrap_or(0) >= 6).unwrap_or(false));

pub fn colored(st:&str, color:&str) -> String {
    let ansi_code = match color {
        "green" => format!("\x1b[{};2;39;176;139m", 38),
        "gray" => format!("\x1b[{};2;169;169;169m", 38),
        _ => format!("\x1b[{};2;255;255;255m", 38),
    };
    format!("{}{}{}", ansi_code, st, "\x1b[0m")
}

#[macro_export]
macro_rules! todo_instr {
    ($x:expr) => {{
        println!("{:08X}", $x);
        Err(1)
    }};
}
