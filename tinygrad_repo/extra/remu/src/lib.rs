use crate::state::StateSnapshot;
use crate::work_group::{WaveContext, WorkGroup};
use std::os::raw::c_char;
use std::slice;
mod helpers;
mod rdna3;
mod state;
mod thread;
mod work_group;

#[no_mangle]
pub extern "C" fn run_asm(lib: *const c_char, lib_sz: u32, gx: u32, gy: u32, gz: u32, lx: u32, ly: u32, lz: u32, args_ptr: *const u64) -> i32 {
    if lib.is_null() || (lib_sz % 4) != 0 {
        panic!("Pointer is null or length is not properly aligned to 4 bytes");
    }
    let kernel = unsafe { slice::from_raw_parts(lib as *const u32, (lib_sz / 4) as usize).to_vec() };
    let dispatch_dim = match (gy != 1, gz != 1) {
        (true, true) => 3,
        (true, false) => 2,
        _ => 1,
    };
    for gx in 0..gx {
        for gy in 0..gy {
            for gz in 0..gz {
                let mut wg = WorkGroup::new(dispatch_dim, [gx, gy, gz], [lx, ly, lz], &kernel, args_ptr);
                if let Err(err) = wg.exec_waves() {
                    return err;
                }
            }
        }
    }
    0
}

// FFI functions for single-stepping comparison tests

#[no_mangle]
pub extern "C" fn wave_create(lib: *const c_char, lib_sz: u32, n_lanes: u32) -> *mut WaveContext {
    if lib.is_null() || (lib_sz % 4) != 0 { return std::ptr::null_mut(); }
    let kernel = unsafe { slice::from_raw_parts(lib as *const u32, (lib_sz / 4) as usize).to_vec() };
    Box::into_raw(Box::new(WaveContext::new(kernel, n_lanes as usize)))
}

#[no_mangle]
pub extern "C" fn wave_step(ctx: *mut WaveContext) -> i32 {
    if ctx.is_null() { return -99; }
    unsafe { (*ctx).step() }
}

#[no_mangle]
pub extern "C" fn wave_get_snapshot(ctx: *const WaveContext, out: *mut StateSnapshot) {
    if ctx.is_null() || out.is_null() { return; }
    unsafe { *out = (*ctx).get_snapshot(); }
}

#[no_mangle]
pub extern "C" fn wave_set_sgpr(ctx: *mut WaveContext, idx: u32, val: u32) {
    if ctx.is_null() || idx >= 128 { return; }
    unsafe { (*ctx).scalar_reg[idx as usize] = val; }
}

#[no_mangle]
pub extern "C" fn wave_set_vgpr(ctx: *mut WaveContext, lane: u32, idx: u32, val: u32) {
    if ctx.is_null() || lane >= 32 || idx >= 256 { return; }
    unsafe { (*ctx).vec_reg.get_lane_mut(lane as usize)[idx as usize] = val; }
}

#[no_mangle]
pub extern "C" fn wave_init_lds(ctx: *mut WaveContext, size: u32) {
    if ctx.is_null() { return; }
    unsafe { (*ctx).lds.data.resize(size as usize, 0); }
}

#[no_mangle]
pub extern "C" fn wave_free(ctx: *mut WaveContext) {
    if !ctx.is_null() { unsafe { drop(Box::from_raw(ctx)); } }
}
