use crate::work_group::WorkGroup;
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
