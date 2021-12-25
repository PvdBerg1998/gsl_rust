use std::os::raw::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub mod integration;
pub mod minimizer;
pub mod nonlinear_fit;

mod error;
pub use error::*;

pub mod bindings {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(deref_nullptr)]

    include!("../bindings.rs");
}

pub fn disable_error_handler() {
    unsafe {
        bindings::gsl_set_error_handler_off();
    }
}

unsafe extern "C" fn trampoline<F: FnMut(f64) -> f64>(x: f64, params: *mut c_void) -> f64 {
    let f: &mut F = &mut *(params as *mut F);
    match catch_unwind(AssertUnwindSafe(move || f(x))) {
        Ok(y) => y,
        Err(_) => f64::NAN,
    }
}
