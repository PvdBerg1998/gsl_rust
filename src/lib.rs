/*
    lib.rs : gsl_rust. A small, safe Rust wrapper around the GSL.
    Copyright (C) 2021 Pim van den Berg

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

use std::os::raw::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub mod bspline;
pub mod fft;
pub mod filter;
pub mod integration;
pub mod interpolation;
pub mod linear_fit;
pub mod minimizer;
pub mod nonlinear_fit;
pub mod sorting;
pub mod special;
pub mod stats;

mod data;
pub(crate) use data::*;
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
