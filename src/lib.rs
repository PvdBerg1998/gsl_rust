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

use std::os::raw::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub mod bspline;
pub mod integration;
pub mod linear_fit;
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
use bindings::*;

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

unsafe fn alloc_filled_vector(data: &[f64]) -> *mut gsl_vector {
    let v = gsl_vector_alloc(data.len() as u64);
    assert!(!v.is_null());
    for (i, x) in data.iter().enumerate() {
        gsl_vector_set(v, i as u64, *x);
    }
    v
}

unsafe fn copy_from_vector<const N: usize>(v: *const gsl_vector) -> [f64; N] {
    let mut data = [0.0; N];
    for i in 0..N {
        data[i] = gsl_vector_get(v, i as u64);
    }
    data
}

unsafe fn copy_diagonal_from_matrix<const N: usize>(m: *const gsl_matrix) -> [f64; N] {
    let mut data = [0.0; N];
    for i in 0..N {
        data[i] = gsl_matrix_get(m, i as u64, i as u64);
    }
    data
}
