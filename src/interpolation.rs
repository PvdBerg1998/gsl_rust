/*
    interpolation.rs
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

use crate::bindings::*;
use crate::*;
use drop_guard::guard;

pub fn interpolate(algorithm: Algorithm, x: &[f64], y: &[f64]) -> Result<Interpolation> {
    unsafe {
        if x.len() == 0 || y.len() == 0 {
            return Err(GSLError::Invalid);
        }
        if x.len() != y.len() {
            return Err(GSLError::Invalid);
        }

        // todo: data must be monotonic

        // Amount of datapoints
        let n = x.len();

        //let workspace = guard(gsl_interp_alloc(T, n))

        //Ok(Interpolation { raw })
        todo!()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Algorithm {
    Linear,
    Steffen,
}

pub struct Interpolation {
    workspace: *mut gsl_interp,
}

// GSL is thread safe
unsafe impl Send for Interpolation {}
unsafe impl Sync for Interpolation {}

impl Drop for Interpolation {
    fn drop(&mut self) {
        unsafe {
            todo!();
        }
    }
}
