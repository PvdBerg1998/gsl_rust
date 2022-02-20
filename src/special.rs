/*
    special.rs
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
use num_complex::Complex64;

pub fn gamma(x: f64) -> Result<ValWithError<f64>> {
    unsafe {
        let mut result = gsl_sf_result { val: 0.0, err: 0.0 };
        GSLError::from_raw(gsl_sf_gamma_e(x, &mut result))?;
        Ok(result.into())
    }
}

pub fn ln_gamma_complex(z: Complex64) -> Result<ValWithError<Complex64>> {
    unsafe {
        let mut ln_r = gsl_sf_result { val: 0.0, err: 0.0 };
        let mut arg = gsl_sf_result { val: 0.0, err: 0.0 };
        GSLError::from_raw(gsl_sf_lngamma_complex_e(z.re, z.im, &mut ln_r, &mut arg))?;

        Ok(ValWithError {
            val: Complex64::from_polar(ln_r.val, arg.val),
            err: Complex64::from_polar(ln_r.err, arg.err),
        })
    }
}

pub fn gamma_complex(z: Complex64) -> Result<ValWithError<Complex64>> {
    unsafe {
        let mut ln_r = gsl_sf_result { val: 0.0, err: 0.0 };
        let mut arg = gsl_sf_result { val: 0.0, err: 0.0 };
        GSLError::from_raw(gsl_sf_lngamma_complex_e(z.re, z.im, &mut ln_r, &mut arg))?;

        Ok(ValWithError {
            val: Complex64::from_polar(ln_r.val.exp(), arg.val),
            err: Complex64::from_polar(ln_r.err.exp(), arg.err),
        })
    }
}

pub fn hurwitz_zeta(s: f64, a: f64) -> Result<ValWithError<f64>> {
    unsafe {
        let mut result = gsl_sf_result { val: 0.0, err: 0.0 };
        GSLError::from_raw(gsl_sf_hzeta_e(s, a, &mut result))?;
        Ok(result.into())
    }
}

#[test]
fn test_gamma() {
    disable_error_handler();

    approx::assert_abs_diff_eq!(
        gamma(5.0).unwrap().val,
        gamma_complex(Complex64::from(5.0)).unwrap().val.re,
        epsilon = 1.0e-9
    );

    approx::assert_abs_diff_eq!(
        gamma_complex(Complex64::new(1.0, 1.0)).unwrap().val.re,
        0.49801566811835,
        epsilon = 1.0e-9
    );
    approx::assert_abs_diff_eq!(
        gamma_complex(Complex64::new(1.0, 1.0)).unwrap().val.im,
        -0.1549498283018,
        epsilon = 1.0e-9
    );
}
