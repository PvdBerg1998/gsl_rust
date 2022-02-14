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
use num_traits::One;

pub fn gamma(x: f64) -> Result<ValWithError<f64>> {
    unsafe {
        let mut result = gsl_sf_result { val: 0.0, err: 0.0 };
        GSLError::from_raw(gsl_sf_gamma_e(x, &mut result))?;
        Ok(result.into())
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

pub fn hurwitz_zeta_complex(s: Complex64, a: f64) -> Result<ValWithError<Complex64>> {
    if s == Complex64::one() {
        return Err(GSLError::Domain);
    }
    if a <= 0.0 {
        return Err(GSLError::Domain);
    }

    let c_a = Complex64::from(a);

    // NIST 25.11.29. The integral is split into a real and imaginary part.

    let part1 = Complex64::from(0.5) / c_a.powc(s)
        + c_a.powc(Complex64::one() - s) / (s - Complex64::one());

    let re_integrand = |x: f64| -> f64 {
        let t = (x / a).atan();
        let log = (a.powi(2) + x.powi(2)).ln();

        let rez = s.re;
        let imz = s.im;

        ((0.5 * imz * log).cos() * (imz * t).cosh() * (rez * t).sin()
            + (rez * t).cos() * (0.5 * imz * log).sin() * (imz * t).sinh())
            / (((std::f64::consts::TAU * x).exp() - 1.0)
                * ((a.powi(2) + x.powi(2)).powf(0.5 * rez)))
    };

    let im_integrand = |x: f64| -> f64 {
        let t = (x / a).atan();
        let log = (a.powi(2) + x.powi(2)).ln();

        let rez = s.re;
        let imz = s.im;

        (-(0.5 * imz * log).sin() * (imz * t).cosh() * (rez * t).sin()
            + (rez * t).cos() * (0.5 * imz * log).cos() * (imz * t).sinh())
            / (((std::f64::consts::TAU * x).exp() - 1.0)
                * ((a.powi(2) + x.powi(2)).powf(0.5 * rez)))
    };

    let re_part2 = integration::qagiu(0.0, re_integrand)?;
    let im_part2 = integration::qagiu(0.0, im_integrand)?;

    let part2 = Complex64::new(2.0 * re_part2.val, 2.0 * im_part2.val);
    let err = Complex64::new(2.0 * re_part2.err, 2.0 * im_part2.err);

    Ok(ValWithError {
        val: part1 + part2,
        err,
    })
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

#[test]
fn test_hurwitz_zeta_compare_rs_gsl() {
    disable_error_handler();

    for i in 1..100 {
        let s = i as f64 + 0.1;

        for a in 1..10 {
            let a = a as f64;

            let rust_zeta = hurwitz_zeta_complex(Complex64::from(s), a).unwrap().val.re;
            let gsl_zeta = hurwitz_zeta(s, a).unwrap().val;

            approx::assert_abs_diff_eq!(rust_zeta, gsl_zeta, epsilon = 1.0e-9);
        }
    }
}

#[test]
fn test_hurwitz_zeta() {
    disable_error_handler();

    let z = hurwitz_zeta_complex(Complex64::new(5.0, 5.0), 1.5)
        .unwrap()
        .val;

    // Impressive accuracy of the approximation!
    approx::assert_abs_diff_eq!(z.re, -0.057538922474198148085);
    approx::assert_abs_diff_eq!(z.im, -0.108623432041536612067)
}
