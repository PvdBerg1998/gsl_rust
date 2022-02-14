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

pub fn gamma(x: f64) -> Result<f64> {
    unsafe {
        let mut result = gsl_sf_result { val: 0.0, err: 0.0 };
        GSLError::from_raw(gsl_sf_gamma_e(x, &mut result))?;
        Ok(result.val)
    }
}

pub fn gamma_complex(z: Complex64) -> Result<Complex64> {
    unsafe {
        let mut ln_r = gsl_sf_result { val: 0.0, err: 0.0 };
        let mut arg = gsl_sf_result { val: 0.0, err: 0.0 };
        GSLError::from_raw(gsl_sf_lngamma_complex_e(z.re, z.im, &mut ln_r, &mut arg))?;
        Ok(Complex64::from_polar(ln_r.val.exp(), arg.val))
    }
}

pub fn hurwitz_zeta(s: f64, a: f64) -> Result<f64> {
    unsafe {
        let mut result = gsl_sf_result { val: 0.0, err: 0.0 };
        GSLError::from_raw(gsl_sf_hzeta_e(s, a, &mut result))?;
        Ok(result.val)
    }
}

// This shows very poor convengence
// pub fn hurwitz_zeta_complex(s: Complex64, a: f64, n: usize) -> Result<Complex64> {
//     if s == Complex64::one() {
//         return Err(GSLError::Domain);
//     }
//     if s.re <= 0.0 {
//         return Err(GSLError::Domain);
//     }
//     if a <= 0.0 {
//         return Err(GSLError::Domain);
//     }

//     let a = Complex64::from(a);

//     let mut acc = Complex64::zero();
//     for k in 0..=n {
//         acc += (Complex64::from(k as f64) + a).powc(-s);
//     }
//     acc += (Complex64::from(n as f64) + a).powc(Complex64::one() - s) / (s - Complex64::one());

//     Ok(acc)
// }

pub fn hurwitz_zeta_complex(s: Complex64, a: f64) -> Result<Complex64> {
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
                * ((a.powi(2) + x.powi(2)).powf(rez / 2.0)))
    };

    let im_integrand = |x: f64| -> f64 {
        let t = (x / a).atan();
        let log = (a.powi(2) + x.powi(2)).ln();

        let rez = s.re;
        let imz = s.im;

        (-(0.5 * imz * log).sin() * (imz * t).cosh() * (rez * t).sin()
            + (rez * t).cos() * (0.5 * imz * log).cos() * (imz * t).sinh())
            / (((std::f64::consts::TAU * x).exp() - 1.0)
                * ((a.powi(2) + x.powi(2)).powf(rez / 2.0)))
    };

    let re_part2 = 2.0 * integration::qagiu(0.0, re_integrand)?;
    let im_part2 = 2.0 * integration::qagiu(0.0, im_integrand)?;
    let part2 = Complex64::new(re_part2, im_part2);

    Ok(part1 + part2)
}

#[test]
fn test_gamma() {
    disable_error_handler();

    approx::assert_abs_diff_eq!(
        gamma(5.0).unwrap(),
        gamma_complex(Complex64::from(5.0)).unwrap().re,
        epsilon = 1.0e-9
    );

    approx::assert_abs_diff_eq!(
        gamma_complex(Complex64::new(1.0, 1.0)).unwrap().re,
        0.49801566811835,
        epsilon = 1.0e-9
    );
    approx::assert_abs_diff_eq!(
        gamma_complex(Complex64::new(1.0, 1.0)).unwrap().im,
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

            let rust_zeta = hurwitz_zeta_complex(Complex64::from(s), a).unwrap().re;
            let gsl_zeta = hurwitz_zeta(s, a).unwrap();

            approx::assert_abs_diff_eq!(rust_zeta, gsl_zeta, epsilon = 1.0e-9);
        }
    }
}

#[test]
fn test_hurwitz_zeta() {
    disable_error_handler();

    let z = hurwitz_zeta_complex(Complex64::new(5.0, 5.0), 1.5).unwrap();

    // Impressive accuracy of the approximation!
    approx::assert_abs_diff_eq!(z.re, -0.057538922474198148085);
    approx::assert_abs_diff_eq!(z.im, -0.108623432041536612067)
}
