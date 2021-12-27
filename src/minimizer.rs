/*
    minimizer.rs
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

pub fn minimize<F: FnMut(f64) -> f64, C: FnMut(MinimizerCallback)>(
    a: f64,
    b: f64,
    x0: f64,
    f: F,
) -> Result<f64> {
    minimize_ext(100, a, b, x0, 1.0e-9, 0.0, f, |_| {})
}

pub fn minimize_ext<F: FnMut(f64) -> f64, C: FnMut(MinimizerCallback)>(
    max_iter: usize,
    a: f64,
    b: f64,
    x0: f64,
    epsabs: f64,
    epsrel: f64,
    mut f: F,
    mut callback: C,
) -> Result<f64> {
    unsafe {
        let minimizer = guard(
            gsl_min_fminimizer_alloc(gsl_min_fminimizer_brent),
            |minimizer| {
                gsl_min_fminimizer_free(minimizer);
            },
        );
        assert!(!minimizer.is_null());

        let mut gsl_f = gsl_function_struct {
            function: Some(trampoline::<F>),
            params: &mut f as *mut _ as *mut _,
        };

        let status = gsl_min_fminimizer_set(*minimizer, &mut gsl_f, x0, a, b);
        GSLError::from_raw(status)?;

        let mut iter = 0;
        loop {
            let status = gsl_min_fminimizer_iterate(*minimizer);
            GSLError::from_raw(status)?;

            let x_lower = gsl_min_fminimizer_x_lower(*minimizer);
            let x_upper = gsl_min_fminimizer_x_upper(*minimizer);
            let y_lower = gsl_min_fminimizer_f_lower(*minimizer);
            let y_upper = gsl_min_fminimizer_f_upper(*minimizer);
            let x = gsl_min_fminimizer_x_minimum(*minimizer);
            let y = gsl_min_fminimizer_f_minimum(*minimizer);

            callback(MinimizerCallback {
                iter,
                lower_bound: (x_lower, y_lower),
                upper_bound: (x_upper, y_upper),
                minimum: (x, y),
            });

            let status = gsl_min_test_interval(x_lower, x_upper, epsabs, epsrel);
            if GSLError::from_raw(status).is_ok() {
                return Ok(x);
            }

            iter += 1;
            if iter >= max_iter {
                return Err(GSLError::MaxIteration);
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MinimizerCallback {
    pub iter: usize,
    pub lower_bound: (f64, f64),
    pub upper_bound: (f64, f64),
    pub minimum: (f64, f64),
}

#[test]
fn test_minimizer() {
    disable_error_handler();

    approx::assert_abs_diff_eq!(
        minimize_ext(100, 1.0, 6.0, 4.0, 1.0e-6, 0.0, |x| x.sin(), |_| {}).unwrap(),
        std::f64::consts::PI * 3.0 / 2.0,
        epsilon = 1.0e-6
    );
}

#[test]
fn test_invalid_params() {
    disable_error_handler();

    // No iterations
    minimize_ext(0, 1.0, 6.0, 4.0, 1.0e-6, 0.0, |x| x.sin(), |_| {}).unwrap_err();

    // Empty domain
    minimize_ext(100, 0.0, 0.0, 4.0, 1.0e-6, 0.0, |x| x.sin(), |_| {}).unwrap_err();

    // Nonsense guess
    minimize_ext(0, 1.0, 6.0, std::f64::NAN, 1.0e-6, 0.0, |x| x.sin(), |_| {}).unwrap_err();
}
