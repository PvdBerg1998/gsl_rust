/*
    linear_fit.rs
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

pub fn linear_fit<X, F: FnMut(&X) -> [f64; P], const P: usize>(
    x: &[X],
    y: &[f64],
    f: F,
) -> Result<FitResult<P>> {
    unsafe {
        // Amount of datapoints
        assert_eq!(x.len(), y.len());
        let n = x.len();

        // Allocate workspace
        let workspace = guard(gsl_multifit_linear_alloc(n as u64, P as u64), |workspace| {
            gsl_multifit_linear_free(workspace);
        });

        // Prepare storage
        let mut c = Vector::zeroes(P);
        let mut covariance = Matrix::zeroes(P, P);

        // Prepare linear system matrix: system_ij = f_j(x_i)
        let system = Matrix::new(x.iter().flat_map(f), n, P);

        // Convert y data to GSL format
        let gsl_y = gsl_vector_from_ref(y);

        // Solve the linear system using SVD
        let mut chisq = 0.0f64;
        let status = gsl_multifit_linear(
            system.as_gsl(),
            &gsl_y,
            c.as_gsl_mut(),
            covariance.as_gsl_mut(),
            &mut chisq,
            *workspace,
        );

        // Calculate mean and total sum of squares wrt mean
        let mean = gsl_stats_mean(gsl_y.data, gsl_y.stride, n as u64);
        let tss = gsl_stats_tss_m(gsl_y.data, gsl_y.stride, n as u64, mean);

        let result = FitResult {
            params: c.to_array(),
            covariance: covariance.to_2d_array(),
            residual_squared: chisq,
            mean,
            r_squared: 1.0 - chisq / tss,
        };

        GSLError::from_raw(status)?;
        Ok(result)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FitResult<const P: usize> {
    pub params: [f64; P],
    pub covariance: [[f64; P]; P],
    pub residual_squared: f64,
    pub mean: f64,
    pub r_squared: f64,
}

#[test]
fn test_fit_1() {
    disable_error_handler();

    fn model(a: f64, b: f64, c: f64, x: f64) -> f64 {
        a + b * x + c * x.powi(2)
    }

    let a = 10.0;
    let b = 2.0;
    let c = 2.0;

    let x = (0..100).map(|x| x as f64 / 10.0).collect::<Vec<_>>();
    let y = x.iter().map(|&x| model(a, b, c, x)).collect::<Vec<_>>();

    let fit = linear_fit(&x, &y, |&x| [1.0, x, x.powi(2)]).unwrap();

    dbg!(fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-6);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-6);
    approx::assert_abs_diff_eq!(fit.params[2], c, epsilon = 1.0e-6);
}

#[test]
fn test_fit_2() {
    disable_error_handler();
    fastrand::seed(0);

    fn model(a: f64, b: f64, c: f64, x: f64) -> f64 {
        a + b * x + c * x.powi(2)
    }

    let a = 10.0;
    let b = 2.0;
    let c = 2.0;

    let x = (0..100).map(|x| x as f64 / 10.0).collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|&x| model(a, b, c, x) + 0.068 * (fastrand::f64() * 2.0 - 1.0))
        .collect::<Vec<_>>();

    let fit = linear_fit(&x, &y, |&x| [1.0, x, x.powi(2)]).unwrap();

    dbg!(fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-2);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-2);
    approx::assert_abs_diff_eq!(fit.params[2], c, epsilon = 1.0e-2);
}
