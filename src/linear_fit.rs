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

/*

    Linear fitting does not make use of const generics,
    because it is used as a tool for e.g. BSpline fitting.
    This requires a runtime amount of parameters.
    Also, these applications can require a relatively large amount of parameters,
    making stack allocation less attractive.
    The convenience function `linear_fit_p` is available to add compile time checking to `f`.

*/

pub fn linear_fit_p<X, F: FnMut(&X) -> Result<[f64; P]>, const P: usize>(
    x: &[X],
    y: &[f64],
    mut f: F,
) -> Result<FitResult> {
    linear_fit(P, x, y, |x, p| {
        let params = f(x)?;
        p.copy_from_slice(&params);
        Ok(())
    })
}

pub fn polynomial_basis<const P: usize>(x: &f64) -> Result<[f64; P]> {
    let mut basis = [*x; P];
    for i in 0..P {
        basis[i] = basis[i].powi(i as i32);
    }
    Ok(basis)
}

pub fn linear_fit<X, F: FnMut(&X, &mut [f64]) -> Result<()>>(
    p: usize,
    x: &[X],
    y: &[f64],
    mut f: F,
) -> Result<FitResult> {
    unsafe {
        if p == 0 {
            return Err(GSLError::Invalid);
        }
        if x.len() == 0 || y.len() == 0 {
            return Err(GSLError::Invalid);
        }
        if x.len() != y.len() {
            return Err(GSLError::Invalid);
        }

        // Amount of datapoints
        let n = x.len();

        // Allocate workspace
        let workspace = guard(gsl_multifit_linear_alloc(n as u64, p as u64), |workspace| {
            gsl_multifit_linear_free(workspace);
        });
        assert!(!workspace.is_null());

        // Prepare storage
        let mut c = Vector::zeroes(p);
        let mut covariance = Matrix::zeroes(p, p);

        // Prepare linear system matrix: system_ij = f_j(x_i)
        let data = x
            .iter()
            .map(|x| {
                let mut p = vec![0.0; p];
                f(x, &mut p)?;
                Ok(p)
            })
            .collect::<Result<Vec<_>>>()?;
        let system = Matrix::new(data.into_iter().flatten(), n, p);

        // Convert y data to GSL format
        let gsl_y = gsl_vector::from(y);

        // Solve the linear system using SVD
        let mut chisq = 0.0f64;
        GSLError::from_raw(gsl_multifit_linear(
            system.as_gsl(),
            &gsl_y,
            c.as_gsl_mut(),
            covariance.as_gsl_mut(),
            &mut chisq,
            *workspace,
        ))?;

        // Calculate mean and total sum of squares wrt mean
        let mean = stats::mean(y);
        let tss = gsl_stats_tss_m(gsl_y.data, gsl_y.stride, gsl_y.size, mean);

        let mut residuals = Vector::zeroes(x.len());
        GSLError::from_raw(gsl_multifit_linear_residuals(
            system.as_gsl(),
            &gsl_y,
            c.as_gsl(),
            residuals.as_gsl_mut(),
        ))?;

        Ok(FitResult {
            params: c.to_boxed_slice(),
            covariance: covariance.to_boxed_slice(),
            residuals: residuals.to_boxed_slice(),
            residual_squared: chisq,
            mean,
            r_squared: 1.0 - chisq / tss,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitResult {
    pub params: Box<[f64]>,
    pub covariance: Box<[f64]>,
    pub residuals: Box<[f64]>,
    pub residual_squared: f64,
    pub mean: f64,
    pub r_squared: f64,
}

impl FitResult {
    pub fn covariance(&self, i: usize, j: usize) -> f64 {
        (self.covariance)[i * self.params.len() + j]
    }

    pub fn uncertainty(&self, i: usize) -> f64 {
        self.covariance(i, i).sqrt()
    }
}

/*
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FitResultP<const P: usize> {
    pub params: [f64; P],
    pub covariance: [[f64; P]; P],
    pub residual_squared: f64,
    pub mean: f64,
    pub r_squared: f64,
}
*/

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

    // let fit = linear_fit(3, &x, &y, |&x, p| {
    //     p.copy_from_slice(&[1.0, x, x.powi(2)]);
    //     Ok(())
    // })
    // .unwrap();

    let fit = linear_fit_p(&x, &y, polynomial_basis::<3>).unwrap();

    dbg!(&fit);

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

    let fit = linear_fit(3, &x, &y, |&x, p| {
        p.copy_from_slice(&[1.0, x, x.powi(2)]);
        Ok(())
    })
    .unwrap();

    dbg!(&fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-2);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-2);
    approx::assert_abs_diff_eq!(fit.params[2], c, epsilon = 1.0e-2);
}

#[test]
fn test_invalid_params() {
    disable_error_handler();

    // No data
    linear_fit(3, &[], &[], |&x, p| {
        p.copy_from_slice(&[1.0, x, x.powi(2)]);
        Ok(())
    })
    .unwrap_err();

    // No params
    linear_fit(0, &[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0], |&_, _| Ok(())).unwrap_err();
}
