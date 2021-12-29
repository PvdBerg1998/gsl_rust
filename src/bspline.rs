/*
    bspline.rs
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
use linear_fit::*;
use std::fmt;

/// `k + 1` is equal to the spline order
pub fn fit_bspline(
    k: usize,
    a: f64,
    b: f64,
    nbreak: usize,
    x: &[f64],
    y: &[f64],
) -> Result<BSpline> {
    unsafe {
        if k == 0 {
            return Err(GSLError::Invalid);
        }
        if nbreak < 2 {
            return Err(GSLError::Invalid);
        }
        if b <= a {
            return Err(GSLError::Domain);
        }

        // Allocate workspace
        let workspace = gsl_bspline_alloc(k as u64, nbreak as u64);
        assert!(!workspace.is_null());
        let ncoeffs = gsl_bspline_ncoeffs(workspace) as usize;

        // Calculate knots
        // The domain is equal to [x_k, x_{n-k}]
        // Thus if we want to pass through a and b,
        // we need to have k repetitions of the first and last knot.
        // knots(1:k) = a
        // knots(k+1:k+l-1) = a + i*delta, i = 1 .. l - 1
        // knots(n+1:n+k) = b
        GSLError::from_raw(gsl_bspline_knots_uniform(a, b, workspace))?;

        // Cache vector for basis spline values
        let mut b = Vector::zeroes(ncoeffs);

        // Build the linear system and fit it
        let fit = linear_fit(ncoeffs, x, y, |&x, p| {
            // Evaluate all basis splines at this position and store them in b
            GSLError::from_raw(gsl_bspline_eval(x, b.as_gsl_mut(), workspace))?;
            p.copy_from_slice(&b);
            Ok(())
        })?;

        Ok(BSpline { fit, workspace })
    }
}

pub struct BSpline {
    pub fit: FitResult,
    // We own the associated workspace as it contains a nontrivial amount of data
    workspace: *mut gsl_bspline_workspace,
}

impl BSpline {
    pub fn fit(k: usize, a: f64, b: f64, nbreak: usize, x: &[f64], y: &[f64]) -> Result<Self> {
        fit_bspline(k, a, b, nbreak, x, y)
    }

    pub fn eval<const DV: usize>(&self, x: &[f64]) -> Result<BSplineEvaluation<DV>> {
        unsafe {
            let ncoeffs = gsl_bspline_ncoeffs(self.workspace) as usize;

            let mut db = Matrix::zeroes(ncoeffs, DV + 1);
            let mut b = Vector::zeroes(ncoeffs);
            let c = gsl_vector::from(self.fit.params.as_ref());
            let covariance = gsl_matrix::from_slice(&self.fit.covariance, ncoeffs, ncoeffs);

            let mut y = vec![0.0; x.len()].into_boxed_slice();
            let mut y_err = vec![0.0; x.len()].into_boxed_slice();
            let mut dv = vec![[0.0; DV]; x.len()].into_boxed_slice();
            let mut dv_err = vec![[0.0; DV]; x.len()].into_boxed_slice();

            for (i, x) in x.iter().copied().enumerate() {
                // Evaluate all basis splines and their derivatives at this position,
                // and store them in db.
                GSLError::from_raw(gsl_bspline_deriv_eval(
                    x,
                    DV as u64,
                    db.as_gsl_mut(),
                    self.workspace,
                ))?;

                // 0th order derivative is special cased for convenience
                gsl_matrix_get_col(b.as_gsl_mut(), db.as_gsl(), 0);
                GSLError::from_raw(gsl_multifit_linear_est(
                    b.as_gsl(),
                    &c,
                    &covariance,
                    &mut y[i],
                    &mut y_err[i],
                ))?;

                // Then, take all the derivatives
                for j in 0..DV {
                    gsl_matrix_get_col(b.as_gsl_mut(), db.as_gsl(), j as u64 + 1);
                    GSLError::from_raw(gsl_multifit_linear_est(
                        b.as_gsl(),
                        &c,
                        &covariance,
                        &mut dv[i][j],
                        &mut dv_err[i][j],
                    ))?;
                }
            }

            Ok(BSplineEvaluation {
                y,
                y_err,
                dv,
                dv_err,
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BSplineEvaluation<const DV: usize> {
    pub y: Box<[f64]>,
    pub y_err: Box<[f64]>,
    pub dv: Box<[[f64; DV]]>,
    pub dv_err: Box<[[f64; DV]]>,
}

impl<const DV: usize> BSplineEvaluation<DV> {
    pub fn dv_flat<'a>(&'a self) -> &'a [f64] {
        // Safety: [[T; N], [T; N], ...] is equal to [T; M*N]
        // For DV=0 the pointer will still be aligned thanks to Box
        unsafe { std::slice::from_raw_parts(self.dv.as_ptr() as *const _, self.dv.len() * DV) }
    }

    pub fn dv_err_flat<'a>(&'a self) -> &'a [f64] {
        // Safety: see dv_flat
        unsafe {
            std::slice::from_raw_parts(self.dv_err.as_ptr() as *const _, self.dv_err.len() * DV)
        }
    }
}

impl fmt::Debug for BSpline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BSpline")
            .field("fit", &self.fit)
            .finish_non_exhaustive()
    }
}

impl Drop for BSpline {
    fn drop(&mut self) {
        unsafe {
            gsl_bspline_free(self.workspace);
        }
    }
}

// GSL is thread safe
unsafe impl Send for BSpline {}
unsafe impl Sync for BSpline {}

#[test]
fn test_bspline_fit_1() {
    disable_error_handler();

    fn model(a: f64, b: f64, x: f64) -> f64 {
        (a * x + b).sin()
    }

    fn model_dv(a: f64, b: f64, x: f64) -> f64 {
        a * (a * x + b).cos()
    }

    let a = 10.0;
    let b = 2.0;

    let x = (0..100).map(|x| x as f64 / 100.0).collect::<Vec<_>>();
    let y = x.iter().map(|&x| model(a, b, x)).collect::<Vec<_>>();

    let spline = fit_bspline(4, 0.0, 1.0, 10, &x, &y).unwrap();

    dbg!(&spline);

    assert!(spline.fit.r_squared > 0.99999);

    // Check interpolation accuracy
    let interpolated_x = (0..1000).map(|x| x as f64 / 1000.0).collect::<Vec<_>>();
    let interpolated_y = spline.eval::<0>(&interpolated_x).unwrap().y;
    for (x, interpolated_y) in interpolated_x.iter().zip(interpolated_y.iter()) {
        let y = model(a, b, *x);
        approx::assert_abs_diff_eq!(y, interpolated_y, epsilon = 1.0e-2);
    }

    // Check derivative
    let derivative = spline.eval::<1>(&x).unwrap().dv;
    for (x, [spline_dv]) in x.iter().zip(derivative.iter()) {
        let dv = model_dv(a, b, *x);
        // Derivative amplifies any fitting errors so we have to be a bit more lenient
        approx::assert_abs_diff_eq!(dv, spline_dv, epsilon = 0.3);
    }
}

#[test]
fn miri_test_bspline_eval_flat() {
    let eval = BSplineEvaluation {
        y: vec![].into_boxed_slice(),
        y_err: vec![].into_boxed_slice(),
        dv: vec![[0.0], [1.0], [2.0]].into_boxed_slice(),
        dv_err: vec![[0.0], [1.0], [2.0]].into_boxed_slice(),
    };
    assert_eq!(eval.dv_flat(), &[0.0, 1.0, 2.0]);
    assert_eq!(eval.dv_err_flat(), &[0.0, 1.0, 2.0]);

    let eval = BSplineEvaluation {
        y: vec![].into_boxed_slice(),
        y_err: vec![].into_boxed_slice(),
        dv: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]].into_boxed_slice(),
        dv_err: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]].into_boxed_slice(),
    };
    assert_eq!(eval.dv_flat(), &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    assert_eq!(eval.dv_err_flat(), &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

    let eval = BSplineEvaluation {
        y: vec![].into_boxed_slice(),
        y_err: vec![].into_boxed_slice(),
        dv: vec![[]].into_boxed_slice(),
        dv_err: vec![[]].into_boxed_slice(),
    };
    assert_eq!(eval.dv_flat(), &[]);
    assert_eq!(eval.dv_err_flat(), &[]);
}

#[test]
fn test_invalid_params() {
    disable_error_handler();

    // 0th order spline
    fit_bspline(0, 0.0, 1.0, 10, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]).unwrap_err();

    // Too few breakpoints
    fit_bspline(4, 0.0, 1.0, 1, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]).unwrap_err();

    // No data
    fit_bspline(4, 0.0, 1.0, 10, &[], &[]).unwrap_err();

    // Empty domain
    fit_bspline(4, 0.0, 0.0, 10, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]).unwrap_err();
}
