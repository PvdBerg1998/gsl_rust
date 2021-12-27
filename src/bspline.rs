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
pub fn fit_bspline<const NCOEFFS: usize>(
    k: usize,
    a: f64,
    b: f64,
    x: &[f64],
    y: &[f64],
) -> Result<BSpline<NCOEFFS>> {
    unsafe {
        if k == 0 {
            return Err(GSLError::Invalid);
        }
        if NCOEFFS <= 2 || NCOEFFS - 2 <= k {
            return Err(GSLError::Invalid);
        }

        let nbreak = NCOEFFS as u64 + 2 - k as u64;

        // Allocate workspace
        let workspace = gsl_bspline_alloc(k as u64, nbreak);
        assert!(!workspace.is_null());

        // Calculate knots associated with uniformly distributed breakpoints
        GSLError::from_raw(gsl_bspline_knots_uniform(a, b, workspace))?;

        // Cache vector for basis spline values
        let mut b = Vector::zeroes(NCOEFFS);

        // Build the linear system and fit it
        let fit = linear_fit(x, y, |&x| {
            // Evaluate all basis splines at this position and store them in b
            GSLError::from_raw(gsl_bspline_eval(x, b.as_gsl_mut(), workspace))?;
            Ok(b.to_array::<NCOEFFS>())
        })?;

        Ok(BSpline { fit, workspace })
    }
}

pub struct BSpline<const NCOEFFS: usize> {
    pub fit: FitResult<NCOEFFS>,
    // We own the associated workspace as it contains a nontrivial amount of data
    workspace: *mut gsl_bspline_workspace,
}

impl<const NCOEFFS: usize> BSpline<NCOEFFS> {
    pub fn fit(k: usize, a: f64, b: f64, x: &[f64], y: &[f64]) -> Result<Self> {
        fit_bspline(k, a, b, x, y)
    }

    pub fn eval(&self, x: &[f64]) -> Result<Box<[(f64, f64)]>> {
        unsafe {
            let mut b = Vector::zeroes(NCOEFFS);
            let c = gsl_vector_from_ref(&self.fit.params);
            let covariance = gsl_matrix_from_ref(&self.fit.covariance);

            x.iter()
                .copied()
                .map(|x| {
                    // Evaluate all basis splines at this position and store them in b
                    GSLError::from_raw(gsl_bspline_eval(x, b.as_gsl_mut(), self.workspace))?;

                    // Evaluate fit at this x value
                    let mut y = 0.0f64;
                    let mut y_err = 0.0f64;
                    GSLError::from_raw(gsl_multifit_linear_est(
                        b.as_gsl(),
                        &c,
                        &covariance,
                        &mut y,
                        &mut y_err,
                    ))?;
                    Ok((y, y_err))
                })
                .collect()
        }
    }

    /// Returns y, y_err, dv, dv_err
    pub fn eval_derivative(&self, x: &[f64]) -> Result<Box<[(f64, f64, f64, f64)]>> {
        unsafe {
            // 0th and 1st derivatives
            let order = 2;

            let mut db = Matrix::zeroes(NCOEFFS, order);
            let mut b = Vector::zeroes(NCOEFFS);
            let c = gsl_vector_from_ref(&self.fit.params);
            let covariance = gsl_matrix_from_ref(&self.fit.covariance);

            x.iter()
                .copied()
                .map(|x| {
                    // Evaluate all basis splines and their derivatives at this position,
                    // and store them in db.
                    GSLError::from_raw(gsl_bspline_deriv_eval(
                        x,
                        1,
                        db.as_gsl_mut(),
                        self.workspace,
                    ))?;

                    // Since we calculated the 0th derivative as well,
                    // we also return it
                    gsl_matrix_get_col(b.as_gsl_mut(), db.as_gsl(), 0);
                    let mut y = 0.0f64;
                    let mut y_err = 0.0f64;
                    GSLError::from_raw(gsl_multifit_linear_est(
                        b.as_gsl(),
                        &c,
                        &covariance,
                        &mut y,
                        &mut y_err,
                    ))?;

                    // Then, get the first derivative
                    gsl_matrix_get_col(b.as_gsl_mut(), db.as_gsl(), 1);
                    let mut dv = 0.0f64;
                    let mut dv_err = 0.0f64;
                    GSLError::from_raw(gsl_multifit_linear_est(
                        b.as_gsl(),
                        &c,
                        &covariance,
                        &mut dv,
                        &mut dv_err,
                    ))?;

                    Ok((y, y_err, dv, dv_err))
                })
                .collect()
        }
    }
}

impl<const NCOEFFS: usize> fmt::Debug for BSpline<NCOEFFS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BSpline")
            .field("NCOEFFS", &NCOEFFS)
            .field("fit", &self.fit)
            .finish_non_exhaustive()
    }
}

impl<const NCOEFFS: usize> Drop for BSpline<NCOEFFS> {
    fn drop(&mut self) {
        unsafe {
            gsl_bspline_free(self.workspace);
        }
    }
}

// GSL is thread safe
unsafe impl<const NCOEFFS: usize> Send for BSpline<NCOEFFS> {}
unsafe impl<const NCOEFFS: usize> Sync for BSpline<NCOEFFS> {}

#[test]
fn test_bspline_fit_1() {
    disable_error_handler();

    const NCOEFFS: usize = 12;

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

    let spline = fit_bspline::<NCOEFFS>(4, 0.0, 1.0, &x, &y).unwrap();

    dbg!(&spline);

    assert!(spline.fit.r_squared > 0.99999);

    // Check interpolation accuracy
    let interpolated_x = (0..1000).map(|x| x as f64 / 1000.0).collect::<Vec<_>>();
    let interpolated_y = spline.eval(&interpolated_x).unwrap();
    for (x, (interpolated_y, _err)) in interpolated_x.iter().zip(interpolated_y.iter()) {
        let y = model(a, b, *x);
        approx::assert_abs_diff_eq!(y, interpolated_y, epsilon = 1.0e-2);
    }

    // Check derivative
    let derivative = spline.eval_derivative(&x).unwrap();
    for (x, (spline_y, _y_err, spline_dv, _dv_err)) in x.iter().zip(derivative.iter()) {
        let y = model(a, b, *x);
        let dv = model_dv(a, b, *x);

        // Derivative amplifies any fitting errors so we have to be a bit more lenient
        approx::assert_abs_diff_eq!(y, spline_y, epsilon = 1.0e-2);
        approx::assert_abs_diff_eq!(dv, spline_dv, epsilon = 0.3);
    }
}

#[test]
fn test_invalid_params() {
    disable_error_handler();

    // 0th order spline
    fit_bspline::<10>(0, 0.0, 1.0, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]).unwrap_err();

    // Too few coefficients
    fit_bspline::<1>(4, 0.0, 1.0, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]).unwrap_err();

    // No data
    fit_bspline::<10>(4, 0.0, 1.0, &[], &[]).unwrap_err();

    // Empty domain
    fit_bspline::<1>(4, 0.0, 0.0, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]).unwrap_err();
}
