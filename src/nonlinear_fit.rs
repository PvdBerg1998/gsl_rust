/*
    nonlinear_fit.rs
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
use std::panic::{catch_unwind, AssertUnwindSafe};

pub type HyperParams = gsl_multifit_nlinear_parameters;

pub fn nonlinear_fit<X, F: FnMut(&X, [f64; P]) -> Result<f64>, const P: usize>(
    p0: [f64; P],
    x: &[X],
    y: &[f64],
    f: F,
) -> Result<FitResult<P>> {
    nonlinear_fit_ext(
        100,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        p0,
        x,
        y,
        f,
        None::<fn(FitCallback<P>)>,
    )
}

pub fn nonlinear_fit_ext<
    X,
    F: FnMut(&X, [f64; P]) -> Result<f64>,
    C: FnMut(FitCallback<P>),
    const P: usize,
>(
    max_iter: usize,
    xtol: f64,
    gtol: f64,
    ftol: f64,
    hyper_params: HyperParams,
    p0: [f64; P],
    x: &[X],
    y: &[f64],
    f: F,
    mut callback: Option<C>,
) -> Result<FitResult<P>> {
    unsafe {
        if P == 0 {
            return Err(GSLError::Invalid);
        }
        if x.len() == 0 || y.len() == 0 {
            return Err(GSLError::Invalid);
        }

        // Amount of datapoints
        assert_eq!(x.len(), y.len());
        let n = x.len() as u64;

        // Allocate workspace
        let workspace = guard(
            gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &hyper_params, n, P as u64),
            |workspace| {
                gsl_multifit_nlinear_free(workspace);
            },
        );
        assert!(!workspace.is_null());

        // Information we need inside the trampolines
        let mut ffi_params = FFIParams {
            f,
            x,
            y,
            error: GSL_SUCCESS,
            panicked: false,
        };

        // Function to be optimized
        let mut fdf = gsl_multifit_nlinear_fdf {
            f: Some(fit_f::<X, F, P>),
            df: None,
            fvv: None,
            n,
            p: P as u64,
            params: &mut ffi_params as *mut _ as *mut _,
            nevalf: 0,
            nevaldf: 0,
            nevalfvv: 0,
        };

        // Init workspace
        let param_guess = gsl_vector_from_ref(&p0);
        GSLError::from_raw(gsl_multifit_nlinear_init(
            &param_guess,
            &mut fdf,
            *workspace,
        ))?;

        // Initial cost function chi^2_0
        let mut chisq0 = 0.0f64;
        {
            let start_residuals = gsl_multifit_nlinear_residual(*workspace);
            GSLError::from_raw(gsl_blas_ddot(start_residuals, start_residuals, &mut chisq0))?;
        }

        let mut _info = 0i32;
        let status = gsl_multifit_nlinear_driver(
            max_iter as u64,
            xtol,
            gtol,
            ftol,
            if callback.is_some() {
                Some(fit_callback::<C, P>)
            } else {
                None
            },
            &mut callback as *mut _ as *mut c_void,
            &mut _info,
            *workspace,
        );

        if ffi_params.panicked {
            return Err(GSLError::BadFunction);
        }
        GSLError::from_raw(ffi_params.error)?; // Could disable this if you want less error checking
        GSLError::from_raw(status)?;

        /*

             Extract fit information

        */

        // Numerical fit results
        let fit_result = gsl_multifit_nlinear_position(*workspace);
        let fit_jacobian = gsl_multifit_nlinear_jac(*workspace);
        let fit_residuals = gsl_multifit_nlinear_residual(*workspace);

        // Fit evaluation statistics
        let fit_niter = gsl_multifit_nlinear_niter(*workspace);
        let fit_neval_f = fdf.nevalf;

        // Final cost function chi^2_1
        let mut chisq1 = 0.0f64;
        GSLError::from_raw(gsl_blas_ddot(fit_residuals, fit_residuals, &mut chisq1))?;

        // Calculate variance-covariance matrix
        let mut fit_covariance = Matrix::zeroes(P, P);
        GSLError::from_raw(gsl_multifit_nlinear_covar(
            fit_jacobian,
            0.0,
            fit_covariance.as_gsl_mut(),
        ))?;
        GSLError::from_raw(gsl_matrix_scale(
            fit_covariance.as_gsl_mut(),
            chisq1 / (n as f64 - P as f64),
        ))?;

        // Calculate mean and total sum of squares wrt mean
        let gsl_y = gsl_vector_from_ref(y);
        let mean = gsl_stats_mean(gsl_y.data, gsl_y.stride, n as u64);
        let tss = gsl_stats_tss_m(gsl_y.data, gsl_y.stride, n as u64, mean);

        let result = FitResult {
            params: gsl_vector_to_array(fit_result),
            covariance: fit_covariance.to_2d_array(),
            niter: fit_niter,
            neval_f: fit_neval_f,
            initial_residual_squared: chisq0,
            final_residual_squared: chisq1,
            mean,
            r_squared: 1.0 - chisq1 / tss,
        };

        Ok(result)
    }
}

struct FFIParams<'a, 'b, F, X> {
    f: F,
    x: &'a [X],
    y: &'b [f64],
    error: i32,
    panicked: bool,
}

unsafe extern "C" fn fit_f<X, F: FnMut(&X, [f64; P]) -> Result<f64>, const P: usize>(
    params: *const gsl_vector,
    ffi_params: *mut c_void,
    out: *mut gsl_vector,
) -> i32 {
    let ffi_params: &mut FFIParams<'_, '_, F, X> = &mut *(ffi_params as *mut _);
    let params = gsl_vector_to_array(params);

    for (i, (x, y)) in ffi_params.x.iter().zip(ffi_params.y.iter()).enumerate() {
        let val = catch_unwind(AssertUnwindSafe(|| (ffi_params.f)(x, params)));
        let err = match val {
            Ok(Ok(y)) => y,
            Ok(Err(e)) => {
                let e = e.into();
                ffi_params.error = e;
                return e;
            }
            Err(_) => {
                ffi_params.panicked = true;
                return GSL_EBADFUNC;
            }
        } - *y;
        gsl_vector_set(out, i as u64, err);
    }

    GSL_SUCCESS
}

/*
unsafe extern "C" fn fit_j<
    X,
    F: FnMut(&X, [f64; P]) -> Result<f64>,
    J: FnMut(&X, [f64; P]) -> Result<[f64; P]>,
    const P: usize,
>(
    params: *const gsl_vector,
    ffi_params: *mut c_void,
    out: *mut gsl_matrix,
) -> i32 {
    let ffi_params: &mut FFIParams<'_, '_, F, J, X> = &mut *(ffi_params as *mut _);
    let params = gsl_vector_to_array(params);

    for (i, x) in ffi_params.x.iter().enumerate() {
        let val = catch_unwind(AssertUnwindSafe(|| (ffi_params.j)(x, params)));

        let dvs = match val {
            Ok(Ok(dvs)) => dvs,
            Ok(Err(e)) => {
                let e = e.into();
                ffi_params.error = e;
                return e;
            }
            Err(_) => {
                ffi_params.panicked = true;
                return GSL_EBADFUNC;
            }
        };

        for (j, &dv) in dvs.iter().enumerate() {
            gsl_matrix_set(out, i as u64, j as u64, dv);
        }
    }

    GSL_SUCCESS
}
*/

unsafe extern "C" fn fit_callback<C: FnMut(FitCallback<P>), const P: usize>(
    iter: u64,
    callback: *mut c_void,
    workspace: *const gsl_multifit_nlinear_workspace,
) {
    let callback: &mut Option<C> = &mut *(callback as *mut _);
    let callback = match callback {
        Some(callback) => callback,
        None => std::hint::unreachable_unchecked(),
    };

    let params = gsl_multifit_nlinear_position(workspace);

    let residuals = gsl_multifit_nlinear_residual(workspace);
    let mut chisq = 0.0f64;
    let _ = gsl_blas_ddot(residuals, residuals, &mut chisq);

    let mut rcond = 0.0;
    let _ = gsl_multifit_nlinear_rcond(&mut rcond, workspace);

    let _ = catch_unwind(AssertUnwindSafe(|| {
        callback(FitCallback {
            iter: iter as usize,
            params: gsl_vector_to_array(params),
            cond: 1.0 / rcond,
            residual_squared: chisq,
        });
    }));
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FitCallback<const P: usize> {
    pub iter: usize,
    pub params: [f64; P],
    pub cond: f64,
    pub residual_squared: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FitResult<const P: usize> {
    pub params: [f64; P],
    pub covariance: [[f64; P]; P],
    pub niter: u64,
    pub neval_f: u64,
    pub initial_residual_squared: f64,
    pub final_residual_squared: f64,
    pub mean: f64,
    pub r_squared: f64,
}

impl Default for HyperParams {
    fn default() -> Self {
        unsafe { gsl_multifit_nlinear_default_parameters() }
    }
}

#[test]
fn test_nlfit_1() {
    disable_error_handler();

    for i in 0..10 {
        fn model(a: f64, b: f64, x: f64) -> f64 {
            a + b * x + (a * b) * x.powi(2)
        }

        let a = 1.0 + i as f64;
        let b = 1.0 + i as f64;

        let x = (0..1000).map(|x| x as f64 / 100.0).collect::<Vec<_>>();
        let y = x.iter().map(|&x| model(a, b, x)).collect::<Vec<_>>();

        let fit = nonlinear_fit_ext(
            1000,
            1.0e-9,
            1.0e-9,
            1.0e-9,
            HyperParams::default(),
            [10.0, 5.0],
            &x,
            &y,
            |&x, [a, b]| Ok(model(a, b, x)),
            Some(|callback| {
                dbg!(callback);
            }),
        )
        .unwrap();

        dbg!(fit);

        approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-3);
        approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-3);
    }
}

#[test]
fn test_nlfit_2() {
    disable_error_handler();

    fn model(a: f64, b: f64, x: f64) -> f64 {
        (a * x + b).sin()
    }

    let a = 10.0;
    let b = 2.0;

    let x = (0..100).map(|x| x as f64 / 100.0).collect::<Vec<_>>();
    let y = x.iter().map(|&x| model(a, b, x)).collect::<Vec<_>>();

    let fit = nonlinear_fit_ext(
        1000,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [9.0, 1.0],
        &x,
        &y,
        |&x, [a, b]| Ok(model(a, b, x)),
        Some(|callback| {
            dbg!(callback);
        }),
    )
    .unwrap();

    dbg!(fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-3);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-3);
}

#[test]
fn test_nlfit_3() {
    disable_error_handler();
    fastrand::seed(0);

    fn model(a: f64, b: f64, c: f64, x: f64) -> f64 {
        a * (-b * x).exp() + c
    }

    let a = 5.0;
    let b = 1.5;
    let c = 1.0;

    let x = (0..100).map(|x| x as f64 / 100.0 * 3.0).collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|&x| model(a, b, c, x) + 0.068 * (fastrand::f64() * 2.0 - 1.0))
        .collect::<Vec<_>>();

    let fit = nonlinear_fit_ext(
        1000,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [1.0, 1.0, 0.0],
        &x,
        &y,
        |&x, [a, b, c]| Ok(model(a, b, c, x)),
        Some(|callback| {
            dbg!(callback);
        }),
    )
    .unwrap();

    dbg!(fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-2);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-2);
}

#[test]
fn test_nlfit_panic() {
    disable_error_handler();

    let fit = nonlinear_fit_ext(
        1000,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [1.0],
        &[0, 1, 2],
        &[0.0; 3],
        |_, [_]| panic!(),
        Some(|callback| {
            dbg!(callback);
        }),
    )
    .unwrap_err();

    assert_eq!(fit, GSLError::BadFunction);
}

#[test]
fn test_nlfit_error() {
    disable_error_handler();

    let fit = nonlinear_fit_ext(
        1000,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [1.0],
        &[0, 1, 2],
        &[0.0; 3],
        |_, [_]| Err(GSLError::Fault),
        Some(|callback| {
            dbg!(callback);
        }),
    )
    .unwrap_err();

    assert_eq!(fit, GSLError::Fault);
}

#[test]
fn test_invalid_params() {
    disable_error_handler();

    // No iterations
    nonlinear_fit_ext(
        0,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [1.0],
        &[0, 1, 2],
        &[0.0; 3],
        |_, [_]| Ok(0.0),
        Some(|callback| {
            dbg!(callback);
        }),
    )
    .unwrap_err();

    // No parameters
    nonlinear_fit_ext(
        1000,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [],
        &[0, 1, 2],
        &[0.0; 3],
        |_, []| Ok(0.0),
        Some(|callback| {
            dbg!(callback);
        }),
    )
    .unwrap_err();

    // No data
    nonlinear_fit_ext(
        0,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [1.0],
        &[],
        &[],
        |_: &(), [_]| Ok(0.0),
        Some(|callback| {
            dbg!(callback);
        }),
    )
    .unwrap_err();
}
