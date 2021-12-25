use crate::bindings::*;
use crate::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub type HyperParams = gsl_multifit_nlinear_parameters;

pub fn nonlinear_fit<
    X,
    F: FnMut(&X, [f64; P]) -> Result<f64>,
    //J: FnMut(X, [f64; P]) -> Result<[f64; P]>,
    C: FnMut(FitCallback<P>) -> (),
    const P: usize,
>(
    max_iter: usize,
    xtol: f64,
    gtol: f64,
    ftol: f64,
    hyper_params: HyperParams,
    p0: [f64; P],
    data: &[(X, f64)],
    f: F,
    //j: J,
    callback: C,
) -> Result<FitResult<P>> {
    unsafe {
        // Define fit method and parameters
        let fit_type = gsl_multifit_nlinear_trust;

        // Amount of datapoints
        let n = data.len() as u64;

        // Allocate workspace
        let workspace = gsl_multifit_nlinear_alloc(
            fit_type,
            &hyper_params as *const _,
            n,        // amount of datapoints
            P as u64, // amount of parameters
        );
        assert!(!workspace.is_null());

        // Initial parameter guess
        let param_guess = gsl_vector_alloc(P as u64);
        assert!(!param_guess.is_null());
        for (i, &p) in p0.iter().enumerate() {
            gsl_vector_set(param_guess, i as u64, p);
        }

        // Information we need inside the trampolines
        //let mut ffi_params = (f, j, data);
        let mut ffi_params = (f, data);

        // Function to be optimized
        let mut fdf = gsl_multifit_nlinear_fdf {
            f: Some(fit_f::<X, F, P>), // Some(fit_f::<X, F, J, P>),
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
        gsl_multifit_nlinear_init(param_guess, &mut fdf as *mut _, workspace);

        // Initial cost function chi^2_0
        let start_residuals = gsl_multifit_nlinear_residual(workspace);
        let mut chisq0 = 0.0f64;
        gsl_blas_ddot(start_residuals, start_residuals, &mut chisq0 as *mut _);
        drop(start_residuals);

        let mut _info = 0i32;
        let status = gsl_multifit_nlinear_driver(
            max_iter as u64,
            xtol,
            gtol,
            ftol,
            Some(fit_callback::<C, P>),
            &callback as *const _ as *mut c_void,
            &mut _info as *mut _,
            workspace,
        );

        /*

             Extract fit information

        */

        // Numerical fit results
        let fit_result = gsl_multifit_nlinear_position(workspace);
        let fit_jacobian = gsl_multifit_nlinear_jac(workspace);
        let fit_residuals = gsl_multifit_nlinear_residual(workspace);

        // Fit evaluation statistics
        let fit_niter = gsl_multifit_nlinear_niter(workspace);
        let fit_neval_f = fdf.nevalf;
        //let fit_neval_df = fdf.nevaldf;

        // Final cost function chi^2_1
        let mut chisq1 = 0.0f64;
        gsl_blas_ddot(fit_residuals, fit_residuals, &mut chisq1 as *mut _);

        // Allocate variance-covariance matrix
        let fit_covariance = gsl_matrix_alloc(P as u64, P as u64);
        assert!(!fit_covariance.is_null());

        // Calculate variance-covariance matrix
        gsl_multifit_nlinear_covar(fit_jacobian, 0.0, fit_covariance);
        gsl_matrix_scale(fit_covariance, chisq1 / (n as f64 - P as f64));

        // Calculate variance of data itself
        let mean = data.iter().map(|(_, y)| *y).sum::<f64>() / n as f64;
        let variance = data
            .iter()
            .map(|(_, y)| *y)
            .map(|y| (y - mean).powi(2))
            .sum::<f64>();

        // R^2 "goodness of fit"
        let r_squared = 1.0 - chisq1 / variance;

        // Extract fitted parameters
        let mut param_cache = [0.0; P];
        for i in 0..P {
            param_cache[i] = gsl_vector_get(fit_result, i as u64);
        }

        // Extract parameter uncertainties
        let mut param_sigma_cache = [0.0; P];
        for i in 0..P {
            param_sigma_cache[i] = gsl_matrix_get(fit_covariance, i as u64, i as u64).sqrt();
        }

        let result = FitResult {
            params: param_cache,
            uncertainties: param_sigma_cache,
            niter: fit_niter,
            neval_f: fit_neval_f,
            //neval_j: fit_neval_df,
            initial_residual_squared: chisq0,
            final_residual_squared: chisq1,
            final_residual_variance: chisq1 / (n as f64 - P as f64),
            mean,
            r_squared,
        };

        // Free memory
        gsl_matrix_free(fit_covariance);
        gsl_multifit_nlinear_free(workspace);
        gsl_vector_free(param_guess);

        GSLError::from_raw(status)?;
        Ok(result)
    }
}

unsafe extern "C" fn fit_f<
    X,
    F: FnMut(&X, [f64; P]) -> Result<f64>,
    //J: FnMut(X, [f64; P]) -> Result<[f64; P]>,
    const P: usize,
>(
    params: *const gsl_vector,
    ffi_params: *mut c_void,
    out: *mut gsl_vector,
) -> i32 {
    //let (f, _j, data): &mut (F, J, &[(X, f64)]) = &mut *(ffi_params as *mut _);
    let (f, data): &mut (F, &[(X, f64)]) = &mut *(ffi_params as *mut _);

    let mut param_cache = [0.0; P];
    for i in 0..P {
        param_cache[i] = gsl_vector_get(params, i as u64);
    }

    for (i, (x, y)) in data.iter().enumerate() {
        let val = catch_unwind(AssertUnwindSafe(|| f(x, param_cache)));
        let err = match val {
            Ok(Ok(y)) => y,
            Ok(Err(e)) => return e.into(),
            Err(_) => return GSLError::BadFunction.into(),
        } - *y;
        gsl_vector_set(out, i as u64, err);
    }

    GSL_SUCCESS
}

#[allow(dead_code)]
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
    let (_f, j, data): &mut (F, J, &[(X, f64)]) = &mut *(ffi_params as *mut _);

    let mut param_cache = [0.0; P];
    for i in 0..P {
        param_cache[i] = gsl_vector_get(params, i as u64);
    }

    for (i, (x, _y)) in data.iter().enumerate() {
        let val = catch_unwind(AssertUnwindSafe(|| j(x, param_cache)));

        let dvs = match val {
            Ok(Ok(dvs)) => dvs,
            Ok(Err(e)) => return e.into(),
            Err(_) => return GSLError::BadFunction.into(),
        };

        for (j, &dv) in dvs.iter().enumerate() {
            gsl_matrix_set(out, i as u64, j as u64, dv);
        }
    }

    GSL_SUCCESS
}

unsafe extern "C" fn fit_callback<C: FnMut(FitCallback<P>) -> (), const P: usize>(
    iter: u64,
    callback: *mut c_void,
    workspace: *const gsl_multifit_nlinear_workspace,
) {
    let params = gsl_multifit_nlinear_position(workspace);
    let mut param_cache = [0.0; P];
    for i in 0..P {
        param_cache[i] = gsl_vector_get(params, i as u64);
    }

    let residuals = gsl_multifit_nlinear_residual(workspace);
    let mut chisq = 0.0f64;
    gsl_blas_ddot(residuals, residuals, &mut chisq as *mut _);

    let mut rcond = 0.0;
    gsl_multifit_nlinear_rcond(&mut rcond as *mut _, workspace);

    let callback: &mut C = &mut *(callback as *mut _);
    let _ = catch_unwind(AssertUnwindSafe(|| {
        callback(FitCallback {
            iter: iter as usize,
            params: param_cache,
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
    pub uncertainties: [f64; P],
    pub niter: u64,
    pub neval_f: u64,
    //pub neval_j: u64,
    pub initial_residual_squared: f64,
    pub final_residual_squared: f64,
    pub final_residual_variance: f64,
    pub mean: f64,
    pub r_squared: f64,
}

impl Default for HyperParams {
    fn default() -> Self {
        unsafe { gsl_multifit_nlinear_default_parameters() }
    }
}

#[test]
fn test_fit() {
    disable_error_handler();

    for i in 0..10 {
        fn model(a: f64, b: f64, x: f64) -> f64 {
            a + b * x + (a * b) * x.powi(2)
        }

        let a = 1.0 + i as f64;
        let b = 1.0 + i as f64;

        let data = (0..1000)
            .map(|x| x as f64 / 100.0)
            .map(|x| (x, model(a, b, x)))
            .collect::<Vec<_>>();

        let fit = nonlinear_fit(
            1000,
            1.0e-9,
            1.0e-9,
            1.0e-9,
            HyperParams::default(),
            [10.0, 5.0],
            &data,
            |&x, [a, b]| Ok(model(a, b, x)),
            /*|x, [a, b]| {
                let dmda = 1.0 + b * x.powi(2);
                let dmdb = x + a * x.powi(2);
                Ok([dmda, dmdb])
            },*/
            |_| {},
        )
        .unwrap();

        //dbg!(fit);

        approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-3);
        approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-3);
    }
}

#[test]
fn test_fit_2() {
    disable_error_handler();

    fn model(a: f64, b: f64, x: f64) -> f64 {
        (a * x + b).sin()
    }

    let a = 10.0;
    let b = 2.0;

    let data = (0..100)
        .map(|x| x as f64 / 100.0)
        .map(|x| (x, model(a, b, x)))
        .collect::<Vec<_>>();

    let fit = nonlinear_fit(
        1000,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [9.0, 1.0],
        &data,
        |&x, [a, b]| Ok(model(a, b, x)),
        /*|x, [a, b]| {
            let dmda = (a * x + b).cos() * x;
            let dmdb = (a * x + b).cos();
            Ok([dmda, dmdb])
        },*/
        |_| {},
    )
    .unwrap();

    dbg!(fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-3);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-3);
}

#[test]
fn test_fit_3() {
    disable_error_handler();
    fastrand::seed(0);

    fn model(a: f64, b: f64, c: f64, x: f64) -> f64 {
        a * (-b * x).exp() + c
    }

    let a = 5.0;
    let b = 1.5;
    let c = 1.0;

    let data = (0..100)
        .map(|x| x as f64 / 100.0 * 3.0)
        .map(|x| (x, model(a, b, c, x) + 0.2 * (fastrand::f64() * 2.0 - 1.0)))
        .collect::<Vec<_>>();

    let fit = nonlinear_fit(
        1000,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        HyperParams::default(),
        [1.0, 1.0, 0.0],
        &data,
        |&x, [a, b, c]| Ok(model(a, b, c, x)),
        /*|x, [a, b]| {
            let dmda = (a * x + b).cos() * x;
            let dmdb = (a * x + b).cos();
            Ok([dmda, dmdb])
        },*/
        |callback| {
            dbg!(callback);
        },
    )
    .unwrap();

    dbg!(fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-2);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-2);
}
