use crate::bindings::*;
use crate::*;
use std::panic::{catch_unwind, RefUnwindSafe};

pub fn nonlinear_fit<
    F: Fn(f64, [f64; P]) -> Result<f64> + RefUnwindSafe,
    //J: Fn(f64, [f64; P]) -> Result<[f64; P]> + RefUnwindSafe,
    C: Fn(FitCallback<P>) -> () + RefUnwindSafe,
    const P: usize,
>(
    max_iter: usize,
    xtol: f64,
    gtol: f64,
    ftol: f64,
    p0: [f64; P],
    data: &[(f64, f64)],
    f: F,
    //j: J,
    callback: C,
) -> Result<[f64; P]> {
    unsafe {
        // Define fit method and parameters
        let fit_type = gsl_multifit_nlinear_trust;
        let fit_params = gsl_multifit_nlinear_default_parameters();

        // Amount of datapoints
        let n = data.len() as u64;

        // Allocate workspace
        let workspace = gsl_multifit_nlinear_alloc(
            fit_type,
            &fit_params as *const _,
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
        //let ffi_params = (f, j, data);
        let ffi_params = (f, data);

        // Function to be optimized
        let mut fdf = gsl_multifit_nlinear_fdf {
            f: Some(fit_f::<F, P>), // Some(fit_f::<F, J, P>),
            df: None,
            fvv: None,
            n,
            p: P as u64,
            params: &ffi_params as *const _ as *mut c_void,
            nevalf: 0,
            nevaldf: 0,
            nevalfvv: 0,
        };

        // Init workspace
        gsl_multifit_nlinear_init(param_guess, &mut fdf as *mut _, workspace);

        // Initial chisq
        let start_residuals = gsl_multifit_nlinear_residual(workspace);
        let mut chisq0 = 0.0;
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

        // Extract fit information
        let fit_result = gsl_multifit_nlinear_position(workspace);
        // todo:
        //let fit_niter = gsl_multifit_nlinear_niter(workspace);
        //let fit_jacobian = gsl_multifit_nlinear_jac(workspace);
        //let fit_residuals = gsl_multifit_nlinear_residual(workspace);
        //let fit_residual_sum = gsl_vector_sum(fit_residuals);

        //let fit_covariance = gsl_matrix_alloc(P as u64, P as u64);
        //assert!(!fit_covariance.is_null());

        //gsl_multifit_nlinear_covar(jacobian, 0.0, fit_covariance);

        let mut param_cache = [0.0; P];
        for i in 0..P {
            param_cache[i] = gsl_vector_get(fit_result, i as u64);
        }
        // todo other statistics

        gsl_multifit_nlinear_free(workspace);
        gsl_vector_free(param_guess);

        GSLError::from_raw(status)?;
        Ok(param_cache)
    }
}

unsafe extern "C" fn fit_f<
    F: Fn(f64, [f64; P]) -> Result<f64> + RefUnwindSafe,
    //J: Fn(f64, [f64; P]) -> Result<[f64; P]> + RefUnwindSafe,
    const P: usize,
>(
    params: *const gsl_vector,
    ffi_params: *mut c_void,
    out: *mut gsl_vector,
) -> i32 {
    //let (f, _j, data): &(F, J, &[(f64, f64)]) = &*(ffi_params as *const _);
    let (f, data): &(F, &[(f64, f64)]) = &*(ffi_params as *const _);

    let mut param_cache = [0.0; P];
    for i in 0..P {
        param_cache[i] = gsl_vector_get(params, i as u64);
    }

    for (i, &(x, y)) in data.iter().enumerate() {
        let val = catch_unwind(move || f(x, param_cache));
        let err = match val {
            Ok(Ok(y)) => y,
            Ok(Err(e)) => return e.into(),
            Err(_) => return GSLError::BadFunction.into(),
        } - y;
        gsl_vector_set(out, i as u64, err);
    }

    GSL_SUCCESS
}

#[allow(dead_code)]
unsafe extern "C" fn fit_j<
    F: Fn(f64, [f64; P]) -> Result<f64> + RefUnwindSafe,
    J: Fn(f64, [f64; P]) -> Result<[f64; P]> + RefUnwindSafe,
    const P: usize,
>(
    params: *const gsl_vector,
    ffi_params: *mut c_void,
    out: *mut gsl_matrix,
) -> i32 {
    let (_f, j, data): &(F, J, &[(f64, f64)]) = &*(ffi_params as *const _);

    let mut param_cache = [0.0; P];
    for i in 0..P {
        param_cache[i] = gsl_vector_get(params, i as u64);
    }

    for (i, &(x, _y)) in data.iter().enumerate() {
        let val = catch_unwind(move || j(x, param_cache));

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

unsafe extern "C" fn fit_callback<C: Fn(FitCallback<P>) -> () + RefUnwindSafe, const P: usize>(
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
    let residual_norm = gsl_blas_dnrm2(residuals);

    let mut rcond = 0.0;
    gsl_multifit_nlinear_rcond(&mut rcond as *mut _, workspace);

    let callback: &C = &*(callback as *const _);
    let _ = catch_unwind(move || {
        callback(FitCallback {
            iter: iter as usize,
            params: param_cache,
            cond: 1.0 / rcond,
            residual_norm,
        });
    });
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FitCallback<const P: usize> {
    pub iter: usize,
    pub params: [f64; P],
    pub cond: f64,
    pub residual_norm: f64,
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

        let fit_params = nonlinear_fit(
            1000,
            1.0e-9,
            1.0e-9,
            1.0e-9,
            [10.0, 5.0],
            &data,
            |x, [a, b]| Ok(model(a, b, x)),
            /*|x, [a, b]| {
                let dmda = 1.0 + b * x.powi(2);
                let dmdb = x + a * x.powi(2);
                Ok([dmda, dmdb])
            },*/
            |_| {},
        )
        .unwrap();

        approx::assert_abs_diff_eq!(fit_params[0], a, epsilon = 1.0e-3);
        approx::assert_abs_diff_eq!(fit_params[1], b, epsilon = 1.0e-3);
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

    let fit_params = nonlinear_fit(
        1000,
        1.0e-9,
        1.0e-9,
        1.0e-9,
        [9.0, 1.0],
        &data,
        |x, [a, b]| Ok(model(a, b, x)),
        /*|x, [a, b]| {
            let dmda = (a * x + b).cos() * x;
            let dmdb = (a * x + b).cos();
            Ok([dmda, dmdb])
        },*/
        |_| {},
    )
    .unwrap();

    dbg!(fit_params);

    approx::assert_abs_diff_eq!(fit_params[0], a, epsilon = 1.0e-3);
    approx::assert_abs_diff_eq!(fit_params[1], b, epsilon = 1.0e-3);
}
