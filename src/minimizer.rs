use crate::bindings::*;
use crate::*;
use drop_guard::guard;
use std::panic::RefUnwindSafe;

pub fn minimize<F: Fn(f64) -> f64 + RefUnwindSafe, C: Fn(MinimizerCallback) -> ()>(
    max_iter: usize,
    a: f64,
    b: f64,
    x0: f64,
    epsabs: f64,
    epsrel: f64,
    f: F,
    callback: C,
) -> Result<f64> {
    unsafe {
        let minimizer = gsl_min_fminimizer_alloc(gsl_min_fminimizer_brent);
        assert!(!minimizer.is_null());
        let _free_minimizer = guard(minimizer, |minimizer| {
            gsl_min_fminimizer_free(minimizer);
        });

        let mut gsl_f = gsl_function_struct {
            function: Some(trampoline::<F>),
            params: &f as *const _ as *mut _,
        };

        let status = gsl_min_fminimizer_set(minimizer, &mut gsl_f as *mut _, x0, a, b);
        GSLError::from_raw(status)?;

        let mut iter = 0;
        loop {
            let status = gsl_min_fminimizer_iterate(minimizer);
            GSLError::from_raw(status)?;

            let lower_bound = gsl_min_fminimizer_x_lower(minimizer);
            let upper_bound = gsl_min_fminimizer_x_upper(minimizer);
            let x = gsl_min_fminimizer_x_minimum(minimizer);

            callback(MinimizerCallback {
                iter,
                lower_bound,
                upper_bound,
                x,
            });

            let status = gsl_min_test_interval(lower_bound, upper_bound, epsabs, epsrel);
            if let Ok(_) = GSLError::from_raw(status) {
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
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub x: f64,
}

#[test]
fn test_minimizer() {
    disable_error_handler();
    approx::assert_abs_diff_eq!(
        minimize(100, 1.0, 6.0, 4.0, 1.0e-6, 0.0, |x| x.sin(), |_| {}).unwrap(),
        std::f64::consts::PI * 3.0 / 2.0,
        epsilon = 1.0e-6
    );
}
