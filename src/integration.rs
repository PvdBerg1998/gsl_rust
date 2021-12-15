use crate::bindings::*;
use crate::*;
use std::panic::RefUnwindSafe;

pub fn qag_gk61<F: Fn(f64) -> f64 + RefUnwindSafe>(
    workspace_size: usize,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    f: F,
) -> Result<f64> {
    unsafe {
        let workspace = gsl_integration_workspace_alloc(workspace_size as u64);
        assert!(!workspace.is_null());

        let gsl_f = gsl_function_struct {
            function: Some(trampoline::<F>),
            params: &f as *const _ as *mut _,
        };

        let mut result = 0.0f64;
        let mut final_abserr = 0.0f64;

        let status = gsl_integration_qag(
            &gsl_f as *const _,
            a,
            b,
            epsabs,
            epsrel,
            workspace_size as u64,
            GSL_INTEG_GAUSS61 as c_int,
            workspace,
            &mut result as *mut _,
            &mut final_abserr as *mut _,
        );

        gsl_integration_workspace_free(workspace);

        GSLError::from_raw(status)?;
        Ok(result)
    }
}

#[test]
fn test_qag65() {
    disable_error_handler();
    approx::assert_abs_diff_eq!(
        qag_gk61(4, 0.0, 1.0, 1.0e-6, 0.0, |x| x.powi(3) + x).unwrap(),
        0.75,
        epsilon = 1.0e-6
    );
}
