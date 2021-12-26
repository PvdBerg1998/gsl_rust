use crate::bindings::*;
use crate::*;
use drop_guard::guard;

pub fn qag<F: FnMut(f64) -> f64>(
    workspace_size: usize,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    rule: GaussKronrodRule,
    mut f: F,
) -> Result<f64> {
    unsafe {
        let workspace = gsl_integration_workspace_alloc(workspace_size as u64);
        assert!(!workspace.is_null());
        let _free_workspace = guard(workspace, |workspace| {
            gsl_integration_workspace_free(workspace);
        });

        let gsl_f = gsl_function_struct {
            function: Some(trampoline::<F>),
            params: &mut f as *mut _ as *mut _,
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
            rule as c_int,
            workspace,
            &mut result as *mut _,
            &mut final_abserr as *mut _,
        );

        GSLError::from_raw(status)?;
        Ok(result)
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GaussKronrodRule {
    Gauss15 = GSL_INTEG_GAUSS15,
    Gauss21 = GSL_INTEG_GAUSS21,
    Gauss31 = GSL_INTEG_GAUSS31,
    Gauss41 = GSL_INTEG_GAUSS41,
    Gauss51 = GSL_INTEG_GAUSS51,
    Gauss61 = GSL_INTEG_GAUSS61,
}

#[test]
fn test_qag65() {
    disable_error_handler();
    approx::assert_abs_diff_eq!(
        qag(4, 0.0, 1.0, 1.0e-6, 0.0, GaussKronrodRule::Gauss61, |x| x
            .powi(3)
            + x)
        .unwrap(),
        0.75,
        epsilon = 1.0e-6
    );
}
