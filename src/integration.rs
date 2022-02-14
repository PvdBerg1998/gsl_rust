/*
    integration.rs
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

pub fn qag<F: FnMut(f64) -> f64>(a: f64, b: f64, f: F) -> Result<ValWithError<f64>> {
    qag_ext(16, a, b, 1.0e-9, 0.0, GaussKronrodRule::Gauss15, f)
}

pub fn qag_ext<F: FnMut(f64) -> f64>(
    workspace_size: usize,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    rule: GaussKronrodRule,
    mut f: F,
) -> Result<ValWithError<f64>> {
    unsafe {
        if workspace_size == 0 {
            return Err(GSLError::Invalid);
        }

        let workspace = guard(
            gsl_integration_workspace_alloc(workspace_size as u64),
            |workspace| {
                gsl_integration_workspace_free(workspace);
            },
        );
        assert!(!workspace.is_null());

        let gsl_f = gsl_function_struct {
            function: Some(trampoline::<F>),
            params: &mut f as *mut _ as *mut _,
        };

        let mut result = 0.0f64;
        let mut final_abserr = 0.0f64;

        GSLError::from_raw(gsl_integration_qag(
            &gsl_f,
            a,
            b,
            epsabs,
            epsrel,
            workspace_size as u64,
            rule as _,
            *workspace,
            &mut result,
            &mut final_abserr,
        ))?;

        Ok(ValWithError {
            val: result,
            err: final_abserr,
        })
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GaussKronrodRule {
    Gauss15 = GSL_INTEG_GAUSS15 as u32,
    Gauss21 = GSL_INTEG_GAUSS21 as u32,
    Gauss31 = GSL_INTEG_GAUSS31 as u32,
    Gauss41 = GSL_INTEG_GAUSS41 as u32,
    Gauss51 = GSL_INTEG_GAUSS51 as u32,
    Gauss61 = GSL_INTEG_GAUSS61 as u32,
}

pub fn qagiu<F: FnMut(f64) -> f64>(a: f64, f: F) -> Result<ValWithError<f64>> {
    qagiu_ext(32, a, 1.0e-9, 0.0, f)
}

pub fn qagiu_ext<F: FnMut(f64) -> f64>(
    workspace_size: usize,
    a: f64,
    epsabs: f64,
    epsrel: f64,
    mut f: F,
) -> Result<ValWithError<f64>> {
    unsafe {
        if workspace_size == 0 {
            return Err(GSLError::Invalid);
        }

        let workspace = guard(
            gsl_integration_workspace_alloc(workspace_size as u64),
            |workspace| {
                gsl_integration_workspace_free(workspace);
            },
        );
        assert!(!workspace.is_null());

        let gsl_f = gsl_function_struct {
            function: Some(trampoline::<F>),
            params: &mut f as *mut _ as *mut _,
        };

        let mut result = 0.0f64;
        let mut final_abserr = 0.0f64;

        // Mutability: gsl_f is not actually modified, the header definition is poor.
        GSLError::from_raw(gsl_integration_qagiu(
            &gsl_f as *const _ as *mut _,
            a,
            epsabs,
            epsrel,
            workspace_size as u64,
            *workspace,
            &mut result,
            &mut final_abserr,
        ))?;

        Ok(ValWithError {
            val: result,
            err: final_abserr,
        })
    }
}

#[test]
fn test_qag65() {
    disable_error_handler();

    approx::assert_abs_diff_eq!(
        qag_ext(4, 0.0, 1.0, 1.0e-6, 0.0, GaussKronrodRule::Gauss61, |x| x
            .powi(3)
            + x)
        .unwrap()
        .val,
        0.75,
        epsilon = 1.0e-6
    );
}

#[test]
fn test_qagiu() {
    disable_error_handler();

    approx::assert_abs_diff_eq!(
        (qagiu(0.0, |x| { (-x.powi(2)).exp() }).unwrap().val * 2.0).powi(2),
        std::f64::consts::PI,
        epsilon = 1.0e-6
    );
}

#[test]
fn test_invalid_params() {
    disable_error_handler();

    // Empty workspace
    qag_ext(0, 0.0, 1.0, 1.0e-6, 0.0, GaussKronrodRule::Gauss61, |x| {
        x.powi(3) + x
    })
    .unwrap_err();
}
