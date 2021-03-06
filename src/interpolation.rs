/*
    interpolation.rs
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

/// This function sorts and deduplicates the given data using the mean.
///
/// For more control, use `interpolate_monotonic` and perform sorting/deduplication manually.
pub fn interpolate(
    algorithm: Algorithm,
    derivative: Derivative,
    mut x: Vec<f64>,
    mut y: Vec<f64>,
    x_eval: &[f64],
) -> Result<Vec<f64>> {
    if x.len() != y.len() {
        return Err(GSLError::Invalid);
    }

    sorting::sort_xy(&mut x, &mut y);
    let (x, y) = sorting::dedup_x_mean(&x, &y)?;

    interpolate_monotonic(algorithm, derivative, &x, &y, x_eval)
}

/// This function assumes the data is sorted and free of duplicates.
pub fn interpolate_monotonic(
    algorithm: Algorithm,
    derivative: Derivative,
    x: &[f64],
    y: &[f64],
    x_eval: &[f64],
) -> Result<Vec<f64>> {
    unsafe {
        if x.len() != y.len() {
            return Err(GSLError::Invalid);
        }

        // Amount of datapoints
        let n = x.len();

        // Allocate workspaces
        let algorithm = match algorithm {
            Algorithm::Linear => gsl_interp_linear,
            Algorithm::Steffen => gsl_interp_steffen,
        };

        // Check required amount of datapoints
        if n < gsl_interp_type_min_size(algorithm) as usize {
            return Err(GSLError::Invalid);
        }

        let workspace = guard(gsl_interp_alloc(algorithm, n as u64), |workspace| {
            gsl_interp_free(workspace);
        });
        assert!(!workspace.is_null());

        let accel = guard(gsl_interp_accel_alloc(), |accel| {
            gsl_interp_accel_free(accel);
        });
        assert!(!accel.is_null());

        GSLError::from_raw(gsl_interp_init(
            *workspace,
            x.as_ptr(),
            y.as_ptr(),
            n as u64,
        ))?;

        x_eval
            .iter()
            .map(|&x_eval| {
                let mut y_eval = 0.0;

                GSLError::from_raw(match derivative {
                    Derivative::None => gsl_interp_eval_e(
                        *workspace,
                        x.as_ptr(),
                        y.as_ptr(),
                        x_eval,
                        *accel,
                        &mut y_eval,
                    ),
                    Derivative::First => gsl_interp_eval_deriv_e(
                        *workspace,
                        x.as_ptr(),
                        y.as_ptr(),
                        x_eval,
                        *accel,
                        &mut y_eval,
                    ),
                    Derivative::Second => gsl_interp_eval_deriv2_e(
                        *workspace,
                        x.as_ptr(),
                        y.as_ptr(),
                        x_eval,
                        *accel,
                        &mut y_eval,
                    ),
                })
                .map(|_| y_eval)
            })
            .collect()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Algorithm {
    Linear,
    Steffen,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Derivative {
    None,
    First,
    Second,
}

#[test]
fn test_linear_fit() {
    disable_error_handler();

    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [0.0, 2.0, 4.0, 6.0, 8.0];
    let x_eval = [0.5, 1.5, 2.5, 3.5];
    let expected = [1.0, 3.0, 5.0, 7.0];

    for (y_eval, y_expected) in
        interpolate_monotonic(Algorithm::Linear, Derivative::None, &x, &y, &x_eval)
            .unwrap()
            .iter()
            .zip(expected.iter())
    {
        approx::assert_abs_diff_eq!(y_eval, y_expected);
    }
}

#[test]
fn test_invalid_params() {
    disable_error_handler();

    // No data
    interpolate_monotonic(Algorithm::Linear, Derivative::None, &[], &[], &[0.0]).unwrap_err();

    // Outside domain
    interpolate_monotonic(
        Algorithm::Linear,
        Derivative::None,
        &[0.0, 1.0, 2.0],
        &[0.0; 3],
        &[100.0],
    )
    .unwrap_err();
}
