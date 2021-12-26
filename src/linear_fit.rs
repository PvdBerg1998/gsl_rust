use crate::bindings::*;
use crate::*;
use drop_guard::guard;

pub fn linear_fit<X, F: FnMut(&X) -> [f64; P], const P: usize>(
    data: &[(X, f64)],
    mut f: F,
) -> Result<FitResult<P>> {
    unsafe {
        // Amount of datapoints
        let n = data.len();

        // Allocate workspace and storage for data
        let workspace = gsl_multifit_linear_alloc(n as u64, P as u64);
        let x = gsl_matrix_alloc(n as u64, P as u64);
        let y = gsl_vector_alloc(n as u64);
        let c = gsl_vector_alloc(P as u64);
        let covariance = gsl_matrix_alloc(P as u64, P as u64);

        assert!(!workspace.is_null());
        assert!(!x.is_null());
        assert!(!y.is_null());
        assert!(!c.is_null());
        assert!(!covariance.is_null());

        let _free = guard(
            (workspace, x, y, c, covariance),
            |(workspace, x, y, c, covariance)| {
                gsl_multifit_linear_free(workspace);
                gsl_matrix_free(x);
                gsl_vector_free(y);
                gsl_vector_free(c);
                gsl_matrix_free(covariance);
            },
        );

        // Copy data into GSL vector
        for (i, (_x, data_y)) in data.iter().enumerate() {
            gsl_vector_set(y, i as u64, *data_y);
        }

        // Prepare linear system matrix: X_ij = f_j(i)
        // The i-th value of predictor variable f_j
        // i in 0..n
        // j in 0..P
        for (i, (data_x, _y)) in data.iter().enumerate() {
            for (j, f_j) in f(data_x).into_iter().enumerate() {
                gsl_matrix_set(x, i as u64, j as u64, f_j);
            }
        }

        // Solve the linear system using SVD
        let mut chisq = 0.0f64;
        let status = gsl_multifit_linear(x, y, c, covariance, &mut chisq as *mut _, workspace);

        // Calculate mean and total sum of squares
        let mean = data.iter().map(|(_, y)| *y).sum::<f64>() / n as f64;
        let tss = data
            .iter()
            .map(|(_, y)| *y)
            .map(|y| (y - mean).powi(2))
            .sum::<f64>();

        // R^2 "goodness of fit"
        let r_squared = 1.0 - chisq / tss;

        // Extract fitted parameters
        let mut param_cache = [0.0; P];
        for i in 0..P {
            param_cache[i] = gsl_vector_get(c, i as u64);
        }

        // Extract parameter uncertainties
        let mut param_sigma_cache = [0.0; P];
        for i in 0..P {
            param_sigma_cache[i] = gsl_matrix_get(covariance, i as u64, i as u64).sqrt();
        }

        let result = FitResult {
            params: param_cache,
            uncertainties: param_sigma_cache,
            residual_squared: chisq,
            residual_variance: chisq / (n as f64 - P as f64),
            mean,
            r_squared,
        };

        GSLError::from_raw(status)?;
        Ok(result)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FitResult<const P: usize> {
    pub params: [f64; P],
    pub uncertainties: [f64; P],
    pub residual_squared: f64,
    pub residual_variance: f64,
    pub mean: f64,
    pub r_squared: f64,
}

#[test]
fn test_fit_1() {
    disable_error_handler();

    fn model(a: f64, b: f64, c: f64, x: f64) -> f64 {
        a + b * x + c * x.powi(2)
    }

    let a = 10.0;
    let b = 2.0;
    let c = 2.0;

    let data = (0..100)
        .map(|x| x as f64 / 10.0)
        .map(|x| (x, model(a, b, c, x)))
        .collect::<Vec<_>>();

    let fit = linear_fit(&data, |&x| [1.0, x, x.powi(2)]).unwrap();

    dbg!(fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-6);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-6);
    approx::assert_abs_diff_eq!(fit.params[2], c, epsilon = 1.0e-6);
}

#[test]
fn test_fit_2() {
    disable_error_handler();
    fastrand::seed(0);

    fn model(a: f64, b: f64, c: f64, x: f64) -> f64 {
        a + b * x + c * x.powi(2)
    }

    let a = 10.0;
    let b = 2.0;
    let c = 2.0;

    let data = (0..100)
        .map(|x| x as f64 / 10.0)
        .map(|x| (x, model(a, b, c, x) + 0.068 * (fastrand::f64() * 2.0 - 1.0)))
        .collect::<Vec<_>>();

    let fit = linear_fit(&data, |&x| [1.0, x, x.powi(2)]).unwrap();

    dbg!(fit);

    approx::assert_abs_diff_eq!(fit.params[0], a, epsilon = 1.0e-2);
    approx::assert_abs_diff_eq!(fit.params[1], b, epsilon = 1.0e-2);
    approx::assert_abs_diff_eq!(fit.params[2], c, epsilon = 1.0e-2);
}
