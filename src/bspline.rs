use crate::bindings::*;
use crate::*;
use linear_fit::*;
use std::fmt;

pub fn fit_bspline<const NCOEFFS: usize>(
    k: u64,
    a: f64,
    b: f64,
    data: &[(f64, f64)],
) -> Result<BSpline<NCOEFFS>> {
    unsafe {
        let nbreak = NCOEFFS as u64 + 2 - k;

        // Allocate workspace
        let workspace = gsl_bspline_alloc(k, nbreak);
        assert!(!workspace.is_null());

        // Calculate knots associated with uniformly distributed breakpoints
        gsl_bspline_knots_uniform(a, b, workspace);

        // Allocate vector for basis spline values
        let b = gsl_vector_alloc(NCOEFFS as u64);
        assert!(!b.is_null());

        let fit = linear_fit(data, |&x| {
            // Evaluate all basis splines at this position and store them in b
            gsl_bspline_eval(x, b, workspace);
            copy_from_vector::<NCOEFFS>(b)
        })?;

        Ok(BSpline { fit, workspace, b })
    }
}

#[derive(PartialEq)]
pub struct BSpline<const NCOEFFS: usize> {
    pub fit: FitResult<NCOEFFS>,
    workspace: *mut gsl_bspline_workspace, // Not Copy/Clone!
    b: *mut gsl_vector,
}

impl<const NCOEFFS: usize> BSpline<NCOEFFS> {
    pub fn eval(&self, x: &[f64]) -> Box<[(f64, f64)]> {
        unsafe {
            let c = alloc_filled_vector(&self.fit.params);

            let data = x
                .iter()
                .copied()
                .map(|x| {
                    // Evaluate all basis splines at this position and store them in b
                    gsl_bspline_eval(x, self.b, self.workspace);

                    // y = b.c
                    let mut y = 0.0;
                    gsl_blas_ddot(self.b, c, &mut y as *mut _);
                    (x, y)
                })
                .collect();

            gsl_vector_free(c);

            data
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
            gsl_vector_free(self.b);
            gsl_bspline_free(self.workspace);
        }
    }
}

#[test]
fn test_bspline_fit_1() {
    disable_error_handler();

    const NCOEFFS: usize = 12;

    fn model(a: f64, b: f64, x: f64) -> f64 {
        (a * x + b).sin()
    }

    let a = 10.0;
    let b = 2.0;

    let data = (0..100)
        .map(|x| x as f64 / 100.0)
        .map(|x| (x, model(a, b, x)))
        .collect::<Vec<_>>();

    let spline = fit_bspline::<NCOEFFS>(4, 0.0, 1.0, &data).unwrap();

    dbg!(&spline);

    assert!(spline.fit.r_squared > 0.99999);

    let interpolated_x = (0..1000).map(|x| x as f64 / 1000.0).collect::<Vec<_>>();
    let interpolated = spline.eval(&interpolated_x);

    for &(x, interpolated_y) in interpolated.iter() {
        let y = model(a, b, x);
        approx::assert_abs_diff_eq!(y, interpolated_y, epsilon = 1.0e-2);
    }
}
