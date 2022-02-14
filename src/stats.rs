/*
    stats.rs
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

//use crate::bindings::*;

pub fn mean(x: &[f64]) -> f64 {
    // unsafe {
    //     let gsl_x = gsl_vector::from(x);
    //     gsl_stats_mean(gsl_x.data, gsl_x.stride, gsl_x.size)
    // }
    x.iter().copied().sum::<f64>() / x.len() as f64
}

pub fn variance(x: &[f64]) -> f64 {
    // unsafe {
    //     let gsl_x = gsl_vector::from(x);
    //     gsl_stats_variance(gsl_x.data, gsl_x.stride, gsl_x.size)
    // }
    variance_mean(x, mean(x))
}

pub fn variance_mean(x: &[f64], mean: f64) -> f64 {
    // unsafe {
    //     let gsl_x = gsl_vector::from(x);
    //     gsl_stats_variance_m(gsl_x.data, gsl_x.stride, gsl_x.size, mean)
    // }
    x.iter()
        .copied()
        .map(|xi| xi - mean)
        .map(|x| x.powi(2))
        .sum::<f64>()
        / (x.len() - 1) as f64
}

#[test]
fn test_variance_compare_rs_gsl() {
    let x = [1.0, 2.0, 3.0, 4.0, 10.0, 200.0, -10.0, 0.0];
    let gsl_variance = unsafe {
        use crate::bindings::*;

        let gsl_x = gsl_vector::from(&x as &[f64]);
        gsl_stats_variance(gsl_x.data, gsl_x.stride, gsl_x.size)
    };
    let rust_variance = variance(&x);
    approx::assert_abs_diff_eq!(gsl_variance, rust_variance);
}
