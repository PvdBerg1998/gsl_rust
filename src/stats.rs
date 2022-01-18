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

use crate::bindings::*;

pub fn mean(x: &[f64]) -> f64 {
    unsafe {
        let gsl_x = gsl_vector::from(x);
        gsl_stats_mean(gsl_x.data, gsl_x.stride, gsl_x.size)
    }
}

pub fn variance(x: &[f64]) -> f64 {
    unsafe {
        let gsl_x = gsl_vector::from(x);
        gsl_stats_variance(gsl_x.data, gsl_x.stride, gsl_x.size)
    }
}

pub fn variance_mean(x: &[f64], mean: f64) -> f64 {
    unsafe {
        let gsl_x = gsl_vector::from(x);
        gsl_stats_variance_m(gsl_x.data, gsl_x.stride, gsl_x.size, mean)
    }
}
