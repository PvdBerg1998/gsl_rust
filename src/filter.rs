/*
    filter.rs
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

pub fn median(width: usize, data: &mut [f64]) -> Result<()> {
    unsafe {
        if width == 0 {
            return Err(GSLError::Invalid);
        }
        if data.is_empty() {
            return Ok(());
        }

        let workspace = guard(gsl_filter_median_alloc(width as u64), |workspace| {
            gsl_filter_median_free(workspace);
        });
        assert!(!workspace.is_null());

        let mut gsl_data = gsl_vector::from(&*data);
        gsl_filter_median(
            gsl_filter_end_t_GSL_FILTER_END_PADVALUE,
            &gsl_data,
            &mut gsl_data,
            *workspace,
        );

        Ok(())
    }
}

/// Impulse filter
/// See https://www.gnu.org/software/gsl/doc/html/filter.html#impulse-detection-filter
///
/// Tuning parameter `t`:
/// - `t = 0`: equal to median filter
/// - `t = inf`: equal to identity filter
///
/// Returns: amount of outliers filtered
pub fn impulse(width: usize, t: f64, scale: ImpulseFilterScale, data: &mut [f64]) -> Result<usize> {
    unsafe {
        if width == 0 {
            return Err(GSLError::Invalid);
        }
        if data.is_empty() {
            return Ok(0);
        }
        if t < 0.0 {
            return Err(GSLError::Invalid);
        }

        let workspace = guard(gsl_filter_impulse_alloc(width as u64), |workspace| {
            gsl_filter_impulse_free(workspace);
        });
        assert!(!workspace.is_null());

        let mut gsl_data = gsl_vector::from(&*data);
        let mut x_median = Vector::zeroes(data.len());
        let mut x_sigma = Vector::zeroes(data.len());
        let mut outliers = 0;

        gsl_filter_impulse(
            gsl_filter_end_t_GSL_FILTER_END_PADVALUE,
            scale as _,
            t,
            &gsl_data,
            &mut gsl_data,
            x_median.as_gsl_mut(),
            x_sigma.as_gsl_mut(),
            &mut outliers,
            std::ptr::null_mut(),
            *workspace,
        );

        Ok(outliers as usize)
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImpulseFilterScale {
    MedianAbsoluteDeviation = gsl_filter_scale_t_GSL_FILTER_SCALE_MAD as u32,
    InterQuartileRange = gsl_filter_scale_t_GSL_FILTER_SCALE_IQR as u32,
    SnStatistic = gsl_filter_scale_t_GSL_FILTER_SCALE_SN as u32,
    QnStatistic = gsl_filter_scale_t_GSL_FILTER_SCALE_QN as u32,
}
