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

pub fn median(width: usize, data: &[f64]) -> Result<Vec<f64>> {
    unsafe {
        if width == 0 {
            return Err(GSLError::Invalid);
        }

        let workspace = guard(gsl_filter_median_alloc(width as u64), |workspace| {
            gsl_filter_median_free(workspace);
        });
        assert!(!workspace.is_null());

        let gsl_in = gsl_vector::from(&*data);
        let mut gsl_out = Vector::zeroes(data.len());

        gsl_filter_median(
            gsl_filter_end_t_GSL_FILTER_END_PADVALUE,
            &gsl_in,
            gsl_out.as_gsl_mut(),
            *workspace,
        );

        Ok(gsl_out.to_vec())
    }
}
