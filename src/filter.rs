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

pub fn median_filter(width: usize, data: &mut [f64]) -> Result<()> {
    unsafe {
        if width == 0 {
            return Err(GSLError::Invalid);
        }

        let workspace = guard(gsl_filter_median_alloc(width as u64), |workspace| {
            gsl_filter_median_free(workspace);
        });
        assert!(!workspace.is_null());

        // In-place mutation is allowed
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
