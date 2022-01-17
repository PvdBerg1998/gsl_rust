/*
    sorting.rs
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

pub fn sort_xy(x: &mut [f64], y: &mut [f64]) {
    unsafe {
        let gsl_x = gsl_vector::from(&*x);
        let gsl_y = gsl_vector::from(&*y);

        // Mutability: the vectors arent actually modified, the header definition is poor.
        gsl_sort_vector2(&gsl_x as *const _ as *mut _, &gsl_y as *const _ as *mut _);
    }
}

/// This function assumes the data is sorted and uses the mean as a reducing function.
///
/// See `dedup_x`.
pub fn dedup_x_mean<X>(x: &[X], y: &[f64]) -> Result<(Box<[X]>, Box<[f64]>)>
where
    X: PartialEq + Clone,
{
    dedup_x(x, y, |bin| unsafe {
        let gsl_bin = gsl_vector::from(bin);
        let mean = gsl_stats_mean(gsl_bin.data, gsl_bin.stride, gsl_bin.size);
        mean
    })
}

/// This function assumes the data is sorted.
pub fn dedup_x<X, Y, F: FnMut(&[Y]) -> Y>(
    x: &[X],
    y: &[Y],
    mut reduce: F,
) -> Result<(Box<[X]>, Box<[Y]>)>
where
    X: PartialEq + Clone,
    Y: Clone,
{
    if x.len() != y.len() {
        return Err(GSLError::Invalid);
    }

    // Amount of datapoints
    let n = x.len();

    // Handle edge cases
    if n == 0 {
        return Ok((Box::new([]), Box::new([])));
    }

    // Allocate buffer, with enough capacity to host x/y if they have no duplicates
    let mut buf_x = Vec::with_capacity(n);
    let mut buf_y = Vec::with_capacity(n);

    let mut counting_duplicates = false;
    let mut block_start = 0;

    for (i, window) in x.windows(2).enumerate() {
        let [a, b] = [&window[0], &window[1]];

        // Boundary cases
        let eq = a == b;
        match (eq, counting_duplicates) {
            (false, false) => {
                // Continue counting regular
                continue;
            }
            (true, false) => {
                // Stop counting regular, start counting duplicates
                counting_duplicates = true;

                buf_x.extend_from_slice(&x[block_start..i]);
                buf_y.extend_from_slice(&y[block_start..i]);

                block_start = i;
            }
            (true, true) => {
                // Continue counting duplicates
                continue;
            }
            (false, true) => {
                // Stop counting duplicates, start counting regular
                counting_duplicates = false;

                buf_x.push(x[block_start].clone());
                buf_y.push(reduce(&y[block_start..=i]));

                block_start = i + 1;
            }
        }
    }

    // Final block
    if counting_duplicates {
        buf_x.push(x[block_start].clone());
        buf_y.push(reduce(&y[block_start..]));
    } else {
        buf_x.extend_from_slice(&x[block_start..]);
        buf_y.extend_from_slice(&y[block_start..]);
    }

    Ok((buf_x.into_boxed_slice(), buf_y.into_boxed_slice()))
}

#[test]
fn test_sort_simple() {
    disable_error_handler();

    let mut x = vec![3.0, 2.0, 1.0];
    let mut y = vec![3.0, 2.0, 1.0];

    sort_xy(&mut x, &mut y);

    for (i, y) in y.iter().enumerate() {
        assert_eq!(*y as usize, i + 1);
    }
}

#[test]
fn test_dedup() {
    disable_error_handler();

    let x = ['a', 'a', 'a', 'b', 'c', 'd', 'e', 'f', 'f', 'g', 'g', 'h'];
    let y = x;

    let (xdedup, ydedup) = dedup_x(&x, &y, |set| (set.len() as u8 + '0' as u8) as char).unwrap();

    dbg!(&xdedup, &ydedup);
    assert_eq!(xdedup.as_ref(), &['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
    assert_eq!(ydedup.as_ref(), &['3', 'b', 'c', 'd', 'e', '2', '2', 'h']);
}
