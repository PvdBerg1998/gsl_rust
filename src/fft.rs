/*
    fft.rs
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
use num_complex::Complex64;

pub fn fft64_packed(data: &mut [f64]) -> Result<()> {
    unsafe {
        let n = data.len();

        // Only radix 2 is implemented
        if n % 2 != 0 {
            return Err(GSLError::Invalid);
        }

        // Deal with empty data
        if n == 0 {
            return Ok(());
        }

        // Transform data in place
        GSLError::from_raw(gsl_fft_real_radix2_transform(
            data.as_mut_ptr(),
            1,
            n as u64,
        ))?;

        Ok(())
    }
}

pub fn fft64_unpack_iter(half_complex: &[f64]) -> impl Iterator<Item = Complex64> + '_ {
    /*
    complex[0].real    =    data[0]
    complex[0].imag    =    0
    complex[1].real    =    data[1]
    complex[1].imag    =    data[n-1]
    ...............         ................
    complex[k].real    =    data[k]
    complex[k].imag    =    data[n-k]
    ...............         ................
    complex[n/2].real  =    data[n/2]
    complex[n/2].imag  =    0
    ...............         ................
    complex[k'].real   =    data[k]        k' = n - k
    complex[k'].imag   =   -data[n-k]
    ...............         ................
    complex[n-1].real  =    data[1]
    complex[n-1].imag  =   -data[n-1]
    */

    // We can infer the DFT length from the half complex packing
    let n = half_complex.len();
    let n_dft = n / 2 + 1;
    (0..n_dft).map(move |i| {
        let re = half_complex[i];

        // First and last element do not have their imaginary part stored
        let im = if i == 0 || i == n / 2 {
            0.0
        } else {
            half_complex[n - i]
        };

        Complex64::new(re, im)
    })
}

pub fn fft64_unpack(half_complex: &[f64]) -> Vec<Complex64> {
    fft64_unpack_iter(half_complex).collect()
}

pub fn fft64_unpack_norm(half_complex: &[f64]) -> Vec<f64> {
    fft64_unpack_iter(half_complex).map(|z| z.norm()).collect()
}

#[test]
fn test_fft() {
    // Generate test data
    let mut y = (0..2u64.pow(14)) // 16384
        .map(|x| x as f64 / 100.0 * std::f64::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();

    // In place transform
    fft64_packed(&mut y).unwrap();
    let fft = fft64_unpack_norm(&y);

    // f = k/T so 1=k/(16384 / 100) -> k=164
    assert!(fft[164] > fft[163]);
    assert!(fft[164] > fft[165]);
}
