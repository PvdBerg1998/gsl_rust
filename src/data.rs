/*
    data.rs
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
use std::fmt;
use std::iter::FromIterator;
use std::ops::Deref;
use std::ops::DerefMut;

pub fn gsl_vector_from_ref(data: &[f64]) -> gsl_vector {
    let size = data.len() as u64;
    gsl_vector {
        size,
        stride: 1,
        data: data as *const _ as *mut _,
        block: std::ptr::null_mut(),
        owner: 0,
    }
}

pub fn gsl_matrix_from_ref<const M: usize, const N: usize>(data: &[[f64; N]; M]) -> gsl_matrix {
    gsl_matrix {
        size1: M as u64,
        size2: N as u64,
        tda: N as u64, // No trailing empty slots per row
        data: data.as_ptr() as *mut _,
        block: std::ptr::null_mut(),
        owner: 0,
    }
}

pub struct Vector {
    data: *mut [f64],
    gsl: gsl_vector,
}

impl Vector {
    pub fn new<T: IntoIterator<Item = f64>>(data: T) -> Self {
        let data = data.into_iter().collect::<Box<[f64]>>();
        let size = data.len() as u64;
        let data = Box::into_raw(data);

        let gsl = gsl_vector {
            size,
            stride: 1,
            data: data as *mut _,
            block: std::ptr::null_mut(),
            owner: 0,
        };
        Vector { data, gsl }
    }

    pub fn zeroes(n: usize) -> Self {
        Vector::new(vec![0.0; n])
    }

    pub fn to_array<const N: usize>(&self) -> [f64; N] {
        assert_eq!(self.deref().len(), N);
        self.deref().try_into().unwrap()
    }

    pub fn as_gsl(&self) -> *const gsl_vector {
        &self.gsl
    }

    pub fn as_gsl_mut(&mut self) -> *mut gsl_vector {
        &mut self.gsl
    }
}

impl fmt::Debug for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl FromIterator<f64> for Vector {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        Vector::new(iter)
    }
}

impl Deref for Vector {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        unsafe {
            // It requires unsafe to dereference a possible *mut to our data,
            // so we can assume we have unique access.
            &*self.data
        }
    }
}

impl DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            // Same argument as Deref
            &mut *self.data
        }
    }
}

impl Clone for Vector {
    fn clone(&self) -> Self {
        Vector::new(self.iter().copied())
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        unsafe {
            // Same argument as Deref
            drop(Box::from_raw(self.data));
        }
    }
}

pub struct Matrix {
    data: *mut [f64],
    m: usize,
    n: usize,
    gsl: gsl_matrix,
}

impl Matrix {
    /// `m` by `n` matrix
    ///
    /// Row length `n`
    ///
    /// Column length `m`
    ///
    /// Assumed to be stored as row major
    pub fn new<T: IntoIterator<Item = f64>>(data: T, m: usize, n: usize) -> Self {
        let data = data.into_iter().collect::<Box<[f64]>>();
        assert_eq!(m * n, data.len());
        let data = Box::into_raw(data);

        let gsl = gsl_matrix {
            size1: m as u64,
            size2: n as u64,
            tda: n as u64, // No trailing empty slots per row
            data: data as *mut _,
            block: std::ptr::null_mut(),
            owner: 0,
        };
        Matrix { data, m, n, gsl }
    }

    /// Gets element `X_ij` from `X_00` to `X_mn`
    ///
    /// `i` runs from `0` to `m` (vertical, row index)
    ///
    /// `j` runs from `0` to `n` (horizontal, column index)
    pub fn elem_ij(&self, i: usize, j: usize) -> f64 {
        self.deref()[i * self.n + j]
    }

    /// `m` by `n` matrix
    ///
    /// Row length `n`
    ///
    /// Column length `m`
    pub fn zeroes(m: usize, n: usize) -> Self {
        Matrix::new(vec![0.0; m * n], m, n)
    }

    pub fn to_2d_array<const M: usize, const N: usize>(&self) -> [[f64; N]; M] {
        assert_eq!(self.n, N);
        assert_eq!(self.m, M);

        // Safety: we checked the dimensions and the memory layout is the same as a 1d array
        unsafe { *(self.data as *const _ as *const [[f64; N]; M]) }
    }

    pub fn as_gsl(&self) -> *const gsl_matrix {
        &self.gsl
    }

    pub fn as_gsl_mut(&mut self) -> *mut gsl_matrix {
        &mut self.gsl
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Deref for Matrix {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        unsafe {
            // It requires unsafe to dereference a possible *mut to our data,
            // so we can assume we have unique access.
            &*self.data
        }
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            // Same argument as Deref
            &mut *self.data
        }
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Matrix::new(self.iter().copied(), self.m, self.n)
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            // Same argument as Deref
            drop(Box::from_raw(self.data));
        }
    }
}
