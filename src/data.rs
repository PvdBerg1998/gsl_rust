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
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;

pub struct Vector {
    // We own this data on the heap via Box.
    // It is stored as a pointer to avoid aliasing issues when handing out a *mut
    // Also, we store the gsl field on the heap to avoid accidentally moving the Vector
    // and thereby invalidating old references.
    data: *mut [f64],
    gsl: *mut gsl_vector,
    _phantom: PhantomData<Box<[f64]>>,
}

impl Vector {
    pub fn new<T: IntoIterator<Item = f64>>(data: T) -> Self {
        let data = data.into_iter().collect::<Box<[f64]>>();
        assert!(data.len() > 0);

        let size = data.len() as u64;
        let data = Box::into_raw(data);

        let gsl = gsl_vector {
            size,
            stride: 1,
            data: data as *mut _,
            block: std::ptr::null_mut(),
            owner: 0,
        };
        let gsl = Box::into_raw(Box::new(gsl));

        Vector {
            data,
            gsl,
            _phantom: PhantomData,
        }
    }

    pub fn zeroes(n: usize) -> Self {
        Vector::new(vec![0.0; n])
    }

    pub fn to_array<const N: usize>(&self) -> [f64; N] {
        assert_eq!(self.deref().len(), N);

        // Safety: we checked the length
        unsafe { *(self.data as *const [f64; N]) }
    }

    pub fn to_boxed_slice(&self) -> Box<[f64]> {
        self.deref().to_owned().into_boxed_slice()
    }

    pub fn as_gsl(&self) -> *const gsl_vector {
        self.gsl
    }

    pub fn as_gsl_mut(&mut self) -> *mut gsl_vector {
        self.gsl
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
            drop(Box::from_raw(self.gsl));
        }
    }
}

pub struct Matrix {
    // We own this data on the heap via Box.
    // It is stored as a pointer to avoid aliasing issues when handing out a *mut
    // Also, we store the gsl field on the heap to avoid accidentally moving the Vector
    // and thereby invalidating old references.
    data: *mut [f64],
    gsl: *mut gsl_matrix,
    m: usize,
    n: usize,
    _phantom: PhantomData<Box<[f64]>>,
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
        assert!(m > 0);
        assert!(n > 0);

        let data = Box::into_raw(data);

        let gsl = gsl_matrix {
            size1: m as u64,
            size2: n as u64,
            tda: n as u64, // No trailing empty slots per row
            data: data as *mut _,
            block: std::ptr::null_mut(),
            owner: 0,
        };
        let gsl = Box::into_raw(Box::new(gsl));

        Matrix {
            data,
            gsl,
            m,
            n,
            _phantom: PhantomData,
        }
    }

    pub fn from_2d_boxed_slice(data: &[Box<[f64]>]) -> Self {
        let m = data.len();
        assert!(m > 0);
        let n = data[0].len();
        assert!(n > 0);

        // Check uniformity
        for i in 0..m {
            assert_eq!(data[i].len(), n);
        }

        Matrix::new(data.iter().map(|row| row.iter()).flatten().copied(), m, n)
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

    pub fn to_2d_boxed_slice(&self) -> Box<[Box<[f64]>]> {
        let mut out = vec![vec![0.0; self.n].into_boxed_slice(); self.m].into_boxed_slice();
        for i in 0..self.m {
            for j in 0..self.n {
                out[i][j] = self.elem_ij(i, j);
            }
        }
        out
    }

    pub fn to_boxed_slice(&self) -> Box<[f64]> {
        self.deref().to_owned().into_boxed_slice()
    }

    pub fn as_gsl(&self) -> *const gsl_matrix {
        self.gsl
    }

    pub fn as_gsl_mut(&mut self) -> *mut gsl_matrix {
        self.gsl
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

impl<const M: usize, const N: usize> From<[[f64; N]; M]> for Matrix {
    fn from(data: [[f64; N]; M]) -> Self {
        Matrix::new(data.into_iter().flatten(), M, N)
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            // Same argument as Deref
            drop(Box::from_raw(self.data));
            drop(Box::from_raw(self.gsl));
        }
    }
}

impl gsl_vector {
    /// # Safety
    /// The gsl_vector is assumed to be valid
    pub unsafe fn to_array<const N: usize>(this: *const Self) -> [f64; N] {
        let gsl_vector {
            size,
            stride,
            data: ptr,
            block: _,
            owner: _,
        } = *this;

        assert_eq!(N as u64, size);

        if stride == 1 {
            // We can just copy the whole block in one go
            *(ptr as *const _ as *const [f64; N])
        } else {
            // Data layout is nontrivial
            let mut data = [0.0; N];
            for i in 0..N {
                data[i] = gsl_vector_get(this, i as u64);
            }
            data
        }
    }
}

impl From<&[f64]> for gsl_vector {
    fn from(data: &[f64]) -> Self {
        assert!(data.len() > 0);

        let size = data.len() as u64;
        gsl_vector {
            size,
            stride: 1,
            data: data as *const _ as *mut _,
            block: std::ptr::null_mut(),
            owner: 0,
        }
    }
}

impl gsl_matrix {
    pub fn from_slice(data: &[f64], m: usize, n: usize) -> Self {
        assert_eq!(m * n, data.len());
        assert!(m > 0);
        assert!(n > 0);

        gsl_matrix {
            size1: m as u64,
            size2: n as u64,
            tda: n as u64, // No trailing empty slots per row
            data: data.as_ptr() as *mut _,
            block: std::ptr::null_mut(),
            owner: 0,
        }
    }

    /// # Safety
    /// The gsl_matrix is assumed to be valid
    pub unsafe fn to_2d_array<const M: usize, const N: usize>(this: *const Self) -> [[f64; N]; M] {
        let gsl_matrix {
            size1,
            size2,
            tda,
            data: ptr,
            block: _,
            owner: _,
        } = *this;

        assert_eq!(M as u64, size1);
        assert_eq!(N as u64, size2);

        if tda == N as u64 {
            // We can just copy the whole block in one go
            *(ptr as *const _ as *const [[f64; N]; M])
        } else {
            // Data layout is nontrivial
            let mut data = [[0.0; N]; M];
            for i in 0..M {
                for j in 0..N {
                    data[i][j] = gsl_matrix_get(this, i as u64, j as u64);
                }
            }
            data
        }
    }
}

impl<const M: usize, const N: usize> From<&[[f64; N]; M]> for gsl_matrix {
    fn from(data: &[[f64; N]; M]) -> Self {
        assert!(M > 0);
        assert!(N > 0);

        gsl_matrix {
            size1: M as u64,
            size2: N as u64,
            tda: N as u64, // No trailing empty slots per row
            data: data.as_ptr() as *mut _,
            block: std::ptr::null_mut(),
            owner: 0,
        }
    }
}

#[test]
fn test_gsl_vector_wrapper() {
    unsafe {
        // Initialize a vector entirely using GSL
        let v = gsl_vector_alloc(10);
        for i in 0..10 {
            gsl_vector_set(v, i as u64, i as f64);
        }

        dbg!(*v);

        // Convert the GSL vector to a rust array
        let arr = gsl_vector::to_array::<10>(v);
        assert_eq!(arr, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Convert a Rust array reference to a GSL vector
        let vr = gsl_vector::from(arr.as_slice());
        assert_eq!(gsl_vector_equal(v, &vr), 1);

        dbg!(&vr);

        // Construct the same vector using the Rust wrapper
        let v2 = Vector::new(arr);
        assert_eq!(gsl_vector_equal(v, v2.as_gsl()), 1);
        assert_eq!(v2.to_array::<10>(), arr);

        dbg!(&v2);

        // Moving the wrapper should not matter
        let old_ptr = (*v2.as_gsl()).data;
        let v3 = Box::new(v2);
        assert_eq!(gsl_vector_equal(v, v3.as_gsl()), 1);
        assert_eq!(old_ptr, (*v3.as_gsl()).data);

        gsl_vector_free(v);
    }
}

#[test]
fn miri_test_gsl_vector_wrapper() {
    unsafe {
        // Define test array
        let arr = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Convert a Rust array reference to a GSL vector
        let vr = gsl_vector::from(arr.as_slice());
        let _vr = gsl_vector::from(arr.as_slice());

        // Convert back and check equality
        let arr2 = gsl_vector::to_array::<10>(&vr);
        assert_eq!(arr, arr2);

        // Construct the same vector using the Rust wrapper
        let mut v2 = Vector::new(arr);
        let _1 = v2.as_gsl_mut();
        let _2 = v2.as_gsl_mut();

        // Moving the wrapper should not matter
        let old_ptr = (*v2.as_gsl()).data;
        let v3 = Box::new(v2);
        assert_eq!(old_ptr, (*v3.as_gsl()).data);
    }
}

#[test]
fn test_gsl_matrix_wrapper() {
    unsafe {
        // Initialize a MxN = 2x3 matrix entirely using GSL
        let m = gsl_matrix_alloc(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                gsl_matrix_set(m, i as u64, j as u64, (i * 10 + j) as f64);
            }
        }

        dbg!(*m);

        // Convert the GSL matrix to a rust 2d array
        let arr = gsl_matrix::to_2d_array::<2, 3>(m);
        assert_eq!(arr, [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]);

        // Convert a Rust array reference to a GSL matrix
        let mr = gsl_matrix::from(&arr);
        assert_eq!(gsl_matrix_equal(m, &mr), 1);

        dbg!(&mr);

        // Construct the same matrix using the Rust wrapper
        let m2 = Matrix::new(arr.iter().flatten().copied(), 2, 3);
        assert_eq!(gsl_matrix_equal(m, m2.as_gsl()), 1);
        assert_eq!(m2.to_2d_array::<2, 3>(), arr);
        let runtime_m2 = m2.to_2d_boxed_slice();
        let runtime_m2_rec = Matrix::from_2d_boxed_slice(&runtime_m2);

        // Check Rust wrapper elements
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m2.elem_ij(i, j), arr[i][j]);
                assert_eq!(runtime_m2[i][j], arr[i][j]);
                assert_eq!(runtime_m2_rec.elem_ij(i, j), arr[i][j]);
            }
        }

        dbg!(&m2);

        // Moving the wrapper should not matter
        let old_ptr = (*m2.as_gsl()).data;
        let m3 = Box::new(m2);
        assert_eq!(gsl_matrix_equal(m, m3.as_gsl()), 1);
        assert_eq!(old_ptr, (*m3.as_gsl()).data);

        gsl_matrix_free(m);
    }
}

#[test]
fn miri_test_gsl_matrix_wrapper() {
    unsafe {
        // Define test matrix
        let arr = [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]];

        // Convert a Rust array reference to a GSL matrix
        let mr = gsl_matrix::from(&arr);
        let _mr = gsl_matrix::from(&arr);

        // Convert back and check equality
        let arr2 = gsl_matrix::to_2d_array::<2, 3>(&mr);
        assert_eq!(arr, arr2);

        // Construct the same matrix using the Rust wrapper
        let mut m2 = Matrix::new(arr.iter().flatten().copied(), 2, 3);
        let _1 = m2.as_gsl_mut();
        let _2 = m2.as_gsl_mut();
        let runtime_m2 = m2.to_2d_boxed_slice();
        let runtime_m2_rec = Matrix::from_2d_boxed_slice(&runtime_m2);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m2.elem_ij(i, j), arr[i][j]);
                assert_eq!(runtime_m2[i][j], arr[i][j]);
                assert_eq!(runtime_m2_rec.elem_ij(i, j), arr[i][j]);
            }
        }

        // Moving the wrapper should not matter
        let old_ptr = (*m2.as_gsl()).data;
        let m3 = Box::new(m2);
        assert_eq!(old_ptr, (*m3.as_gsl()).data);
    }
}

#[test]
#[should_panic]
fn test_zero_sized_vector() {
    Vector::new([]);
}

#[test]
#[should_panic]
fn test_zero_sized_vector_ref() {
    let _ = gsl_vector::from([].as_slice());
}

#[test]
#[should_panic]
fn test_zero_sized_matrix() {
    Matrix::new([], 0, 0);
}

#[test]
#[should_panic]
fn test_zero_sized_matrix_ref() {
    gsl_matrix::from_slice([].as_slice(), 0, 0);
}

#[test]
#[should_panic]
fn test_zero_sized_matrix_ref2() {
    let _ = gsl_matrix::from(&[[], []]);
}
