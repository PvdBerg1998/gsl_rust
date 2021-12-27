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

/// # Safety
/// The gsl_vector is assumed to be valid
pub unsafe fn gsl_vector_to_array<const N: usize>(v: *const gsl_vector) -> [f64; N] {
    let gsl_vector {
        size,
        stride,
        data: ptr,
        block: _,
        owner: _,
    } = *v;

    assert_eq!(N as u64, size);
    if N == 0 {
        return [0.0; N];
    }

    if stride == 1 {
        // We can just copy the whole block in one go
        *(ptr as *const _ as *const [f64; N])
    } else {
        // Data layout is nontrivial
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = gsl_vector_get(v, i as u64);
        }
        data
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

/// # Safety
/// The gsl_matrix is assumed to be valid
pub unsafe fn gsl_matrix_to_2d_array<const M: usize, const N: usize>(
    m: *const gsl_matrix,
) -> [[f64; N]; M] {
    let gsl_matrix {
        size1,
        size2,
        tda,
        data: ptr,
        block: _,
        owner: _,
    } = *m;

    assert_eq!(M as u64, size1);
    assert_eq!(N as u64, size2);
    if N == 0 && M == 0 {
        return [[0.0; N]; M];
    }

    if tda == N as u64 {
        // We can just copy the whole block in one go
        *(ptr as *const _ as *const [[f64; N]; M])
    } else {
        // Data layout is nontrivial
        let mut data = [[0.0; N]; M];
        for i in 0..M {
            for j in 0..N {
                data[i][j] = gsl_matrix_get(m, i as u64, j as u64);
            }
        }
        data
    }
}

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

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            // Same argument as Deref
            drop(Box::from_raw(self.data));
            drop(Box::from_raw(self.gsl));
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
        let arr = gsl_vector_to_array::<10>(v);
        assert_eq!(arr, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Convert a Rust array reference to a GSL vector
        let vr = gsl_vector_from_ref(&arr);
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
        let arr = gsl_matrix_to_2d_array::<2, 3>(m);
        assert_eq!(arr, [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]);

        // Convert a Rust array reference to a GSL matrix
        let mr = gsl_matrix_from_ref::<2, 3>(&arr);
        assert_eq!(gsl_matrix_equal(m, &mr), 1);

        dbg!(&mr);

        // Construct the same matrix using the Rust wrapper
        let m2 = Matrix::new(arr.iter().flatten().copied(), 2, 3);
        assert_eq!(gsl_matrix_equal(m, m2.as_gsl()), 1);
        assert_eq!(m2.to_2d_array::<2, 3>(), arr);

        // Check Rust wrapper elements
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m2.elem_ij(i, j), arr[i][j]);
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
