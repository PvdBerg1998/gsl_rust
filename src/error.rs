/*
    error.rs
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
use std::error::Error;
use std::fmt;
use std::os::raw::*;

pub type Result<T> = std::result::Result<T, GSLError>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GSLError {
    Failure,
    /// iteration has not converged
    Continue,
    /// input domain error, e.g sqrt(-1)
    Domain,
    /// output range error, e.g. exp(1e100)
    Range,
    /// invalid pointer
    Fault,
    /// invalid argument supplied by user
    Invalid,
    /// generic failure
    Failed,
    /// factorization failed
    Factorization,
    /// sanity check failed - shouldn't happen
    Sanity,
    /// malloc failed
    NoMemory,
    /// problem with user-supplied function
    BadFunction,
    /// iterative process is out of control
    RunAway,
    /// exceeded max number of iterations
    MaxIteration,
    /// tried to divide by zero
    ZeroDiv,
    /// user specified an invalid tolerance
    BadTolerance,
    /// failed to reach the specified tolerance
    Tolerance,
    /// underflow
    UnderFlow,
    /// overflow
    OverFlow,
    /// loss of accuracy
    Loss,
    /// failed because of roundoff error
    Round,
    /// matrix, vector lengths are not conformant
    BadLength,
    /// matrix not square
    NotSquare,
    /// apparent singularity detected
    Singularity,
    /// integral or series is divergent
    Diverge,
    /// requested feature is not supported by the hardware
    Unsupported,
    /// requested feature not (yet) implemented
    Unimplemented,
    /// cache limit exceeded
    Cache,
    /// table limit exceeded
    Table,
    /// iteration is not making progress towards solution
    NoProgress,
    /// jacobian evaluations are not improving the solution
    NoProgressJacobian,
    /// cannot reach the specified tolerance in F
    ToleranceF,
    /// cannot reach the specified tolerance in X
    ToleranceX,
    /// cannot reach the specified tolerance in gradient
    ToleranceG,
    /// cannot reach the specified tolerance in gradient
    #[allow(clippy::upper_case_acronyms)]
    EOF,
    /// Unknown value.
    Unknown(i32),
}

impl Into<c_int> for GSLError {
    fn into(self) -> c_int {
        match self {
            Self::Failure => GSL_FAILURE,
            Self::Continue => GSL_CONTINUE,
            Self::Domain => GSL_EDOM,
            Self::Range => GSL_ERANGE,
            Self::Fault => GSL_EFAULT,
            Self::Invalid => GSL_EINVAL,
            Self::Failed => GSL_EFAILED,
            Self::Factorization => GSL_EFACTOR,
            Self::Sanity => GSL_ESANITY,
            Self::NoMemory => GSL_ENOMEM,
            Self::BadFunction => GSL_EBADFUNC,
            Self::RunAway => GSL_ERUNAWAY,
            Self::MaxIteration => GSL_EMAXITER,
            Self::ZeroDiv => GSL_EZERODIV,
            Self::BadTolerance => GSL_EBADTOL,
            Self::Tolerance => GSL_ETOL,
            Self::UnderFlow => GSL_EUNDRFLW,
            Self::OverFlow => GSL_EOVRFLW,
            Self::Loss => GSL_ELOSS,
            Self::Round => GSL_EROUND,
            Self::BadLength => GSL_EBADLEN,
            Self::NotSquare => GSL_ENOTSQR,
            Self::Singularity => GSL_ESING,
            Self::Diverge => GSL_EDIVERGE,
            Self::Unsupported => GSL_EUNSUP,
            Self::Unimplemented => GSL_EUNIMPL,
            Self::Cache => GSL_ECACHE,
            Self::Table => GSL_ETABLE,
            Self::NoProgress => GSL_ENOPROG,
            Self::NoProgressJacobian => GSL_ENOPROGJ,
            Self::ToleranceF => GSL_ETOLF,
            Self::ToleranceX => GSL_ETOLX,
            Self::ToleranceG => GSL_ETOLG,
            Self::EOF => GSL_EOF,
            Self::Unknown(x) => x,
        }
    }
}

impl GSLError {
    pub(crate) fn from_raw(raw: c_int) -> Result<()> {
        match raw {
            GSL_SUCCESS => Ok(()),
            GSL_FAILURE => Err(Self::Failure),
            GSL_CONTINUE => Err(Self::Continue),
            GSL_EDOM => Err(Self::Domain),
            GSL_ERANGE => Err(Self::Range),
            GSL_EFAULT => Err(Self::Fault),
            GSL_EINVAL => Err(Self::Invalid),
            GSL_EFAILED => Err(Self::Failed),
            GSL_EFACTOR => Err(Self::Factorization),
            GSL_ESANITY => Err(Self::Sanity),
            GSL_ENOMEM => Err(Self::NoMemory),
            GSL_EBADFUNC => Err(Self::BadFunction),
            GSL_ERUNAWAY => Err(Self::RunAway),
            GSL_EMAXITER => Err(Self::MaxIteration),
            GSL_EZERODIV => Err(Self::ZeroDiv),
            GSL_EBADTOL => Err(Self::BadTolerance),
            GSL_ETOL => Err(Self::Tolerance),
            GSL_EUNDRFLW => Err(Self::UnderFlow),
            GSL_EOVRFLW => Err(Self::OverFlow),
            GSL_ELOSS => Err(Self::Loss),
            GSL_EROUND => Err(Self::Round),
            GSL_EBADLEN => Err(Self::BadLength),
            GSL_ENOTSQR => Err(Self::NotSquare),
            GSL_ESING => Err(Self::Singularity),
            GSL_EDIVERGE => Err(Self::Diverge),
            GSL_EUNSUP => Err(Self::Unsupported),
            GSL_EUNIMPL => Err(Self::Unimplemented),
            GSL_ECACHE => Err(Self::Cache),
            GSL_ETABLE => Err(Self::Table),
            GSL_ENOPROG => Err(Self::NoProgress),
            GSL_ENOPROGJ => Err(Self::NoProgressJacobian),
            GSL_ETOLF => Err(Self::ToleranceF),
            GSL_ETOLX => Err(Self::ToleranceX),
            GSL_ETOLG => Err(Self::ToleranceG),
            GSL_EOF => Err(Self::EOF),
            x => Err(Self::Unknown(x)),
        }
    }
}

impl Error for GSLError {}

impl fmt::Display for GSLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
