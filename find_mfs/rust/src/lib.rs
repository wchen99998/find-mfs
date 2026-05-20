#![cfg_attr(not(feature = "extension-module"), allow(dead_code))]

mod chemistry;
mod decomposer;
mod filters;
mod finder;
mod formula;
pub mod fragmentation_tree;
mod isospec_ffi;
mod isotope;
mod prior;
mod query;
mod static_data;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
mod python_bindings;

#[cfg(feature = "extension-module")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_bindings::register(m)
}
