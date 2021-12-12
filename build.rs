use std::path::PathBuf;

fn main() {
    let out = cmake::Config::new(std::fs::canonicalize("./gsl").unwrap())
        .define("NO_AMPL_BINDINGS", "")
        .build();

    let mut lib = out.clone();
    lib.push("lib");

    let mut headers = out.clone();
    headers.push("include/gsl");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=gsl");
    println!("cargo:rustc-link-lib=static=gslcblas");

    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", headers.display()))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
