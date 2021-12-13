fn main() {
    std::env::remove_var("NUM_JOBS");
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
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .blocklist_item("FP_.*")
        .blocklist_item(".*long_double.*")
        .allowlist_function("gsl.*")
        .allowlist_type("gsl.*")
        .allowlist_var("gsl.*")
        .allowlist_function("GSL.*")
        .allowlist_type("GSL.*")
        .allowlist_var("GSL.*")
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("bindings.rs")
        .expect("Couldn't write bindings!");
}
