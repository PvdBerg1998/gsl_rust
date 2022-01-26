fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");

    std::env::remove_var("NUM_JOBS");
    let mut build = cmake::Config::new("./gsl");

    let out = build
        .define("NO_AMPL_BINDINGS", "1")
        .define("GSL_DISABLE_TESTS", "1")
        .always_configure(true)
        .profile("Release")
        .build();

    let mut lib = out.clone();
    lib.push("lib");

    let mut headers1 = out.clone();
    headers1.push("include");

    let mut headers2 = out.clone();
    headers2.push("include/gsl");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=gsl");
    println!("cargo:rustc-link-lib=static=gslcblas");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", headers1.display()))
        .clang_arg(format!("-I{}", headers2.display()))
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

    // This does not get generated unless gsl_vector_double is the only header in wrapper.h
    // This only happens on my laptop
    // let mut f = std::fs::OpenOptions::new()
    //     .append(true)
    //     .open("bindings.rs")
    //     .unwrap();
    // writeln!(
    //     &mut f,
    //     "extern \"C\" {{\n\
    //     \x20   pub fn gsl_vector_sum(a: *const gsl_vector) -> f64;\n\
    //     }}"
    // )
    // .unwrap();
}
