# gsl_rust

This is a safe Rust wrapper around the GSL.

It only supports a subset of the API which I need for private projects.

GSL is bundled and gets compiled from source, then statically linked. This can take a while, especially if Cargo decides to recompile.
