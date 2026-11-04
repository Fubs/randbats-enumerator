//! With the `gpu` feature, builds slot 5's class sums and slot 6's node
//! loop (`cuda/`) with nvcc and links cudart statically. `CUDA_PATH` (or
//! `CUDA_HOME`) names the toolkit, by default the one whose nvcc is on
//! `PATH`. `TEAMGEN_CUDA_ARCH=sm_89` names the GPU to build for, by default
//! the one in the machine (`-arch=native`).
fn main() {
    if std::env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }
    for f in ["cuda/slot5.cu", "cuda/slot5.h", "cuda/six.cu", "cuda/six.h"] {
        println!("cargo:rerun-if-changed={f}");
    }
    for k in ["CUDA_PATH", "CUDA_HOME", "TEAMGEN_CUDA_ARCH"] {
        println!("cargo:rerun-if-env-changed={k}");
    }
    let arch = std::env::var("TEAMGEN_CUDA_ARCH").ok();
    println!("cargo:rerun-if-env-changed=TG_SIX_BOUNDS");
    let bounds = std::env::var_os("TG_SIX_BOUNDS").is_some();
    let minb: Vec<(&str, String)> = ["TG_SIX_MINB", "TG_SIX_MINB_BIG"]
        .into_iter()
        .filter_map(|k| {
            println!("cargo:rerun-if-env-changed={k}");
            std::env::var(k).ok().map(|v| (k, v))
        })
        .collect();
    let cuda = cuda_root();
    for (file, lib) in [
        ("cuda/slot5.cu", "teamgen_slot5"),
        ("cuda/six.cu", "teamgen_six"),
    ] {
        let mut build = cc::Build::new();
        if bounds {
            build.define("TG_SIX_BOUNDS", None);
        }
        for (k, v) in &minb {
            build.define(k, v.as_str());
        }
        build
            .cuda(true)
            .debug(false)
            .flag("-lineinfo")
            .flag("-O3")
            .flag("--allow-unsupported-compiler")
            .flag("-std=c++17");
        match &arch {
            Some(sm) => {
                let n = sm.trim_start_matches("sm_");
                build.flag("-gencode").flag(format!("arch=compute_{n},code=sm_{n}"));
            }
            None => {
                build.flag("-arch=native");
            }
        }
        build.file(file).compile(lib);
    }
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=rt");
}
/// `CUDA_PATH`, `CUDA_HOME`, or the toolkit above the nvcc on PATH.
fn cuda_root() -> String {
    for k in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(v) = std::env::var(k) && !v.is_empty() {
            return v;
        }
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .find(|dir| dir.join("nvcc").is_file())
        .and_then(|bin| bin.parent().map(|root| root.display().to_string()))
        .unwrap_or_else(|| "/usr/local/cuda".into())
}
