# Vendored crates

`whisper-rs` 0.16.0 and `whisper-rs-sys` 0.15.0 (Unlicense; whisper.cpp is
MIT, see `whisper-rs-sys/whisper.cpp/LICENSE`), wired in through
`[patch.crates-io]` in the workspace `Cargo.toml`. They are only built with
`detent-indexer --features whisper`.

Upstream does not build on `x86_64-pc-windows-gnu`. Every change is marked
`detent patch` in the source:

| Problem on MinGW | Fix |
|---|---|
| `/utf-8` (an MSVC-only flag) is passed to GCC, which treats it as an input file | added only when the *target* env is `msvc` (`whisper-rs-sys/build.rs`) |
| libclang finds neither its resource headers nor the MinGW headers, so bindgen falls back to bundled Linux bindings whose layouts are wrong on Windows (`long` is 4 bytes) | bindgen gets the C compiler's own `#include <...>` search list (`gnu_system_includes`) |
| ggml's CMake drops the `lib` prefix on WIN32, so `static=ggml` finds no `libggml.a` | `libggml*.a` copies are added after the build (`add_lib_prefix`) |
| Rust enums mirror C enums as `u32` on windows-gnu, but C enums are `int` on every Windows ABI | `repr(i32)` on all Windows targets (`whisper-rs/src/common_logging.rs`, `whisper_grammar.rs`) |
| A dynamic `libstdc++-6.dll` binds to whichever copy is first on PATH at run time (Git for Windows ships an old one), killing the process with `STATUS_ENTRYPOINT_NOT_FOUND` | libstdc++ and winpthread are linked statically on windows-gnu (`link_static_mingw_runtime`) |

To keep the checkout small, only the CPU and BLAS ggml backends are kept
(CUDA, Metal, Vulkan, SYCL, OpenCL and the others were removed from
`whisper-rs-sys/whisper.cpp/ggml/src`). The GPU features of these crates
therefore do not build from this copy.

CI covers MSVC (`native-whisper-windows`) and MinGW
(`native-whisper-windows-gnu`), plus Linux and macOS.
