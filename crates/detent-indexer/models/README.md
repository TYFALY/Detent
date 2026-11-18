# Embedded models

`ultraface-rfb-320.onnx` is `version-RFB-320.onnx` from
[Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB),
unchanged (SHA-256 `34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017`).
It is distributed under the MIT license; see the upstream repository's `LICENSE`.

The `faces` feature compiles it into the binary (`include_bytes!`) and runs it
with [tract](https://github.com/sonos/tract), so face detection needs no
download and no native runtime.
