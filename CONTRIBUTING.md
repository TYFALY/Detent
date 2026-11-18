# Contributing to detent

Thanks for helping. Bug reports, footage that breaks the solver, docs fixes and code are all welcome.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting a bug

Open an issue with the **Bug report** template. The most useful reports include:

- the `detent --version` output, your OS, and `ffmpeg -version` (first line);
- the exact command or MCP tool call;
- the EDL, and the `index.json` if you can share it (it holds timings and analysis, not your video);
- what you expected and what you got (a frame grab or a short clip helps a lot for visual bugs).

Security problems go through [SECURITY.md](SECURITY.md), not public issues.

## Suggesting a feature

Open a **Feature request** issue and describe the edit you are trying to make, not only the API you want. detent's rule is that the model decides *what* to keep and the solver decides *exactly how*, so new options should be semantic intents (presets, styles, named moments), never raw pixel coordinates, frame numbers or FFmpeg filters.

## Development setup

You need:

- Rust 1.91 or newer (`rustup update stable`);
- FFmpeg and ffprobe on `PATH` for the integration tests (they skip with a message when FFmpeg is missing).

```bash
git clone https://github.com/TYFALY/detent.git
cd detent
cargo build --workspace
cargo test --workspace
```

Optional features (`whisper`, `onnx`) are described in the README under **Feature flags**.

### Workspace layout

| Crate | Role |
|---|---|
| `detent-core` | EDL schema, solver, camera springs, captions, DSP. No I/O |
| `detent-indexer` | Probing, audio analysis, shots, faces/saliency, freeze detection, ASR adapters |
| `detent-renderer` | Timeline → FFmpeg filtergraph, encoder discovery, rendering |
| `detent-mcp` | The MCP server (JSON-RPC over stdio) |
| `detent-cli` | The `detent` command |

## Before you open a pull request

CI runs these, so run them locally first:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Guidelines:

- **Keep the solver deterministic.** The same index and EDL must always give the same timeline. No randomness, wall-clock time or hash-map iteration order in solver output.
- **Add a test with every fix or feature.** Solver changes should keep the property tests in `detent-core` passing and, where it fits, add a new invariant to them. Rendering changes should keep the exact frame-count and A/V-sync end-to-end tests green.
- **Schema changes are API changes.** EDL fields are strict (unknown fields are rejected), so new fields need a default and docs in the README, and the files in `examples/` must still parse (`crates/detent-core/tests/examples.rs` checks them).
- **Never commit media you do not own.** Test fixtures must be generated (synthetic video, TTS voices) or freely licensed.
- **One topic per pull request.** Small PRs get reviewed faster.

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) as the history does: `feat: ...`, `fix(renderer): ...`, `docs: ...`, `test: ...`, `chore: ...`.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
