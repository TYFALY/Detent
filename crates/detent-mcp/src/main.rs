//! `detent-mcp`: stdio MCP server binary. Logs go to stderr (stdout carries
//! the protocol). Set `RUST_LOG=detent=debug` for verbose output.

use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();
    detent_mcp::serve_stdio(detent_mcp::ServerSettings::default()).await
}
