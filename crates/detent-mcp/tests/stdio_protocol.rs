//! Drives the real `detent-mcp` binary over stdio JSON-RPC, the same way
//! Claude Code does. The media round-trip is skipped without FFmpeg.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Client {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_detent-mcp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut c = Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        };
        let init = c.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "detent-test", "version": "0"}
            }),
        );
        assert!(
            init["result"]["capabilities"]["tools"].is_object(),
            "{init}"
        );
        assert!(
            init["result"]["instructions"]
                .as_str()
                .unwrap()
                .contains("solve_and_validate_edl")
        );
        c.notify("notifications/initialized");
        c
    }

    fn send(&mut self, msg: &Value) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn notify(&mut self, method: &str) {
        self.send(&json!({"jsonrpc": "2.0", "method": method}));
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        loop {
            let mut line = String::new();
            assert!(
                self.stdout.read_line(&mut line).unwrap() > 0,
                "server closed stdout"
            );
            let v: Value = serde_json::from_str(&line).unwrap();
            if v["id"] == json!(id) {
                return v;
            }
        }
    }

    fn call(&mut self, tool: &str, args: Value) -> Value {
        let r = self.request("tools/call", json!({"name": tool, "arguments": args}));
        r["result"].clone()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn text(result: &Value) -> Value {
    let t = result["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(t).unwrap()
}

#[test]
fn lists_tools_with_strict_schemas() {
    let mut c = Client::start();
    let r = c.request("tools/list", json!({}));
    let tools = r["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(
        names.len(),
        4,
        "index_media, inspect_keyframes, solve_and_validate_edl, render_timeline: {names:?}"
    );
    let solve = tools
        .iter()
        .find(|t| t["name"] == "solve_and_validate_edl")
        .unwrap();
    let schema = solve["inputSchema"].to_string();
    assert!(schema.contains("SmoothTrackSubject") && schema.contains("hook_score"));
}

#[test]
fn rejects_hallucinated_fields_at_the_boundary() {
    let mut c = Client::start();
    let r = c.request(
        "tools/call",
        json!({"name": "solve_and_validate_edl", "arguments": {
            "index_path": "nope.json",
            "edl": {"clips": [{"start_ms": 0, "end_ms": 1000, "hook_score": 0.5,
                     "editorial_note": "x", "ffmpeg_filter": "zoompan=z=2"}]}
        }}),
    );
    // Either a JSON-RPC error or an isError result: never a success.
    let failed = r.get("error").is_some() || r["result"]["isError"] == json!(true);
    assert!(failed, "{r}");
    assert!(r.to_string().contains("ffmpeg_filter"), "{r}");
}

fn fixture(dir: &Path) -> Option<std::path::PathBuf> {
    let ok = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg("testsrc2=s=480x270:r=25:d=6")
        .args(["-f", "lavfi", "-i"])
        .arg("aevalsrc='if(between(t,2.0,2.6),0.0004*random(0),0.3*sin(2*PI*200*t))':s=48000:d=6")
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ])
        .arg(dir.join("clip.mp4"))
        .status()
        .is_ok_and(|s| s.success());
    ok.then(|| dir.join("clip.mp4"))
}

#[test]
fn index_solve_render_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let Some(media) = fixture(dir.path()) else {
        eprintln!("skipping: ffmpeg not found");
        return;
    };
    let mut c = Client::start();

    let r = c.call("index_media", json!({"path": media.display().to_string()}));
    assert_ne!(r["isError"], json!(true), "{r}");
    let idx = text(&r);
    assert_eq!(idx["digest"]["duration_ms"].as_u64().unwrap() / 100, 60);
    assert_eq!(
        r["content"][1]["type"], "image",
        "contact sheet is attached"
    );
    let index_path = idx["index_path"].as_str().unwrap().to_owned();

    // Out-of-range clip → structured issues the model can act on.
    let bad = c.call(
        "solve_and_validate_edl",
        json!({"index_path": index_path, "edl": {"clips": [
            {"start_ms": 5000, "end_ms": 9000, "hook_score": 0.9, "editorial_note": "too long"}]}}),
    );
    assert_eq!(bad["isError"], json!(true));
    assert!(
        text(&bad)["issues"][0]["path"]
            .as_str()
            .unwrap()
            .starts_with("/clips/0")
    );

    let ok = c.call(
        "solve_and_validate_edl",
        json!({"index_path": index_path, "edl": {
            "clips": [
                {"start_ms": 2700, "end_ms": 5500, "hook_score": 0.9, "editorial_note": "hook"},
                {"start_ms": 300, "end_ms": 2100, "hook_score": 0.4, "editorial_note": "setup"}
            ],
            "motion_cues": [{"clip": 0, "preset": "SlowCreep"}],
            "output": {"aspect": "Square1x1", "height": 360}
        }}),
    );
    assert_ne!(ok["isError"], json!(true), "{ok}");
    let solved = text(&ok);
    assert_eq!(solved["report"]["cuts"].as_array().unwrap().len(), 2);
    let timeline = solved["timeline_path"].as_str().unwrap().to_owned();

    let out = dir.path().join("out.mp4");
    let r = c.call(
        "render_timeline",
        json!({"resolved_timeline_path": timeline, "output_path": out.display().to_string(),
               "quality": "draft", "encoder": "libx264"}),
    );
    assert_ne!(r["isError"], json!(true), "{r}");
    let m = text(&r);
    assert_eq!(m["encoder"], "libx264");
    assert!(m["frames"].as_u64().unwrap() > 100);
    assert!(out.exists());

    let insp = c.call(
        "inspect_keyframes",
        json!({"index_path": index_path, "start_ms": 1000, "end_ms": 3000, "columns": 2, "rows": 2}),
    );
    assert_ne!(insp["isError"], json!(true), "{insp}");
    assert_eq!(insp["content"][1]["mimeType"], "image/jpeg");
}
