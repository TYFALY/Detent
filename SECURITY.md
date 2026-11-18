# Security Policy

## Supported versions

detent is pre-1.0. Security fixes go into the latest release and the `main` branch only.

| Version | Supported |
|---|---|
| `main` / latest 0.x release | Yes |
| Older 0.x releases | No |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report them privately through GitHub: go to the repository's **Security** tab and choose **Report a vulnerability** ([direct link](https://github.com/TYFALY/detent/security/advisories/new)).

Please include:

- what an attacker can do and under which conditions;
- steps or a minimal input (EDL, index or media file) that reproduces it;
- the detent version, OS and FFmpeg version.

You can expect a first reply within 7 days. Once the issue is confirmed, a fix is prepared privately and released, and you are credited in the advisory unless you would rather stay anonymous.

## Scope

detent runs locally and processes files and JSON that may come from someone else, so these are in scope:

- crafted EDL, timeline or index JSON, or media files, that cause code execution, writes outside the chosen output path, or injection into FFmpeg arguments or filtergraphs;
- the MCP server (`detent-mcp`) accepting tool input that escapes its schema or reaches the file system or a shell in unexpected ways;
- crashes or unbounded memory/CPU use triggered by small inputs (denial of service on the local machine).

Out of scope: vulnerabilities in FFmpeg, whisper.cpp or ONNX Runtime themselves (report those upstream), and problems that need an attacker who can already run commands on your machine.
