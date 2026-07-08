"""Tiny local web UI to drive the real MatterLedController from a browser.

Why a local server (not a hosted page): the buttons must call the *actual*
``MatterLedController`` — same code path the LLM will use — so verification is
real, not a JS re-implementation. Uses only the stdlib ``http.server`` (no new
dependency) and binds to localhost.

Run::

    uv run python -m matter_platform_led.webui                 # backend from config.toml
    uv run python -m matter_platform_led.webui --backend mock  # force mock
    uv run python -m matter_platform_led.webui --port 8765

Then open http://127.0.0.1:8765 in a browser. Works with `mock` today and with
`chip_tool` once real hardware is commissioned — no UI change needed.
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from matter_platform_led.controller import MatterLedController
from matter_platform_led.exceptions import MatterError

logger = logging.getLogger("matter_platform_led.webui")

# Server-wide config, set in main(). A fresh controller is built per request so
# the mock backend re-reads its state file and there is no shared mutable state.
_CONFIG_PATH: str | None = None
_BACKEND: str | None = None

_PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matter LED 제어</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #f3f4f6; color: #111;
  }
  @media (prefers-color-scheme: dark) {
    body { background: #0f1115; color: #e8e8e8; }
    .card { background: #181b22 !important; }
  }
  .card {
    background: #fff; border-radius: 18px; padding: 32px; width: 340px; text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,.15);
  }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .meta { font-size: 12px; opacity: .6; margin-bottom: 20px; }
  .bulb {
    width: 120px; height: 120px; border-radius: 50%; margin: 8px auto 20px;
    background: radial-gradient(circle at 35% 30%, #555, #222);
    transition: all .25s ease; border: 2px solid rgba(0,0,0,.1);
  }
  .bulb.on {
    background: radial-gradient(circle at 35% 30%, #fff6c0, #ffd21a);
    box-shadow: 0 0 40px 8px rgba(255,210,26,.7);
  }
  .bulb.unreachable { background: repeating-linear-gradient(45deg,#888,#888 6px,#666 6px,#666 12px); }
  .state { font-size: 22px; font-weight: 600; margin-bottom: 20px; letter-spacing: .05em; }
  .row { display: flex; gap: 8px; margin-bottom: 8px; }
  button {
    flex: 1; padding: 12px 0; border: 0; border-radius: 10px; font-size: 14px; font-weight: 600;
    cursor: pointer; color: #fff; transition: filter .15s;
  }
  button:hover { filter: brightness(1.08); }
  button:active { transform: translateY(1px); }
  .on-btn { background:#16a34a; } .off-btn { background:#4b5563; }
  .toggle-btn { background:#2563eb; } .status-btn { background:#7c3aed; }
  .comm-btn { background:#d97706; width:100%; }
  .log { margin-top:16px; font-size:11px; text-align:left; opacity:.7; min-height:16px; white-space:pre-wrap; }
  .err { color:#ef4444; }
</style>
</head>
<body>
  <div class="card">
    <h1>Matter LED 제어</h1>
    <div class="meta" id="meta">backend: … · node: …</div>
    <div class="bulb" id="bulb"></div>
    <div class="state" id="state">…</div>
    <div class="row">
      <button class="on-btn" onclick="cmd('on')">켜기</button>
      <button class="off-btn" onclick="cmd('off')">끄기</button>
    </div>
    <div class="row">
      <button class="toggle-btn" onclick="cmd('toggle')">토글</button>
      <button class="status-btn" onclick="refresh()">상태읽기</button>
    </div>
    <div class="row"><button class="comm-btn" onclick="cmd('commission')">커미셔닝(페어링)</button></div>
    <div class="log" id="log"></div>
  </div>
<script>
  let busy = false;   // one request at a time (each may spawn a chip-tool process)
  async function call(path) {
    const r = await fetch(path, { method: 'POST' });
    return await r.json();
  }
  function render(d) {
    const bulb = document.getElementById('bulb');
    const state = document.getElementById('state');
    document.getElementById('meta').textContent = `backend: ${d.backend} · node: ${d.node_id} · ep: ${d.endpoint_id}`;
    bulb.className = 'bulb' + (d.reachable === false ? ' unreachable' : (d.on ? ' on' : ''));
    if (d.error) { state.innerHTML = '<span class="err">ERROR</span>'; }
    else { state.textContent = d.on ? 'ON' : 'OFF'; }
  }
  function log(msg, isErr) {
    const el = document.getElementById('log');
    el.innerHTML = (isErr ? '<span class="err">'+msg+'</span>' : msg);
  }
  async function refresh(silent) {
    if (busy) return;
    busy = true;
    try {
      const d = await call('/api/status');
      render(d);
      if (!silent) {
        const s = 'status: ' + (d.on ? 'ON' : 'OFF') + (d.reachable === false ? ' (unreachable)' : '');
        log(d.error ? d.error : s, !!d.error);
      }
    } finally { busy = false; }
  }
  async function cmd(name) {
    if (busy) return;
    busy = true;
    try {
      const d = await call('/api/' + name);
      render(d);
      log(d.error ? d.error : name + ' ✔', !!d.error);
    } finally { busy = false; }
  }
  refresh();
  // Live mirror: re-read the real device state so external changes (phone,
  // another controller) also show up. Skipped while a request is in flight.
  setInterval(() => refresh(true), 3000);
</script>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # quieter default logging
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_page(self) -> None:
        body = _PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path in ("/", "/index.html"):
            self._send_page()
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self) -> None:
        action = urlparse(self.path).path.removeprefix("/api/")
        if action not in ("status", "on", "off", "toggle", "commission"):
            self._send_json({"error": "unknown action"}, code=404)
            return
        self._send_json(_run_action(action))


def _run_action(action: str) -> dict:
    """Build a controller, run one action, and return a JSON-able state dict."""
    try:
        with MatterLedController.from_config(_CONFIG_PATH, backend_override=_BACKEND) as ctrl:
            cfg = ctrl._config  # noqa: SLF001 — read-only display fields
            base = {"backend": cfg.backend, "node_id": cfg.node_id, "endpoint_id": cfg.endpoint_id}
            if action == "on":
                ctrl.on()
            elif action == "off":
                ctrl.off()
            elif action == "toggle":
                ctrl.toggle()
            elif action == "commission":
                ctrl.commission()
            st = ctrl.status()
            return {**base, "on": st.on, "reachable": st.reachable}
    except MatterError as exc:
        # Still try to surface backend metadata even on failure.
        return {"error": str(exc), "backend": _BACKEND or "?", "node_id": "?", "endpoint_id": "?"}


def main(argv: list[str] | None = None) -> int:
    """Start the local web UI server. Blocks until Ctrl-C."""
    parser = argparse.ArgumentParser(prog="python -m matter_platform_led.webui")
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--backend", default=None, choices=["mock", "chip_tool"], help="override backend")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    global _CONFIG_PATH, _BACKEND
    _CONFIG_PATH = args.config
    _BACKEND = args.backend

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}"
    logger.info("Matter LED web UI on %s  (backend override=%s)  — Ctrl-C to stop", url, args.backend or "config")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
