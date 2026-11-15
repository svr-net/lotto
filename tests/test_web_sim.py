import http.server, json, os, shutil, subprocess, threading

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
WASM = os.path.join(ROOT, "results", "exports", "lotto_sim.wasm")


def _have_browser():
    if not shutil.which("node"):
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0


@pytest.mark.skipif(not (os.path.exists(WASM) and _have_browser()), reason="needs results/exports/lotto_sim.wasm, node and playwright")
def test_webassembly_and_webgpu_fill_the_same_histograms(tmp_path):
    site = tmp_path / "site"
    (site / "data").mkdir(parents=True)
    shutil.copy(os.path.join(ROOT, "results", "report.html"), site / "index.html")
    for d in ("results", os.path.join("results", "exports")):
        for f in os.listdir(os.path.join(ROOT, d)):
            if f.endswith((".json", ".wasm", ".csv", ".txt")):
                shutil.copy(os.path.join(ROOT, d, f), site / "data" / f)
    handler = lambda *a: http.server.SimpleHTTPRequestHandler(*a, directory=str(site))  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = subprocess.run(["node", os.path.join(ROOT, "tests", "web_sim_check.js"), f"http://127.0.0.1:{server.server_address[1]}"],
                             check=True, capture_output=True, text=True, timeout=300).stdout
    finally:
        server.shutdown()
    r = json.loads(out.strip().splitlines()[-1])
    assert r["errors"] == [] and r["table"] == 6, r
    for run in r["runs"]:
        assert run["paths"] == 2000
        fair = run["cpu"]
        assert 0 <= fair[0] <= fair[1] <= fair[2] and fair[3] <= fair[4] <= fair[5]  # quantiles in order
        if run["gpu"] is not None:  # a WebGPU adapter was found (SwiftShader in headless Chromium)
            assert run["differing"] == 0 and run["gpu"] == fair
