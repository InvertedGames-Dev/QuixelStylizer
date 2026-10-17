"""Client for the warm headless Blender process. JSON-lines over localhost."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from qs.logging import log_line
from qs.version import TOOL_DIR

DEFAULT_PORT = 18765
SERVER_SCRIPT = Path(TOOL_DIR) / "qs_blender_server.py"
# Must match SERVER_VERSION in qs_blender_server.py.
EXPECTED_SERVER_VERSION = "rewrap-bake-6"


def find_blender() -> str | None:
    env = os.environ.get("BLENDER_EXE") or os.environ.get("BLENDER")
    candidates = [env] if env else []
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    for ver in ("5.2", "5.1", "5.0", "4.5", "4.4", "4.3", "4.2"):
        candidates.append(os.path.join(pf, "Blender Foundation", f"Blender {ver}", "blender.exe"))
    candidates.append(os.path.join(pf, "Blender Foundation", "Blender", "blender.exe"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


class BlenderClient:
    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.proc = None

    def alive(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=0.4):
                return True
        except OSError:
            return False

    def _send(self, payload: dict, timeout: float) -> dict:
        data = (json.dumps(payload) + "\n").encode("utf-8")
        with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as sock:
            sock.sendall(data)
            buf = b""
            sock.settimeout(timeout)
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
        if not buf.strip():
            raise RuntimeError("Blender server closed the connection")
        reply = json.loads(buf.decode("utf-8"))
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or "blender job failed")
        return reply.get("result") or {}

    def _server_version(self) -> str | None:
        try:
            return self._send({"op": "ping"}, timeout=1.0).get("version")
        except Exception:
            return None

    def start(self) -> bool:
        if self.alive():
            if self._server_version() == EXPECTED_SERVER_VERSION:
                return True
            # A warm server from an older build would silently run the old
            # job code. Ask it to quit and start the current one instead.
            log_line("blender server version mismatch; restarting it")
            try:
                self._send({"op": "quit"}, timeout=2.0)
            except Exception:
                pass
            for _ in range(30):
                if not self.alive():
                    break
                time.sleep(0.1)
            if self.alive():
                return False
        exe = find_blender()
        if not exe:
            log_line("blender executable not found")
            return False
        if not SERVER_SCRIPT.is_file():
            log_line(f"missing {SERVER_SCRIPT}")
            return False
        creation = 0
        if sys.platform == "win32":
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [exe, "-b", "--python", str(SERVER_SCRIPT), "--", "--port", str(self.port)],
            cwd=str(TOOL_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation,
        )
        for _ in range(40):
            if self._server_version() == EXPECTED_SERVER_VERSION:
                log_line(f"blender server pid={self.proc.pid} port={self.port}")
                return True
            time.sleep(0.25)
        log_line("blender server did not accept a connection")
        return False

    def stop(self):
        try:
            self.call("quit", timeout=2)
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def call(self, op: str, timeout: float = 120, **kwargs) -> dict:
        if not self.start():
            raise RuntimeError("Blender is not available")
        return self._send({"op": op, **kwargs}, timeout=timeout)


_client = None


def client() -> BlenderClient:
    global _client
    if _client is None:
        _client = BlenderClient()
    return _client


def convert_to_gltf(src: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    result = client().call("convert", src=src, dest=dest)
    return result.get("dest") or dest


def reuv(src: str, dest: str, method: str = "smart") -> str:
    result = client().call("reuv", src=src, dest=dest, method=method)
    return result.get("dest") or dest


def bake(high: str, low: str, dest_dir: str, cage_offset: float = 0.02,
         maps: dict | None = None, size: int = 1024, base_name: str = "Baked") -> dict:
    os.makedirs(dest_dir, exist_ok=True)
    return client().call(
        "bake", high=high, low=low, dest=dest_dir, cage_offset=cage_offset,
        maps=maps or {}, size=int(size), base_name=base_name, timeout=300)


def rewrap_bake(high: str, low: str, dest_dir: str, dest_mesh: str, cage_offset: float = 0.02,
                maps: dict | None = None, size: int = 1024, base_name: str = "Baked") -> dict:
    """Unwrap `low` fresh in Blender and bake the hi-poly mesh's maps and normals onto it."""
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dest_mesh) or ".", exist_ok=True)
    return client().call(
        "rewrap_bake", high=high, low=low, dest=dest_dir, dest_mesh=dest_mesh,
        cage_offset=cage_offset, maps=maps or {}, size=int(size),
        base_name=base_name, timeout=300)


def export_fbx(src: str, dest: str, lods: list | None = None) -> str:
    result = client().call("export_fbx", src=src, dest=dest, lods=lods or [])
    return result.get("dest") or dest


def reference_render(src: str, dest: str, alb: str | None = None) -> str:
    result = client().call("render", src=src, dest=dest, alb=alb)
    return result.get("dest") or dest
