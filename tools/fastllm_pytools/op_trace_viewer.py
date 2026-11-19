from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import tempfile
import threading
import webbrowser
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.background import BackgroundTask

from .op_trace_archive import (
    MAX_SUMMARY_TOKENS,
    PHASE_ORDER,
    Session,
    TraceFormatError,
    default_archive_root,
    delete_session as delete_trace_session,
    import_trace as import_trace_file,
    list_sessions,
    pack,
)
from .op_trace_ui import PAGE_HTML


def _is_local_request(request: Request) -> bool:
    origin = request.headers.get("origin", "")
    if origin:
        origin_host = urlparse(origin).hostname or ""
        if origin_host not in ("127.0.0.1", "::1", "localhost"):
            return False
    client = getattr(request, "client", None)
    host = getattr(client, "host", "") if client is not None else ""
    return host in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost")


def _error(status: int, code: str, message: str = "", hint: str = ""):
    payload = {"error": code}
    if message:
        payload["message"] = message
    if hint:
        payload["hint"] = hint
    return JSONResponse(content=payload, status_code=status)


def _valid_phase(phase: str | None) -> bool:
    return not phase or phase in PHASE_ORDER


def _export_filename(session: str, fmt: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", session).strip("._")
    if not name:
        name = "op-trace"
    return f"{name}.{fmt}.json"


def _within(parent: str, child: str) -> bool:
    try:
        return (os.path.commonpath([
            os.path.normcase(os.path.realpath(parent)),
            os.path.normcase(os.path.realpath(child)),
        ]) == os.path.normcase(os.path.realpath(parent)))
    except ValueError:
        return False


class Viewer:
    def __init__(self, root: str | None = None, read_only: bool = False):
        self.root = os.path.abspath(os.path.expanduser(root or default_archive_root()))
        self.read_only = bool(read_only)
        self.session = None
        self.path = None
        self.lock = threading.RLock()
        self.packing = False
        self.pack_error = ""
        self.pack_path = ""
        self.pack_source = ""

    def close(self):
        with self.lock:
            session = self.session
            self.session = None
            self.path = None
            if session is not None:
                session.close()

    def open_trace(self, path: str):
        path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        session = Session.open(path, recover=not self.read_only, read_only=self.read_only)
        with self.lock:
            old = self.session
            self.session = session
            self.path = path
            self.pack_error = ""
            self.pack_path = ""
            if old is not None:
                old.close()
        meta = session.meta
        archive = meta.get("archive") or {}
        packed = bool(archive.get("packed")) or path.lower().endswith(".ftrace")
        return {
            "session": str(meta.get("session") or ""),
            "path": path,
            "meta": meta,
            "sealed": bool(meta.get("sealed")),
            "packed": packed,
            "archive": archive,
            "read_only": bool(self.read_only or session.read_only),
            "has_summary": bool(session.entries),
            "range": session.range(),
            "chunks": len(session.entries),
        }

    def status(self):
        with self.lock:
            session = self.session
            if session is None:
                return {
                    "supported": True,
                    "enabled": False,
                    "mode": "viewer",
                    "session": None,
                    "path": None,
                    "sealed": False,
                    "packed": False,
                    "archive": {},
                    "read_only": self.read_only,
                    "root": self.root,
                    "packing": self.packing,
                    "pack_error": self.pack_error,
                    "packed_path": self.pack_path,
                }
            archive = session.meta.get("archive") or {}
            packed = bool(archive.get("packed")) or str(self.path or "").lower().endswith(".ftrace")
            return {
                "supported": True,
                "enabled": False,
                "mode": "viewer",
                "session": str(session.meta.get("session") or ""),
                "path": self.path,
                "sealed": bool(session.meta.get("sealed")),
                "packed": packed,
                "archive": archive,
                "read_only": bool(self.read_only or session.read_only),
                "root": self.root,
                "packing": self.packing,
                "pack_error": self.pack_error,
                "packed_path": self.pack_path,
                "range": session.range(),
                "chunks": len(session.entries),
            }

    def pack_current(self):
        with self.lock:
            session = self.session
            path = self.path
            if session is None or not path:
                raise TraceFormatError("no session")
            if self.read_only:
                raise PermissionError("viewer is read-only")
            if not os.path.isdir(path):
                return {"status": "already_packed", "path": path}
            if self.packing:
                raise RuntimeError("packing")
            if not session.sealed or session.read_only:
                raise RuntimeError("session_active")
            self.packing = True
            self.pack_error = ""
            self.pack_source = path
        thread = threading.Thread(target=self._pack_worker, args=(path,), daemon=True)
        thread.start()
        return {"status": "packing", "path": path}

    def delete_session(self, path: str):
        if self.read_only:
            raise PermissionError("viewer is read-only")
        closed = {"value": False}

        def before_delete(target: str, kind: str, meta: dict):
            with self.lock:
                if self.packing:
                    raise RuntimeError("packing")
                session = None
                if self.path and _within(target, self.path):
                    session = self.session
                    self.session = None
                    self.path = None
                    closed["value"] = True
            if session is not None:
                session.close()

        result = delete_trace_session(path, self.root, before_delete=before_delete)
        with self.lock:
            if closed["value"]:
                self.pack_error = ""
                self.pack_path = ""
                self.pack_source = ""
        result["closed"] = closed["value"]
        return result

    def _pack_worker(self, path: str):
        try:
            output = pack(path)
            with self.lock:
                self.packing = False
                self.pack_path = output
                self.pack_error = ""
                if self.session is not None and self.path == path:
                    try:
                        with open(os.path.join(path, "meta.json"), "r",
                                  encoding="utf-8-sig") as handle:
                            self.session.meta = json.load(handle)
                    except Exception:
                        pass
        except Exception as error:
            with self.lock:
                self.packing = False
                self.pack_error = str(error)


def create_app(viewer: Viewer) -> FastAPI:
    app = FastAPI(title="FastLLM Operator Trace Viewer")

    def guard(request: Request):
        if not _is_local_request(request):
            return _error(403, "forbidden",
                          "Operator trace is only available from localhost")
        return None

    def session_or_error():
        with viewer.lock:
            session = viewer.session
        if session is None:
            return None, _error(400, "no_session", "Open a trace first")
        return session, None

    @app.get("/", response_class=HTMLResponse)
    def page(request: Request):
        denied = guard(request)
        if denied:
            return HTMLResponse("<h1>403</h1><p>Viewer is localhost only.</p>",
                                status_code=403)
        return HTMLResponse(PAGE_HTML)

    @app.get("/debug/op-trace", response_class=HTMLResponse)
    def debug_page(request: Request):
        return page(request)

    @app.get("/favicon.ico")
    def favicon(request: Request):
        denied = guard(request)
        if denied:
            return denied
        return JSONResponse(content={}, status_code=204)

    @app.get("/api/op-trace/status")
    def status(request: Request):
        denied = guard(request)
        if denied:
            return denied
        return viewer.status()

    @app.get("/api/op-trace/snapshot")
    def snapshot(request: Request):
        denied = guard(request)
        if denied:
            return denied
        with viewer.lock:
            session = viewer.session
            session_id = str(session.meta.get("session") or "") if session else ""
            sealed = bool(session.meta.get("sealed")) if session else False
            last_seq = session.visible_last_seq if session else 0
            archive = dict((session.meta.get("archive") or {})) if session else {}
        return {
            "supported": True,
            "enabled": False,
            "mode": "viewer",
            "capacity": 0,
            "size": 0,
            "pinned": 0,
            "pinned_capacity": 0,
            "count": 0,
            "first_seq": 0,
            "last_seq": last_seq,
            "dropped": int(archive.get("dropped") or 0),
            "lost": False,
            "session_id": session_id,
            "archive": {
                "enabled": False,
                "committed_seq": last_seq,
                "chunks": int(archive.get("chunks") or 0),
                "bytes": int(archive.get("bytes") or 0),
                "dropped": int(archive.get("dropped") or 0),
                "state": "viewer",
                "sealed": sealed,
                "packed": bool(archive.get("packed")),
            },
            "events": [],
        }

    @app.get("/api/op-trace/summary")
    def summary(request: Request, max_tokens: int = MAX_SUMMARY_TOKENS):
        denied = guard(request)
        if denied:
            return denied
        session, missing = session_or_error()
        if missing:
            return missing
        if not _valid_phase(request.query_params.get("phase")):
            return _error(400, "invalid_phase")
        try:
            return session.summary(max_tokens)
        except TraceFormatError as error:
            return _error(400, "invalid_trace", str(error))

    @app.get("/api/op-trace/export")
    def export_trace(request: Request, format: str = "chrome"):
        denied = guard(request)
        if denied:
            return denied
        session, missing = session_or_error()
        if missing:
            return missing
        if format not in ("chrome", "perfetto"):
            return _error(400, "invalid_format", "format must be chrome or perfetto")
        session_id = str(session.meta.get("session") or "")
        filename = _export_filename(session_id, format)
        temp_dir = tempfile.mkdtemp(prefix="fastllm-op-trace-export-")
        target = os.path.join(temp_dir, filename)
        try:
            session.export_chrome_trace(target)
        except (TraceFormatError, OSError) as error:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return _error(400, "export_failed", str(error))
        except Exception as error:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return _error(500, "export_failed", str(error))
        return FileResponse(
            target,
            media_type="application/json",
            filename=filename,
            background=BackgroundTask(shutil.rmtree, temp_dir, True),
        )

    @app.get("/api/op-trace/archive/page")
    def archive_page(request: Request, from_seq: int = 0, limit: int = 8192,
                     phase: str | None = None, req: int | None = None,
                     token: int | None = None, op: str | None = None,
                     min_dur_us: int = 0, from_us: int | None = None,
                     to_us: int | None = None):
        denied = guard(request)
        if denied:
            return denied
        session, missing = session_or_error()
        if missing:
            return missing
        if not _valid_phase(phase):
            return _error(400, "invalid_phase")
        try:
            return session.page(from_seq=from_seq, limit=limit, phase=phase,
                                req=req, token=token, op=op,
                                min_dur_us=min_dur_us, from_us=from_us,
                                to_us=to_us)
        except (TraceFormatError, ValueError) as error:
            return _error(400, "invalid_page", str(error))

    @app.get("/api/op-trace/archive/find")
    def archive_find(request: Request, phase: str | None = None,
                     req: int | None = None, token: int | None = None):
        denied = guard(request)
        if denied:
            return denied
        session, missing = session_or_error()
        if missing:
            return missing
        if not _valid_phase(phase):
            return _error(400, "invalid_phase")
        if phase is None and req is None and token is None:
            return _error(400, "missing_filter")
        try:
            return session.find(phase=phase, req=req, token=token)
        except (TraceFormatError, ValueError) as error:
            return _error(400, "invalid_find", str(error))

    @app.post("/api/op-trace/open")
    async def open_trace(request: Request):
        denied = guard(request)
        if denied:
            return denied
        try:
            payload = await request.json()
        except Exception:
            return _error(400, "invalid_request")
        path = payload.get("path") if isinstance(payload, dict) else None
        if not path:
            return _error(400, "invalid_request", "path is required")
        try:
            return viewer.open_trace(path)
        except (TraceFormatError, OSError) as error:
            return _error(400, "invalid_trace", str(error))

    @app.post("/api/op-trace/close")
    def close_trace(request: Request):
        denied = guard(request)
        if denied:
            return denied
        viewer.close()
        return {"closed": True, "mode": "viewer"}

    @app.post("/api/op-trace/pick")
    async def pick(request: Request):
        denied = guard(request)
        if denied:
            return denied
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        mode = payload.get("mode", "file") if isinstance(payload, dict) else "file"
        if mode not in ("file", "dir"):
            return _error(400, "invalid_request", "mode must be file or dir")
        from . import op_trace_dialogs
        result = op_trace_dialogs.pick_path(mode)
        if result.get("cancelled"):
            return {"cancelled": True}
        if result.get("path"):
            return {"path": result["path"]}
        error = result.get("error", "dialog_failed")
        if error == "dialog_timeout":
            return _error(408, error, "The file dialog timed out")
        if error == "dialog_busy":
            return _error(409, error, "Another file dialog is open")
        if error == "dialog_unavailable":
            return _error(409, error, hint="请输入路径")
        return _error(500, error, result.get("message", ""))

    @app.post("/api/op-trace/pack")
    def pack_trace(request: Request):
        denied = guard(request)
        if denied:
            return denied
        try:
            return viewer.pack_current()
        except PermissionError as error:
            return _error(403, "read_only", str(error))
        except TraceFormatError as error:
            return _error(400, "no_session", str(error))
        except RuntimeError as error:
            code = str(error)
            if code == "packing":
                return _error(409, "packing", "Packing is already running")
            if code == "session_active":
                return _error(409, "session_active", "Close tracing first")
            return _error(400, "invalid_request", code)

    @app.post("/api/op-trace/import")
    async def import_trace(request: Request, name: str = ""):
        denied = guard(request)
        if denied:
            return denied
        if viewer.read_only:
            return _error(403, "read_only", "viewer is read-only")
        filename = name or request.headers.get("x-filename", "")
        if not filename:
            return _error(400, "invalid_request", "name is required")
        temp = None
        try:
            os.makedirs(viewer.root, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=".op-trace-import-", suffix=".tmp",
                                        dir=viewer.root)
            size = 0
            with os.fdopen(fd, "wb") as handle:
                async for chunk in request.stream():
                    if chunk:
                        handle.write(chunk)
                        size += len(chunk)
            if not size:
                return _error(400, "invalid_request", "empty upload")
            return import_trace_file(temp, filename, viewer.root)
        except TraceFormatError as error:
            return _error(400, "invalid_trace", str(error))
        except OSError as error:
            return _error(409, "import_failed", str(error))
        finally:
            if temp:
                try:
                    if os.path.exists(temp):
                        os.remove(temp)
                except OSError:
                    pass

    @app.post("/api/op-trace/delete")
    async def delete_trace(request: Request):
        denied = guard(request)
        if denied:
            return denied
        try:
            payload = await request.json()
        except Exception:
            return _error(400, "invalid_request")
        path = payload.get("path") if isinstance(payload, dict) else None
        if not path:
            return _error(400, "invalid_request", "path is required")
        try:
            return viewer.delete_session(path)
        except PermissionError as error:
            text = str(error)
            code = "read_only" if text == "viewer is read-only" else "forbidden"
            return _error(403, code, text)
        except FileNotFoundError as error:
            return _error(404, "not_found", str(error))
        except TraceFormatError as error:
            text = str(error)
            if text == "session is still recording":
                return _error(409, "session_active", "Close tracing first")
            return _error(400, "invalid_session", text)
        except RuntimeError as error:
            if str(error) == "packing":
                return _error(409, "packing", "Packing is already running")
            return _error(400, "invalid_request", str(error))
        except OSError as error:
            return _error(409, "delete_failed", str(error))

    @app.get("/api/op-trace/sessions")
    def sessions(request: Request, root: str | None = None):
        denied = guard(request)
        if denied:
            return denied
        target = root or viewer.root
        try:
            return {"root": os.path.abspath(os.path.expanduser(target)),
                    "read_only": viewer.read_only,
                    "sessions": list_sessions(target)}
        except (OSError, TraceFormatError) as error:
            return _error(400, "invalid_root", str(error))

    @app.post("/api/op-trace/enabled")
    async def enabled(request: Request):
        denied = guard(request)
        if denied:
            return denied
        return {"supported": True, "enabled": False, "mode": "viewer"}

    @app.post("/api/op-trace/clear")
    def clear(request: Request):
        denied = guard(request)
        if denied:
            return denied
        return _error(403, "read_only", "Viewer cannot clear a trace")

    return app


def _port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def choose_port(host: str, port: int) -> int:
    if port == 0:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])
    for offset in range(10):
        candidate = port + offset
        if _port_available(host, candidate):
            return candidate
    raise OSError(f"no free port in {port}..{port + 9}")


def _open_browser(url: str):
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ftllm op-trace-view",
                                     description="Open a FastLLM operator trace")
    parser.add_argument("--trace", default="", help="trace directory or .ftrace file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", default="", help="archive root for session listing")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args(argv)

    viewer = Viewer(root=args.root or None, read_only=args.read_only)
    if args.trace:
        try:
            opened = viewer.open_trace(args.trace)
            print(f"[op-trace-view] opened {opened['session']} ({opened['path']})")
        except (TraceFormatError, OSError) as error:
            print(f"[op-trace-view] cannot open {args.trace}: {error}")
            return 2
    try:
        port = choose_port(args.host, args.port)
    except OSError as error:
        print(f"[op-trace-view] {error}")
        return 2
    url = f"http://{args.host}:{port}/"
    app = create_app(viewer)
    print(f"[op-trace-view] listening on {url}")
    print(f"[op-trace-view] archive root: {viewer.root}")
    if not args.no_browser:
        timer = threading.Timer(0.6, _open_browser, args=(url,))
        timer.daemon = True
        timer.start()
    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=port, log_level="warning")
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
