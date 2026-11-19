import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import fastllm_pytools.server as server
from test.api.test_op_trace_archive import _write_session


def _read_meta(path):
    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _wait_packed(path, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        meta = _read_meta(path)
        if (meta.get("archive") or {}).get("packed"):
            return meta
        time.sleep(0.05)
    return _read_meta(path)


class ServerPackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="op trace server ")

    def tearDown(self):
        self.temp.cleanup()

    def test_pack_queue_handles_unsealed_dead_session(self):
        path = _write_session(self.temp.name, name="20260101-000000-queue", sealed=False)
        self.assertTrue(server._queue_op_trace_pack(path))
        meta = _wait_packed(path)
        self.assertTrue(meta["archive"]["packed"])
        self.assertTrue(os.path.isfile(os.path.join(
            self.temp.name, meta["archive"]["ftrace"])))

    def test_status_and_summary_endpoints(self):
        path = _write_session(self.temp.name, name="20260101-000000-status", sealed=True)
        fake_llm = types.ModuleType("fastllm_pytools.llm")
        fake_llm.get_op_trace_enabled = lambda: False
        fake_llm.op_trace_supported = lambda: True
        fake_llm.get_op_trace_session = lambda: {
            "session": "20260101-000000-status",
            "dir": path,
            "sealed": True,
            "archive": {"state": "sealed", "committed_seq": 13},
        }
        fake_llm.get_op_trace_summary = lambda max_tokens: {
            "schema": 1, "phases": {}, "tokens": []}
        client = TestClient(server.app, client=("127.0.0.1", 12345))
        with patch.dict(sys.modules, {"fastllm_pytools.llm": fake_llm}):
            status = client.get("/api/op-trace/status")
            summary = client.get("/api/op-trace/summary")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["mode"], "live")
        self.assertEqual(status.json()["session"], "20260101-000000-status")
        self.assertIn("archive", status.json())
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["schema"], 1)

    def test_disable_endpoint_queues_pack(self):
        path = _write_session(self.temp.name, name="20260101-000000-disable", sealed=False)
        fake_llm = types.ModuleType("fastllm_pytools.llm")
        fake_llm.get_op_trace_enabled = Mock(side_effect=[True, False])
        fake_llm.op_trace_supported = lambda: True
        fake_llm.set_op_trace_enabled = lambda enabled: True
        client = TestClient(server.app, client=("127.0.0.1", 12345))
        with patch.dict(sys.modules, {"fastllm_pytools.llm": fake_llm}), \
                patch.object(server, "_op_trace_session_snapshot",
                             return_value={"dir": path}):
            response = client.post("/api/op-trace/enabled", json={"enabled": False})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["enabled"])
        meta = _wait_packed(path)
        self.assertTrue(meta["archive"]["packed"])

    def test_clear_endpoint_queues_previous_session(self):
        path = _write_session(self.temp.name, name="20260101-000000-clear", sealed=False)
        fake_llm = types.ModuleType("fastllm_pytools.llm")
        fake_llm.get_op_trace_enabled = lambda: True
        fake_llm.clear_op_trace = lambda: True
        client = TestClient(server.app, client=("127.0.0.1", 12345))
        with patch.dict(sys.modules, {"fastllm_pytools.llm": fake_llm}), \
                patch.object(server, "_op_trace_session_snapshot",
                             return_value={"dir": path}):
            response = client.post("/api/op-trace/clear")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["supported"])
        meta = _wait_packed(path)
        self.assertTrue(meta["archive"]["packed"])

    def test_maintenance_packs_sealed_session_and_skips_live_owner(self):
        root = self.temp.name
        packed_path = _write_session(root, name="20260101-000000-packed", sealed=True)
        live_path = _write_session(root, name="20260101-000001-live", sealed=False)
        live_meta = _read_meta(live_path)
        live_meta["pid"] = os.getpid()
        with open(os.path.join(live_path, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(live_meta, handle)
        with patch.dict(os.environ, {"FASTLLM_OP_TRACE_DIR": root}, clear=False):
            server._op_trace_maintenance()
        meta = _wait_packed(packed_path)
        self.assertTrue(meta["archive"]["packed"])
        self.assertFalse(_read_meta(live_path)["archive"]["packed"])


if __name__ == "__main__":
    unittest.main()
