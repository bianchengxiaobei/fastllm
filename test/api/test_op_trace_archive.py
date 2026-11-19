import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from fastapi.testclient import TestClient


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools import op_trace_archive
from fastllm_pytools import op_trace_dialogs
from fastllm_pytools import op_trace_static
from fastllm_pytools import op_trace_viewer


PHASE_CODE = {"none": 0, "prefill": 1, "decode": 2, "mixed": 3}


def _u16(value):
    return struct.pack("<H", value)


def _u32(value):
    return struct.pack("<I", value)


def _u64(value):
    return struct.pack("<Q", value)


def _varint(value):
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _zigzag(value):
    encoded = value << 1
    return _varint(encoded if value >= 0 else ~encoded)


def _encode_chunk(events):
    strings = []
    string_ids = {}

    def string_id(value):
        if value not in string_ids:
            string_ids[value] = len(strings)
            strings.append(value)
        return string_ids[value]

    op_ids = []
    dev_ids = []
    weight_ids = []
    io_ids = []
    tid_values = []
    tid_ids = {}
    tid_ids_list = []
    for event in events:
        op_ids.append(string_id(event["op"]))
        dev_ids.append(string_id(event["dev"]))
        weight_ids.append(0xFFFFFFFF if not event["weight"] else string_id(event["weight"]))
        io_ids.append(0xFFFFFFFF if not event["io"] else string_id(event["io"]))
        tid = int(event["tid"])
        if tid not in tid_ids:
            tid_ids[tid] = len(tid_values)
            tid_values.append(tid)
        tid_ids_list.append(tid_ids[tid])
    string_table = bytearray(_u16(len(strings)))
    for item in strings:
        raw = item.encode("utf-8")
        string_table += _u16(len(raw))
        string_table += raw
    payload = bytearray()
    payload.append(len(tid_values))
    for value in tid_values:
        payload += _u64(value)
    previous_seq = events[0]["seq"]
    for index, event in enumerate(events):
        payload += _varint(0 if index == 0 else event["seq"] - previous_seq)
        previous_seq = event["seq"]
    base_us = events[0]["t_us"]
    previous_us = 0
    for index, event in enumerate(events):
        current = event["t_us"] - base_us
        payload += _zigzag(current if index == 0 else current - previous_us)
        previous_us = current
    for event in events:
        payload += _varint(event["dur_ns"])
    for value in tid_ids_list:
        payload.append(value)
    for event in events:
        payload += _zigzag(event["layer"])
    for event in events:
        payload += _zigzag(event["req"])
    for event in events:
        payload += _zigzag(event["token"])
    for event in events:
        payload.append(1 if event.get("gap") else 0)
    for event in events:
        payload.append(PHASE_CODE[event["phase"]])
    for value in op_ids:
        payload += _u16(value)
    for value in dev_ids:
        payload += _u16(value)
    for value in weight_ids:
        payload += _u32(value)
    for value in io_ids:
        payload += _u32(value)
    for event in events:
        payload += _varint(event.get("xfer_ns", 0))
    for event in events:
        payload += _varint(event.get("reshape_ns", 0))
    for event in events:
        payload += _varint(event.get("run_ns", 0))
    for event in events:
        payload += _varint(event.get("sync_ns", 0))
    header = b"FOTR" + _u16(1) + _u16(0) + _u32(len(events)) + \
        _u64(events[0]["seq"]) + _u64(events[0]["t_us"] * 1000) + \
        _u32(len(string_table)) + _u32(len(payload))
    return header + string_table + payload


def _chunks():
    return [
        [
            {"seq": 1, "t_us": 1000000, "dur_ns": 5000, "tid": 7, "layer": -1,
             "phase": "prefill", "req": 0, "token": -1, "op": "Linear",
             "dev": "cuda", "weight": "w0", "io": ""},
            {"seq": 2, "t_us": 1000010, "dur_ns": 7000, "tid": 7, "layer": -1,
             "phase": "prefill", "req": 0, "token": -1, "op": "Attention",
             "dev": "cuda", "weight": "", "io": ""},
            {"seq": 3, "t_us": 1000020, "dur_ns": 3000, "tid": 7, "layer": 0,
             "phase": "prefill", "req": 0, "token": -1, "op": "Norm",
             "dev": "cpu", "weight": "", "io": "output=x[1] f32"},
            {"seq": 4, "t_us": 1000100, "dur_ns": 4000, "tid": 8, "layer": 0,
             "phase": "decode", "req": 0, "token": 0, "op": "Linear",
             "dev": "cuda", "weight": "w0", "io": "", "run_ns": 3000,
             "xfer_ns": 500, "sync_ns": 100},
            {"seq": 5, "t_us": 1000110, "dur_ns": 25000, "tid": 8, "layer": 0,
             "phase": "decode", "req": 0, "token": 0, "op": "[gap] host",
             "dev": "host", "weight": "", "io": "", "gap": True},
            {"seq": 6, "t_us": 1000150, "dur_ns": 6000, "tid": 8, "layer": 1,
             "phase": "decode", "req": 0, "token": 0, "op": "Attention",
             "dev": "cuda", "weight": "", "io": "", "reshape_ns": 1000},
            {"seq": 7, "t_us": 1000170, "dur_ns": 2000, "tid": 9, "layer": 2,
             "phase": "mixed", "req": 1, "token": -1, "op": "Norm",
             "dev": "cpu", "weight": "", "io": ""},
            {"seq": 8, "t_us": 1000180, "dur_ns": 1000, "tid": 9, "layer": -1,
             "phase": "none", "req": -1, "token": -1, "op": "Unknown",
             "dev": "", "weight": "", "io": ""},
        ],
        [
            {"seq": 9, "t_us": 1000200, "dur_ns": 5000, "tid": 8, "layer": 1,
             "phase": "decode", "req": 0, "token": 0, "op": "Linear",
             "dev": "cuda", "weight": "w0", "io": ""},
            {"seq": 10, "t_us": 1000220, "dur_ns": 4000, "tid": 8, "layer": 0,
             "phase": "decode", "req": 0, "token": 1, "op": "Linear",
             "dev": "cuda", "weight": "w0", "io": ""},
            {"seq": 11, "t_us": 1000230, "dur_ns": 3000, "tid": 8, "layer": 1,
             "phase": "decode", "req": 1, "token": 2, "op": "Attention",
             "dev": "cuda", "weight": "", "io": ""},
        ],
        [
            {"seq": 12, "t_us": 1000300, "dur_ns": 2000, "tid": 9, "layer": 2,
             "phase": "decode", "req": 0, "token": 1, "op": "Norm",
             "dev": "cpu", "weight": "", "io": ""},
            {"seq": 13, "t_us": 1000310, "dur_ns": 1000, "tid": 9, "layer": 0,
             "phase": "decode", "req": 1, "token": 3, "op": "Linear",
             "dev": "cuda", "weight": "w0", "io": ""},
        ],
    ]


def _write_session(root, name="20260101-000000-1", sealed=False):
    path = os.path.join(root, name)
    os.makedirs(os.path.join(path, "chunks"))
    for index, events in enumerate(_chunks(), 1):
        with open(os.path.join(path, "chunks", f"{index:05d}.bin"), "wb") as handle:
            handle.write(_encode_chunk(events))
    meta = {
        "schema": 1,
        "session": name,
        "pid": 99999999,
        "host": socket.gethostname(),
        "created_at_ns": 1000,
        "wall_base_ns": 1000,
        "ended_at_ns": None,
        "sealed": sealed,
        "model": "test-model",
        "config": {"chunk_events": 8, "flush_ms": 200, "io": True,
                   "gaps": True, "gap_us": 20},
        "archive": {"committed_seq": 0, "chunks": 0, "bytes": 0, "dropped": 0,
                    "state": "recording", "stopped_reason": "", "packed": False,
                    "ftrace": None, "pack_error": ""},
    }
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    return path


class TraceArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="op trace ")
        self.root = self.temp.name
        self.path = _write_session(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def open_session(self):
        return op_trace_archive.Session.open(self.path)

    def test_recover_and_decode(self):
        session = self.open_session()
        try:
            self.assertTrue(session.sealed)
            self.assertEqual(session.visible_last_seq, 13)
            self.assertEqual(len(session.entries), 3)
            page = session.page(limit=100)
            self.assertEqual(len(page["events"]), 13)
            self.assertEqual(page["events"][4]["seq"], 5)
            self.assertEqual(page["events"][4]["gap"], 1)
            self.assertEqual(page["events"][3]["dur_ns"], 4000)
            self.assertEqual(page["events"][7]["phase"], "none")
            self.assertEqual(page["events"][2]["io"], "output=x[1] f32")
            self.assertEqual(page["events"][0]["layer"], -1)
        finally:
            session.close()

    def test_paging_and_filters(self):
        session = self.open_session()
        try:
            page = session.page(from_seq=0, limit=5)
            self.assertEqual([item["seq"] for item in page["events"]], [1, 2, 3, 4, 5])
            self.assertEqual(page["next_seq"], 6)
            self.assertTrue(page["has_more"])
            self.assertEqual(page["total_events"], 13)
            page = session.page(from_seq=6, limit=100)
            self.assertEqual([item["seq"] for item in page["events"]],
                             [6, 7, 8, 9, 10, 11, 12, 13])
            self.assertFalse(page["has_more"])
            page = session.page(from_seq=7, limit=4)
            self.assertEqual([item["seq"] for item in page["events"]], [7, 8, 9, 10])
            page = session.page(phase="decode", limit=100)
            self.assertEqual([item["seq"] for item in page["events"]],
                             [4, 5, 6, 9, 10, 11, 12, 13])
            page = session.page(token=0, limit=100)
            self.assertEqual([item["seq"] for item in page["events"]], [4, 5, 6, 9])
            page = session.page(op="Linear", limit=100)
            self.assertEqual([item["seq"] for item in page["events"]], [1, 4, 9, 10, 13])
            page = session.page(min_dur_us=10, limit=100)
            self.assertEqual([item["seq"] for item in page["events"]], [5])
            page = session.page(phase="prefill", req=9, limit=100)
            self.assertEqual(page["events"], [])
            page = session.page(from_us=1000100, to_us=1000200, limit=100)
            self.assertEqual([item["seq"] for item in page["events"]], [4, 5, 6, 7, 8, 9])
        finally:
            session.close()

    def test_find(self):
        session = self.open_session()
        try:
            found = session.find(phase="prefill")
            self.assertEqual(found["first_seq"], 1)
            self.assertEqual(found["last_seq"], 3)
            self.assertEqual(found["count"], 3)
            found = session.find(req=0, token=0)
            self.assertEqual(found["first_seq"], 4)
            self.assertEqual(found["last_seq"], 9)
            self.assertEqual(found["count"], 4)
            found = session.find(token=3)
            self.assertEqual(found["first_seq"], 13)
            self.assertEqual(found["last_seq"], 13)
            found = session.find(phase="none")
            self.assertEqual(found["first_seq"], 8)
            self.assertEqual(found["count"], 1)
            self.assertEqual(session.find(phase="prefill", req=7), {})
        finally:
            session.close()

    def test_summary_matches_scan(self):
        session = self.open_session()
        try:
            fast = session.summary()
            scanned = op_trace_archive._merge_summary(
                session._scan_entries(), op_trace_archive.MAX_SUMMARY_TOKENS,
                "20260101-000000-1")
            self.assertEqual(fast["phases"], scanned["phases"])
            self.assertEqual(fast["tokens"], scanned["tokens"])
            self.assertEqual(fast["ops"], scanned["ops"])
            self.assertEqual(fast["gaps"], scanned["gaps"])
            self.assertEqual(fast["devices"], scanned["devices"])
            self.assertEqual(fast["tokens_total"], 4)
            token = next(item for item in fast["tokens"]
                         if item["req"] == 0 and item["token"] == 0)
            self.assertEqual(token["first_seq"], 4)
            self.assertEqual(token["last_seq"], 9)
            self.assertEqual(token["events"], 4)
            self.assertEqual(token["gap_attr_us"], 25)
            self.assertGreaterEqual(token["gap_us"], 25)
            self.assertTrue(token["has_split"])
        finally:
            session.close()

    def test_export_chrome_trace(self):
        session = self.open_session()
        try:
            output = os.path.join(self.root, "trace.chrome.json")
            self.assertEqual(session.export_chrome_trace(output), 13)
        finally:
            session.close()
        with open(output, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["displayTimeUnit"], "ms")
        self.assertEqual(data["metadata"]["event_count"], 13)
        events = data["traceEvents"]
        complete = [item for item in events if item.get("ph") == "X"]
        self.assertEqual(len(complete), 13)
        self.assertEqual(complete[0]["name"], "Linear")
        self.assertEqual(complete[0]["cat"], "prefill")
        self.assertEqual(complete[0]["ts"], 0)
        self.assertEqual(complete[0]["dur"], 5)
        self.assertEqual(complete[0]["tid"], 7)
        self.assertEqual(complete[0]["args"]["seq"], 1)
        self.assertEqual(complete[0]["args"]["layer"], -1)
        self.assertEqual(complete[0]["args"]["io"], "")
        gap = next(item for item in complete
                   if item["args"]["seq"] == 5)
        self.assertEqual(gap["cat"], "gap")
        self.assertEqual(gap["dur"], 25)
        self.assertEqual(gap["args"]["gap"], 1)
        metadata = [item for item in events if item.get("ph") == "M"]
        self.assertTrue(any(item["name"] == "process_name" for item in metadata))
        self.assertTrue(any(item["name"] == "thread_name" and item["tid"] == 7
                            for item in metadata))
        self.assertEqual({item["args"]["name"] for item in metadata
                          if item["name"] == "thread_name"},
                         {"tid 7", "tid 8", "tid 9"})

    def test_pack_and_open_ftrace(self):
        session = self.open_session()
        session.close()
        output = op_trace_archive.pack(self.path)
        self.assertTrue(os.path.isfile(output))
        self.assertEqual(os.path.splitext(output)[1], ".ftrace")
        with zipfile.ZipFile(output, "r") as archive:
            names = archive.namelist()
            self.assertIn("meta.json", names)
            self.assertIn("chunks.idx", names)
            self.assertIn("chunks/00001.bin", names)
            self.assertFalse(any(name.endswith(".tmp") for name in names))
            packed_meta = json.loads(archive.read("meta.json").decode("utf-8"))
            self.assertTrue(packed_meta["archive"]["packed"])
        with open(os.path.join(self.path, "meta.json"), "r", encoding="utf-8") as handle:
            source_meta = json.load(handle)
        self.assertTrue(source_meta["archive"]["packed"])
        self.assertTrue(source_meta["archive"]["sha256"])
        direct = self.open_session()
        try:
            direct_page = direct.page(limit=100)["events"]
            direct_summary = direct.summary()
        finally:
            direct.close()
        packed = op_trace_archive.Session.open(output)
        try:
            self.assertTrue(packed.sealed)
            self.assertEqual(packed.page(limit=100)["events"], direct_page)
            self.assertEqual(packed.summary()["tokens_total"],
                             direct_summary["tokens_total"])
            found = packed.find(req=1, token=2)
            self.assertEqual(found["first_seq"], 11)
        finally:
            packed.close()

    def test_recover_repairs_index_and_skips_tmp(self):
        op_trace_archive.recover(self.path)
        index_path = os.path.join(self.path, "chunks.idx")
        with open(index_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        with open(index_path, "w", encoding="utf-8") as handle:
            handle.writelines(lines[1:])
            handle.write('{"broken"')
        with open(os.path.join(self.path, "chunks", "00003.bin"), "r+b") as handle:
            handle.truncate(20)
        with open(os.path.join(self.path, "chunks", "00009.bin.tmp"), "wb") as handle:
            handle.write(b"not a chunk")
        meta = op_trace_archive.recover(self.path)
        self.assertTrue(meta["sealed"])
        self.assertEqual(meta["archive"]["chunks"], 1)
        self.assertEqual(meta["archive"]["committed_seq"], 11)
        session = op_trace_archive.Session.open(self.path)
        try:
            self.assertEqual(session.visible_last_seq, 11)
            self.assertEqual(len(session.page(limit=100)["events"]), 3)
            self.assertEqual(len(session.entries), 1)
        finally:
            session.close()

    def test_missing_index_is_rebuilt(self):
        op_trace_archive.recover(self.path)
        os.remove(os.path.join(self.path, "chunks.idx"))
        session = self.open_session()
        try:
            self.assertEqual(len(session.entries), 3)
            self.assertEqual(session.visible_last_seq, 13)
        finally:
            session.close()
        self.assertTrue(os.path.isfile(os.path.join(self.path, "chunks.idx")))

    def test_active_session_is_read_only_until_committed(self):
        op_trace_archive.recover(self.path)
        meta_path = os.path.join(self.path, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["sealed"] = False
        meta["pid"] = os.getpid()
        meta["host"] = socket.gethostname()
        meta["archive"]["committed_seq"] = 8
        meta["archive"]["state"] = "recording"
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        session = self.open_session()
        try:
            self.assertFalse(session.sealed)
            self.assertTrue(session.read_only)
            self.assertEqual(session.visible_last_seq, 8)
            page = session.page(limit=100)
            self.assertEqual([item["seq"] for item in page["events"]],
                             list(range(1, 9)))
            self.assertEqual(session.summary()["phases"]["decode"]["events"], 3)
        finally:
            session.close()
        with open(meta_path, "r", encoding="utf-8") as handle:
            self.assertFalse(json.load(handle)["sealed"])

    def test_list_sessions(self):
        other = _write_session(self.root, "20260102-000000-2", sealed=True)
        op_trace_archive.recover(self.path)
        result = op_trace_archive.list_sessions(self.root)
        names = {item["session"] for item in result}
        self.assertIn("20260101-000000-1", names)
        self.assertIn("20260102-000000-2", names)
        self.assertTrue(any(item["path"] == os.path.abspath(other) for item in result))
        self.assertTrue(all(item["kind"] == "dir" for item in result))

    def test_delete_session_artifacts_and_guards(self):
        op_trace_archive.recover(self.path)
        active = _write_session(self.root, "20260103-000000-active", sealed=False)
        meta_path = os.path.join(active, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["pid"] = os.getpid()
        meta["host"] = socket.gethostname()
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        with self.assertRaises(op_trace_archive.TraceFormatError):
            op_trace_archive.delete_session(active, self.root)
        with self.assertRaises(PermissionError):
            op_trace_archive.delete_session(os.path.join(self.root, "..", "outside"),
                                            self.root)
        packed = op_trace_archive.pack(self.path)
        self.assertTrue(os.path.isfile(packed))
        removed = op_trace_archive.delete_session(packed, self.root)
        self.assertEqual(removed["kind"], "ftrace")
        self.assertFalse(os.path.exists(packed))
        removed = op_trace_archive.delete_session(self.path, self.root)
        self.assertEqual(removed["kind"], "dir")
        self.assertFalse(os.path.exists(self.path))

    def test_import_trace_sanitizes_name_and_avoids_collision(self):
        op_trace_archive.recover(self.path)
        source = op_trace_archive.pack(self.path)
        with self.assertRaises(op_trace_archive.TraceFormatError):
            op_trace_archive.import_trace(source, "not-a-trace.zip", self.root)
        imported = op_trace_archive.import_trace(
            source, "../my trace!.ftrace", self.root)
        self.assertEqual(os.path.dirname(imported["path"]), self.root)
        self.assertEqual(os.path.basename(imported["path"]), "my trace_.ftrace")
        self.assertTrue(os.path.isfile(imported["path"]))
        self.assertGreater(imported["size_bytes"], 0)
        second = op_trace_archive.import_trace(
            op_trace_archive.pack(self.path), "my trace_.ftrace", self.root)
        self.assertEqual(os.path.basename(second["path"]), "my trace_-1.ftrace")


class ViewerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="op trace viewer ")
        self.root = self.temp.name
        self.path = _write_session(self.root)
        op_trace_archive.recover(self.path)
        self.viewer = op_trace_viewer.Viewer(root=self.root)
        self.app = op_trace_viewer.create_app(self.viewer)
        self.client = TestClient(self.app, client=("127.0.0.1", 12345))

    def tearDown(self):
        self.viewer.close()
        self.temp.cleanup()

    def test_viewer_routes(self):
        status = self.client.get("/api/op-trace/status")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["mode"], "viewer")
        self.assertIsNone(status.json()["session"])
        self.assertEqual(self.client.get("/api/op-trace/summary").status_code, 400)
        opened = self.client.post("/api/op-trace/open", json={"path": self.path})
        self.assertEqual(opened.status_code, 200)
        self.assertEqual(opened.json()["session"], "20260101-000000-1")
        self.assertTrue(opened.json()["has_summary"])
        summary = self.client.get("/api/op-trace/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["tokens_total"], 4)
        page = self.client.get("/api/op-trace/archive/page?limit=3")
        self.assertEqual(len(page.json()["events"]), 3)
        self.assertTrue(page.json()["has_more"])
        found = self.client.get("/api/op-trace/archive/find?req=1&token=2")
        self.assertEqual(found.json()["first_seq"], 11)
        sessions = self.client.get("/api/op-trace/sessions")
        self.assertEqual(sessions.status_code, 200)
        self.assertEqual(len(sessions.json()["sessions"]), 1)
        self.assertEqual(sessions.json()["sessions"][0]["kind"], "dir")
        self.assertGreater(sessions.json()["sessions"][0]["size_bytes"], 0)
        snapshot = self.client.get("/api/op-trace/snapshot")
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.json()["mode"], "viewer")
        self.assertEqual(snapshot.json()["events"], [])
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.post("/api/op-trace/clear").status_code, 403)

    def test_viewer_status_archive_and_close(self):
        opened = self.client.post("/api/op-trace/open", json={"path": self.path})
        self.assertEqual(opened.status_code, 200)
        self.assertIn("archive", opened.json())
        status = self.client.get("/api/op-trace/status").json()
        self.assertEqual(status["session"], "20260101-000000-1")
        self.assertIn("archive", status)
        closed = self.client.post("/api/op-trace/close")
        self.assertEqual(closed.status_code, 200)
        self.assertTrue(closed.json()["closed"])
        self.assertIsNone(self.client.get("/api/op-trace/status").json()["session"])

    def test_viewer_delete_current_session(self):
        opened = self.client.post("/api/op-trace/open", json={"path": self.path})
        self.assertEqual(opened.status_code, 200)
        deleted = self.client.post("/api/op-trace/delete", json={"path": self.path})
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["closed"])
        self.assertEqual(deleted.json()["kind"], "dir")
        self.assertFalse(os.path.exists(self.path))
        self.assertIsNone(self.client.get("/api/op-trace/status").json()["session"])
        self.assertEqual(self.client.get("/api/op-trace/sessions").json()["sessions"], [])

    def test_viewer_import_trace(self):
        packed = op_trace_archive.pack(self.path)
        with open(packed, "rb") as handle:
            payload = handle.read()
        imported = self.client.post(
            "/api/op-trace/import?name=uploaded.ftrace", content=payload,
            headers={"content-type": "application/octet-stream"})
        self.assertEqual(imported.status_code, 200)
        data = imported.json()
        self.assertTrue(data["path"].endswith("uploaded.ftrace"))
        self.assertTrue(os.path.isfile(data["path"]))
        opened = self.client.post("/api/op-trace/open", json={"path": data["path"]})
        self.assertEqual(opened.status_code, 200)
        self.assertEqual(opened.json()["session"], "20260101-000000-1")
        sessions = self.client.get("/api/op-trace/sessions").json()["sessions"]
        self.assertTrue(any(item["path"] == data["path"] for item in sessions))
        self.assertEqual(len(sessions), 3)
        bad = self.client.post(
            "/api/op-trace/import?name=bad.ftrace", content=b"not a zip",
            headers={"content-type": "application/octet-stream"})
        self.assertEqual(bad.status_code, 400)
        empty = self.client.post(
            "/api/op-trace/import?name=empty.ftrace", content=b"",
            headers={"content-type": "application/octet-stream"})
        self.assertEqual(empty.status_code, 400)
        readonly = op_trace_viewer.Viewer(root=self.root, read_only=True)
        try:
            ro_client = TestClient(op_trace_viewer.create_app(readonly),
                                   client=("127.0.0.1", 12345))
            denied = ro_client.post(
                "/api/op-trace/import?name=readonly.ftrace", content=payload,
                headers={"content-type": "application/octet-stream"})
            self.assertEqual(denied.status_code, 403)
        finally:
            readonly.close()

    def test_viewer_export_chrome_trace(self):
        opened = self.client.post("/api/op-trace/open", json={"path": self.path})
        self.assertEqual(opened.status_code, 200)
        response = self.client.get("/api/op-trace/export?format=chrome")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        self.assertIn(".chrome.json", response.headers["content-disposition"])
        data = response.json()
        self.assertEqual(data["metadata"]["event_count"], 13)
        self.assertEqual(len([item for item in data["traceEvents"]
                              if item.get("ph") == "X"]), 13)
        perfetto = self.client.get("/api/op-trace/export?format=perfetto")
        self.assertEqual(perfetto.status_code, 200)
        self.assertEqual(perfetto.json()["metadata"]["event_count"], 13)
        invalid = self.client.get("/api/op-trace/export?format=bad")
        self.assertEqual(invalid.status_code, 400)
        self.client.post("/api/op-trace/close")
        missing = self.client.get("/api/op-trace/export?format=chrome")
        self.assertEqual(missing.status_code, 400)

    def test_viewer_rejects_remote(self):
        remote = TestClient(self.app, client=("10.0.0.1", 12345))
        self.assertEqual(remote.get("/api/op-trace/status").status_code, 403)
        self.assertEqual(remote.get("/").status_code, 403)

    def test_viewer_pick_mapping(self):
        with patch.object(op_trace_dialogs, "pick_path", return_value={"path": "x.ftrace"}):
            response = self.client.post("/api/op-trace/pick", json={"mode": "file"})
        self.assertEqual(response.json()["path"], "x.ftrace")
        with patch.object(op_trace_dialogs, "pick_path", return_value={"cancelled": True}):
            response = self.client.post("/api/op-trace/pick", json={"mode": "dir"})
        self.assertTrue(response.json()["cancelled"])
        with patch.object(op_trace_dialogs, "pick_path",
                          return_value={"error": "dialog_unavailable"}):
            response = self.client.post("/api/op-trace/pick", json={"mode": "file"})
        self.assertEqual(response.status_code, 409)
        with patch.object(op_trace_dialogs, "pick_path",
                          return_value={"error": "dialog_timeout"}):
            response = self.client.post("/api/op-trace/pick", json={"mode": "file"})
        self.assertEqual(response.status_code, 408)

    def test_viewer_opens_ftrace(self):
        output = op_trace_archive.pack(self.path)
        opened = self.client.post("/api/op-trace/open", json={"path": output})
        self.assertEqual(opened.status_code, 200)
        self.assertTrue(opened.json()["packed"])
        self.assertEqual(self.client.get("/api/op-trace/summary").status_code, 200)
        exported = self.client.get("/api/op-trace/export?format=chrome")
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.json()["metadata"]["event_count"], 13)
        packed = self.client.post("/api/op-trace/pack")
        self.assertEqual(packed.status_code, 200)
        self.assertEqual(packed.json()["status"], "already_packed")
        invalid = self.client.post("/api/op-trace/open",
                                   json={"path": os.path.join(self.root, "missing")})
        self.assertEqual(invalid.status_code, 400)

    def test_viewer_pack(self):
        opened = self.client.post("/api/op-trace/open", json={"path": self.path})
        self.assertEqual(opened.status_code, 200)
        response = self.client.post("/api/op-trace/pack")
        self.assertEqual(response.status_code, 200)
        for _ in range(100):
            status = self.client.get("/api/op-trace/status").json()
            if not status["packing"]:
                break
            import time
            time.sleep(0.02)
        self.assertFalse(status["packing"])
        self.assertTrue(status["packed"])
        self.assertTrue(os.path.isfile(status["packed_path"]))

    def test_viewer_imports_no_native_modules(self):
        code = (
            "import sys;"
            "import fastllm_pytools.op_trace_viewer;"
            "print(int('fastllm_pytools.llm' in sys.modules),"
            "int('fastllm_pytools.server' in sys.modules))"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS_DIR
        completed = subprocess.run([sys.executable, "-c", code], env=env,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "0 0")


class StaticViewerTest(unittest.TestCase):
    def test_build_static_html(self):
        html = op_trace_static.build_static_html()
        self.assertIn("__opTraceStatic", html)
        self.assertIn("staticFile", html)
        self.assertIn("DecompressionStream", html)
        self.assertIn("window.fetch", html)
        self.assertNotIn("http://127.0.0.1", html)
        self.assertEqual(html.count("<script>"), 2)

    def test_static_cli_fast_path(self):
        with tempfile.TemporaryDirectory(prefix="op trace static ") as root:
            output = os.path.join(root, "viewer.html")
            code = (
                "import sys\n"
                "from fastllm_pytools import cli\n"
                "try:\n"
                "    cli.main(['op-trace-static', '--output', " + repr(output) + "])\n"
                "except SystemExit as error:\n"
                "    code=error.code\n"
                "print(int('fastllm_pytools.llm' in sys.modules),"
                "int('fastllm_pytools.server' in sys.modules))"
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = TOOLS_DIR
            completed = subprocess.run([sys.executable, "-c", code], env=env,
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip().splitlines()[-1], "0 0")
            self.assertTrue(os.path.isfile(output))
            with open(output, "r", encoding="utf-8") as handle:
                self.assertIn("__opTraceStatic", handle.read())


class DialogAndCliTest(unittest.TestCase):
    def test_windows_dialog_path(self):
        completed = type("Completed", (), {"returncode": 0,
                                            "stdout": "C:\\数据 目录\\a.ftrace\r\n".encode("utf-8"),
                                            "stderr": b""})()
        with patch.object(op_trace_dialogs.platform, "system", return_value="Windows"), \
                patch.object(op_trace_dialogs.shutil, "which", return_value="powershell.exe"), \
                patch.object(op_trace_dialogs, "_run", return_value=completed):
            result = op_trace_dialogs.pick_path("file")
        self.assertEqual(result["path"], "C:\\数据 目录\\a.ftrace")

    def test_dialog_cancel(self):
        completed = type("Completed", (), {"returncode": 0,
                                            "stdout": b"",
                                            "stderr": b""})()
        with patch.object(op_trace_dialogs.platform, "system", return_value="Windows"), \
                patch.object(op_trace_dialogs.shutil, "which", return_value="powershell.exe"), \
                patch.object(op_trace_dialogs, "_run", return_value=completed):
            result = op_trace_dialogs.pick_path("file")
        self.assertTrue(result["cancelled"])

    def test_linux_dialog_unavailable(self):
        with patch.object(op_trace_dialogs.platform, "system", return_value="Linux"), \
                patch.dict(os.environ, {}, clear=True):
            result = op_trace_dialogs.pick_path("file")
        self.assertEqual(result["error"], "dialog_unavailable")

    def test_dialog_timeout_and_busy(self):
        with patch.object(op_trace_dialogs.platform, "system", return_value="Windows"), \
                patch.object(op_trace_dialogs.shutil, "which", return_value="powershell.exe"), \
                patch.object(op_trace_dialogs, "_run",
                             side_effect=subprocess.TimeoutExpired("x", 1)):
            result = op_trace_dialogs.pick_path("file", timeout=1)
        self.assertEqual(result["error"], "dialog_timeout")
        acquired = op_trace_dialogs._DIALOG_LOCK.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            result = op_trace_dialogs.pick_path("file")
            self.assertEqual(result["error"], "dialog_busy")
        finally:
            op_trace_dialogs._DIALOG_LOCK.release()

    def test_cli_fast_path(self):
        code = (
            "import sys;"
            "from fastllm_pytools import cli;"
            "code=0\n"
            "try:\n"
            "    cli.main(['op-trace-view','--help'])\n"
            "except SystemExit as error:\n"
            "    code=error.code\n"
            "print(code, int('fastllm_pytools.llm' in sys.modules),"
            "int('fastllm_pytools.server' in sys.modules))"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS_DIR
        completed = subprocess.run([sys.executable, "-c", code], env=env,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip().splitlines()[-1], "0 0 0")


if __name__ == "__main__":
    unittest.main()
