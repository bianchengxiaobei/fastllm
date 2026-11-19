from __future__ import annotations

import bisect
import hashlib
import json
import os
import posixpath
import shutil
import socket
import struct
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict


SCHEMA_VERSION = 1
CHUNK_VERSION = 1
CHUNK_MAGIC = b"FOTR"
PHASE_ORDER = ("prefill", "decode", "mixed", "none")
PHASE_BY_CODE = ("none", "prefill", "decode", "mixed")
MAX_SUMMARY_TOKENS = 8192
MAX_PAGE_EVENTS = 200000
MAX_CACHE_CHUNKS = 8
MAX_CHUNK_OPS = 32
MAX_CHUNK_GAPS = 32
MAX_CHUNK_DEVICES = 64
MAX_SUMMARY_OPS = 64
MAX_SUMMARY_GAPS = 32
MAX_SUMMARY_DEVICES = 64
PACK_LOCK = threading.Lock()


class TraceFormatError(Exception):
    pass


def default_archive_root() -> str:
    value = os.environ.get("FASTLLM_OP_TRACE_DIR", "").strip()
    if value:
        return os.path.abspath(os.path.expanduser(value))
    return os.path.abspath(os.path.join(os.getcwd(), ".fastllm", "op_trace"))


def _json_loads(text: str, name: str):
    try:
        return json.loads(text)
    except Exception as error:
        raise TraceFormatError(f"{name}: invalid json: {error}") from error


def _read_text(source, name: str):
    try:
        data = source.read(name)
    except FileNotFoundError:
        return None
    return data.decode("utf-8-sig")


def _normalize_relpath(value) -> str:
    text = str(value or "").replace("\\", "/")
    path = posixpath.normpath(text)
    if path in ("", ".", "..") or path.startswith("../") or path.startswith("/"):
        raise TraceFormatError(f"invalid chunk path: {value!r}")
    if not path.startswith("chunks/"):
        raise TraceFormatError(f"invalid chunk path: {value!r}")
    if ":" in path.split("/", 1)[0]:
        raise TraceFormatError(f"invalid chunk path: {value!r}")
    return path


def _coerce_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float_us(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, 0, int(pid))
            if not handle:
                return kernel32.GetLastError() == 5
            kernel32.CloseHandle(handle)
            return True
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _owner_alive(meta: dict) -> bool:
    pid = _coerce_int(meta.get("pid"), 0)
    if pid <= 0:
        return False
    host = str(meta.get("host") or "")
    if host and host.casefold() != socket.gethostname().casefold():
        return False
    return _pid_alive(pid)


def owner_alive(meta: dict) -> bool:
    return _owner_alive(meta)


class _DirSource:
    is_dir = True

    def __init__(self, root: str):
        self.root = root

    def read(self, name: str) -> bytes:
        path = os.path.join(self.root, name.replace("/", os.sep))
        with open(path, "rb") as handle:
            return handle.read()

    def size(self, name: str) -> int:
        return os.path.getsize(os.path.join(self.root, name.replace("/", os.sep)))

    def exists(self, name: str) -> bool:
        return os.path.isfile(os.path.join(self.root, name.replace("/", os.sep)))

    def list_chunks(self):
        root = os.path.join(self.root, "chunks")
        try:
            names = os.listdir(root)
        except FileNotFoundError:
            return []
        result = []
        for name in names:
            if not name.endswith(".bin"):
                continue
            stem = name[:-4]
            if len(stem) == 5 and stem.isdigit():
                result.append("chunks/" + name)
        result.sort()
        return result

    def fingerprint(self):
        meta = os.path.join(self.root, "meta.json")
        index = os.path.join(self.root, "chunks.idx")
        try:
            meta_stat = os.stat(meta)
        except FileNotFoundError:
            meta_stat = None
        try:
            index_stat = os.stat(index)
        except FileNotFoundError:
            index_stat = None
        return (
            meta_stat.st_mtime_ns if meta_stat else 0,
            meta_stat.st_size if meta_stat else 0,
            index_stat.st_mtime_ns if index_stat else 0,
            index_stat.st_size if index_stat else 0,
        )

    def close(self):
        return None


class _ZipSource:
    is_dir = False

    def __init__(self, path: str):
        self.path = path
        self.zip = zipfile.ZipFile(path, "r")
        self.names = {}
        for name in self.zip.namelist():
            normalized = name.replace("\\", "/")
            if normalized.startswith("/") or normalized.startswith("../"):
                continue
            self.names[normalized] = name

    def read(self, name: str) -> bytes:
        normalized = name.replace("\\", "/")
        member = self.names.get(normalized)
        if member is None:
            raise FileNotFoundError(normalized)
        return self.zip.read(member)

    def size(self, name: str) -> int:
        normalized = name.replace("\\", "/")
        member = self.names.get(normalized)
        if member is None:
            raise FileNotFoundError(normalized)
        return self.zip.getinfo(member).file_size

    def exists(self, name: str) -> bool:
        return name.replace("\\", "/") in self.names

    def list_chunks(self):
        result = []
        for name in self.names:
            if not name.startswith("chunks/") or not name.endswith(".bin"):
                continue
            stem = name[len("chunks/"):-4]
            if len(stem) == 5 and stem.isdigit():
                result.append(name)
        result.sort()
        return result

    def fingerprint(self):
        stat = os.stat(self.path)
        return (stat.st_mtime_ns, stat.st_size)

    def close(self):
        try:
            self.zip.close()
        except Exception:
            pass


class _ByteReader:
    __slots__ = ("data", "pos", "end")

    def __init__(self, data: bytes, pos: int = 0, end: int | None = None):
        self.data = data
        self.pos = pos
        self.end = len(data) if end is None else end

    def take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > self.end:
            raise TraceFormatError("truncated chunk")
        value = self.data[self.pos:self.pos + size]
        self.pos += size
        return value

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def varint(self) -> int:
        value = 0
        shift = 0
        for _ in range(10):
            byte = self.u8()
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                return value
            shift += 7
        raise TraceFormatError("varint overflow")

    def zigzag(self) -> int:
        value = self.varint()
        return (value >> 1) ^ -(value & 1)


def _decode_chunk(data: bytes, name: str = "") -> tuple[dict, list[dict]]:
    if len(data) < 36:
        raise TraceFormatError(f"{name}: chunk header is truncated")
    magic, version, flags, count, first_seq, base_t_ns, strings_size, payload_size = \
        struct.unpack_from("<4sHHIQQII", data, 0)
    if magic != CHUNK_MAGIC:
        raise TraceFormatError(f"{name}: invalid chunk magic")
    if version != CHUNK_VERSION:
        raise TraceFormatError(f"{name}: unsupported chunk version {version}")
    if count <= 0:
        raise TraceFormatError(f"{name}: empty chunk")
    if first_seq <= 0:
        raise TraceFormatError(f"{name}: invalid first seq")
    total = 36 + strings_size + payload_size
    if total > len(data):
        raise TraceFormatError(f"{name}: chunk is truncated")
    payload_start = 36 + strings_size
    payload_end = total

    strings_reader = _ByteReader(data, 36, payload_start)
    string_count = strings_reader.u16()
    strings = []
    for _ in range(string_count):
        length = strings_reader.u16()
        raw = strings_reader.take(length)
        try:
            strings.append(raw.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise TraceFormatError(f"{name}: invalid utf-8 string") from error
    if strings_reader.pos != payload_start:
        raise TraceFormatError(f"{name}: string table length mismatch")

    reader = _ByteReader(data, payload_start, payload_end)
    tid_count = reader.u8()
    tids = [reader.u64() for _ in range(tid_count)]

    seqs = []
    seq = first_seq
    for index in range(count):
        delta = reader.varint()
        if index == 0:
            seq = first_seq
        else:
            seq += delta
            if delta == 0:
                raise TraceFormatError(f"{name}: seq is not increasing")
        seqs.append(seq)

    base_us = base_t_ns // 1000
    event_us = []
    previous_us = 0
    for index in range(count):
        delta = reader.zigzag()
        current = delta if index == 0 else previous_us + delta
        previous_us = current
        event_us.append(base_us + current)

    durations = [reader.varint() for _ in range(count)]
    tid_ids = [reader.u8() for _ in range(count)]
    layers = [reader.zigzag() for _ in range(count)]
    requests = [reader.zigzag() for _ in range(count)]
    tokens = [reader.zigzag() for _ in range(count)]
    gaps = [reader.u8() for _ in range(count)]
    phases = [reader.u8() for _ in range(count)]
    op_ids = [reader.u16() for _ in range(count)]
    dev_ids = [reader.u16() for _ in range(count)]
    weight_ids = [reader.u32() for _ in range(count)]
    io_ids = [reader.u32() for _ in range(count)]
    xfer = [reader.varint() for _ in range(count)]
    reshape = [reader.varint() for _ in range(count)]
    run = [reader.varint() for _ in range(count)]
    sync = [reader.varint() for _ in range(count)]
    if reader.pos != payload_end:
        raise TraceFormatError(f"{name}: payload length mismatch")

    def string_at(value: int) -> str:
        if value >= len(strings):
            raise TraceFormatError(f"{name}: string id out of range")
        return strings[value]

    events = []
    for index in range(count):
        phase_code = phases[index]
        if phase_code >= len(PHASE_BY_CODE):
            raise TraceFormatError(f"{name}: invalid phase code")
        tid_id = tid_ids[index]
        tid = tids[tid_id] if tid_id < len(tids) else 0
        event = {
            "seq": seqs[index],
            "t_us": event_us[index],
            "dur_ns": durations[index],
            "dur_us": durations[index] // 1000,
            "tid": tid,
            "layer": layers[index],
            "phase": PHASE_BY_CODE[phase_code],
            "req": requests[index],
            "token": tokens[index],
            "op": string_at(op_ids[index]),
            "dev": string_at(dev_ids[index]),
            "weight": "" if weight_ids[index] == 0xFFFFFFFF else string_at(weight_ids[index]),
            "io": "" if io_ids[index] == 0xFFFFFFFF else string_at(io_ids[index]),
            "xfer_us": xfer[index] // 1000,
            "reshape_us": reshape[index] // 1000,
            "run_us": run[index] // 1000,
            "sync_us": sync[index] // 1000,
            "_start_ns": event_us[index] * 1000,
            "_xfer_ns": xfer[index],
            "_reshape_ns": reshape[index],
            "_run_ns": run[index],
            "_sync_ns": sync[index],
        }
        if gaps[index]:
            event["gap"] = 1
        events.append(event)

    header = {
        "version": version,
        "flags": flags,
        "event_count": count,
        "first_seq": first_seq,
        "base_t_ns": base_t_ns,
        "strings_size": strings_size,
        "payload_size": payload_size,
        "bytes": total,
    }
    return header, events


def _public_event(event: dict) -> dict:
    return {key: value for key, value in event.items() if not key.startswith("_")}


CHROME_TRACE_PID = 1


def _chrome_meta(name: str, args: dict, tid: int = 0) -> dict:
    return {
        "name": name,
        "ph": "M",
        "pid": CHROME_TRACE_PID,
        "tid": tid,
        "args": args,
    }


def _chrome_trace_event(event: dict, base_us: int) -> dict:
    phase = str(event.get("phase") or "none")
    gap = bool(event.get("gap"))
    args = {
        "seq": int(event.get("seq", 0)),
        "phase": phase,
        "layer": int(event.get("layer", -1)),
        "req": int(event.get("req", -1)),
        "token": int(event.get("token", -1)),
        "dev": str(event.get("dev") or ""),
        "weight": str(event.get("weight") or ""),
        "io": str(event.get("io") or ""),
        "t_us": int(event.get("t_us", 0)),
        "dur_ns": int(event.get("dur_ns", 0)),
        "xfer_us": int(event.get("xfer_us", 0)),
        "reshape_us": int(event.get("reshape_us", 0)),
        "run_us": int(event.get("run_us", 0)),
        "sync_us": int(event.get("sync_us", 0)),
    }
    if gap:
        args["gap"] = 1
    return {
        "name": str(event.get("op") or "event"),
        "cat": "gap" if gap else phase,
        "ph": "X",
        "ts": max(0, int(event.get("t_us", 0)) - int(base_us)),
        "dur": max(0, int(event.get("dur_us", 0))),
        "pid": CHROME_TRACE_PID,
        "tid": max(0, int(event.get("tid", 0))),
        "args": args,
    }


def _new_summary_token(req: int, token: int) -> dict:
    return {
        "req": req,
        "token": token,
        "first_seq": 0,
        "last_seq": 0,
        "events": 0,
        "total_us": 0,
        "latency_us": 0,
        "run_us": 0,
        "xfer_us": 0,
        "sync_us": 0,
        "other_op_us": 0,
        "gap_us": 0,
        "gap_attr_us": 0,
        "cuda_us": 0,
        "cpu_us": 0,
        "other_us": 0,
        "start_us": 0,
        "end_us": 0,
        "has_split": False,
    }


def _add_token_event(target: dict, event: dict):
    if target["events"] == 0:
        target["first_seq"] = event["seq"]
        target["start_us"] = event["t_us"]
    target["last_seq"] = event["seq"]
    target["events"] += 1
    duration = int(event.get("dur_ns", 0))
    target["total_us"] += duration // 1000
    if event.get("gap"):
        target["gap_attr_us"] += duration // 1000
    else:
        target["has_split"] = True
        target["run_us"] += int(event.get("_run_ns", 0)) // 1000
        target["xfer_us"] += int(event.get("_xfer_ns", 0)) // 1000
        target["sync_us"] += int(event.get("_sync_ns", 0)) // 1000
        split = int(event.get("_run_ns", 0)) + int(event.get("_xfer_ns", 0)) + \
            int(event.get("_sync_ns", 0))
        if duration > split:
            target["other_op_us"] += (duration - split) // 1000
        if event.get("dev") == "cuda":
            target["cuda_us"] += duration // 1000
        elif event.get("dev") == "cpu":
            target["cpu_us"] += duration // 1000
        else:
            target["other_us"] += duration // 1000
    end = event["t_us"] + duration // 1000
    if target["events"] == 1 or end > target["end_us"]:
        target["end_us"] = end


def _finalize_token(target: dict):
    target["latency_us"] = max(0, int(target["end_us"]) - int(target["start_us"]))
    target["gap_us"] = max(0, int(target["latency_us"]) - int(target["total_us"]))


def _add_op(target: dict, phase: str, op: str, duration_ns: int):
    key = (phase, op)
    item = target.get(key)
    if item is None:
        item = {"phase": phase, "op": op, "total_us": 0, "count": 0, "max_us": 0, "_ns": 0}
        target[key] = item
    item["_ns"] += duration_ns
    item["count"] += 1
    item["total_us"] = item["_ns"] // 1000
    item["max_us"] = max(item["max_us"], duration_ns // 1000)


def _ops_json(target: dict, limit: int):
    items = sorted(target.values(), key=lambda item: (item["_ns"], item["op"]), reverse=True)
    result = []
    for item in items[:limit]:
        result.append({
            "phase": item["phase"],
            "op": item["op"],
            "total_us": int(item["total_us"]),
            "count": int(item["count"]),
            "max_us": int(item["max_us"]),
        })
    return result


def _build_index_entry(file: str, events: list[dict], size: int) -> dict:
    if not events:
        raise TraceFormatError("cannot index an empty chunk")
    phases = {phase: 0 for phase in PHASE_ORDER}
    phase_ns = {phase: 0 for phase in PHASE_ORDER}
    requests = {}
    tokens = {}
    ops = {}
    gaps = {}
    devices = {}
    t0_us = 0
    t1_us = 0
    for event in events:
        phase = event.get("phase") or "none"
        if phase not in phases:
            phase = "none"
        duration = int(event.get("dur_ns", 0))
        phases[phase] += 1
        phase_ns[phase] += duration
        start_us = int(event.get("t_us", 0))
        end_us = start_us + duration // 1000
        if t0_us == 0 or start_us < t0_us:
            t0_us = start_us
        if end_us > t1_us:
            t1_us = end_us
        req = int(event.get("req", -1))
        request = requests.get(req)
        if request is None:
            requests[req] = {"req": req, "first_seq": event["seq"], "last_seq": event["seq"]}
        else:
            request["last_seq"] = event["seq"]
        token_value = int(event.get("token", -1))
        if phase == "decode" and token_value >= 0:
            key = (req, token_value)
            target = tokens.get(key)
            if target is None:
                target = _new_summary_token(req, token_value)
                tokens[key] = target
            _add_token_event(target, event)
        if event.get("gap"):
            _add_op(gaps, phase, str(event.get("op", "")), duration)
        else:
            _add_op(ops, phase, str(event.get("op", "")), duration)
        dev = str(event.get("dev", ""))
        device = devices.get(dev)
        if device is None:
            devices[dev] = {"dev": dev, "events": 1, "_ns": duration}
        else:
            device["events"] += 1
            device["_ns"] += duration
    token_list = []
    for key in sorted(tokens):
        target = tokens[key]
        _finalize_token(target)
        token_list.append({
            "req": target["req"],
            "token": target["token"],
            "first_seq": target["first_seq"],
            "last_seq": target["last_seq"],
            "events": target["events"],
            "total_us": target["total_us"],
            "latency_us": target["latency_us"],
            "run_us": target["run_us"],
            "xfer_us": target["xfer_us"],
            "sync_us": target["sync_us"],
            "gap_us": target["gap_us"],
            "gap_attr_us": target["gap_attr_us"],
            "cuda_us": target["cuda_us"],
            "cpu_us": target["cpu_us"],
            "other_us": target["other_us"],
            "other_op_us": target["other_op_us"],
            "start_us": target["start_us"],
            "end_us": target["end_us"],
            "has_split": target["has_split"],
        })
    device_list = []
    for item in sorted(devices.values(), key=lambda value: (value["_ns"], value["dev"]), reverse=True):
        device_list.append({"dev": item["dev"], "events": item["events"], "us": item["_ns"] // 1000})
    return {
        "file": file,
        "first_seq": int(events[0]["seq"]),
        "last_seq": int(events[-1]["seq"]),
        "t0_us": t0_us,
        "t1_us": t1_us,
        "phases": phases,
        "phase_us": {phase: phase_ns[phase] // 1000 for phase in PHASE_ORDER},
        "reqs": [requests[key] for key in sorted(requests)],
        "tokens": token_list,
        "ops": _ops_json(ops, MAX_CHUNK_OPS),
        "gaps": _ops_json(gaps, MAX_CHUNK_GAPS),
        "devices": device_list[:MAX_CHUNK_DEVICES],
        "bytes": int(size),
    }


def _parse_index_text(text: str, tolerant: bool = True):
    entries = []
    invalid = False
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            path = _normalize_relpath(item.get("file"))
            first_seq = int(item.get("first_seq"))
            last_seq = int(item.get("last_seq"))
            if first_seq <= 0 or last_seq < first_seq:
                raise ValueError("invalid seq range")
            phases = {phase: _coerce_int((item.get("phases") or {}).get(phase), 0)
                      for phase in PHASE_ORDER}
            entry = dict(item)
            entry["file"] = path
            entry["first_seq"] = first_seq
            entry["last_seq"] = last_seq
            entry["phases"] = phases
            entry.setdefault("t0_us", 0)
            entry.setdefault("t1_us", 0)
            entry.setdefault("reqs", [])
            entry.setdefault("tokens", [])
            entry.setdefault("ops", [])
            entries.append(entry)
        except Exception:
            invalid = True
            if tolerant:
                break
            raise TraceFormatError("invalid chunks.idx entry")
    entries.sort(key=lambda entry: (entry["first_seq"], entry["last_seq"]))
    seen = set()
    unique = []
    for entry in entries:
        if entry["file"] in seen:
            invalid = True
            continue
        seen.add(entry["file"])
        unique.append(entry)
    return unique, invalid


def _scan_source(source):
    entries = []
    for name in source.list_chunks():
        try:
            data = source.read(name)
            _, events = _decode_chunk(data, name)
            entries.append(_build_index_entry(name, events, len(data)))
        except TraceFormatError:
            continue
    entries.sort(key=lambda entry: entry["first_seq"])
    return entries


def _chunk_number(name: str):
    base = posixpath.basename(name)
    if not base.endswith(".bin"):
        return None
    stem = base[:-4]
    return int(stem) if len(stem) == 5 and stem.isdigit() else None


def _decode_indexed_chunk(source, name: str):
    data = source.read(name)
    header, events = _decode_chunk(data, name)
    if int(header.get("event_count", 0)) != len(events):
        raise TraceFormatError(f"{name}: event count mismatch")
    for index, event in enumerate(events):
        if int(event["seq"]) != int(header["first_seq"]) + index:
            raise TraceFormatError(f"{name}: non-contiguous seq")
    return events, len(data)


def _write_atomic(path: str, data: bytes):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01 * (attempt + 1))


def recover(path: str) -> dict:
    path = os.path.abspath(os.fspath(path))
    if not os.path.isdir(path):
        session = Session.open(path, recover=False, read_only=True)
        try:
            return dict(session.meta)
        finally:
            session.close()
    meta_path = os.path.join(path, "meta.json")
    if not os.path.isfile(meta_path):
        raise TraceFormatError("meta.json not found")
    with open(meta_path, "r", encoding="utf-8-sig") as handle:
        meta = _json_loads(handle.read(), "meta.json")
    if _coerce_int(meta.get("schema"), 0) != SCHEMA_VERSION:
        raise TraceFormatError("unsupported trace schema")
    source = _DirSource(path)
    index_text = _read_text(source, "chunks.idx")
    indexed = []
    if index_text is not None:
        indexed, _ = _parse_index_text(index_text, tolerant=True)
    entries = []
    if indexed:
        previous_last = 0
        previous_number = None
        for entry in indexed:
            number = _chunk_number(entry["file"])
            if number is None:
                break
            if previous_number is not None and number != previous_number + 1:
                break
            if previous_last and entry["first_seq"] != previous_last + 1:
                break
            try:
                events, size = _decode_indexed_chunk(source, entry["file"])
            except TraceFormatError:
                break
            if int(entry["first_seq"]) != int(events[0]["seq"]) or \
                    int(entry["last_seq"]) != int(events[-1]["seq"]):
                break
            entries.append(_build_index_entry(entry["file"], events, size))
            previous_last = int(events[-1]["seq"])
            previous_number = number
    else:
        previous_last = 0
        previous_number = None
        for name in source.list_chunks():
            number = _chunk_number(name)
            if number is None:
                break
            if previous_number is not None and number != previous_number + 1:
                break
            try:
                events, size = _decode_indexed_chunk(source, name)
            except TraceFormatError:
                break
            if previous_last and int(events[0]["seq"]) != previous_last + 1:
                break
            entries.append(_build_index_entry(name, events, size))
            previous_last = int(events[-1]["seq"])
            previous_number = number
    index_lines = []
    bytes_total = 0
    for entry in entries:
        bytes_total += _coerce_int(entry.get("bytes"), 0)
        index_lines.append(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
    index_text = "".join(line + "\n" for line in index_lines)
    _write_atomic(os.path.join(path, "chunks.idx"), index_text.encode("utf-8"))

    archive = meta.setdefault("archive", {})
    archive["committed_seq"] = int(entries[-1]["last_seq"]) if entries else 0
    archive["chunks"] = len(entries)
    archive["bytes"] = bytes_total
    archive["dropped"] = _coerce_int(archive.get("dropped"), 0)
    if not meta.get("sealed"):
        meta["sealed"] = True
        archive["state"] = "sealed" if archive.get("state") in ("recording", "idle") \
            else archive.get("state", "sealed")
        if meta.get("ended_at_ns") is None:
            meta["ended_at_ns"] = time.time_ns()
        elif not meta.get("ended_at_ns"):
            meta["ended_at_ns"] = time.time_ns()
    if not archive.get("state"):
        archive["state"] = "sealed"
    pack_error = str(archive.get("pack_error") or "")
    packed = bool(archive.get("packed"))
    ftrace = archive.get("ftrace")
    if packed and ftrace:
        candidate = os.fspath(ftrace)
        if not os.path.isabs(candidate):
            candidate = os.path.join(path, candidate)
        if not os.path.isfile(candidate) and candidate != path:
            packed = False
            pack_error = "missing_ftrace"
            ftrace = None
    archive["packed"] = packed
    archive["ftrace"] = ftrace
    archive["pack_error"] = pack_error
    _write_atomic(meta_path, json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    source.close()
    return meta


def _merge_summary(entries, max_tokens: int, session: str):
    phases = {phase: {"events": 0, "us": 0, "tokens": 0} for phase in PHASE_ORDER}
    tokens = {}
    ops = {}
    gaps = {}
    devices = {}
    for entry in entries:
        entry_phases = entry.get("phases") or {}
        entry_phase_us = entry.get("phase_us") or {}
        for phase in PHASE_ORDER:
            phases[phase]["events"] += _coerce_int(entry_phases.get(phase), 0)
            phases[phase]["us"] += _coerce_int(entry_phase_us.get(phase), 0)
        for token in entry.get("tokens") or []:
            req = _coerce_int(token.get("req"), -1)
            value = _coerce_int(token.get("token"), -1)
            if value < 0:
                continue
            key = (req, value)
            target = tokens.get(key)
            if target is None:
                target = _new_summary_token(req, value)
                tokens[key] = target
            first_seq = _coerce_int(token.get("first_seq"), 0)
            last_seq = _coerce_int(token.get("last_seq"), 0)
            if target["events"] == 0:
                target["first_seq"] = first_seq
                target["start_us"] = _coerce_int(token.get("start_us"), 0)
            else:
                target["first_seq"] = min(target["first_seq"], first_seq)
            target["last_seq"] = max(target["last_seq"], last_seq)
            target["events"] += _coerce_int(token.get("events"), 0)
            for field in ("total_us", "run_us", "xfer_us", "sync_us", "other_op_us",
                          "gap_attr_us", "cuda_us", "cpu_us", "other_us"):
                target[field] += _coerce_int(token.get(field), 0)
            target["start_us"] = min(target["start_us"], _coerce_int(token.get("start_us"), 0)) \
                if target["events"] else target["start_us"]
            target["end_us"] = max(target["end_us"], _coerce_int(token.get("end_us"), 0))
            target["has_split"] = target["has_split"] or bool(token.get("has_split"))
        for item in entry.get("ops") or []:
            duration = _coerce_int(item.get("total_us"), 0) * 1000
            key = (str(item.get("phase", "none")), str(item.get("op", "")))
            target = ops.get(key)
            if target is None:
                target = {"phase": key[0], "op": key[1], "total_us": 0, "count": 0,
                          "max_us": 0, "_ns": 0}
                ops[key] = target
            target["_ns"] += duration
            target["total_us"] = target["_ns"] // 1000
            target["count"] += _coerce_int(item.get("count"), 0)
            target["max_us"] = max(target["max_us"], _coerce_int(item.get("max_us"), 0))
        for item in entry.get("gaps") or []:
            duration = _coerce_int(item.get("total_us"), 0) * 1000
            key = (str(item.get("phase", "none")), str(item.get("op", "")))
            target = gaps.get(key)
            if target is None:
                target = {"phase": key[0], "op": key[1], "total_us": 0, "count": 0,
                          "max_us": 0, "_ns": 0}
                gaps[key] = target
            target["_ns"] += duration
            target["total_us"] = target["_ns"] // 1000
            target["count"] += _coerce_int(item.get("count"), 0)
            target["max_us"] = max(target["max_us"], _coerce_int(item.get("max_us"), 0))
        for item in entry.get("devices") or []:
            key = str(item.get("dev", ""))
            target = devices.get(key)
            if target is None:
                target = {"dev": key, "events": 0, "_ns": 0}
                devices[key] = target
            target["events"] += _coerce_int(item.get("events"), 0)
            target["_ns"] += _coerce_int(item.get("us"), 0) * 1000
    total_tokens = len(tokens)
    if max_tokens <= 0:
        max_tokens = MAX_SUMMARY_TOKENS
    max_tokens = min(max_tokens, MAX_SUMMARY_TOKENS)
    token_list = []
    for target in tokens.values():
        _finalize_token(target)
        token_list.append({
            "req": target["req"],
            "token": target["token"],
            "first_seq": target["first_seq"],
            "last_seq": target["last_seq"],
            "events": target["events"],
            "total_us": target["total_us"],
            "latency_us": target["latency_us"],
            "run_us": target["run_us"],
            "xfer_us": target["xfer_us"],
            "sync_us": target["sync_us"],
            "other_op_us": target["other_op_us"],
            "gap_us": target["gap_us"],
            "gap_attr_us": target["gap_attr_us"],
            "cuda_us": target["cuda_us"],
            "cpu_us": target["cpu_us"],
            "other_us": target["other_us"],
            "start_us": target["start_us"],
            "end_us": target["end_us"],
            "has_split": target["has_split"],
        })
    token_list.sort(key=lambda item: (item["end_us"], item["last_seq"]), reverse=True)
    selected = token_list[:max_tokens]
    phases["decode"]["tokens"] = total_tokens
    device_list = [
        {"dev": item["dev"], "events": item["events"], "us": item["_ns"] // 1000}
        for item in sorted(devices.values(), key=lambda value: (value["_ns"], value["dev"]),
                           reverse=True)
    ]
    return {
        "schema": SCHEMA_VERSION,
        "session": session,
        "phases": phases,
        "tokens": selected,
        "tokens_total": total_tokens,
        "tokens_returned": len(selected),
        "tokens_truncated": total_tokens > len(selected),
        "ops": _ops_json(ops, MAX_SUMMARY_OPS),
        "gaps": _ops_json(gaps, MAX_SUMMARY_GAPS),
        "devices": device_list[:MAX_SUMMARY_DEVICES],
    }


class Session:
    def __init__(self, path: str, source, meta: dict, entries: list[dict],
                 read_only: bool, recover_allowed: bool):
        self.path = path
        self.source = source
        self.meta = meta
        self.entries = entries
        self.read_only = read_only
        self.recover_allowed = recover_allowed
        self.sealed = bool(meta.get("sealed"))
        archive = meta.get("archive") or {}
        self.visible_last_seq = _coerce_int(archive.get("committed_seq"), 0)
        if self.sealed:
            self.visible_last_seq = entries[-1]["last_seq"] if entries else 0
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._summary_cache = None
        self._closed = False
        for entry in self._visible_entries()[:2]:
            try:
                self._decode(entry["file"])
            except TraceFormatError:
                break

    @classmethod
    def open(cls, path: str, recover: bool = True, read_only: bool = False) -> "Session":
        path = os.path.abspath(os.fspath(path))
        if not os.path.exists(path):
            raise TraceFormatError(f"path not found: {path}")
        if os.path.isdir(path):
            return cls._open_dir(path, recover, read_only)
        if not path.lower().endswith(".ftrace"):
            raise TraceFormatError("trace file must end with .ftrace")
        return cls._open_zip(path)

    @classmethod
    def _open_zip(cls, path: str) -> "Session":
        try:
            source = _ZipSource(path)
        except (OSError, zipfile.BadZipFile) as error:
            raise TraceFormatError(f"invalid .ftrace file: {error}") from error
        try:
            meta_text = _read_text(source, "meta.json")
            if meta_text is None:
                raise TraceFormatError("meta.json not found in .ftrace")
            meta = _json_loads(meta_text, "meta.json")
            if _coerce_int(meta.get("schema"), 0) != SCHEMA_VERSION:
                raise TraceFormatError("unsupported trace schema")
            index_text = _read_text(source, "chunks.idx")
            if index_text is None:
                entries = _scan_source(source)
            else:
                entries, invalid = _parse_index_text(index_text, tolerant=False)
                if invalid:
                    raise TraceFormatError("invalid chunks.idx")
                if not entries:
                    entries = _scan_source(source)
            for entry in entries:
                if not source.exists(entry["file"]):
                    raise TraceFormatError(f"missing chunk: {entry['file']}")
            if not meta.get("sealed"):
                meta["sealed"] = True
            return cls(path, source, meta, entries, True, False)
        except Exception:
            source.close()
            raise

    @classmethod
    def _open_dir(cls, path: str, recover_allowed: bool, read_only: bool) -> "Session":
        source = _DirSource(path)
        meta_text = _read_text(source, "meta.json")
        if meta_text is None:
            raise TraceFormatError("meta.json not found")
        meta = _json_loads(meta_text, "meta.json")
        if _coerce_int(meta.get("schema"), 0) != SCHEMA_VERSION:
            raise TraceFormatError("unsupported trace schema")
        index_text = _read_text(source, "chunks.idx")
        entries = []
        invalid = False
        if index_text is not None:
            entries, invalid = _parse_index_text(index_text, tolerant=True)
        chunk_files = source.list_chunks()
        missing = [entry["file"] for entry in entries if not source.exists(entry["file"])]
        owner_alive = _owner_alive(meta)
        needs_recover = bool(
            (index_text is None and chunk_files) or
            invalid or
            missing or
            (chunk_files and not entries) or
            (not meta.get("sealed") and not owner_alive)
        )
        if needs_recover and owner_alive:
            needs_recover = False
        if needs_recover and recover_allowed and not read_only:
            source.close()
            recover(path)
            source = _DirSource(path)
            meta_text = _read_text(source, "meta.json")
            meta = _json_loads(meta_text or "", "meta.json")
            index_text = _read_text(source, "chunks.idx")
            entries = []
            if index_text is not None:
                entries, _ = _parse_index_text(index_text, tolerant=False)
        elif needs_recover and (index_text is None or not entries):
            entries = _scan_source(source)
        if not meta.get("sealed"):
            if not owner_alive and (read_only or not recover_allowed):
                meta["sealed"] = True
            else:
                meta["sealed"] = False
        for entry in entries:
            if not source.exists(entry["file"]):
                if recover_allowed and not read_only and not owner_alive:
                    source.close()
                    recover(path)
                    return cls._open_dir(path, False, read_only)
                raise TraceFormatError(f"missing chunk: {entry['file']}")
        return cls(path, source, meta, entries, bool(read_only or not meta.get("sealed")),
                   recover_allowed)

    def _visible_entries(self):
        if self.sealed:
            return list(self.entries)
        if self.visible_last_seq <= 0:
            return []
        return [entry for entry in self.entries if entry["first_seq"] <= self.visible_last_seq]

    def _cache_key(self):
        return (self.source.fingerprint(), self.visible_last_seq, len(self.entries))

    def _decode(self, name: str):
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                self._cache.move_to_end(name)
                return cached
        data = self.source.read(name)
        header, events = _decode_chunk(data, name)
        result = (header, events)
        with self._lock:
            self._cache[name] = result
            while len(self._cache) > MAX_CACHE_CHUNKS:
                self._cache.popitem(last=False)
        return result

    def _scan_entries(self):
        result = []
        for entry in self._visible_entries():
            try:
                _, events = self._decode(entry["file"])
                result.append(_build_index_entry(entry["file"], events,
                                                 _coerce_int(entry.get("bytes"), 0)))
            except TraceFormatError:
                continue
        return result

    def _summary_entries(self):
        entries = self._visible_entries()
        for entry in entries:
            if "phase_us" not in entry or "devices" not in entry or "gaps" not in entry:
                return self._scan_entries()
        return entries

    def summary(self, max_tokens: int = MAX_SUMMARY_TOKENS) -> dict:
        max_tokens = int(max_tokens or 0)
        if max_tokens <= 0:
            max_tokens = MAX_SUMMARY_TOKENS
        max_tokens = min(max_tokens, MAX_SUMMARY_TOKENS)
        key = (self._cache_key(), max_tokens)
        with self._lock:
            if self._summary_cache and self._summary_cache[0] == key:
                return self._summary_cache[1]
            result = _merge_summary(self._summary_entries(), max_tokens,
                                    str(self.meta.get("session") or ""))
            self._summary_cache = (key, result)
            return result

    def page(self, from_seq: int = 0, limit: int = 8192, phase: str | None = None,
             req: int | None = None, token: int | None = None, op: str | None = None,
             min_dur_us: int = 0, from_us: int | None = None,
             to_us: int | None = None) -> dict:
        from_seq = max(0, int(from_seq or 0))
        limit = int(limit or 8192)
        limit = max(1, min(limit, MAX_PAGE_EVENTS))
        entries = self._visible_entries()
        total_events = sum(sum(_coerce_int((entry.get("phases") or {}).get(item), 0)
                               for item in PHASE_ORDER) for entry in entries)
        if not entries or self.visible_last_seq <= 0 or from_seq > self.visible_last_seq:
            return {
                "schema": SCHEMA_VERSION,
                "session": str(self.meta.get("session") or ""),
                "range": self.range(),
                "events": [],
                "next_seq": from_seq if from_seq <= self.visible_last_seq else 0,
                "has_more": False,
                "total_events": total_events,
            }
        last_seqs = [entry["last_seq"] for entry in entries]
        start_index = bisect.bisect_left(last_seqs, from_seq) if from_seq > 0 else 0
        events = []
        last_seq = 0
        has_more = False
        for entry in entries[start_index:]:
            if entry["last_seq"] < from_seq:
                continue
            _, chunk_events = self._decode(entry["file"])
            for event in chunk_events:
                seq = int(event["seq"])
                if seq < from_seq:
                    continue
                if not self.sealed and seq > self.visible_last_seq:
                    continue
                if phase and event.get("phase") != phase:
                    continue
                if req is not None and int(event.get("req", -1)) != int(req):
                    continue
                if token is not None and int(event.get("token", -1)) != int(token):
                    continue
                if op is not None and event.get("op") != op:
                    continue
                if int(event.get("dur_us", 0)) < int(min_dur_us or 0):
                    continue
                if from_us is not None and int(event.get("t_us", 0)) < int(from_us):
                    continue
                if to_us is not None and int(event.get("t_us", 0)) > int(to_us):
                    continue
                events.append(_public_event(event))
                last_seq = seq
                if len(events) >= limit:
                    has_more = last_seq < self.visible_last_seq
                    break
            if len(events) >= limit:
                has_more = last_seq < self.visible_last_seq
                break
        return {
            "schema": SCHEMA_VERSION,
            "session": str(self.meta.get("session") or ""),
            "range": self.range(),
            "events": events,
            "next_seq": last_seq + 1 if events else 0,
            "has_more": bool(has_more),
            "total_events": total_events,
        }

    def export_chrome_trace(self, path: str) -> int:
        target = os.path.abspath(os.path.expanduser(os.fspath(path)))
        parent = os.path.dirname(target) or os.getcwd()
        os.makedirs(parent, exist_ok=True)
        entries = self._visible_entries()
        base_us = min((_coerce_int(entry.get("t0_us"), 0) for entry in entries),
                      default=0)
        first_seq = 0
        last_seq = 0
        last_t_us = 0
        event_count = 0
        fd, temp_path = tempfile.mkstemp(prefix=".op-trace-chrome-",
                                         suffix=".tmp", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                fd = -1
                first = True

                def emit(item: dict):
                    nonlocal first
                    if not first:
                        handle.write(",")
                    handle.write(json.dumps(item, ensure_ascii=False,
                                            separators=(",", ":")))
                    first = False

                session = str(self.meta.get("session") or "")
                handle.write('{"traceEvents":[')
                emit(_chrome_meta("process_name", {
                    "name": session or "FastLLM op trace",
                    "model": str(self.meta.get("model") or ""),
                }))
                seen_tids = set()
                for entry in entries:
                    _, events = self._decode(entry["file"])
                    for event in events:
                        seq = int(event["seq"])
                        if not self.sealed and seq > self.visible_last_seq:
                            continue
                        tid = max(0, int(event.get("tid", 0)))
                        if tid not in seen_tids:
                            seen_tids.add(tid)
                            emit(_chrome_meta("thread_name", {
                                "name": f"tid {tid}",
                            }, tid))
                        emit(_chrome_trace_event(event, base_us))
                        if not first_seq:
                            first_seq = seq
                        last_seq = seq
                        last_t_us = max(last_t_us,
                                        int(event.get("t_us", 0)) +
                                        int(event.get("dur_us", 0)))
                        event_count += 1
                metadata = {
                    "schema": SCHEMA_VERSION,
                    "session": session,
                    "model": str(self.meta.get("model") or ""),
                    "event_count": event_count,
                    "first_seq": first_seq,
                    "last_seq": last_seq,
                    "from_us": base_us,
                    "to_us": last_t_us,
                }
                handle.write('],"displayTimeUnit":"ms","metadata":')
                handle.write(json.dumps(metadata, ensure_ascii=False,
                                        separators=(",", ":")))
                handle.write("}\n")
            os.replace(temp_path, target)
        except Exception:
            if fd >= 0:
                os.close(fd)
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise
        return event_count

    def find(self, phase: str | None = None, req: int | None = None,
             token: int | None = None) -> dict:
        entries = self._visible_entries()
        if not entries:
            return {}
        if token is not None:
            if phase and phase != "decode":
                return {}
            first_seq = 0
            last_seq = 0
            t0_us = 0
            t1_us = 0
            count = 0
            for entry in entries:
                for item in entry.get("tokens") or []:
                    if int(item.get("token", -1)) != int(token):
                        continue
                    if req is not None and int(item.get("req", -1)) != int(req):
                        continue
                    item_first = _coerce_int(item.get("first_seq"), 0)
                    item_last = _coerce_int(item.get("last_seq"), 0)
                    if not first_seq or (item_first and item_first < first_seq):
                        first_seq = item_first
                    last_seq = max(last_seq, item_last)
                    item_t0 = _coerce_int(item.get("start_us"), 0)
                    item_t1 = _coerce_int(item.get("end_us"), 0)
                    if not t0_us or (item_t0 and item_t0 < t0_us):
                        t0_us = item_t0
                    t1_us = max(t1_us, item_t1)
                    count += _coerce_int(item.get("events"), 0)
            if not count:
                return {}
            return {"first_seq": first_seq, "last_seq": last_seq,
                    "t0_us": t0_us, "t1_us": t1_us, "count": count}
        if req is not None and not phase:
            first_seq = 0
            last_seq = 0
            t0_us = 0
            t1_us = 0
            for entry in entries:
                for item in entry.get("reqs") or []:
                    if int(item.get("req", -1)) != int(req):
                        continue
                    item_first = _coerce_int(item.get("first_seq"), 0)
                    item_last = _coerce_int(item.get("last_seq"), 0)
                    if not first_seq or (item_first and item_first < first_seq):
                        first_seq = item_first
                    last_seq = max(last_seq, item_last)
                    t0_us = t0_us or _coerce_int(entry.get("t0_us"), 0)
                    t1_us = max(t1_us, _coerce_int(entry.get("t1_us"), 0))
            if not first_seq:
                return {}
            return {"first_seq": first_seq, "last_seq": last_seq,
                    "t0_us": t0_us, "t1_us": t1_us}
        candidates = []
        for entry in entries:
            if phase and _coerce_int((entry.get("phases") or {}).get(phase), 0) <= 0:
                continue
            candidates.append(entry)
        first_seq = 0
        last_seq = 0
        t0_us = 0
        t1_us = 0
        count = 0
        for entry in candidates:
            _, chunk_events = self._decode(entry["file"])
            for event in chunk_events:
                if not self.sealed and int(event["seq"]) > self.visible_last_seq:
                    continue
                if phase and event.get("phase") != phase:
                    continue
                if req is not None and int(event.get("req", -1)) != int(req):
                    continue
                if token is not None and int(event.get("token", -1)) != int(token):
                    continue
                seq = int(event["seq"])
                if not first_seq:
                    first_seq = seq
                last_seq = seq
                event_t0 = int(event.get("t_us", 0))
                event_t1 = event_t0 + int(event.get("dur_us", 0))
                if not t0_us or event_t0 < t0_us:
                    t0_us = event_t0
                t1_us = max(t1_us, event_t1)
                count += 1
        if not count:
            return {}
        return {"first_seq": first_seq, "last_seq": last_seq,
                "t0_us": t0_us, "t1_us": t1_us, "count": count}

    def range(self):
        entries = self._visible_entries()
        if not entries:
            return {"from_us": 0, "to_us": 0}
        return {
            "from_us": min(_coerce_int(entry.get("t0_us"), 0) for entry in entries),
            "to_us": max(_coerce_int(entry.get("t1_us"), 0) for entry in entries),
        }

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            self._summary_cache = None
            self.source.close()


def pack(path: str, output: str | None = None) -> str:
    path = os.path.abspath(os.fspath(path))
    with PACK_LOCK:
        if not os.path.isdir(path):
            if not path.lower().endswith(".ftrace"):
                raise TraceFormatError("only a session directory can be packed")
            if output is None:
                return path
            output = os.path.abspath(os.fspath(output))
            if os.path.abspath(path) == output:
                return output
            temp = output + ".tmp"
            shutil.copyfile(path, temp)
            os.replace(temp, output)
            return output

        meta_path = os.path.join(path, "meta.json")
        with open(meta_path, "r", encoding="utf-8-sig") as handle:
            meta = _json_loads(handle.read(), "meta.json")
        if _coerce_int(meta.get("schema"), 0) != SCHEMA_VERSION:
            raise TraceFormatError("unsupported trace schema")
        if not meta.get("sealed"):
            if _owner_alive(meta):
                raise TraceFormatError("session is still recording")
            meta = recover(path)
        index_path = os.path.join(path, "chunks.idx")
        if not os.path.isfile(index_path):
            meta = recover(path)
        session_id = str(meta.get("session") or os.path.basename(path))
        if output is None:
            output = os.path.join(os.path.dirname(path), session_id + ".ftrace")
        output = os.path.abspath(os.fspath(output))
        if os.path.isfile(output) and _validate_ftrace(output):
            _update_pack_meta(path, True, output, "", _sha256(output))
            return output
        chunk_names = []
        chunk_root = os.path.join(path, "chunks")
        if os.path.isdir(chunk_root):
            for name in os.listdir(chunk_root):
                if name.endswith(".bin") and len(name) == 9 and name[:-4].isdigit():
                    chunk_names.append("chunks/" + name)
        chunk_names.sort()
        total_bytes = sum(os.path.getsize(os.path.join(path, item.replace("/", os.sep)))
                          for item in chunk_names)
        try:
            free = shutil.disk_usage(os.path.dirname(output)).free
        except OSError:
            free = total_bytes
        if free < total_bytes:
            _update_pack_meta(path, False, None, "not_enough_disk", "")
            raise OSError("not enough disk space to pack trace")
        with open(index_path, "rb") as handle:
            index_bytes = handle.read()
        zip_meta = json.loads(json.dumps(meta))
        archive = zip_meta.setdefault("archive", {})
        archive["packed"] = True
        archive["ftrace"] = os.path.basename(output)
        archive["pack_error"] = ""
        archive.pop("sha256", None)
        temp = output + ".tmp"
        if os.path.exists(temp):
            os.remove(temp)
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as archive_zip:
                archive_zip.writestr("meta.json",
                                     json.dumps(zip_meta, ensure_ascii=False,
                                                separators=(",", ":")).encode("utf-8"))
                archive_zip.writestr("chunks.idx", index_bytes)
                for name in chunk_names:
                    archive_zip.write(os.path.join(path, name.replace("/", os.sep)),
                                      arcname=name)
            os.replace(temp, output)
        except Exception as error:
            try:
                if os.path.exists(temp):
                    os.remove(temp)
            except OSError:
                pass
            _update_pack_meta(path, False, None, str(error), "")
            raise
        if not _validate_ftrace(output):
            _update_pack_meta(path, False, None, "pack_validation_failed", "")
            raise TraceFormatError("packed trace validation failed")
        _update_pack_meta(path, True, output, "", _sha256(output))
        return output


def _update_pack_meta(path: str, packed: bool, ftrace: str | None,
                      error: str, digest: str):
    meta_path = os.path.join(path, "meta.json")
    with open(meta_path, "r", encoding="utf-8-sig") as handle:
        meta = _json_loads(handle.read(), "meta.json")
    archive = meta.setdefault("archive", {})
    archive["packed"] = bool(packed)
    archive["ftrace"] = os.path.abspath(ftrace) if ftrace else None
    archive["pack_error"] = str(error or "")
    if digest:
        archive["sha256"] = digest
    else:
        archive.pop("sha256", None)
    _write_atomic(meta_path, json.dumps(meta, ensure_ascii=False,
                                          separators=(",", ":")).encode("utf-8"))


def _validate_ftrace(path: str) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = {name.replace("\\", "/") for name in archive.namelist()}
            if "meta.json" not in names or "chunks.idx" not in names:
                return False
            meta = _json_loads(archive.read("meta.json").decode("utf-8-sig"), "meta.json")
            if _coerce_int(meta.get("schema"), 0) != SCHEMA_VERSION:
                return False
            text = archive.read("chunks.idx").decode("utf-8-sig")
            entries, invalid = _parse_index_text(text, tolerant=False)
            if invalid:
                return False
            for entry in entries:
                if entry["file"] not in names:
                    return False
            return True
    except Exception:
        return False


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _path_size(path: str, kind: str) -> int:
    if kind != "dir":
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    try:
        for current, _, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(current, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _delete_target(path: str, root: str | None = None):
    root = os.path.realpath(os.path.abspath(os.path.expanduser(
        os.fspath(root or default_archive_root()))))
    target = os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        within = os.path.commonpath([root, target]) == root
    except ValueError:
        within = False
    if (not within or target == root or
            os.path.normcase(os.path.dirname(target)) != os.path.normcase(root)):
        raise PermissionError("only sessions directly under the trace root can be deleted")
    if os.path.isdir(target):
        meta_path = os.path.join(target, "meta.json")
        if not os.path.isfile(meta_path):
            raise TraceFormatError("not a session directory")
        with open(meta_path, "r", encoding="utf-8-sig") as handle:
            meta = _json_loads(handle.read(), "meta.json")
        if _coerce_int(meta.get("schema"), 0) != SCHEMA_VERSION:
            raise TraceFormatError("unsupported trace schema")
        if not meta.get("sealed") and _owner_alive(meta):
            raise TraceFormatError("session is still recording")
        return target, "dir", meta
    if os.path.isfile(target):
        if not target.lower().endswith(".ftrace") or not _validate_ftrace(target):
            raise TraceFormatError("not a valid .ftrace file")
        with zipfile.ZipFile(target, "r") as archive:
            meta = _json_loads(archive.read("meta.json").decode("utf-8-sig"), "meta.json")
        return target, "ftrace", meta
    raise FileNotFoundError(f"path not found: {target}")


def delete_session(path: str, root: str | None = None, before_delete=None):
    target, kind, meta = _delete_target(path, root)
    if before_delete is not None:
        before_delete(target, kind, meta)
    if kind == "dir":
        shutil.rmtree(target)
    else:
        os.remove(target)
    return {
        "path": target,
        "kind": kind,
        "session": str(meta.get("session") or os.path.basename(target)),
    }


_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
}


def _safe_import_name(filename: str) -> str:
    base = os.path.basename(str(filename or "").replace("\\", "/"))
    base = "".join(ch if (ch.isalnum() or ch in "._- ") else "_" for ch in base)
    base = base.strip(" .")
    if not base or not base.lower().endswith(".ftrace"):
        raise TraceFormatError("import file must end with .ftrace")
    stem = base[:-7].strip(" .")[:120].rstrip(" .") or "imported"
    if stem.casefold() in _WINDOWS_RESERVED_NAMES:
        stem = "trace-" + stem
    return stem + ".ftrace"


def _move_import(source: str, target: str):
    try:
        os.link(source, target)
    except FileExistsError:
        raise
    except OSError:
        created = False
        try:
            with open(source, "rb") as src, open(target, "xb") as dst:
                created = True
                shutil.copyfileobj(src, dst, 1024 * 1024)
        except BaseException:
            if created:
                try:
                    os.remove(target)
                except OSError:
                    pass
            raise
    try:
        os.remove(source)
    except OSError:
        pass


def import_trace(source: str, filename: str, root: str | None = None):
    source = os.path.abspath(os.path.expanduser(os.fspath(source)))
    if not os.path.isfile(source):
        raise FileNotFoundError(f"path not found: {source}")
    if not _validate_ftrace(source):
        raise TraceFormatError("not a valid .ftrace file")
    root = os.path.abspath(os.path.expanduser(os.fspath(root or default_archive_root())))
    os.makedirs(root, exist_ok=True)
    name = _safe_import_name(filename)
    stem = name[:-7]
    target = ""
    for index in range(10000):
        suffix = "" if index == 0 else f"-{index}"
        target = os.path.join(root, stem + suffix + ".ftrace")
        try:
            _move_import(source, target)
            break
        except FileExistsError:
            continue
    else:
        raise TraceFormatError("too many imported files with the same name")
    with zipfile.ZipFile(target, "r") as archive:
        meta = _json_loads(archive.read("meta.json").decode("utf-8-sig"), "meta.json")
    return {
        "path": target,
        "session": str(meta.get("session") or stem),
        "size_bytes": os.path.getsize(target),
    }


def list_sessions(root: str | None = None):
    root = os.path.abspath(os.path.expanduser(os.fspath(root or default_archive_root())))
    result = []
    candidates = []
    if os.path.isfile(os.path.join(root, "meta.json")):
        candidates.append(("dir", root))
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        names = []
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "meta.json")):
            candidates.append(("dir", path))
        elif os.path.isfile(path) and name.lower().endswith(".ftrace"):
            candidates.append(("ftrace", path))
    for kind, path in candidates:
        try:
            if kind == "dir":
                with open(os.path.join(path, "meta.json"), "r", encoding="utf-8-sig") as handle:
                    meta = _json_loads(handle.read(), "meta.json")
                packed = bool((meta.get("archive") or {}).get("packed"))
                ftrace = (meta.get("archive") or {}).get("ftrace")
            else:
                with zipfile.ZipFile(path, "r") as archive:
                    meta = _json_loads(archive.read("meta.json").decode("utf-8-sig"),
                                       "meta.json")
                packed = True
                ftrace = path
            if _coerce_int(meta.get("schema"), 0) != SCHEMA_VERSION:
                continue
            archive = meta.get("archive") or {}
            sealed = bool(meta.get("sealed"))
            try:
                mtime_ns = os.stat(path).st_mtime_ns
            except OSError:
                mtime_ns = 0
            result.append({
                "path": path,
                "kind": kind,
                "session": str(meta.get("session") or os.path.basename(path)),
                "host": meta.get("host"),
                "pid": meta.get("pid"),
                "model": meta.get("model"),
                "sealed": sealed,
                "active": kind == "dir" and not sealed and _owner_alive(meta),
                "packed": packed,
                "ftrace": ftrace,
                "created_at_ns": meta.get("created_at_ns"),
                "ended_at_ns": meta.get("ended_at_ns"),
                "size_bytes": _path_size(path, kind),
                "mtime_ns": mtime_ns,
                "chunks": _coerce_int(archive.get("chunks"), 0),
                "bytes": _coerce_int(archive.get("bytes"), 0),
                "dropped": _coerce_int(archive.get("dropped"), 0),
                "state": archive.get("state", ""),
            })
        except Exception:
            continue
    result.sort(key=lambda item: (item.get("created_at_ns") or 0, item.get("session") or ""),
                reverse=True)
    return result
