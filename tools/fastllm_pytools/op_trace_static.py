from __future__ import annotations

import argparse
import os
import sys

from .op_trace_ui import PAGE_HTML


STATIC_STYLE = r"""
<style>
  #staticHint {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    padding: 8px 16px;
    background: #102030;
    border-bottom: 1px solid #25374a;
    color: #9fb3c8;
    font-size: 12px;
  }
  #staticHint b { color: #e6edf3; }
  #staticHint input[type=file] { color: #a9b7c8; font-size: 12px; }
  #staticHint .static-file { color: #7dd3fc; }
  #openDir, #uploadTrace, #uploadFile, #packTrace, #exportTrace,
  #manageTrace, #sessionPanel, #closeTrace, #pathWrap, #openPath {
    display: none !important;
  }
</style>
"""


STATIC_BOOTSTRAP = r"""
(() => {
  "use strict";

  const PHASES = ["none", "prefill", "decode", "mixed"];
  const MAX_TOKENS = 8192;
  const MAX_PAGE = 200000;
  const UINT32_MAX = 0xFFFFFFFF;
  const textDecoder = new TextDecoder();
  const state = {
    zip: null,
    session: null,
    pendingFile: null,
    pendingToken: "",
    pickResolve: null,
  };

  function response(payload, status) {
    const code = status || 200;
    return {
      ok: code >= 200 && code < 300,
      status: code,
      headers: { get: () => null },
      json: async () => payload,
      text: async () => JSON.stringify(payload),
    };
  }

  function errorResponse(code, message, status) {
    return response({ error: code, message: message || code }, status || 200);
  }

  function fail(message) {
    throw new Error(message);
  }

  function u16(view, pos) {
    return view.getUint16(pos, true);
  }

  function u32(view, pos) {
    return view.getUint32(pos, true);
  }

  function u64(view, pos) {
    const value = view.getBigUint64(pos, true);
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      fail("value is too large");
    }
    return Number(value);
  }

  function u64Time(view, pos) {
    return Number(view.getBigUint64(pos, true));
  }

  function findEocd(view) {
    const length = view.byteLength;
    for (let pos = length - 22; pos >= Math.max(0, length - 65557); pos -= 1) {
      if (u32(view, pos) === 0x06054b50) {
        return pos;
      }
    }
    fail("ZIP end record not found");
  }

  function readZip64(view, eocd) {
    const locator = eocd - 20;
    if (locator < 0 || u32(view, locator) !== 0x07064b50) {
      fail("ZIP64 locator not found");
    }
    const offset = u64(view, locator + 8);
    if (offset + 56 > view.byteLength || u32(view, offset) !== 0x06064b50) {
      fail("ZIP64 end record not found");
    }
    return {
      count: u64(view, offset + 32),
      size: u64(view, offset + 40),
      offset: u64(view, offset + 48),
    };
  }

  function readZip(buffer) {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    const eocd = findEocd(view);
    let count = u16(view, eocd + 10);
    let centralSize = u32(view, eocd + 12);
    let centralOffset = u32(view, eocd + 16);
    if (count === 0xFFFF || centralSize === UINT32_MAX || centralOffset === UINT32_MAX) {
      const zip64 = readZip64(view, eocd);
      count = zip64.count;
      centralSize = zip64.size;
      centralOffset = zip64.offset;
    }
    if (centralOffset + centralSize > bytes.length) {
      fail("ZIP central directory is truncated");
    }
    const files = new Map();
    let pos = centralOffset;
    for (let index = 0; index < count; index += 1) {
      if (pos + 46 > bytes.length || u32(view, pos) !== 0x02014b50) {
        fail("invalid ZIP central directory");
      }
      const method = u16(view, pos + 10);
      let compressedSize = u32(view, pos + 20);
      let uncompressedSize = u32(view, pos + 24);
      const nameLength = u16(view, pos + 28);
      const extraLength = u16(view, pos + 30);
      const commentLength = u16(view, pos + 32);
      let localOffset = u32(view, pos + 42);
      const nameStart = pos + 46;
      const name = textDecoder.decode(bytes.subarray(nameStart, nameStart + nameLength));
      const extraStart = nameStart + nameLength;
      let extraPos = extraStart;
      const extraEnd = extraStart + extraLength;
      while (extraPos + 4 <= extraEnd) {
        const tag = u16(view, extraPos);
        const size = u16(view, extraPos + 2);
        let valuePos = extraPos + 4;
        if (tag === 1) {
          if (uncompressedSize === UINT32_MAX && valuePos + 8 <= extraEnd) {
            uncompressedSize = u64(view, valuePos);
            valuePos += 8;
          }
          if (compressedSize === UINT32_MAX && valuePos + 8 <= extraEnd) {
            compressedSize = u64(view, valuePos);
            valuePos += 8;
          }
          if (localOffset === UINT32_MAX && valuePos + 8 <= extraEnd) {
            localOffset = u64(view, valuePos);
          }
        }
        extraPos += 4 + size;
      }
      files.set(name, {
        method,
        compressedSize,
        uncompressedSize,
        localOffset,
      });
      pos = extraEnd + commentLength;
    }
    return { bytes, view, files };
  }

  async function readEntry(zip, name) {
    const entry = zip.files.get(name);
    if (!entry) {
      fail("ZIP entry not found: " + name);
    }
    const pos = entry.localOffset;
    if (pos + 30 > zip.bytes.length || u32(zip.view, pos) !== 0x04034b50) {
      fail("invalid ZIP local header: " + name);
    }
    const nameLength = u16(zip.view, pos + 26);
    const extraLength = u16(zip.view, pos + 28);
    const start = pos + 30 + nameLength + extraLength;
    const end = start + entry.compressedSize;
    if (end > zip.bytes.length) {
      fail("ZIP entry is truncated: " + name);
    }
    const compressed = zip.bytes.subarray(start, end);
    if (entry.method === 0) {
      return compressed;
    }
    if (entry.method !== 8) {
      fail("unsupported ZIP compression method");
    }
    if (typeof DecompressionStream !== "function") {
      fail("this browser does not support deflate-raw");
    }
    const stream = new Blob([compressed]).stream().pipeThrough(
      new DecompressionStream("deflate-raw"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  async function readTextEntry(zip, name) {
    const bytes = await readEntry(zip, name);
    return textDecoder.decode(bytes);
  }

  function parseIndex(text) {
    const entries = [];
    for (const line of text.split(/\r?\n/)) {
      if (!line.trim()) continue;
      try {
        const item = JSON.parse(line);
        if (item && item.file) entries.push(item);
      } catch (error) {
        fail("invalid chunks.idx");
      }
    }
    return entries;
  }

  function chunkHeader(bytes) {
    if (bytes.length < 36) fail("chunk header is truncated");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (textDecoder.decode(bytes.subarray(0, 4)) !== "FOTR") {
      fail("invalid chunk magic");
    }
    const version = u16(view, 4);
    if (version !== 1) fail("unsupported chunk version");
    const count = u32(view, 8);
    if (!count) fail("empty chunk");
    return {
      count,
      firstSeq: u64(view, 12),
      baseNs: u64Time(view, 20),
      stringsSize: u32(view, 28),
      payloadSize: u32(view, 32),
    };
  }

  function decodeChunk(bytes, name) {
    const header = chunkHeader(bytes);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const payloadStart = 36 + header.stringsSize;
    const total = payloadStart + header.payloadSize;
    if (total > bytes.length) fail("chunk is truncated: " + name);
    const strings = [];
    let stringPos = 36;
    const stringCount = u16(view, stringPos);
    stringPos += 2;
    for (let index = 0; index < stringCount; index += 1) {
      const length = u16(view, stringPos);
      stringPos += 2;
      if (stringPos + length > payloadStart) fail("string table is truncated");
      strings.push(textDecoder.decode(bytes.subarray(stringPos, stringPos + length)));
      stringPos += length;
    }
    if (stringPos !== payloadStart) fail("string table length mismatch");
    let pos = payloadStart;
    function u8() {
      if (pos >= total) fail("chunk payload is truncated");
      return bytes[pos++];
    }
    function nextU16() {
      if (pos + 2 > total) fail("chunk payload is truncated");
      const value = u16(view, pos);
      pos += 2;
      return value;
    }
    function nextU32() {
      if (pos + 4 > total) fail("chunk payload is truncated");
      const value = u32(view, pos);
      pos += 4;
      return value;
    }
    function nextU64() {
      if (pos + 8 > total) fail("chunk payload is truncated");
      const value = u64(view, pos);
      pos += 8;
      return value;
    }
    function nextU64Time() {
      if (pos + 8 > total) fail("chunk payload is truncated");
      const value = u64Time(view, pos);
      pos += 8;
      return value;
    }
    function varint() {
      let value = 0;
      let factor = 1;
      for (let index = 0; index < 10; index += 1) {
        const byte = u8();
        value += (byte & 0x7F) * factor;
        if (!(byte & 0x80)) return value;
        factor *= 128;
      }
      fail("varint overflow");
    }
    function zigzag() {
      const value = varint();
      return value % 2 ? -(value + 1) / 2 : value / 2;
    }
    const tidCount = u8();
    const tids = [];
    for (let index = 0; index < tidCount; index += 1) tids.push(nextU64Time());
    const seqs = [];
    let seq = header.firstSeq;
    for (let index = 0; index < header.count; index += 1) {
      const delta = varint();
      if (index === 0) {
        seq = header.firstSeq;
      } else {
        seq += delta;
        if (!delta) fail("seq is not increasing");
      }
      seqs.push(seq);
    }
    const baseUs = header.baseNs / 1000;
    const times = [];
    let previousUs = 0;
    for (let index = 0; index < header.count; index += 1) {
      const delta = zigzag();
      const current = index === 0 ? delta : previousUs + delta;
      previousUs = current;
      times.push(baseUs + current);
    }
    const durations = [];
    for (let index = 0; index < header.count; index += 1) durations.push(varint());
    const tidIds = [];
    for (let index = 0; index < header.count; index += 1) tidIds.push(u8());
    const layers = [];
    for (let index = 0; index < header.count; index += 1) layers.push(zigzag());
    const requests = [];
    for (let index = 0; index < header.count; index += 1) requests.push(zigzag());
    const tokens = [];
    for (let index = 0; index < header.count; index += 1) tokens.push(zigzag());
    const gaps = [];
    for (let index = 0; index < header.count; index += 1) gaps.push(u8());
    const phases = [];
    for (let index = 0; index < header.count; index += 1) phases.push(u8());
    const opIds = [];
    for (let index = 0; index < header.count; index += 1) opIds.push(nextU16());
    const devIds = [];
    for (let index = 0; index < header.count; index += 1) devIds.push(nextU16());
    const weightIds = [];
    for (let index = 0; index < header.count; index += 1) weightIds.push(nextU32());
    const ioIds = [];
    for (let index = 0; index < header.count; index += 1) ioIds.push(nextU32());
    const xfer = [];
    for (let index = 0; index < header.count; index += 1) xfer.push(varint());
    const reshape = [];
    for (let index = 0; index < header.count; index += 1) reshape.push(varint());
    const run = [];
    for (let index = 0; index < header.count; index += 1) run.push(varint());
    const sync = [];
    for (let index = 0; index < header.count; index += 1) sync.push(varint());
    if (pos !== total) fail("payload length mismatch");
    function stringAt(value) {
      if (value < 0 || value >= strings.length) fail("string id out of range");
      return strings[value];
    }
    const events = [];
    for (let index = 0; index < header.count; index += 1) {
      const phaseCode = phases[index];
      if (phaseCode >= PHASES.length) fail("invalid phase code");
      const tidId = tidIds[index];
      const event = {
        seq: seqs[index],
        t_us: times[index],
        dur_ns: durations[index],
        dur_us: Math.floor(durations[index] / 1000),
        tid: tidId < tids.length ? tids[tidId] : 0,
        layer: layers[index],
        phase: PHASES[phaseCode],
        req: requests[index],
        token: tokens[index],
        op: stringAt(opIds[index]),
        dev: stringAt(devIds[index]),
        weight: weightIds[index] === UINT32_MAX ? "" : stringAt(weightIds[index]),
        io: ioIds[index] === UINT32_MAX ? "" : stringAt(ioIds[index]),
        xfer_us: Math.floor(xfer[index] / 1000),
        reshape_us: Math.floor(reshape[index] / 1000),
        run_us: Math.floor(run[index] / 1000),
        sync_us: Math.floor(sync[index] / 1000),
        _start_ns: times[index] * 1000,
        _xfer_ns: xfer[index],
        _run_ns: run[index],
        _sync_ns: sync[index],
      };
      if (gaps[index]) event.gap = 1;
      events.push(event);
    }
    return events;
  }

  function eventRange(entry) {
    if (entry.t0_us != null && entry.t1_us != null) {
      return [Number(entry.t0_us) || 0, Number(entry.t1_us) || 0];
    }
    return [0, 0];
  }

  function sessionRange() {
    if (!state.session || !state.session.entries.length) {
      return { from_us: 0, to_us: 0 };
    }
    const starts = [];
    const ends = [];
    for (const entry of state.session.entries) {
      const range = eventRange(entry);
      starts.push(range[0]);
      ends.push(range[1]);
    }
    return {
      from_us: Math.min.apply(null, starts),
      to_us: Math.max.apply(null, ends),
    };
  }

  function chunkKey(entry) {
    return entry.file;
  }

  async function getChunk(entry) {
    if (!state.session.chunks.has(chunkKey(entry))) {
      state.session.chunks.set(chunkKey(entry), null);
      const bytes = await readEntry(state.zip, entry.file);
      const events = decodeChunk(bytes, entry.file);
      state.session.chunks.set(chunkKey(entry), events);
    }
    return state.session.chunks.get(chunkKey(entry));
  }

  function publicEvent(event) {
    const result = {};
    for (const key of Object.keys(event)) {
      if (!key.startsWith("_")) result[key] = event[key];
    }
    return result;
  }

  function tokenKey(req, token) {
    return String(req) + ":" + String(token);
  }

  function newToken(req, token) {
    return {
      req,
      token,
      first_seq: 0,
      last_seq: 0,
      events: 0,
      total_us: 0,
      latency_us: 0,
      run_us: 0,
      xfer_us: 0,
      sync_us: 0,
      other_op_us: 0,
      gap_us: 0,
      gap_attr_us: 0,
      cuda_us: 0,
      cpu_us: 0,
      other_us: 0,
      start_us: 0,
      end_us: 0,
      has_split: false,
    };
  }

  function addToken(target, item) {
    const first = Number(item.first_seq) || 0;
    const last = Number(item.last_seq) || 0;
    const start = Number(item.start_us) || 0;
    const end = Number(item.end_us) || 0;
    if (!target.events) {
      target.first_seq = first;
      target.start_us = start;
    } else {
      target.first_seq = Math.min(target.first_seq, first);
      target.start_us = Math.min(target.start_us, start);
    }
    target.last_seq = Math.max(target.last_seq, last);
    target.end_us = Math.max(target.end_us, end);
    target.events += Number(item.events) || 0;
    for (const field of ["total_us", "run_us", "xfer_us", "sync_us",
                         "other_op_us", "gap_attr_us", "cuda_us", "cpu_us",
                         "other_us"]) {
      target[field] += Number(item[field]) || 0;
    }
    target.has_split = target.has_split || !!item.has_split;
  }

  function addEventToken(target, event) {
    if (!target.events) {
      target.first_seq = event.seq;
      target.start_us = event.t_us;
    }
    target.last_seq = event.seq;
    target.events += 1;
    target.total_us += event.dur_us;
    if (!event.gap) {
      target.has_split = true;
      target.run_us += event.run_us;
      target.xfer_us += event.xfer_us;
      target.sync_us += event.sync_us;
      const split = event.run_us + event.xfer_us + event.sync_us;
      if (event.dur_us > split) target.other_op_us += event.dur_us - split;
      if (event.dev === "cuda") target.cuda_us += event.dur_us;
      else if (event.dev === "cpu") target.cpu_us += event.dur_us;
      else target.other_us += event.dur_us;
    } else {
      target.gap_attr_us += event.dur_us;
    }
    target.end_us = Math.max(target.end_us, event.t_us + event.dur_us);
  }

  function finalizeToken(token) {
    token.latency_us = Math.max(0, token.end_us - token.start_us);
    token.gap_us = Math.max(0, token.latency_us - token.total_us);
    return token;
  }

  async function summaryFromEvents(maxTokens) {
    const phases = {};
    for (const phase of PHASES) phases[phase] = { events: 0, us: 0, tokens: 0 };
    const tokens = new Map();
    for (const entry of state.session.entries) {
      for (const event of await getChunk(entry)) {
        const phase = PHASES.includes(event.phase) ? event.phase : "none";
        phases[phase].events += 1;
        phases[phase].us += event.dur_us;
        if (phase !== "decode" || event.token < 0) continue;
        const key = tokenKey(event.req, event.token);
        let target = tokens.get(key);
        if (!target) {
          target = newToken(event.req, event.token);
          tokens.set(key, target);
        }
        addEventToken(target, event);
      }
    }
    phases.decode.tokens = tokens.size;
    const list = Array.from(tokens.values()).map(finalizeToken);
    list.sort((a, b) => b.end_us - a.end_us || b.last_seq - a.last_seq);
    const selected = list.slice(0, maxTokens);
    return {
      schema: 1,
      session: String(state.session.meta.session || ""),
      phases,
      tokens: selected,
      tokens_total: list.length,
      tokens_returned: selected.length,
      tokens_truncated: list.length > selected.length,
    };
  }

  function summaryFromIndex(maxTokens) {
    const entries = state.session.entries;
    for (const entry of entries) {
      const eventCount = PHASES.reduce((total, phase) =>
        total + (Number((entry.phases || {})[phase]) || 0), 0);
      if (eventCount > 0 && !entry.phase_us) return null;
      if ((Number((entry.phases || {}).decode) || 0) > 0 &&
          !Array.isArray(entry.tokens)) {
        return null;
      }
    }
    const phases = {};
    for (const phase of PHASES) phases[phase] = { events: 0, us: 0, tokens: 0 };
    const tokens = new Map();
    for (const entry of entries) {
      for (const phase of PHASES) {
        phases[phase].events += Number((entry.phases || {})[phase]) || 0;
        phases[phase].us += Number((entry.phase_us || {})[phase]) || 0;
      }
      for (const item of entry.tokens || []) {
        const token = Number(item.token);
        if (!(token >= 0)) continue;
        const req = Number(item.req == null ? -1 : item.req);
        const key = tokenKey(req, token);
        let target = tokens.get(key);
        if (!target) {
          target = newToken(req, token);
          tokens.set(key, target);
        }
        addToken(target, item);
      }
    }
    phases.decode.tokens = tokens.size;
    const list = Array.from(tokens.values()).map(finalizeToken);
    list.sort((a, b) => b.end_us - a.end_us || b.last_seq - a.last_seq);
    const selected = list.slice(0, maxTokens);
    return {
      schema: 1,
      session: String(state.session.meta.session || ""),
      phases,
      tokens: selected,
      tokens_total: list.length,
      tokens_returned: selected.length,
      tokens_truncated: list.length > selected.length,
    };
  }

  async function summary(maxTokens) {
    const limit = Math.max(1, Math.min(Number(maxTokens) || MAX_TOKENS, MAX_TOKENS));
    return summaryFromIndex(limit) || await summaryFromEvents(limit);
  }

  function entryEventCount(entry) {
    if (entry.event_count != null) return Number(entry.event_count) || 0;
    return PHASES.reduce((total, phase) =>
      total + (Number((entry.phases || {})[phase]) || 0), 0);
  }

  async function archivePage(params) {
    const fromSeq = Math.max(0, Number(params.get("from_seq")) || 0);
    const limit = Math.max(1, Math.min(Number(params.get("limit")) || 8192, MAX_PAGE));
    const phase = params.get("phase") || "";
    const req = params.has("req") ? Number(params.get("req")) : null;
    const token = params.has("token") ? Number(params.get("token")) : null;
    const op = params.get("op") || "";
    const minDur = Number(params.get("min_dur_us")) || 0;
    const fromUs = params.has("from_us") ? Number(params.get("from_us")) : null;
    const toUs = params.has("to_us") ? Number(params.get("to_us")) : null;
    const entries = state.session.entries;
    const totalEvents = entries.reduce((total, entry) => total + entryEventCount(entry), 0);
    const visibleLast = state.session.visibleLastSeq;
    const range = sessionRange();
    if (!entries.length || !visibleLast || fromSeq > visibleLast) {
      return { schema: 1, session: state.session.meta.session || "", range,
               events: [], next_seq: fromSeq <= visibleLast ? fromSeq : 0,
               has_more: false, total_events: totalEvents };
    }
    const events = [];
    let lastSeq = 0;
    let hasMore = false;
    const candidates = entries.filter((entry) => (Number(entry.last_seq) || 0) >= fromSeq);
    for (const entry of candidates) {
      const chunk = await getChunk(entry);
      for (const event of chunk) {
        if (event.seq < fromSeq) continue;
        if (phase && event.phase !== phase) continue;
        if (req !== null && Number(event.req) !== req) continue;
        if (token !== null && Number(event.token) !== token) continue;
        if (op && event.op !== op) continue;
        if (event.dur_us < minDur) continue;
        if (fromUs !== null && event.t_us < fromUs) continue;
        if (toUs !== null && event.t_us > toUs) continue;
        events.push(publicEvent(event));
        lastSeq = event.seq;
        if (events.length >= limit) {
          hasMore = lastSeq < visibleLast;
          break;
        }
      }
      if (events.length >= limit) break;
    }
    return { schema: 1, session: state.session.meta.session || "", range, events,
             next_seq: events.length ? lastSeq + 1 : 0,
             has_more: hasMore, total_events: totalEvents };
  }

  async function archiveFind(params) {
    const phase = params.get("phase") || "";
    const req = params.has("req") ? Number(params.get("req")) : null;
    const token = params.has("token") ? Number(params.get("token")) : null;
    if (!phase && req === null && token === null) return {};
    let firstSeq = 0;
    let lastSeq = 0;
    let t0 = 0;
    let t1 = 0;
    let count = 0;
    for (const entry of state.session.entries) {
      for (const event of await getChunk(entry)) {
        if (phase && event.phase !== phase) continue;
        if (req !== null && Number(event.req) !== req) continue;
        if (token !== null && Number(event.token) !== token) continue;
        if (!firstSeq) firstSeq = event.seq;
        lastSeq = event.seq;
        if (!t0 || event.t_us < t0) t0 = event.t_us;
        t1 = Math.max(t1, event.t_us + event.dur_us);
        count += 1;
      }
    }
    return count ? { first_seq: firstSeq, last_seq: lastSeq, t0_us: t0,
                     t1_us: t1, count } : {};
  }

  function archiveInfo() {
    const archive = Object.assign({}, state.session.meta.archive || {});
    archive.packed = true;
    return archive;
  }

  function statusPayload() {
    if (!state.session) {
      return { supported: true, enabled: false, mode: "viewer", session: null,
               path: null, sealed: false, packed: false, archive: {},
               read_only: true, root: "", packing: false, pack_error: "",
               packed_path: "" };
    }
    return {
      supported: true,
      enabled: false,
      mode: "viewer",
      session: String(state.session.meta.session || ""),
      path: state.session.path,
      sealed: true,
      packed: true,
      archive: archiveInfo(),
      read_only: true,
      root: "",
      packing: false,
      pack_error: "",
      packed_path: state.session.path,
      range: sessionRange(),
      chunks: state.session.entries.length,
    };
  }

  function openPayload() {
    return Object.assign(statusPayload(), {
      meta: state.session.meta,
      has_summary: state.session.entries.length > 0,
      range: sessionRange(),
      chunks: state.session.entries.length,
    });
  }

  async function loadFile(file) {
    if (!file) fail("请选择 .ftrace 文件");
    const buffer = await file.arrayBuffer();
    const zip = readZip(buffer);
    const metaText = (await readTextEntry(zip, "meta.json")).replace(/^\uFEFF/, "");
    const meta = JSON.parse(metaText);
    if (!meta || !meta.session) fail("meta.json 缺少 session");
    let entries = [];
    if (zip.files.has("chunks.idx")) {
      entries = parseIndex(await readTextEntry(zip, "chunks.idx"));
    }
    if (!entries.length) {
      const names = Array.from(zip.files.keys())
        .filter((name) => name.startsWith("chunks/") && name.endsWith(".bin"))
        .sort();
      for (const name of names) {
        const bytes = await readEntry(zip, name);
        const header = chunkHeader(bytes);
        entries.push({
          file: name,
          first_seq: header.firstSeq,
          last_seq: header.firstSeq + header.count - 1,
          event_count: header.count,
          t0_us: header.baseNs / 1000,
          t1_us: header.baseNs / 1000,
        });
      }
    }
    entries.sort((a, b) => (Number(a.first_seq) || 0) - (Number(b.first_seq) || 0));
    if (!entries.length) fail("没有可读的 chunk");
    const visibleLastSeq = Math.max(
      ...entries.map((entry) => Number(entry.last_seq) || 0));
    state.zip = zip;
    state.session = { meta, entries, path: file.name, chunks: new Map(),
                      visibleLastSeq };
    state.pendingFile = null;
    state.pendingToken = "";
    const nameNode = document.querySelector("#staticName");
    if (nameNode) nameNode.textContent = file.name;
  }

  function pushFile(file) {
    state.pendingFile = file;
    state.pendingToken = "browser:" + Date.now();
  }

  async function pickPayload() {
    if (state.pendingFile) {
      return response({ path: state.pendingToken });
    }
    const input = document.querySelector("#staticFile");
    if (!input) return response({ error: "file_input_missing" }, 500);
    return await new Promise((resolve) => {
      state.pickResolve = (payload) => {
        state.pickResolve = null;
        resolve(payload);
      };
      const timer = setTimeout(() => {
        if (state.pickResolve) {
          const resolve = state.pickResolve;
          state.pickResolve = null;
          state.pickFinish = null;
          resolve(response({ cancelled: true }));
        }
      }, 60000);
      const finish = (payload) => {
        clearTimeout(timer);
        state.pickFinish = null;
        if (state.pickResolve) state.pickResolve(payload);
      };
      state.pickFinish = finish;
      input.value = "";
      input.click();
      window.addEventListener("focus", () => {
        setTimeout(() => {
          if (state.pickResolve && state.pendingFile) {
            finish(response({ path: state.pendingToken }));
          } else if (state.pickResolve) {
            finish(response({ cancelled: true }));
          }
        }, 500);
      }, { once: true });
    });
  }

  function apiPath(parsed) {
    const marker = "/api/op-trace/";
    const index = parsed.pathname.indexOf(marker);
    return index >= 0 ? parsed.pathname.slice(index) : parsed.pathname;
  }

  async function handleApi(url, options) {
    const parsed = new URL(url, location.href);
    const path = apiPath(parsed);
    if (path === "/api/op-trace/status") return response(statusPayload());
    if (path === "/api/op-trace/snapshot") {
      return response({ supported: true, enabled: false, mode: "viewer",
                        capacity: 0, size: 0, pinned: 0, pinned_capacity: 0,
                        count: 0, first_seq: 0,
                        last_seq: state.session ? state.session.visibleLastSeq : 0,
                        dropped: 0, lost: false,
                        session_id: state.session ? state.session.meta.session : "",
                        archive: state.session ? archiveInfo() : {},
                        events: [] });
    }
    if (path === "/api/op-trace/summary") {
      if (!state.session) return errorResponse("no_session", "Open a trace first", 400);
      try {
        return response(await summary(parsed.searchParams.get("max_tokens")));
      } catch (error) {
        return errorResponse("invalid_trace", error.message, 400);
      }
    }
    if (path === "/api/op-trace/archive/page") {
      if (!state.session) return errorResponse("no_session", "Open a trace first", 400);
      try {
        return response(await archivePage(parsed.searchParams));
      } catch (error) {
        return errorResponse("invalid_page", error.message, 400);
      }
    }
    if (path === "/api/op-trace/archive/find") {
      if (!state.session) return errorResponse("no_session", "Open a trace first", 400);
      try {
        return response(await archiveFind(parsed.searchParams));
      } catch (error) {
        return errorResponse("invalid_find", error.message, 400);
      }
    }
    if (path === "/api/op-trace/pick") return await pickPayload();
    if (path === "/api/op-trace/open") {
      if (!state.pendingFile) return errorResponse("no_file", "请选择 .ftrace 文件", 400);
      try {
        await loadFile(state.pendingFile);
        return response(openPayload());
      } catch (error) {
        return errorResponse("invalid_trace", error.message, 400);
      }
    }
    if (path === "/api/op-trace/close") {
      state.session = null;
      state.zip = null;
      state.pendingFile = null;
      const nameNode = document.querySelector("#staticName");
      if (nameNode) nameNode.textContent = "未选择";
      return response({ closed: true, mode: "viewer" });
    }
    if (path === "/api/op-trace/sessions") return response({ root: "", sessions: [] });
    if (path === "/api/op-trace/export") {
      return errorResponse("static_mode", "静态查看器不提供服务端导出", 403);
    }
    if (path === "/api/op-trace/pack") {
      return errorResponse("static_mode", "静态文件已经打包", 403);
    }
    return errorResponse("not_found", "not found", 404);
  }

  const originalFetch = window.fetch.bind(window);
  window.fetch = async (input, options) => {
    const url = typeof input === "string" ? input : input.url;
    const parsed = new URL(url, location.href);
    if (!apiPath(parsed).startsWith("/api/op-trace/")) {
      return originalFetch(input, options);
    }
    return await handleApi(url, options);
  };

  const hint = document.createElement("div");
  hint.id = "staticHint";
  hint.innerHTML = '<b>静态 Trace 查看器</b><span>选择 .ftrace 文件后在本页解析，数据不会上传；Ctrl/Cmd+O 也可打开。</span>' +
    '<input id="staticFile" type="file" accept=".ftrace,application/zip">' +
    '<span id="staticName" class="static-file">未选择</span>';
  const dashboard = document.querySelector(".dashboard");
  if (dashboard) document.body.insertBefore(hint, dashboard);
  else document.body.appendChild(hint);
  const input = document.querySelector("#staticFile");
  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;
    pushFile(file);
    input.value = "";
    if (state.pickFinish) {
      state.pickFinish(response({ path: state.pendingToken }));
    } else {
      const openButton = document.querySelector("#openFile");
      if (openButton) openButton.click();
    }
  });
  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "o") {
      event.preventDefault();
      input.click();
    }
  });

  window.__opTraceStatic = {
    buildStaticHtml: true,
    loadFile,
    getSession: () => state.session,
    version: 1,
  };
})();
"""


def build_static_html() -> str:
    html = PAGE_HTML
    marker = "<script>"
    if marker not in html:
        raise RuntimeError("op trace page template has no script")
    return html.replace(
        marker,
        STATIC_STYLE + "<script>" + STATIC_BOOTSTRAP + "</script>\n" + marker,
        1,
    )


def write_static_html(path: str) -> str:
    target = os.path.abspath(os.path.expanduser(os.fspath(path)))
    parent = os.path.dirname(target) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_static_html())
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ftllm op-trace-static",
        description="Generate a standalone operator trace viewer",
    )
    parser.add_argument("--output", default="op_trace_viewer.html",
                        help="output HTML path")
    args = parser.parse_args(argv)
    try:
        target = write_static_html(args.output)
    except OSError as error:
        print(f"[op-trace-static] {error}")
        return 2
    print(f"[op-trace-static] wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
