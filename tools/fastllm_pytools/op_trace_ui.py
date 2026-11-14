PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FastLLM · 算子 Trace</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #090e14;
    --bg2: #0d141d;
    --panel: #111a24;
    --panel2: #15202c;
    --line: #223142;
    --line2: #192633;
    --text: #dbe5f0;
    --muted: #8493a6;
    --accent: #4ea1ff;
    --accent2: #a78bfa;
    --good: #39c27a;
    --warn: #ffb454;
    --bad: #ff6b8a;
    --cuda: #4ea1ff;
    --cpu: #ffb454;
    --other: #7f8b9d;
  }
  * { box-sizing: border-box; }
  html, body { min-height: 100%; }
  body {
    margin: 0;
    color: var(--text);
    font: 13px/1.45 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
    background:
      radial-gradient(1100px 560px at 18% -8%, rgba(78,161,255,.09), transparent 62%),
      radial-gradient(900px 520px at 100% 0%, rgba(167,139,250,.07), transparent 58%),
      var(--bg);
  }
  .topbar {
    position: sticky; top: 0; z-index: 30;
    padding: 13px 18px 11px;
    border-bottom: 1px solid var(--line);
    background: rgba(9,14,20,.84);
    backdrop-filter: blur(14px);
    display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px 22px;
  }
  .brand { min-width: 230px; }
  .title {
    margin: 0; font-size: 18px; font-weight: 650; letter-spacing: .2px;
    background: linear-gradient(90deg, #e9f1fb, #9fc4ff 65%, #c6b5ff);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .title span { color: var(--muted); font-size: 12px; font-weight: 400; margin-left: 7px; }
  .subtitle { color: var(--muted); font-size: 11px; margin-top: 1px; }
  .controls {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    flex: 1; justify-content: flex-end;
  }
  button {
    background: #16212e; color: var(--text); border: 1px solid #2b3d51;
    border-radius: 7px; padding: 5px 11px; cursor: pointer; font: inherit;
    transition: background .15s, border-color .15s, transform .15s;
  }
  button:hover { background: #1c2b3b; border-color: #3b5570; }
  button:active { transform: translateY(1px); }
  button.primary {
    background: linear-gradient(180deg, #2d6fb5, #1d4f86);
    border-color: #347fc9;
    box-shadow: 0 4px 14px rgba(45,111,181,.25);
  }
  button.primary:hover { background: linear-gradient(180deg, #347fc9, #2462a3); }
  button.ghost {
    background: transparent; color: var(--muted); border-color: #2a3b4d;
  }
  button.ghost:hover { color: var(--text); background: #15202b; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button[hidden] { display: none; }
  label {
    display: inline-flex; align-items: center; gap: 5px;
    color: var(--muted); font-size: 12px; white-space: nowrap;
  }
  select, input {
    background: #0c141d; color: var(--text); border: 1px solid #293b4f;
    border-radius: 6px; padding: 4px 7px; font: inherit; outline: none;
  }
  select:focus, input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(78,161,255,.12); }
  input[type="number"] { width: 74px; }
  input[type="checkbox"] { accent-color: var(--accent); }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 9px; border-radius: 99px; border: 1px solid #33455c;
    color: var(--muted); font-size: 11px; white-space: nowrap;
  }
  .pill.on { color: var(--good); border-color: #2b6b4a; background: #10251b; }
  .pill.on::before {
    content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: var(--good); box-shadow: 0 0 0 0 rgba(57,194,122,.55);
    animation: pulse 1.8s infinite;
  }
  .pill.off { color: var(--muted); }
  .pill.bad { color: var(--bad); border-color: #6b2b3a; background: #251016; }
  .pill[hidden] { display: none; }
  .token-chip {
    color: #ffd28a; border-color: #6b4a2b; background: #241a10; gap: 3px;
  }
  .token-chip button {
    background: transparent; border: 0; color: inherit;
    padding: 0 0 0 2px; font-size: 13px; line-height: 1; cursor: pointer;
  }
  .token-chip button:hover { color: #fff; }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(57,194,122,.5); }
    70% { box-shadow: 0 0 0 6px rgba(57,194,122,0); }
    100% { box-shadow: 0 0 0 0 rgba(57,194,122,0); }
  }
  .stats {
    display: flex; flex-wrap: wrap; align-items: center; gap: 9px 17px;
    padding: 8px 18px; border-bottom: 1px solid var(--line);
    color: var(--muted); font-size: 11px; background: rgba(13,20,29,.72);
  }
  .stat { display: inline-flex; align-items: center; gap: 5px; }
  .stat i {
    font-style: normal; color: #6f7e91; text-transform: uppercase;
    letter-spacing: .4px; font-size: 9px;
  }
  .stat b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat em { color: #3d4d61; font-style: normal; }
  .warn { margin-left: auto; color: var(--warn); font-size: 10px; text-align: right; max-width: 48%; }
  .dashboard {
    display: grid; grid-template-columns: minmax(0, 1fr) 382px;
    gap: 14px; padding: 14px 18px 18px; align-items: start;
  }
  .col-main, .col-side { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .card {
    border: 1px solid var(--line); border-radius: 12px;
    background: linear-gradient(180deg, rgba(23,32,44,.88), rgba(15,22,31,.94));
    box-shadow: 0 12px 30px rgba(0,0,0,.22);
    overflow: hidden;
  }
  .card-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; padding: 9px 12px; border-bottom: 1px solid var(--line2);
    min-height: 38px;
  }
  .card-head h2 {
    margin: 0; font-size: 12px; font-weight: 600; letter-spacing: .5px;
    text-transform: uppercase; color: #c7d5e5;
  }
  .card-head .muted, .muted { color: var(--muted); font-size: 11px; }
  .head-actions { display: inline-flex; align-items: center; gap: 7px; }
  .chart-wrap { position: relative; height: 190px; background: #0f161e; }
  canvas { display: block; }
  .timeline-wrap {
    position: relative; max-height: 56vh;
    overflow: auto; background: #0f161e;
  }
  .empty-hint {
    display: flex; align-items: center; gap: 9px; flex-wrap: wrap;
    padding: 6px 12px; border-bottom: 1px solid #3a2f1c;
    background: #1d1a10; color: var(--warn); font-size: 11px;
  }
  .empty-hint[hidden] { display: none; }
  .empty-hint button { padding: 2px 8px; font-size: 10px; }
  .tooltip {
    position: absolute; display: none; pointer-events: none; z-index: 12;
    max-width: 520px; padding: 7px 9px; border-radius: 8px;
    border: 1px solid #3b4d63; background: rgba(7,12,18,.97); color: var(--text);
    font-size: 11px; box-shadow: 0 12px 30px rgba(0,0,0,.5);
    white-space: pre-wrap; word-break: break-all;
    backdrop-filter: blur(6px);
  }
  .legend { display: inline-flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 10px; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
  .legend i { width: 7px; height: 7px; border-radius: 2px; display: inline-block; }
  .trend-legend {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
    padding: 0 17px 12px; color: var(--muted); font-size: 10px;
  }
  .trend-legend span { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
  .trend-legend i { width: 7px; height: 7px; border-radius: 2px; display: inline-block; }
  .tabs { display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 10px 2px; }
  .tab {
    background: transparent; border: 1px solid transparent; color: var(--muted);
    padding: 4px 8px; border-radius: 6px; font-size: 11px; cursor: pointer;
  }
  .tab:hover { background: #15202b; color: var(--text); }
  .tab.active { background: #132438; border-color: #2b4c72; color: #d4e9ff; }
  .hotspots { padding: 3px 8px 8px; max-height: 438px; overflow: auto; }
  .hot-row {
    padding: 6px 7px; border-radius: 8px; cursor: pointer;
    border: 1px solid transparent; transition: background .12s, border-color .12s;
  }
  .hot-row:hover { background: #152131; }
  .hot-row.focus { background: #12263a; border-color: #2d6fb5; box-shadow: inset 0 0 0 1px rgba(78,161,255,.15); }
  .hot-top { display: flex; align-items: center; gap: 7px; font-size: 12px; }
  .dot { width: 7px; height: 7px; border-radius: 2px; flex: none; display: inline-block; }
  .hot-name {
    flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; color: #d7e3f1;
  }
  .hot-val { color: var(--text); font-size: 11px; font-variant-numeric: tabular-nums; }
  .hot-bar { height: 4px; background: #1b2735; border-radius: 3px; overflow: hidden; margin: 5px 0 3px 14px; }
  .hot-bar i { display: block; height: 100%; border-radius: 3px; }
  .hot-sub {
    display: flex; flex-wrap: wrap; gap: 8px; margin-left: 14px;
    color: #728196; font-size: 10px; font-variant-numeric: tabular-nums;
  }
  .hint { padding: 0 12px 10px; color: #6f7e91; font-size: 10px; }
  .phase-grid {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px; padding: 10px 12px 6px;
  }
  .phase-box {
    border: 1px solid #2a3b50; border-radius: 8px; padding: 7px 8px;
    background: #0f1822; cursor: pointer; min-width: 0;
    transition: background .12s, border-color .12s, box-shadow .12s;
  }
  .phase-box:hover { background: #132232; border-color: #3b5570; }
  .phase-box.prefill { border-left: 3px solid #38bdf8; }
  .phase-box.decode { border-left: 3px solid #ffb454; }
  .phase-box.mixed { grid-column: 1 / -1; border-left: 3px solid #8a93a6; }
  .phase-box.active {
    background: #132438; border-color: #2d6fb5;
    box-shadow: inset 0 0 0 1px rgba(78,161,255,.15);
  }
  .phase-box i {
    display: block; font-style: normal; font-size: 9px; color: #6f7e91;
    text-transform: uppercase; letter-spacing: .4px;
  }
  .phase-box b {
    display: block; margin: 2px 0; font-size: 13px; font-weight: 600;
    font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis;
  }
  .phase-box span {
    display: block; color: var(--muted); font-size: 10px;
    font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis;
  }
  .phase-extra { padding: 0 12px 8px; color: #6f7e91; font-size: 10px; }
  .token-list { max-height: 360px; overflow: auto; padding: 4px 8px 8px; }
  .token-row {
    padding: 6px 7px; border-radius: 7px; cursor: pointer;
    border: 1px solid transparent; transition: background .12s, border-color .12s;
  }
  .token-row:hover { background: #152131; }
  .token-row.selected {
    background: #12263a; border-color: #2d6fb5;
    box-shadow: inset 0 0 0 1px rgba(78,161,255,.12);
  }
  .token-top {
    display: flex; align-items: baseline; gap: 6px; min-width: 0;
    font-size: 11px;
  }
  .token-id {
    font-weight: 600; color: #d4e9ff; white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  .token-meta { color: #6f7e91; font-size: 10px; white-space: nowrap; }
  .token-total {
    margin-left: auto; color: var(--text); font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .token-bar {
    height: 3px; background: #1b2735; border-radius: 2px;
    overflow: hidden; margin: 4px 0 3px;
  }
  .token-bar i {
    display: block; height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, #38bdf8, #ffb454);
  }
  .token-sub {
    color: #728196; font-size: 10px; font-variant-numeric: tabular-nums;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .time-split { padding: 0 12px 10px; }
  .split-stack {
    display: flex; height: 12px; border-radius: 6px; overflow: hidden;
    background: #1b2735; box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
  }
  .split-stack i { display: block; height: 100%; min-width: 1px; transition: width .2s; }
  .split-rows { display: flex; flex-direction: column; gap: 4px; margin-top: 9px; }
  .split-row {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: #b7c5d6; min-width: 0;
  }
  .split-dot { width: 8px; height: 8px; border-radius: 2px; flex: none; }
  .split-row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .split-row .val { margin-left: auto; color: var(--text); font-variant-numeric: tabular-nums; }
  .split-row .pct { width: 48px; text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
  .split-note { color: #728196; font-size: 10px; padding: 0 12px 8px; }
  .empty {
    color: var(--muted); padding: 18px 10px; text-align: center;
    border: 1px dashed #26384c; border-radius: 8px; margin: 10px; font-size: 11px;
  }
  .comp-block { padding: 0 12px 10px; }
  .comp-title {
    font-size: 9px; color: #6f7e91; text-transform: uppercase;
    letter-spacing: .5px; margin: 3px 0 6px;
  }
  .stack {
    display: flex; height: 10px; border-radius: 6px; overflow: hidden;
    background: #1b2735; box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
  }
  .stack i { display: block; height: 100%; min-width: 1px; transition: width .2s; }
  .comp-legend { display: flex; flex-direction: column; gap: 3px; margin-top: 8px; }
  .comp-row {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: #b7c5d6; min-width: 0;
  }
  .comp-row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .comp-row .val { margin-left: auto; color: var(--text); font-variant-numeric: tabular-nums; }
  .comp-row .pct { width: 48px; text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
  .fmt-label {
    font-size: 9px; color: #6f7e91; text-transform: uppercase;
    letter-spacing: .5px; padding: 7px 12px 5px;
  }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 12px; }
  .chip {
    display: inline-flex; align-items: center; gap: 5px;
    border: 1px solid #2a3b50; background: #0f1822; color: #c7d3e1;
    border-radius: 999px; padding: 3px 8px; font-size: 11px;
    transition: border-color .12s, background .12s;
  }
  .chip.clickable { cursor: pointer; }
  .chip.clickable:hover { border-color: #3c6f9f; background: #132232; }
  .chip b { color: #8fc1ff; font-weight: 600; font-variant-numeric: tabular-nums; }
  .conv-list { padding: 3px 12px 10px; }
  .conv {
    display: flex; align-items: center; gap: 7px; padding: 4px 0;
    border-bottom: 1px dashed #1d2a38; font-size: 11px; color: #b7c5d6;
    cursor: pointer;
  }
  .conv:hover { color: #fff; }
  .conv .arrow { color: var(--warn); }
  .conv b { margin-left: auto; color: var(--text); font-variant-numeric: tabular-nums; }
  .detail-card #detail, .detail-card #chain { padding: 0 12px 10px; }
  .sub-head { border-top: 1px solid var(--line2); }
  .hero { display: flex; align-items: baseline; gap: 7px; font-size: 14px; margin-bottom: 7px; }
  .hero b { font-weight: 600; overflow-wrap: anywhere; }
  .hero .muted { font-size: 10px; }
  .metric-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin: 8px 0; }
  .metric { background: #0f1822; border: 1px solid #1e2d3d; border-radius: 7px; padding: 5px 7px; min-width: 0; }
  .metric i {
    display: block; font-style: normal; color: #6f7e91; font-size: 9px;
    text-transform: uppercase; letter-spacing: .3px;
  }
  .metric b { font-size: 11px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .durbar { height: 5px; background: #1b2735; border-radius: 3px; overflow: hidden; margin: 3px 0 9px; }
  .durbar i { display: block; height: 100%; border-radius: 3px; }
  .tags { display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0; }
  .tag {
    display: inline-flex; align-items: center; gap: 4px; max-width: 100%;
    border: 1px solid #2a3b50; background: #0f1822; border-radius: 6px;
    padding: 2px 6px; font-size: 10px; color: #b7c5d6;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .section-title {
    display: flex; justify-content: space-between; gap: 8px;
    font-size: 10px; color: #6f7e91; letter-spacing: .3px; margin: 10px 0 5px;
  }
  .io-block { margin: 6px 0; }
  .io-head { font-size: 9px; color: #6f7e91; text-transform: uppercase; letter-spacing: .4px; margin: 5px 0 2px; }
  .io-line {
    display: grid; grid-template-columns: 24px minmax(0, 1fr) auto;
    gap: 5px; align-items: center; padding: 3px 0;
    border-bottom: 1px solid #1a2634; font-size: 11px;
  }
  .io-line b { font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .io-key { font-size: 8px; color: #6ea8ff; text-transform: uppercase; letter-spacing: .4px; }
  .io-key.out { color: var(--good); }
  .io-meta { color: var(--muted); font-size: 10px; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .dtype-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 5px; margin: 8px 0; }
  .badge {
    font-size: 9px; border: 1px solid #2a3b50; border-radius: 999px;
    padding: 1px 6px; color: var(--muted);
  }
  .badge.warn { color: var(--warn); border-color: #6b4a2b; background: #241a10; }
  .chain-item {
    margin: 0 0 5px 0; padding: 5px 7px; border-left: 2px solid #33455c;
    background: #0f1822; border-radius: 5px; cursor: pointer;
    font-size: 11px; word-break: break-all;
  }
  .chain-item:hover { border-left-color: var(--accent); background: #132232; }
  .chain-item .sub { color: var(--muted); font-size: 10px; }
  .chain-io { color: #728196; font-size: 10px; margin-top: 2px; }
  .table-card {
    margin: 0; border: 1px solid var(--line); border-radius: 12px;
    background: linear-gradient(180deg, rgba(23,32,44,.82), rgba(15,22,31,.92));
    overflow: hidden; box-shadow: 0 12px 30px rgba(0,0,0,.18);
  }
  .table-head {
    display: flex; align-items: center; gap: 10px; padding: 9px 12px;
    border-bottom: 1px solid var(--line2);
  }
  .table-head h2 { margin: 0; font-size: 12px; font-weight: 600; letter-spacing: .5px; text-transform: uppercase; }
  .table-head .muted { margin-left: auto; }
  .table-wrap { overflow: auto; max-height: 44vh; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  thead th {
    position: sticky; top: 0; z-index: 2; text-align: left;
    background: #131d28; color: #7d8da1; font-weight: 500;
    padding: 7px 8px; border-bottom: 1px solid var(--line); white-space: nowrap;
  }
  tbody td { padding: 5px 8px; border-bottom: 1px solid #1a2532; white-space: nowrap; }
  tbody tr { cursor: pointer; transition: background .1s; }
  tbody tr:hover { background: #14202d; }
  tbody tr.selected { background: #172c42; }
  tbody tr.dim td { opacity: .42; }
  tbody tr.selected td, tbody tr.selected.dim td { opacity: 1; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.dur { font-weight: 500; }
  td.io { max-width: 430px; overflow: hidden; text-overflow: ellipsis; color: #a9b7c8; }
  td.fmt { color: #97a7ba; max-width: 150px; overflow: hidden; text-overflow: ellipsis; }
  td.fmt.changed { color: var(--warn); }
  td.phase { color: #9fb3c8; font-size: 10px; }
  td.phase.phase-prefill { color: #7dd3fc; }
  td.phase.phase-decode { color: #ffd28a; }
  td.phase.phase-mixed { color: #a8b3c4; }
  .tag.phase-prefill { color: #7dd3fc; border-color: #2b5a73; }
  .tag.phase-decode { color: #ffd28a; border-color: #6b4a2b; }
  .tag.phase-mixed { color: #a8b3c4; }
  .op-dot { display: inline-block; width: 7px; height: 7px; border-radius: 2px; margin-right: 5px; vertical-align: 1px; }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-thumb { background: #25374a; border-radius: 6px; border: 2px solid #0c1219; }
  ::-webkit-scrollbar-thumb:hover { background: #30465d; }
  ::-webkit-scrollbar-track { background: #0b1118; }
  @media (max-width: 1180px) {
    .dashboard { grid-template-columns: 1fr; }
    .col-side { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .detail-card { grid-column: 1 / -1; }
  }
  @media (max-width: 760px) {
    .topbar { position: static; padding: 12px; }
    .controls { justify-content: flex-start; }
    .stats { padding: 8px 12px; }
    .warn { max-width: 100%; text-align: left; }
    .dashboard { padding: 10px 10px 0; }
    .col-side { grid-template-columns: 1fr; }
    .detail-card { grid-column: auto; }
    .table-card { margin: 0; }
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <div class="title">FastLLM <span>算子级 Trace</span></div>
    <div class="subtitle">实时热点 · 时间线 · 数据格式</div>
  </div>
  <div class="controls">
    <button id="toggle" class="primary" disabled>启用采集</button>
    <button id="clear">清空</button>
    <span id="status" class="pill off">连接中</span>
    <label>窗口
      <select id="window" title="时间线和表格的时间范围">
        <option value="token" selected>单 Token</option>
        <option value="200000">200 ms</option>
        <option value="500000">500 ms</option>
        <option value="1000000">1 s</option>
        <option value="2000000">2 s</option>
        <option value="5000000">5 s</option>
        <option value="8000000">8 s</option>
        <option value="0">全部缓冲</option>
      </select>
    </label>
    <label>视图
      <select id="view">
        <option value="device" selected>设备 / 线程</option>
        <option value="layer">层</option>
      </select>
    </label>
    <label>设备
      <select id="deviceFilter">
        <option value="">全部</option>
      </select>
    </label>
    <label>阶段
      <select id="phaseFilter">
        <option value="">全部</option>
        <option value="prefill">Prefill</option>
        <option value="decode">Decode</option>
        <option value="mixed">Mixed</option>
        <option value="none">未标记</option>
      </select>
    </label>
    <span id="tokenChip" class="pill token-chip" hidden>Decode <b id="tokenChipValue"></b><button id="clearToken" title="清除 token 过滤">×</button></span>
    <label>最小耗时 <input id="minDur" type="number" value="0" min="0" step="1"> µs</label>
    <label>过滤 <input id="filter" type="search" placeholder="op / weight / dtype" style="width:160px"></label>
    <button id="resetFilter" class="ghost" title="清空搜索、设备和耗时筛选">重置</button>
    <button id="locateMatch" class="ghost" title="跳到缓冲区里最近一条匹配当前筛选的事件">最近匹配</button>
    <label><input type="checkbox" id="follow" checked> 跟随最新</label>
    <label><input type="checkbox" id="pause"> 暂停视图</label>
  </div>
</header>
<section class="stats">
  <span class="stat"><i>events/s</i><b id="rate">0</b></span>
  <span class="stat"><i>buffer</i><b id="buffered">0</b><em>/</em><b id="capacity">0</b></span>
  <span class="stat"><i>page</i><b id="loaded">0</b></span>
  <span class="stat"><i>devices</i><b id="deviceCounts">-</b></span>
  <span class="stat"><i>dropped</i><b id="dropped">0</b></span>
  <span class="stat"><i>latest seq</i><b id="latest">-</b></span>
  <span class="stat"><i>窗口事件</i><b id="winEvents">0</b></span>
  <span class="stat"><i>算子累计</i><b id="winTotal">-</b></span>
  <span class="stat"><i>最慢</i><b id="winMax">-</b></span>
  <span class="stat"><i>格式转换</i><b id="convCount">0</b></span>
  <span class="stat"><i>rows</i><b id="rowCount">0</b></span>
  <span id="syncState" class="warn">耗时拆分需同步；未开 FASTLLM_CUDA_SYNC=1 时等待可能为 0 或落入空隙；开启采集有额外开销</span>
</section>

<main class="dashboard">
  <section class="col-main">
    <section class="card">
      <div class="card-head">
        <div>
          <h2>Token 耗时趋势</h2>
          <span class="muted" id="trendNote">-</span>
        </div>
        <div class="head-actions">
          <select id="trendMode" title="趋势统计口径">
            <option value="tokenSplit" selected>按耗时去向</option>
            <option value="token">按 Token 耗时</option>
            <option value="tokenCount">按 Token 事件数</option>
            <option value="time">按时间累计</option>
            <option value="count">按时间事件数</option>
          </select>
        </div>
      </div>
      <div class="chart-wrap" id="trendWrap">
        <canvas id="trend"></canvas>
        <div class="tooltip" id="trendTip"></div>
      </div>
      <div id="trendLegend" class="trend-legend" hidden></div>
    </section>

    <section class="card">
      <div class="card-head">
        <div>
          <h2>算子时间线</h2>
          <span class="muted" id="timelineNote">单 Token · 等待 Decode 数据</span>
        </div>
        <div class="legend">
          <span><i style="background:#4ea1ff"></i>GEMM</span>
          <span><i style="background:#b07cff"></i>MoE</span>
          <span><i style="background:#39c27a"></i>Attn</span>
          <span><i style="background:#ffb454"></i>Norm/Act</span>
          <span><i style="background:#38c7c7"></i>Data</span>
          <span><i style="background:#ff6b8a"></i>Sample</span>
          <span><i style="background:#ff4d6d"></i>慢算子标记</span>
        </div>
      </div>
      <div id="emptyHint" class="empty-hint" hidden>
        <span id="emptyHintText">当前窗口无匹配事件</span>
        <button id="jumpMatch" class="ghost">跳到最近匹配</button>
      </div>
      <div class="timeline-wrap" id="timelineWrap">
        <canvas id="timeline"></canvas>
        <div class="tooltip" id="tooltip"></div>
      </div>
    </section>

    <section class="table-card">
      <div class="table-head">
        <h2>事件列表</h2>
        <span class="muted" id="tableNote">最近 300 条</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>seq</th><th>阶段</th><th class="num" title="相对当前窗口起点">t (ms)</th><th class="num">耗时 µs</th>
              <th>算子</th><th class="num">层</th><th>设备</th><th>权重</th>
              <th>格式</th><th>输入 / 输出</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </section>
  </section>

  <aside class="col-side">
    <section class="card phase-card">
      <div class="card-head">
        <div>
          <h2>Prefill / Decode</h2>
          <span class="muted">全部缓冲 · 当前筛选</span>
        </div>
        <span class="muted" id="phaseNote">-</span>
      </div>
      <div id="phaseStats"></div>
      <div class="card-head sub-head">
        <h2>Decode 每 token</h2>
        <span class="muted" id="tokenNote">-</span>
      </div>
      <div id="tokenList" class="token-list"></div>
      <div class="hint">时间线和表格只显示当前 token；趋势按 token 聚合。MTP/DFlash 一步可能生成多个 token；CUDA Graph replay 不产生逐算子事件。</div>
    </section>

    <section class="card">
      <div class="card-head">
        <div>
          <h2>Token 时间去向</h2>
          <span class="muted" id="splitNote">-</span>
        </div>
      </div>
      <div id="timeSplit" class="time-split"></div>
      <div class="hint">算子=host 调用；传输=输入 ToDevice；等待=显式 CUDA 同步；空隙=墙钟未被事件覆盖。op 内部 D2H/拷贝仍算在算子；直接 CUDA 路径可能没有独立传输阶段；异步下 GPU 执行/传输等待可能落到空隙。</div>
    </section>

    <section class="card">
      <div class="card-head">
        <div>
          <h2>热点排行</h2>
          <span class="muted">按累计耗时</span>
        </div>
        <button id="clearFocus" class="ghost" hidden>清除聚焦</button>
      </div>
      <div class="tabs" id="hotTabs">
        <button class="tab active" data-tab="op">算子</button>
        <button class="tab" data-tab="weight">权重</button>
        <button class="tab" data-tab="layer">层</button>
        <button class="tab" data-tab="dev">设备</button>
      </div>
      <div id="hotspots" class="hotspots"></div>
      <div class="hint">点击行查看该分组最慢事件；统计受上方筛选影响</div>
    </section>

    <section class="card">
      <div class="card-head"><h2>耗时构成</h2><span class="muted">累计耗时</span></div>
      <div id="composition"></div>
    </section>

    <section class="card">
      <div class="card-head"><h2>数据格式</h2><span class="muted">输出 dtype / 转换</span></div>
      <div id="formats"></div>
    </section>

    <section class="card detail-card">
      <div class="card-head"><h2>事件详情</h2></div>
      <div id="detail" class="empty">点击时间线或列表查看</div>
      <div class="card-head sub-head"><h2>数据依赖链</h2><span class="muted">点击节点跳转</span></div>
      <div id="chain" class="empty">-</div>
    </section>
  </aside>
</main>

<script>
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    supported: true,
    enabled: false,
    epoch: null,
    afterSeq: 0,
    lastSeq: 0,
    events: [],
    capacity: 0,
    size: 0,
    dropped: 0,
    cudaSync: null,
    cudaSyncRaw: null,
    cudaGraphRaw: null,
    rate: 0,
    idlePolls: 0,
    filter: "",
    deviceFilter: "",
    phaseFilter: "",
    tokenFilter: null,
    tokenFilterReq: null,
    minDurUs: 0,
    windowMode: "token",
    windowUs: 1000000,
    frozenTokenKey: null,
    view: "device",
    follow: true,
    paused: false,
    selectedSeq: null,
    frozenEnd: 0,
    t0: 0,
    layout: null,
    rateSeq: 0,
    rateTime: 0,
    lastTableDraw: 0,
    lastSideDraw: 0,
    pollTimer: null,
    pollInFlight: false,
    drawPending: false,
    lastData: null,
    hotTab: "op",
    focus: null,
    trendMode: "tokenSplit",
    trendGeom: null,
    trendHover: -1,
    lastAgg: null,
    lastView: null,
  };
  const MAX_EVENTS = 65536;
  const KEEP_US = 3600000000;
  const COLORS = {
    moe: "#b07cff",
    gemm: "#4ea1ff",
    attn: "#39c27a",
    norm: "#ffb454",
    move: "#38c7c7",
    sample: "#ff6b8a",
    other: "#8a93a6",
    hot: "#ff4d6d",
  };
  const CATEGORIES = [
    { key: "moe", label: "MoE", color: COLORS.moe, test: /moe|expert/i },
    { key: "gemm", label: "GEMM", color: COLORS.gemm, test: /linear|matmul|gemm|gemv|fc\b/i },
    { key: "attn", label: "Attn", color: COLORS.attn, test: /attention|flash|wkv|mla|indexer/i },
    { key: "norm", label: "Norm/Act", color: COLORS.norm, test: /norm|silu|gelu|swiglu|act|add|mul\b|rope/i },
    { key: "move", label: "Data", color: COLORS.move, test: /embedding|cat|split|permute|slice|copy|tofloat|todevice|reshape/i },
    { key: "sample", label: "Sample", color: COLORS.sample, test: /sample|topk|softmax|argmax/i },
  ];

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function categoryOf(op) {
    const name = String(op || "");
    for (const item of CATEGORIES) {
      if (item.test.test(name)) return item;
    }
    return { key: "other", label: "其他", color: COLORS.other, test: /$^/ };
  }

  function opColor(op) {
    return categoryOf(op).color;
  }

  function devColor(dev) {
    if (dev === "cuda") return COLORS.gemm;
    if (dev === "cpu") return COLORS.norm;
    return COLORS.other;
  }

  function formatUs(us) {
    if (!isFinite(us) || us <= 0) return "0";
    if (us >= 3600000000) return (us / 3600000000).toFixed(2) + " h";
    if (us >= 60000000) return (us / 60000000).toFixed(1) + " min";
    if (us >= 1000000) return (us / 1000000).toFixed(2) + " s";
    if (us >= 1000) return (us / 1000).toFixed(us >= 10000 ? 1 : 2) + " ms";
    return Math.round(us) + " µs";
  }

  function formatCount(value) {
    const n = Number(value || 0);
    if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1) + "k";
    return String(Math.round(n));
  }

  function formatAgo(us) {
    if (!isFinite(us) || us <= 0) return "0";
    if (us >= 3600000000) return (us / 3600000000).toFixed(2) + "h";
    if (us >= 60000000) return (us / 60000000).toFixed(1) + "min";
    if (us >= 1000000) return (us / 1000000).toFixed(2) + "s";
    const ms = us / 1000;
    return ms.toFixed(ms >= 100 ? 0 : (ms >= 10 ? 1 : 2)) + "ms";
  }

  function heatColor(ratio) {
    const t = Math.max(0, Math.min(1, Number(ratio) || 0));
    return "hsl(" + Math.round(205 - 200 * t) + " 85% 67%)";
  }

  function phaseOf(event) {
    return event.phase || "none";
  }

  function hasTimingBreakdown(event) {
    return Object.prototype.hasOwnProperty.call(event, "run_us") ||
      Object.prototype.hasOwnProperty.call(event, "xfer_us") ||
      Object.prototype.hasOwnProperty.call(event, "sync_us");
  }

  function eventTiming(event) {
    if (!hasTimingBreakdown(event)) return null;
    const runUs = Number(event.run_us) || 0;
    const xferUs = Number(event.xfer_us) || 0;
    const waitUs = Number(event.sync_us) || 0;
    const durUs = Number(event.dur_us) || 0;
    return {
      runUs, xferUs, waitUs,
      otherUs: Math.max(0, durUs - runUs - xferUs - waitUs),
    };
  }

  function phaseLabel(phase) {
    if (phase === "prefill") return "Prefill";
    if (phase === "decode") return "Decode";
    if (phase === "mixed") return "Mixed";
    return "未标记";
  }

  function phaseColor(phase) {
    if (phase === "prefill") return "#38bdf8";
    if (phase === "decode") return "#ffb454";
    if (phase === "mixed") return "#8a93a6";
    return "#6f7e91";
  }

  function requestOf(event) {
    return event.req == null ? -1 : Number(event.req);
  }

  function tokenKey(req, token) {
    return Number(req) + ":" + Number(token);
  }

  function tokenKeyOf(event) {
    return tokenKey(requestOf(event), event.token);
  }

  function tokenLabel(token) {
    const base = "T" + token.token;
    return token.req >= 0 ? "R" + token.req + "·" + base : base;
  }

  function activeTokenKey() {
    if (state.tokenFilter != null) {
      return tokenKey(state.tokenFilterReq, state.tokenFilter);
    }
    if (state.windowMode === "token" &&
        (!state.phaseFilter || state.phaseFilter === "decode") &&
        state.lastView && state.lastView.tokenKey) {
      return state.lastView.tokenKey;
    }
    return null;
  }

  function latestToken() {
    const tokens = buildPhaseTokenStats().tokens;
    return tokens.length ? tokens[0] : null;
  }

  function eventPhaseText(event) {
    const phase = phaseOf(event);
    if (phase === "prefill") return "Prefill";
    if (phase === "decode") return "Decode" + (event.token >= 0 ? " T" + event.token : "");
    if (phase === "mixed") return "Mixed";
    return "-";
  }

  function rowKey(event) {
    if (state.view === "layer") {
      return event.layer >= 0 ? "L" + event.layer : "other";
    }
    return String(event.dev || "?") + " · tid " + event.tid;
  }

  function rowLabel(key) {
    return key === "other" ? "未标注" : key;
  }

  function keyFor(event, tab) {
    switch (tab) {
      case "op": return event.op || "?";
      case "weight": return event.weight || "(无权重)";
      case "layer": return event.layer >= 0 ? "L" + event.layer : "未标注";
      case "dev": return event.dev || "?";
      default: return "";
    }
  }

  function focusMatches(event) {
    return !state.focus || keyFor(event, state.focus.tab) === state.focus.key;
  }

  function filterEvents(events, windowStart, windowEnd, options) {
    const opts = options || {};
    const needle = state.filter.trim().toLowerCase();
    return events.filter((event) => {
      if (!opts.ignoreTime && (event.t_us < windowStart || event.t_us > windowEnd)) return false;
      if (event.dur_us < state.minDurUs) return false;
      if (state.deviceFilter && event.dev !== state.deviceFilter) return false;
      if (!opts.ignorePhase && state.phaseFilter) {
        if (phaseOf(event) !== state.phaseFilter) return false;
      }
      if (!opts.ignoreToken && opts.viewToken != null) {
        if (tokenKeyOf(event) !== opts.viewToken) return false;
      }
      if (!opts.ignoreToken && state.tokenFilter != null) {
        if (Number(event.token) !== state.tokenFilter ||
            requestOf(event) !== state.tokenFilterReq) return false;
      }
      if (needle) {
        const hay = (event.op + " " + (event.weight || "") + " " + (event.io || "") +
          " L" + (event.layer >= 0 ? event.layer : "") + " " + (event.dev || "") +
          " " + phaseOf(event) + " " + (event.token >= 0 ? "T" + event.token : "") +
          " tid" + event.tid).toLowerCase();
        if (hay.indexOf(needle) < 0) return false;
      }
      return true;
    });
  }

  function parseIo(io) {
    const inputs = [];
    const outputs = [];
    String(io || "").split("|").forEach((part) => {
      const match = part.match(/^([A-Za-z0-9_]+)=([^[]*)\[([^\]]*)\]\s*(.*)$/);
      if (!match) return;
      const item = {
        key: match[1],
        name: match[2] || "-",
        dims: match[3] || "",
        dtype: (match[4] || "").trim(),
      };
      if (item.key === "output") outputs.push(item);
      else inputs.push(item);
    });
    return { inputs, outputs };
  }

  function ioOf(event) {
    if (!event._io) event._io = parseIo(event.io);
    return event._io;
  }

  function firstDtype(items) {
    for (const item of items) {
      if (item.dtype) return item.dtype;
    }
    return "";
  }

  function formatOf(event) {
    const parsed = ioOf(event);
    const inD = firstDtype(parsed.inputs);
    const outD = firstDtype(parsed.outputs);
    return { inD, outD, changed: !!inD && !!outD && inD !== outD };
  }

  function trimEvents() {
    if (!state.events.length) return;
    const newest = state.events[state.events.length - 1].t_us;
    const cutoff = newest - KEEP_US;
    let drop = 0;
    while (drop < state.events.length &&
           (state.events[drop].t_us < cutoff || state.events.length - drop > MAX_EVENTS)) {
      drop++;
    }
    if (drop > 0) {
      state.events.splice(0, drop);
      if (state.selectedSeq != null &&
          !state.events.some((event) => event.seq === state.selectedSeq)) {
        state.selectedSeq = null;
        renderDetail(null);
      }
    }
    if (!state.t0 || state.t0 < state.events[0].t_us - KEEP_US) {
      state.t0 = state.events[0].t_us;
    }
  }

  function updateSyncNotice() {
    const el = $("syncState");
    if (!el) return;
    const graph = state.cudaGraphRaw == null
      ? "未显式设置（模型可能自动开启）" : state.cudaGraphRaw;
    if (state.cudaSync) {
      el.textContent = "已检测到 FASTLLM_CUDA_SYNC=1；等待=显式同步时间；CUDA_GRAPH=" +
        graph + "；开启采集有额外开销";
    } else {
      const raw = state.cudaSyncRaw == null ? "未设置" : "'" + state.cudaSyncRaw + "'";
      el.textContent = "未检测到 FASTLLM_CUDA_SYNC=1（服务端值 " + raw +
        "）；等待可能为 0，GPU 执行/传输会落入空隙；CUDA_GRAPH=" + graph;
    }
  }

  function appendSnapshot(data) {
    state.supported = data.supported !== false;
    state.enabled = !!data.enabled;
    state.capacity = Number(data.capacity || 0);
    state.size = Number(data.size || 0);
    state.dropped = Number(data.dropped || 0);
    state.cudaSync = data.cuda_sync === true;
    state.cudaSyncRaw = data.cuda_sync_raw == null ? null : String(data.cuda_sync_raw);
    state.cudaGraphRaw = data.cuda_graph_raw == null ? null : String(data.cuda_graph_raw);
    updateSyncNotice();
    state.lastSeq = Number(data.last_seq || 0);
    const events = Array.isArray(data.events) ? data.events : [];
    if (events.length) {
      state.events = state.events.concat(events);
      state.afterSeq = Number(events[events.length - 1].seq);
      const now = performance.now();
      if (!state.rateTime) {
        state.rateTime = now;
        state.rateSeq = state.afterSeq;
      } else if (now - state.rateTime >= 400) {
        const dt = (now - state.rateTime) / 1000;
        state.rate = Math.max(0, Math.round((state.afterSeq - state.rateSeq) / dt));
        state.rateTime = now;
        state.rateSeq = state.afterSeq;
      }
    } else if (!state.enabled) {
      state.rate = 0;
      state.rateTime = 0;
      state.rateSeq = state.afterSeq;
    }
    trimEvents();
    state.lastData = data;
  }

  async function fetchSnapshot(afterSeq) {
    const response = await fetch(
      "/api/op-trace/snapshot?after_seq=" + encodeURIComponent(afterSeq) + "&max_events=8192",
      { cache: "no-store" });
    if (!response.ok) {
      throw new Error("HTTP " + response.status);
    }
    return await response.json();
  }

  async function poll() {
    state.pollTimer = null;
    if (state.pollInFlight) return;
    state.pollInFlight = true;
    try {
      if (document.hidden || state.paused) {
        schedulePoll();
        return;
      }
      let data = await fetchSnapshot(state.afterSeq);
      if (state.epoch !== null && data.epoch !== state.epoch) {
        state.events = [];
        state.afterSeq = 0;
        state.t0 = 0;
        state.selectedSeq = null;
        state.focus = null;
        renderDetail(null);
        data = await fetchSnapshot(0);
      }
      state.epoch = data.epoch;
      if (data.lost) {
        state.events = [];
        state.afterSeq = 0;
        state.t0 = 0;
        state.selectedSeq = null;
        state.focus = null;
        renderDetail(null);
        data = await fetchSnapshot(0);
        state.epoch = data.epoch;
      }
      appendSnapshot(data);
      let gotEvents = (data.events || []).length > 0;
      let more = 8;
      while (more-- > 0 && state.afterSeq < state.lastSeq) {
        const next = await fetchSnapshot(state.afterSeq);
        const count = (next.events || []).length;
        appendSnapshot(next);
        gotEvents = gotEvents || count > 0;
        if (!count) break;
      }
      if (gotEvents) {
        state.idlePolls = 0;
      } else if (state.enabled) {
        state.idlePolls += 1;
      }
      setStatus();
      scheduleDraw();
      if (Date.now() - state.lastTableDraw > 450) {
        state.lastTableDraw = Date.now();
        drawTable();
      }
    } catch (error) {
      setStatus("连接失败: " + error.message, "bad");
    } finally {
      state.pollInFlight = false;
      schedulePoll();
    }
  }

  function schedulePoll() {
    if (state.pollTimer != null) return;
    const delay = !state.enabled ? 600 :
      (state.idlePolls >= 10 ? 1000 : (state.idlePolls >= 4 ? 400 : 100));
    state.pollTimer = setTimeout(poll, delay);
  }

  function setStatus(message, kind) {
    const pill = $("status");
    if (message) {
      pill.textContent = message;
      pill.className = "pill " + (kind || "off");
      return;
    }
    if (!state.supported) {
      pill.textContent = "原生库不支持";
      pill.className = "pill bad";
    } else if (state.enabled) {
      pill.textContent = "采集中";
      pill.className = "pill on";
    } else {
      pill.textContent = "已关闭";
      pill.className = "pill off";
    }
  }

  function latestTime() {
    if (!state.events.length) return 0;
    return state.events[state.events.length - 1].t_us;
  }

  function paddedSpan(startUs, endUs) {
    const span = Math.max(1, endUs - startUs);
    const pad = Math.max(100, span * 0.02);
    return {
      start: Math.max(0, startUs - pad),
      end: endUs + pad,
      contentSpan: span,
    };
  }

  function phaseSpanView(kind, latest, freeze) {
    const events = filterEvents(state.events, 0, Number.MAX_SAFE_INTEGER,
      { ignoreTime: true, ignoreToken: true })
      .filter((event) => phaseOf(event) === kind);
    if (!events.length) return null;
    let startUs = Infinity;
    let endUs = 0;
    for (const event of events) {
      startUs = Math.min(startUs, event.t_us);
      endUs = Math.max(endUs, event.t_us + (event.dur_us || 0));
    }
    const range = paddedSpan(startUs, endUs);
    const label = (kind === "prefill" ? "Prefill" : "Mixed") +
      " · " + formatUs(range.contentSpan);
    return {
      ...range, latest, freeze, kind, tokenKey: null, token: null, label,
    };
  }

  function computeView() {
    const latest = latestTime();
    const freeze = state.paused || !state.follow;

    if (state.windowMode === "token") {
      const canUseToken = state.phaseFilter !== "prefill" && state.phaseFilter !== "mixed";
      if (canUseToken) {
        const tokens = buildPhaseTokenStats().tokens;
        let token = null;
        if (state.tokenFilter != null) {
          const wanted = tokenKey(state.tokenFilterReq, state.tokenFilter);
          token = tokens.find((item) => item.key === wanted) || null;
        }
        if (!token && state.frozenTokenKey) {
          token = tokens.find((item) => item.key === state.frozenTokenKey) || null;
        }
        if (!token) token = tokens[0] || null;
        if (token) {
          const range = paddedSpan(token.startUs, token.endUs);
          return {
            ...range, latest, freeze,
            kind: "token", tokenKey: token.key, token,
            label: "Token " + tokenLabel(token) + " · " + formatUs(range.contentSpan) +
              " · 算子累计 " + formatUs(token.totalUs),
          };
        }
      }
      const prefill = phaseSpanView("prefill", latest, freeze);
      if (prefill) return prefill;
      const mixed = phaseSpanView("mixed", latest, freeze);
      if (mixed) return mixed;
    }

    if (freeze) {
      if (!state.frozenEnd && latest) state.frozenEnd = latest;
    } else {
      state.frozenEnd = 0;
    }
    const end = freeze ? (state.frozenEnd || latest) : latest;
    const start = state.windowUs > 0
      ? Math.max(0, end - state.windowUs)
      : (state.events.length ? state.events[0].t_us : 0);
    return {
      start, end, latest, freeze,
      kind: "time", tokenKey: null, token: null,
      label: "窗口 " + formatUs(end - start),
    };
  }

  function addAgg(map, key, event) {
    let item = map.get(key);
    if (!item) {
      item = { key, totalUs: 0, count: 0, maxUs: 0, slowest: null };
      map.set(key, item);
    }
    item.totalUs += event.dur_us;
    item.count += 1;
    if (event.dur_us > item.maxUs) {
      item.maxUs = event.dur_us;
      item.slowest = event;
    }
  }

  function aggregate(visible) {
    const maps = { op: new Map(), weight: new Map(), layer: new Map(), dev: new Map() };
    const categories = new Map();
    const outDtypes = new Map();
    const conversions = new Map();
    let totalUs = 0;
    let maxUs = 0;
    let slowest = null;
    let convEvents = 0;
    for (const event of visible) {
      totalUs += event.dur_us;
      if (event.dur_us > maxUs) {
        maxUs = event.dur_us;
        slowest = event;
      }
      addAgg(maps.op, event.op || "?", event);
      addAgg(maps.weight, event.weight || "(无权重)", event);
      addAgg(maps.layer, event.layer >= 0 ? "L" + event.layer : "未标注", event);
      addAgg(maps.dev, event.dev || "?", event);

      const category = categoryOf(event.op);
      let cat = categories.get(category.key);
      if (!cat) {
        cat = {
          key: category.key, label: category.label, color: category.color,
          totalUs: 0, count: 0, maxUs: 0, slowest: null,
        };
        categories.set(category.key, cat);
      }
      cat.totalUs += event.dur_us;
      cat.count += 1;
      if (event.dur_us > cat.maxUs) {
        cat.maxUs = event.dur_us;
        cat.slowest = event;
      }

      const fmt = formatOf(event);
      if (fmt.outD) outDtypes.set(fmt.outD, (outDtypes.get(fmt.outD) || 0) + 1);
      if (fmt.changed) {
        convEvents += 1;
        const key = fmt.inD + " → " + fmt.outD;
        conversions.set(key, (conversions.get(key) || 0) + 1);
      }
    }
    return {
      maps, categories, outDtypes, conversions, convEvents,
      totalUs, maxUs, slowest, count: visible.length,
    };
  }

  function topEntries(map, limit) {
    return Array.from(map.values())
      .sort((a, b) => b.totalUs - a.totalUs || b.count - a.count)
      .slice(0, limit || 10);
  }

  function buildPhaseTokenStats() {
    const base = filterEvents(state.events, 0, Number.MAX_SAFE_INTEGER,
      { ignoreTime: true, ignorePhase: true, ignoreToken: true });
    const phases = {
      prefill: { events: 0, us: 0, tokens: new Set(), tokenUs: 0, tokenLatencyUs: 0 },
      decode: { events: 0, us: 0, tokens: new Set(), tokenUs: 0, tokenLatencyUs: 0 },
      mixed: { events: 0, us: 0, tokens: new Set(), tokenUs: 0, tokenLatencyUs: 0 },
      none: { events: 0, us: 0, tokens: new Set(), tokenUs: 0, tokenLatencyUs: 0 },
    };
    const tokenMap = new Map();
    let grandUs = 0;
    for (const event of base) {
      const phase = phaseOf(event);
      const item = phases[phase] || phases.none;
      item.events += 1;
      item.us += event.dur_us;
      grandUs += event.dur_us;
      if (phase !== "decode" || event.token < 0) {
        continue;
      }
      const req = requestOf(event);
      const key = req + ":" + event.token;
      item.tokens.add(key);
      item.tokenUs += event.dur_us;
      let token = tokenMap.get(key);
      if (!token) {
        token = {
          req, token: Number(event.token), key,
          totalUs: 0, count: 0,
          cudaUs: 0, cpuUs: 0, otherUs: 0, ops: new Map(),
          runUs: 0, xferUs: 0, waitUs: 0, otherOpUs: 0,
          hasSplit: false,
          slowest: null, startUs: event.t_us, endUs: 0,
        };
        tokenMap.set(key, token);
      }
      token.totalUs += event.dur_us;
      token.count += 1;
      if (hasTimingBreakdown(event)) {
        const runUs = Number(event.run_us) || 0;
        const xferUs = Number(event.xfer_us) || 0;
        const waitUs = Number(event.sync_us) || 0;
        token.hasSplit = true;
        token.runUs += runUs;
        token.xferUs += xferUs;
        token.waitUs += waitUs;
        token.otherOpUs += Math.max(0, event.dur_us - runUs - xferUs - waitUs);
      } else {
        token.runUs += event.dur_us;
      }
      if (event.dev === "cuda") token.cudaUs += event.dur_us;
      else if (event.dev === "cpu") token.cpuUs += event.dur_us;
      else token.otherUs += event.dur_us;
      token.startUs = Math.min(token.startUs, event.t_us);
      token.endUs = Math.max(token.endUs, event.t_us + (event.dur_us || 0));
      let op = token.ops.get(event.op);
      if (!op) {
        op = { op: event.op, totalUs: 0, count: 0, maxUs: 0 };
        token.ops.set(event.op, op);
      }
      op.totalUs += event.dur_us;
      op.count += 1;
      if (event.dur_us > op.maxUs) op.maxUs = event.dur_us;
      if (!token.slowest || event.dur_us > token.slowest.dur_us) {
        token.slowest = event;
      }
    }
    const tokens = Array.from(tokenMap.values())
      .sort((a, b) => b.endUs - a.endUs);
    for (const token of tokens) {
      token.latencyUs = Math.max(1, token.endUs - token.startUs);
      token.gapUs = Math.max(0, token.latencyUs - token.totalUs);
      phases.decode.tokenLatencyUs += token.latencyUs;
    }
    return { phases, tokens, grandUs };
  }

  function renderPhaseTokens() {
    const stats = buildPhaseTokenStats();
    const phases = stats.phases;
    const total = stats.grandUs || 1;
    const pct = (us) => total ? (us / total) * 100 : 0;
    const decodeTokens = phases.decode.tokens.size;
    const decodeAvg = decodeTokens ? phases.decode.tokenUs / decodeTokens : 0;
    const phaseBox = (phase, label, main, sub) =>
      "<div class='phase-box " + phase + (state.phaseFilter === phase ? " active" : "") +
      "' data-phase='" + phase + "' title='点击查看 " + label + " 全部事件'>" +
      "<i>" + label + "</i><b>" + main + "</b><span>" + sub + "</span></div>";
    let html = "<div class='phase-grid'>" +
      phaseBox("prefill", "Prefill", formatUs(phases.prefill.us),
        formatCount(phases.prefill.events) + " events · " + pct(phases.prefill.us).toFixed(1) + "%") +
      phaseBox("decode", "Decode", formatUs(phases.decode.us),
        formatCount(decodeTokens) + " tokens · avg ops " +
        (decodeTokens ? formatUs(decodeAvg) : "-") + " · " +
        pct(phases.decode.us).toFixed(1) + "%") +
      "</div>";
    if (phases.mixed.events || phases.none.events) {
      const extra = [];
      if (phases.mixed.events) {
        extra.push("Mixed " + formatUs(phases.mixed.us) + " · " + formatCount(phases.mixed.events) + " events");
      }
      if (phases.none.events) {
        extra.push("未标记 " + formatUs(phases.none.us) + " · " + formatCount(phases.none.events) + " events");
      }
      html += "<div class='phase-extra'>" + extra.join("　") + "</div>";
    }
    $("phaseStats").innerHTML = html;
    $("phaseNote").textContent = decodeTokens ? formatCount(decodeTokens) + " tokens" : "-";
    $("tokenNote").textContent = stats.tokens.length
      ? "共 " + formatCount(stats.tokens.length) + " · 最近 " + Math.min(40, stats.tokens.length)
      : "-";
    if (!stats.tokens.length) {
      const message = phases.decode.events
        ? "Decode 事件没有 token 标记（批量/mixed forward 或旧数据）"
        : "还没有 Decode token 数据；确认服务端已更新并重新采集";
      $("tokenList").innerHTML = "<div class='empty'>" + message + "</div>";
      return;
    }
    const activeKey = activeTokenKey();
    let top = stats.tokens.slice(0, 40);
    if (activeKey && !top.some((token) => token.key === activeKey)) {
      const selected = stats.tokens.find((token) => token.key === activeKey);
      if (selected) top = [selected, ...top.slice(0, 39)];
    }
    const maxTotal = Math.max(1, ...top.map((token) => token.latencyUs));
    $("tokenList").innerHTML = top.map((token) => {
      const focused = activeKey === token.key;
      const reqLabel = token.req >= 0 ? "R" + token.req + " · " : "";
      const splitText = token.hasSplit
        ? "算子 " + formatUs(token.runUs) + " · 传输 " + formatUs(token.xferUs) +
          " · 等待 " + formatUs(token.waitUs) + " · 空隙 " + formatUs(token.gapUs)
        : "未拆分 · 算子累计 " + formatUs(token.totalUs);
      const title = "点击查看该 token 时间线 · " + (token.req >= 0
        ? "request " + token.req + " · decode step " + token.token
        : "decode step " + token.token) +
        " · 耗时 " + formatUs(token.latencyUs) + " · " + splitText;
      return "<div class='token-row" + (focused ? " selected" : "") +
        "' data-token='" + token.token + "' data-req='" + token.req +
        "' title='" + esc(title) + "'>" +
        "<div class='token-top'><span class='token-id'>" + esc(reqLabel + "T" + token.token) +
        "</span><span class='token-meta'>" + token.count + " ops</span>" +
        "<span class='token-total'>" + formatUs(token.latencyUs) + "</span></div>" +
        "<div class='token-bar'><i style='width:" +
        (token.latencyUs / maxTotal * 100).toFixed(2) + "%'></i></div>" +
        "<div class='token-sub'>" + esc(splitText) + "</div></div>";
    }).join("");
  }

  function renderTimeSplit() {
    const stats = buildPhaseTokenStats();
    const key = activeTokenKey();
    const token = (key && stats.tokens.find((item) => item.key === key)) ||
      stats.tokens[0] || null;
    if (!token) {
      $("splitNote").textContent = "-";
      $("timeSplit").innerHTML = "<div class='empty'>还没有 Decode token</div>";
      return;
    }
    let components;
    if (token.hasSplit) {
      components = [
        { key: "runUs", label: "算子", color: "#4ea1ff" },
        { key: "xferUs", label: "传输", color: "#38c7c7" },
        { key: "waitUs", label: "等待", color: "#ffb454" },
        { key: "otherOpUs", label: "其他", color: "#8a93a6" },
        { key: "gapUs", label: "空隙", color: "#3b4b60" },
      ].map((item) => ({ ...item, value: token[item.key] || 0 }));
    } else {
      components = [
        { label: "未拆分", color: "#6f7e91", value: token.totalUs || 0 },
        { label: "空隙", color: "#3b4b60", value: token.gapUs || 0 },
      ];
    }
    const sum = Math.max(1, components.reduce((total, item) => total + item.value, 0));
    const bar = components.map((item) => {
      const width = Math.max(0, (item.value / sum) * 100);
      return "<i style='width:" + width.toFixed(3) + "%;background:" + item.color +
        "' title='" + esc(item.label + " " + formatUs(item.value) + " · " +
        width.toFixed(1) + "%") + "'></i>";
    }).join("");
    const rows = components.map((item) => {
      const pct = (item.value / sum) * 100;
      return "<div class='split-row'><span class='split-dot' style='background:" +
        item.color + "'></span><span class='name'>" + esc(item.label) +
        "</span><span class='val'>" + formatUs(item.value) +
        "</span><span class='pct'>" + pct.toFixed(1) + "%</span></div>";
    }).join("");
    const gapPct = token.latencyUs ? (token.gapUs / token.latencyUs) * 100 : 0;
    let note = tokenLabel(token) + " · " + formatUs(token.latencyUs) +
      (token.hasSplit ? "" : " · 未拆分");
    if (token.hasSplit && gapPct >= 30) {
      if (!token.waitUs) {
        note += " · 等待=0，确认 FASTLLM_CUDA_SYNC=1 已生效";
      } else if (state.cudaSync) {
        note += " · 同步已开，空隙多为未埋点/CPU 时间";
      }
    }
    $("splitNote").textContent = note;
    $("timeSplit").innerHTML =
      "<div class='split-stack'>" + bar + "</div>" +
      "<div class='split-rows'>" + rows + "</div>";
  }

  function renderStats(agg) {
    agg = agg || {
      count: 0, totalUs: 0, maxUs: 0, slowest: null, convEvents: 0,
    };
    if (state.enabled && state.rateTime && performance.now() - state.rateTime > 2500) {
      state.rate = 0;
    }
    $("rate").textContent = state.rate;
    $("buffered").textContent = state.size;
    $("capacity").textContent = state.capacity || "-";
    $("buffered").title = "服务端环形缓冲条数；容量 " + (state.capacity || "-") + " 条，超出后旧事件被覆盖";
    $("loaded").textContent = formatCount(state.events.length);
    $("loaded").title = "页面已加载的事件数（上限 " + MAX_EVENTS + " 条）";
    $("dropped").textContent = state.dropped;
    $("latest").textContent = state.lastSeq || "-";
    $("winEvents").textContent = agg.count || 0;
    $("winTotal").textContent = agg.totalUs ? formatUs(agg.totalUs) : "-";
    $("winTotal").title = "当前窗口内各算子耗时求和，线程/设备之间可能重叠";
    $("winMax").textContent = agg.maxUs ? formatUs(agg.maxUs) : "-";
    $("winMax").title = agg.slowest
      ? "#" + agg.slowest.seq + " · " + agg.slowest.op + " · " + agg.slowest.dev
      : "";
    $("convCount").textContent = agg.convEvents || 0;
    $("rowCount").textContent = state.layout ? state.layout.keys.length : 0;

    const counts = new Map();
    for (const event of state.events) {
      const dev = event.dev || "?";
      counts.set(dev, (counts.get(dev) || 0) + 1);
    }
    const entries = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    $("deviceCounts").textContent = entries.length
      ? entries.map(([dev, count]) => dev + " " + formatCount(count)).join(" / ")
      : "-";
    const select = $("deviceFilter");
    const signature = entries.map(([dev]) => dev).join("|");
    if (select.dataset.signature !== signature) {
      const current = state.deviceFilter;
      select.innerHTML = "<option value=''>全部</option>" + entries.map(([dev, count]) =>
        "<option value='" + esc(dev) + "'>" + esc(dev) + " (" + esc(count) + ")</option>").join("");
      select.dataset.signature = signature;
      select.value = current;
      if (select.value !== current) {
        state.deviceFilter = "";
        select.value = "";
      }
    }
    const toggle = $("toggle");
    toggle.disabled = !state.supported;
    toggle.textContent = state.enabled ? "停止采集" : "启用采集";
    toggle.classList.toggle("primary", !state.enabled);
  }

  function roundRect(ctx, x, y, w, h, r) {
    if (typeof ctx.roundRect === "function") {
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, r);
    } else {
      ctx.beginPath();
      ctx.rect(x, y, w, h);
    }
  }

  function drawTimeline(source, view, agg) {
    let visible = source;
    if (source.length > 18000) {
      const step = Math.ceil(source.length / 18000);
      visible = source.filter((_, index) => index % step === 0);
    }
    const wrap = $("timelineWrap");
    const canvas = $("timeline");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = Math.max(320, wrap.clientWidth - 2);
    const start = view.start;
    const end = view.end;

    const keys = [];
    const seen = new Set();
    for (const event of visible) {
      const key = rowKey(event);
      if (!seen.has(key)) { seen.add(key); keys.push(key); }
    }
    if (state.view === "layer") {
      keys.sort((a, b) => {
        if (a === "other") return 1;
        if (b === "other") return -1;
        return parseInt(a.slice(1), 10) - parseInt(b.slice(1), 10);
      });
    } else {
      keys.sort();
    }
    const rowIndex = new Map(keys.map((key, index) => [key, index]));
    const rowCount = Math.max(1, keys.length);
    const gutter = 160;
    const axis = 24;

    if (!keys.length) {
      const height = 120;
      canvas.style.width = cssWidth + "px";
      canvas.style.height = height + "px";
      canvas.width = Math.floor(cssWidth * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#0f161e";
      ctx.fillRect(0, 0, cssWidth, height);
      ctx.fillStyle = "#7f8b9d";
      ctx.font = "12px Segoe UI, Microsoft YaHei, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("当前窗口没有事件", cssWidth / 2, height / 2);
      state.layout = {
        start, end, gutter: 0, axis: 0, rowHeight: 0, rowCount: 0,
        keys: [], rowIndex: new Map(), visible: [], width: cssWidth, height,
      };
      return;
    }

    const rowHeight = Math.max(11, Math.min(26, Math.floor(320 / rowCount)));
    const height = axis + rowCount * rowHeight + 10;
    canvas.style.width = cssWidth + "px";
    canvas.style.height = height + "px";
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#0f161e";
    ctx.fillRect(0, 0, cssWidth, height);

    const plotWidth = Math.max(40, cssWidth - gutter - 10);
    const span = Math.max(1, end - start);
    const xOf = (time) => gutter + ((time - start) / span) * plotWidth;

    ctx.strokeStyle = "#1b2632";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = gutter + (plotWidth * i) / 10;
      ctx.beginPath(); ctx.moveTo(x, axis); ctx.lineTo(x, height - 6); ctx.stroke();
      ctx.fillStyle = "#6f7e91";
      ctx.font = "10px Segoe UI, Microsoft YaHei, sans-serif";
      ctx.textAlign = i === 0 ? "left" : (i === 10 ? "right" : "center");
      ctx.fillText(formatUs((span * i) / 10), x, 14);
    }

    ctx.textAlign = "left";
    ctx.font = "11px Segoe UI, Microsoft YaHei, sans-serif";
    for (let i = 0; i < keys.length; i++) {
      const y = axis + i * rowHeight;
      if (i % 2 === 1) {
        ctx.fillStyle = "#121a23";
        ctx.fillRect(0, y, cssWidth, rowHeight);
      }
      ctx.fillStyle = "#8493a6";
      const label = rowLabel(keys[i]);
      ctx.fillText(label.length > 23 ? label.slice(0, 22) + "…" : label, 8, y + rowHeight - 6);
      ctx.strokeStyle = "#192430";
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cssWidth, y); ctx.stroke();
    }

    const selectedChain = state.selectedSeq != null ? chainSeqSet(state.selectedSeq) : new Set();
    const focus = state.focus;
    const hotUs = Math.max(2000, (agg ? agg.maxUs : 0) * 0.4);
    for (const event of visible) {
      const index = rowIndex.get(rowKey(event));
      if (index == null) continue;
      const durUs = Math.max(1, event.dur_us || 0);
      const x0 = xOf(event.t_us);
      const x1 = Math.max(x0 + 1, xOf(event.t_us + durUs));
      const y = axis + index * rowHeight + 2;
      const barHeight = Math.max(4, rowHeight - 4);
      const matched = !focus || focusMatches(event);
      ctx.globalAlpha = focus ? (matched ? 0.95 : 0.16) : 0.9;
      ctx.fillStyle = opColor(event.op);
      roundRect(ctx, x0, y, Math.max(1, x1 - x0), barHeight, 2);
      ctx.fill();
      if (barHeight >= 6 && event.dur_us >= hotUs) {
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = COLORS.hot;
        ctx.fillRect(x0, y, Math.max(1, x1 - x0), 2);
      }
      if (event.seq === state.selectedSeq || selectedChain.has(event.seq)) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = event.seq === state.selectedSeq ? "#ffffff" : "#9ecbff";
        ctx.lineWidth = event.seq === state.selectedSeq ? 2 : 1;
        ctx.strokeRect(x0 - 0.5, y - 0.5, Math.max(1, x1 - x0) + 1, barHeight + 1);
      }
    }
    ctx.globalAlpha = 1;

    state.layout = {
      start, end, gutter, axis, rowHeight, rowCount, keys, rowIndex, visible,
      width: cssWidth, height,
    };
  }

  const TREND_TOKEN_LIMIT = 72;

  function drawTokenTrend() {
    const wrap = $("trendWrap");
    const canvas = $("trend");
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, wrap.clientWidth - 2);
    const height = Math.max(160, wrap.clientHeight - 2);
    const pad = { l: 52, r: 14, t: 12, b: 24 };
    const plotWidth = Math.max(40, width - pad.l - pad.r);
    const plotHeight = Math.max(60, height - pad.t - pad.b);
    const stats = buildPhaseTokenStats();
    const selectedKey = activeTokenKey();
    const phaseBlocksTokens = !!state.phaseFilter && state.phaseFilter !== "decode";
    let tokens = phaseBlocksTokens ? [] : stats.tokens.slice(0, TREND_TOKEN_LIMIT);
    if (selectedKey && !tokens.some((token) => token.key === selectedKey)) {
      const selected = stats.tokens.find((token) => token.key === selectedKey);
      if (selected) {
        tokens = [selected, ...tokens.slice(0, TREND_TOKEN_LIMIT - 1)];
      }
    }
    tokens.reverse();
    const split = state.trendMode === "tokenSplit";
    const mode = state.trendMode === "tokenCount" ? "count" : "total";
    const slot = plotWidth / Math.max(1, tokens.length);
    const barWidth = Math.max(2, Math.min(26, slot * 0.62));
    const bars = tokens.map((token, index) => {
      const center = pad.l + (index + 0.5) * slot;
      return {
        token, index, center,
        x: center - barWidth / 2,
        w: barWidth,
        total: token.latencyUs || 0,
        opUs: token.totalUs || 0,
        count: token.count || 0,
        cudaUs: token.cudaUs || 0,
        cpuUs: token.cpuUs || 0,
        otherUs: token.otherUs || 0,
        runUs: token.runUs || 0,
        xferUs: token.xferUs || 0,
        waitUs: token.waitUs || 0,
        otherOpUs: token.otherOpUs || 0,
        gapUs: token.gapUs || 0,
      };
    });
    const maxTotal = Math.max(1, ...tokens.map((token) => token.latencyUs));
    const maxCount = Math.max(1, ...tokens.map((token) => token.count));
    const total = tokens.reduce((sum, token) => sum + token.latencyUs, 0);
    state.trendGeom = {
      type: "token", width, height, dpr, pad, plotWidth, plotHeight,
      slot, tokens, bars, mode, split, maxTotal, maxCount, selectedKey,
      empty: tokens.length === 0, total,
    };
    if (state.trendHover < 0 || state.trendHover >= tokens.length) {
      state.trendHover = -1;
    }
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    if (!tokens.length) {
      $("trendNote").textContent = phaseBlocksTokens
        ? "当前阶段 " + phaseLabel(state.phaseFilter) + " 没有 Decode token 趋势 · 可切到按时间"
        : "没有 Decode token · 可切到按时间，或查看 Prefill 时间线";
    } else {
      const selected = tokens.find((token) => token.key === selectedKey);
      const average = mode === "count"
        ? formatCount(tokens.reduce((sum, token) => sum + token.count, 0) / tokens.length) + " ops"
        : "平均耗时 " + formatUs(total / tokens.length);
      const peak = mode === "count" ? formatCount(maxCount) + " ops" : formatUs(maxTotal);
      $("trendNote").textContent =
        "最近 " + tokens.length + " 个 token · " + average +
        " · 最高 " + peak +
        (selected ? " · 已选 " + tokenLabel(selected) : "") +
        (split && !tokens.some((token) => token.hasSplit) ? " · 当前数据未拆分" : "");
    }
    const legend = $("trendLegend");
    if (split) {
      legend.hidden = false;
      legend.innerHTML =
        "<span><i style='background:#4ea1ff'></i>算子</span>" +
        "<span><i style='background:#38c7c7'></i>传输</span>" +
        "<span><i style='background:#ffb454'></i>等待</span>" +
        "<span><i style='background:#8a93a6'></i>其他</span>" +
        "<span><i style='background:#3b4b60'></i>空隙</span>";
    } else {
      legend.hidden = true;
    }
    paintTrend();
  }

  function drawTrend(visible, view) {
    if (state.trendMode === "token" || state.trendMode === "tokenCount" ||
        state.trendMode === "tokenSplit") {
      drawTokenTrend();
      return;
    }
    const wrap = $("trendWrap");
    const canvas = $("trend");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, wrap.clientWidth - 2);
    const height = Math.max(160, wrap.clientHeight - 2);
    const pad = { l: 52, r: 12, t: 12, b: 22 };
    const plotWidth = Math.max(40, width - pad.l - pad.r);
    const plotHeight = Math.max(60, height - pad.t - pad.b);
    const bucketCount = Math.max(24, Math.min(72, Math.floor(plotWidth / 14)));
    const span = Math.max(1, view.end - view.start);
    const bucketUs = span / bucketCount;
    const buckets = Array.from({ length: bucketCount }, () => ({
      cuda: 0, cpu: 0, other: 0, count: 0,
    }));
    let maxTotal = 0;
    let maxCount = 0;
    let sumCuda = 0;
    let sumCpu = 0;
    let sumOther = 0;

    for (const event of visible) {
      let index = Math.floor((event.t_us - view.start) / bucketUs);
      index = Math.max(0, Math.min(bucketCount - 1, index));
      const bucket = buckets[index];
      const dur = Math.max(0, event.dur_us || 0);
      if (event.dev === "cuda") { bucket.cuda += dur; sumCuda += dur; }
      else if (event.dev === "cpu") { bucket.cpu += dur; sumCpu += dur; }
      else { bucket.other += dur; sumOther += dur; }
      bucket.count += 1;
      const total = bucket.cuda + bucket.cpu + bucket.other;
      if (total > maxTotal) maxTotal = total;
      if (bucket.count > maxCount) maxCount = bucket.count;
    }

    state.trendGeom = {
      width, height, dpr, pad, plotWidth, plotHeight,
      bucketCount, bucketUs, span, start: view.start, end: view.end,
      buckets, mode: state.trendMode, maxTotal, maxCount,
      sumCuda, sumCpu, sumOther, total: sumCuda + sumCpu + sumOther,
      empty: visible.length === 0,
    };
    if (visible.length === 0) {
      state.trendHover = -1;
      $("trendTip").style.display = "none";
    }
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);

    if (visible.length === 0) {
      $("trendNote").textContent = "窗口 " + formatUs(span) + " · 当前窗口没有事件";
    } else {
      $("trendNote").textContent =
        "窗口 " + formatUs(span) + " · 累计 " + formatUs(sumCuda + sumCpu + sumOther) +
        " · cuda " + formatUs(sumCuda) + " / cpu " + formatUs(sumCpu);
    }
    paintTrend();
  }

  function paintTokenTrend() {
    const canvas = $("trend");
    const ctx = canvas.getContext("2d");
    const g = state.trendGeom;
    if (!g || g.type !== "token") return;
    const { width, height, pad, plotWidth, plotHeight, bars, mode, slot } = g;
    ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
    ctx.fillStyle = "#0f161e";
    ctx.fillRect(0, 0, width, height);

    const yMax = Math.max(1, mode === "count" ? g.maxCount : g.maxTotal);
    const yOf = (value) => pad.t + plotHeight - (value / yMax) * plotHeight;

    ctx.strokeStyle = "#1b2632";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = pad.l + (plotWidth * i) / 4;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + plotHeight); ctx.stroke();
    }
    for (let i = 0; i <= 2; i++) {
      const y = pad.t + (plotHeight * i) / 2;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotWidth, y); ctx.stroke();
    }

    if (g.empty) {
      ctx.fillStyle = "#7f8b9d";
      ctx.font = "12px Segoe UI, Microsoft YaHei, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("没有 Decode token", pad.l + plotWidth / 2, pad.t + plotHeight / 2);
      return;
    }

    ctx.font = "10px Segoe UI, Microsoft YaHei, sans-serif";
    ctx.fillStyle = "#6f7e91";
    ctx.textAlign = "right";
    for (let i = 0; i <= 2; i++) {
      const value = yMax * (1 - i / 2);
      ctx.fillText(mode === "count" ? formatCount(value) : formatUs(value),
        pad.l - 7, pad.t + (plotHeight * i) / 2 + 3);
    }

    if (g.selectedKey) {
      const selected = bars.find((bar) => bar.token.key === g.selectedKey);
      if (selected) {
        ctx.fillStyle = "rgba(78,161,255,.14)";
        ctx.fillRect(selected.x - 4, pad.t, selected.w + 8, plotHeight);
      }
    }

    const segments = g.split ? [
      { key: "runUs", color: "#4ea1ff" },
      { key: "xferUs", color: "#38c7c7" },
      { key: "waitUs", color: "#ffb454" },
      { key: "otherOpUs", color: "#8a93a6" },
      { key: "gapUs", color: "#3b4b60" },
    ] : [
      { key: "cudaUs", color: COLORS.gemm },
      { key: "cpuUs", color: COLORS.norm },
      { key: "otherUs", color: COLORS.other },
    ];
    for (let index = 0; index < bars.length; index++) {
      const bar = bars[index];
      const selected = bar.token.key === g.selectedKey;
      const hovered = index === state.trendHover;
      if (mode === "count") {
        const y = yOf(bar.count);
        ctx.fillStyle = selected ? "#8cc6ff" : "#4ea1ff";
        ctx.fillRect(bar.x, y, bar.w, pad.t + plotHeight - y);
      } else {
        const stackTotal = segments.reduce((sum, segment) => sum + (bar[segment.key] || 0), 0);
        const targetHeight = (bar.total / yMax) * plotHeight;
        let base = pad.t + plotHeight;
        if (stackTotal > 0) {
          for (const segment of segments) {
            const value = bar[segment.key];
            if (!value) continue;
            const h = targetHeight * (value / stackTotal);
            ctx.fillStyle = segment.color;
            ctx.fillRect(bar.x, base - h, bar.w, h);
            base -= h;
          }
        } else {
          ctx.fillStyle = COLORS.other;
          ctx.fillRect(bar.x, base - targetHeight, bar.w, targetHeight);
        }
      }
      if (selected || hovered) {
        ctx.strokeStyle = selected ? "#eaf5ff" : "rgba(190,220,255,.7)";
        ctx.lineWidth = selected ? 2 : 1;
        ctx.strokeRect(bar.x - 1, pad.t, bar.w + 2, plotHeight);
        if (selected) {
          ctx.fillStyle = "#eaf5ff";
          ctx.fillRect(bar.center - 1, pad.t - 4, 2, 4);
        }
      }
    }

    const labelEvery = Math.max(1, Math.ceil(bars.length / 8));
    ctx.textAlign = "center";
    for (let index = 0; index < bars.length; index++) {
      const bar = bars[index];
      const selected = bar.token.key === g.selectedKey;
      if (index % labelEvery !== 0 && !selected) continue;
      ctx.fillStyle = selected ? "#8cc6ff" : "#6f7e91";
      ctx.fillText(tokenLabel(bar.token), bar.center, height - 6);
    }

    if (state.trendHover >= 0 && state.trendHover < bars.length) {
      const bar = bars[state.trendHover];
      ctx.strokeStyle = "rgba(190,220,255,.55)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(bar.center, pad.t);
      ctx.lineTo(bar.center, pad.t + plotHeight);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function paintTrend() {
    const canvas = $("trend");
    const ctx = canvas.getContext("2d");
    const g = state.trendGeom;
    if (!g) return;
    if (g.type === "token") {
      paintTokenTrend();
      return;
    }
    const { width, height, pad, plotWidth, plotHeight, bucketCount } = g;
    ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
    ctx.fillStyle = "#0f161e";
    ctx.fillRect(0, 0, width, height);

    const yMax = Math.max(1, g.mode === "count" ? g.maxCount : g.maxTotal);
    const xOf = (index) => pad.l + ((index + 0.5) / bucketCount) * plotWidth;
    const yOf = (value) => pad.t + plotHeight - (value / yMax) * plotHeight;

    ctx.strokeStyle = "#1b2632";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = pad.l + (plotWidth * i) / 4;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + plotHeight); ctx.stroke();
    }
    for (let i = 0; i <= 2; i++) {
      const y = pad.t + (plotHeight * i) / 2;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotWidth, y); ctx.stroke();
    }

    if (g.empty) {
      ctx.fillStyle = "#7f8b9d";
      ctx.font = "12px Segoe UI, Microsoft YaHei, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("当前窗口没有事件", pad.l + plotWidth / 2, pad.t + plotHeight / 2);
      return;
    }

    ctx.font = "10px Segoe UI, Microsoft YaHei, sans-serif";
    ctx.fillStyle = "#6f7e91";
    ctx.textAlign = "right";
    for (let i = 0; i <= 2; i++) {
      const value = yMax * (1 - i / 2);
      ctx.fillText(g.mode === "count" ? formatCount(value) : formatUs(value), pad.l - 7, pad.t + (plotHeight * i) / 2 + 3);
    }
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const x = pad.l + (plotWidth * i) / 4;
      const ago = g.end - (g.start + g.span * i / 4);
      const label = i === 4 ? "现在" : "-" + formatAgo(ago);
      ctx.fillText(label, x, height - 6);
    }

    if (g.mode === "count") {
      ctx.beginPath();
      ctx.moveTo(xOf(0), pad.t + plotHeight);
      for (let i = 0; i < bucketCount; i++) {
        ctx.lineTo(xOf(i), yOf(g.buckets[i].count));
      }
      ctx.lineTo(xOf(bucketCount - 1), pad.t + plotHeight);
      ctx.closePath();
      const gradient = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotHeight);
      gradient.addColorStop(0, "rgba(78,161,255,.42)");
      gradient.addColorStop(1, "rgba(78,161,255,.03)");
      ctx.fillStyle = gradient;
      ctx.fill();
      ctx.beginPath();
      for (let i = 0; i < bucketCount; i++) {
        const x = xOf(i);
        const y = yOf(g.buckets[i].count);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "#6eb4ff";
      ctx.lineWidth = 1.6;
      ctx.stroke();
    } else {
      const series = [
        { key: "cuda", color: COLORS.gemm },
        { key: "cpu", color: COLORS.norm },
        { key: "other", color: COLORS.other },
      ];
      const base = new Array(bucketCount).fill(0);
      for (const item of series) {
        const values = g.buckets.map((bucket) => bucket[item.key]);
        const top = base.map((value, index) => value + values[index]);
        ctx.beginPath();
        for (let i = 0; i < bucketCount; i++) {
          const x = xOf(i);
          const y = yOf(top[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        for (let i = bucketCount - 1; i >= 0; i--) {
          ctx.lineTo(xOf(i), yOf(base[i]));
        }
        ctx.closePath();
        ctx.globalAlpha = 0.18;
        ctx.fillStyle = item.color;
        ctx.fill();
        ctx.globalAlpha = 0.95;
        ctx.beginPath();
        for (let i = 0; i < bucketCount; i++) {
          const x = xOf(i);
          const y = yOf(top[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        for (let i = 0; i < bucketCount; i++) base[i] = top[i];
      }
      ctx.globalAlpha = 1;
    }

    if (state.trendHover >= 0 && state.trendHover < bucketCount) {
      const x = xOf(state.trendHover);
      ctx.strokeStyle = "rgba(190,220,255,.55)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + plotHeight); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function renderHotspots(agg) {
    const clear = $("clearFocus");
    clear.hidden = !state.focus;
    document.querySelectorAll("#hotTabs .tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === state.hotTab);
    });
    const entries = agg.count ? topEntries(agg.maps[state.hotTab], 10) : [];
    if (!entries.length) {
      $("hotspots").innerHTML = "<div class='empty'>当前窗口没有事件</div>";
      return;
    }
    const top = Math.max(1, entries[0].totalUs);
    const total = Math.max(1, agg.totalUs);
    $("hotspots").innerHTML = entries.map((entry) => {
      const color = state.hotTab === "dev"
        ? devColor(entry.key)
        : opColor(entry.slowest ? entry.slowest.op : entry.key);
      const share = (entry.totalUs / total) * 100;
      const focused = state.focus && state.focus.tab === state.hotTab && state.focus.key === entry.key;
      return "<div class='hot-row" + (focused ? " focus" : "") + "' data-key='" + esc(entry.key) + "'>" +
        "<div class='hot-top'>" +
        "<span class='dot' style='background:" + color + "'></span>" +
        "<span class='hot-name' title='" + esc(entry.key) + "'>" + esc(entry.key) + "</span>" +
        "<span class='hot-val'>" + formatUs(entry.totalUs) + "</span>" +
        "</div>" +
        "<div class='hot-bar'><i style='width:" + ((entry.totalUs / top) * 100).toFixed(2) +
        "%;background:" + color + "'></i></div>" +
        "<div class='hot-sub'><span>" + share.toFixed(1) + "%</span>" +
        "<span>n=" + formatCount(entry.count) + "</span>" +
        "<span>avg " + formatUs(entry.totalUs / entry.count) + "</span>" +
        "<span>max " + formatUs(entry.maxUs) + "</span></div>" +
        "</div>";
    }).join("");
  }

  function compBlock(title, entries, total, labelOf, colorOf) {
    if (!entries.length || !total) {
      return "<div class='comp-block'><div class='comp-title'>" + esc(title) +
        "</div><div class='empty'>当前窗口没有事件</div></div>";
    }
    const segments = entries.map((entry) => {
      const pct = (entry.totalUs / total) * 100;
      return "<i style='width:" + pct.toFixed(3) + "%;background:" + colorOf(entry) +
        "' title='" + esc(labelOf(entry) + " · " + formatUs(entry.totalUs) + " · " + pct.toFixed(1) + "%") +
        "'></i>";
    }).join("");
    const rows = entries.map((entry) => {
      const pct = (entry.totalUs / total) * 100;
      return "<div class='comp-row'><span class='dot' style='background:" + colorOf(entry) +
        "'></span><span class='name'>" + esc(labelOf(entry)) + "</span>" +
        "<span class='val'>" + formatUs(entry.totalUs) + "</span>" +
        "<span class='pct'>" + pct.toFixed(1) + "%</span></div>";
    }).join("");
    return "<div class='comp-block'><div class='comp-title'>" + esc(title) +
      "</div><div class='stack'>" + segments + "</div>" +
      "<div class='comp-legend'>" + rows + "</div></div>";
  }

  function renderComposition(agg) {
    if (!agg.count) {
      $("composition").innerHTML = "<div class='empty'>当前窗口没有事件</div>";
      return;
    }
    const cats = Array.from(agg.categories.values()).sort((a, b) => b.totalUs - a.totalUs);
    const devs = Array.from(agg.maps.dev.values()).sort((a, b) => b.totalUs - a.totalUs);
    $("composition").innerHTML =
      compBlock("按类别", cats, agg.totalUs,
        (entry) => entry.label, (entry) => entry.color) +
      compBlock("按设备", devs, agg.totalUs,
        (entry) => entry.key, (entry) => devColor(entry.key));
  }

  function renderFormats(agg) {
    if (!agg.count) {
      $("formats").innerHTML = "<div class='empty'>当前窗口没有事件</div>";
      return;
    }
    const dtypes = Array.from(agg.outDtypes.entries())
      .sort((a, b) => b[1] - a[1]).slice(0, 6);
    const convs = Array.from(agg.conversions.entries())
      .sort((a, b) => b[1] - a[1]).slice(0, 6);
    let html = "<div class='fmt-label'>输出 dtype</div>";
    html += dtypes.length
      ? "<div class='chips'>" + dtypes.map(([dtype, count]) =>
          "<span class='chip clickable' data-filter='" + esc(dtype) +
          "' title='筛选包含 " + esc(dtype) + " 的事件'>" + esc(dtype) +
          " <b>" + esc(count) + "</b></span>").join("") + "</div>"
      : "<div class='hint'>没有可解析的 dtype</div>";
    html += "<div class='fmt-label'>dtype 转换（首个输入 → 首个输出）</div>";
    html += convs.length
      ? "<div class='conv-list'>" + convs.map(([pair, count]) => {
          const parts = pair.split(" → ");
          return "<div class='conv' data-filter='" + esc(parts[0] || "") +
            "' title='筛选包含 " + esc(parts[0] || "") + " 的事件'>" +
            "<span>" + esc(parts[0] || "?") + " <span class='arrow'>→</span> " +
            esc(parts[1] || "?") + "</span><b>" + esc(count) + "</b></div>";
        }).join("") + "</div>"
      : "<div class='hint'>未发现 dtype 变化</div>";
    $("formats").innerHTML = html;
  }

  function producerOf(name, beforeSeq) {
    for (let i = state.events.length - 1; i >= 0; i--) {
      const event = state.events[i];
      if (event.seq >= beforeSeq) continue;
      const parsed = ioOf(event);
      if (parsed.outputs.some((out) => out.name === name)) return event;
    }
    return null;
  }

  function buildChain(event) {
    const steps = [];
    const seen = new Set();
    function walk(current, depth) {
      if (!current || depth > 8 || seen.has(current.seq)) return;
      seen.add(current.seq);
      const parsed = ioOf(current);
      for (const input of parsed.inputs) {
        if (!input.name || input.name === "-") continue;
        const producer = producerOf(input.name, current.seq);
        if (producer) walk(producer, depth + 1);
      }
      steps.push({ event: current, depth });
    }
    walk(event, 0);
    return steps;
  }

  function chainSeqSet(seq) {
    const event = state.events.find((item) => item.seq === seq);
    if (!event) return new Set();
    return new Set(buildChain(event).map((step) => step.event.seq));
  }

  function ioRows(items, kind) {
    if (!items.length) return "<div class='muted'>无</div>";
    return items.map((item) =>
      "<div class='io-line'>" +
      "<span class='io-key " + kind + "'>" + (kind === "out" ? "out" : "in") + "</span>" +
      "<b title='" + esc(item.name) + "'>" + esc(item.name) + "</b>" +
      "<span class='io-meta'>[" + esc(item.dims) + "] " + esc(item.dtype || "?") + "</span>" +
      "</div>").join("");
  }

  function renderDetail(event) {
    const detail = $("detail");
    const chain = $("chain");
    if (!event) {
      detail.className = "empty";
      detail.textContent = "点击时间线或列表查看";
      chain.className = "empty";
      chain.textContent = "-";
      return;
    }
    detail.className = "";
    const parsed = ioOf(event);
    const fmt = formatOf(event);
    const opAgg = state.lastAgg && state.lastAgg.maps.op.get(event.op);
    const share = state.lastAgg && state.lastAgg.totalUs
      ? (event.dur_us / state.lastAgg.totalUs) * 100 : 0;
    const maxUs = state.lastAgg ? state.lastAgg.maxUs : event.dur_us;
    const opAvg = opAgg ? opAgg.totalUs / opAgg.count : 0;
    const opMax = opAgg ? opAgg.maxUs : 0;
    const opCount = opAgg ? opAgg.count : 0;
    const timing = eventTiming(event);
    const timingMetrics = timing
      ? "<div class='metric'><i>运行</i><b>" + formatUs(timing.runUs) + "</b></div>" +
        "<div class='metric'><i>传输</i><b>" + formatUs(timing.xferUs) + "</b></div>" +
        "<div class='metric'><i>同步等待</i><b>" + formatUs(timing.waitUs) + "</b></div>" +
        "<div class='metric'><i>其他</i><b>" + formatUs(timing.otherUs) + "</b></div>"
      : "";
    detail.innerHTML =
      "<div class='hero'><span class='dot' style='background:" + opColor(event.op) + "'></span>" +
      "<b>" + esc(event.op) + "</b><span class='muted'>#" + esc(event.seq) + "</span></div>" +
      "<div class='metric-grid'>" +
      "<div class='metric'><i>耗时</i><b>" + formatUs(event.dur_us) + "</b></div>" +
      "<div class='metric'><i>占窗口</i><b>" + share.toFixed(1) + "%</b></div>" +
      "<div class='metric'><i>同算子平均</i><b>" + (opCount ? formatUs(opAvg) : "-") + "</b></div>" +
      timingMetrics +
      "</div>" +
      "<div class='durbar'><i style='width:" +
      Math.min(100, maxUs ? (event.dur_us / maxUs) * 100 : 0).toFixed(2) +
      "%;background:" + heatColor(maxUs ? event.dur_us / maxUs : 0) + "'></i></div>" +
      "<div class='section-title'><span>同算子次数 " + formatCount(opCount) +
      "</span><span>最大 " + (opCount ? formatUs(opMax) : "-") + "</span></div>" +
      "<div class='tags'>" +
      "<span class='tag'><span class='dot' style='background:" + devColor(event.dev) + "'></span>" +
      esc(event.dev) + " · tid " + esc(event.tid) + "</span>" +
      "<span class='tag " + (phaseOf(event) !== "none" ? "phase-" + phaseOf(event) : "") +
      "'>" + esc(phaseLabel(phaseOf(event)) + (event.token >= 0 ? " · T" + event.token : "")) +
      "</span>" +
      (requestOf(event) >= 0 ? "<span class='tag'>R" + esc(requestOf(event)) + "</span>" : "") +
      "<span class='tag'>" + (event.layer >= 0 ? "layer " + esc(event.layer) : "layer -") + "</span>" +
      (event.weight ? "<span class='tag' title='" + esc(event.weight) + "'>" + esc(event.weight) + "</span>" : "") +
      "</div>" +
      "<div class='dtype-flow'>" +
      "<span class='chip'>" + esc(fmt.inD || "?") + "</span>" +
      "<span class='muted'>→</span>" +
      "<span class='chip'>" + esc(fmt.outD || "?") + "</span>" +
      "<span class='badge" + (fmt.changed ? " warn" : "") + "'>" +
      (fmt.changed ? "格式转换" : (fmt.inD && fmt.outD ? "同 dtype" : "dtype 未知")) + "</span>" +
      "</div>" +
      "<div class='io-block'><div class='io-head'>输入</div>" + ioRows(parsed.inputs, "in") +
      "<div class='io-head'>输出</div>" + ioRows(parsed.outputs, "out") + "</div>";

    const steps = buildChain(event);
    if (!steps.length) {
      chain.className = "empty";
      chain.textContent = "没有可用的依赖信息";
      return;
    }
    chain.className = "";
    chain.innerHTML = steps.map((step) => {
      const item = step.event;
      return "<div class='chain-item' data-seq='" + esc(item.seq) + "' style='margin-left:" +
        (step.depth * 10) + "px'>" +
        "<span class='op-dot' style='background:" + opColor(item.op) + "'></span>" +
        "<b>" + esc(item.op) + "</b> <span class='sub'>#" + esc(item.seq) +
        " · " + formatUs(item.dur_us) + (item.layer >= 0 ? " · L" + esc(item.layer) : "") +
        "</span><div class='chain-io'>" + esc(item.io || "") + "</div></div>";
    }).join("");
    chain.querySelectorAll(".chain-item").forEach((node) => {
      node.addEventListener("click", () => {
        const seq = Number(node.dataset.seq);
        const target = state.events.find((item) => item.seq === seq);
        if (target) selectEvent(target);
      });
    });
  }

  function selectEvent(event) {
    state.selectedSeq = event.seq;
    const view = computeView();
    const visible = filterEvents(state.events, view.start, view.end,
      { viewToken: view.tokenKey });
    state.lastView = view;
    state.lastAgg = aggregate(visible);
    renderDetail(event);
    drawTable();
    scheduleDraw();
  }

  function draw() {
    const view = computeView();
    const visible = filterEvents(state.events, view.start, view.end,
      { viewToken: view.tokenKey });
    const agg = aggregate(visible);
    state.lastView = view;
    state.lastAgg = agg;
    $("timelineNote").textContent = view.label;
    const emptyHint = $("emptyHint");
    const jump = $("jumpMatch");
    let showHint = false;
    if (!visible.length && state.events.length) {
      const matchedAll = filterEvents(state.events, 0, Number.MAX_SAFE_INTEGER);
      if (matchedAll.length) {
        const last = matchedAll[matchedAll.length - 1];
        const ago = latestTime() - last.t_us;
        $("emptyHintText").textContent =
          "当前 " + formatUs(view.end - view.start) + " 窗口无匹配事件；缓冲区有 " +
          formatCount(matchedAll.length) + " 条匹配，最近一条在 -" + formatAgo(ago) +
          "（" + last.op + " · " + last.dev + "）";
        jump.hidden = false;
        emptyHint.hidden = false;
        showHint = true;
      } else if (state.deviceFilter || state.filter.trim() || state.minDurUs ||
                 state.phaseFilter || state.tokenFilter != null) {
        $("emptyHintText").textContent =
          "当前缓冲区没有匹配事件；可检索历史最多 " +
          formatCount(Math.min(state.capacity || MAX_EVENTS, MAX_EVENTS)) +
          " 条，更早的事件可能已被覆盖";
        jump.hidden = true;
        emptyHint.hidden = false;
        showHint = true;
      }
    }
    if (!showHint) emptyHint.hidden = true;

    drawTimeline(visible, view, agg);
    drawTrend(visible, view);
    renderStats(agg);
    const now = Date.now();
    if (now - state.lastSideDraw >= 350) {
      state.lastSideDraw = now;
      renderHotspots(agg);
      renderComposition(agg);
      renderFormats(agg);
      renderPhaseTokens();
      renderTimeSplit();
    }
  }

  function scheduleDraw() {
    if (state.drawPending) return;
    state.drawPending = true;
    requestAnimationFrame(() => {
      state.drawPending = false;
      draw();
    });
  }

  function drawTable() {
    const view = computeView();
    const visible = filterEvents(state.events, view.start, view.end,
      { viewToken: view.tokenKey });
    const maxDur = visible.reduce((max, event) => Math.max(max, event.dur_us), 1);
    const rows = visible.slice(-300).reverse();
    $("tableNote").textContent =
      "最近 " + rows.length + " 条 · 窗口 " + visible.length + " 条";
    const focus = state.focus;
    $("rows").innerHTML = rows.map((event) => {
      const selected = event.seq === state.selectedSeq ? " selected" : "";
      const dim = focus && !focusMatches(event) ? " dim" : "";
      const t = view.start != null ? ((event.t_us - view.start) / 1000).toFixed(2) : "-";
      const fmt = formatOf(event);
      const formatText = fmt.inD || fmt.outD
        ? (fmt.inD || "?") + " → " + (fmt.outD || "?")
        : "-";
      const phaseText = eventPhaseText(event);
      const phaseTitle = phaseText + (requestOf(event) >= 0 ? " · request " + requestOf(event) : "");
      return "<tr data-seq='" + esc(event.seq) + "' class='" + (selected + dim).trim() + "'>" +
        "<td class='num'>" + esc(event.seq) + "</td>" +
        "<td class='phase phase-" + esc(phaseOf(event)) + "' title='" + esc(phaseTitle) + "'>" +
        esc(phaseText) + "</td>" +
        "<td class='num'>" + esc(t) + "</td>" +
        "<td class='num dur' style='color:" + heatColor(event.dur_us / maxDur) + "'>" +
        esc(event.dur_us) + "</td>" +
        "<td><span class='op-dot' style='background:" + opColor(event.op) + "'></span>" +
        esc(event.op) + "</td>" +
        "<td class='num'>" + (event.layer >= 0 ? esc(event.layer) : "-") + "</td>" +
        "<td>" + esc(event.dev) + "</td>" +
        "<td>" + esc(event.weight || "-") + "</td>" +
        "<td class='fmt" + (fmt.changed ? " changed" : "") + "' title='" + esc(formatText) + "'>" +
        esc(formatText) + "</td>" +
        "<td class='io' title='" + esc(event.io || "") + "'>" + esc(event.io || "-") + "</td>" +
        "</tr>";
    }).join("");
  }

  function showTooltip(event, clientX, clientY) {
    const tip = $("tooltip");
    if (!event) { tip.style.display = "none"; return; }
    tip.innerHTML =
      "<b>" + esc(event.op) + "</b>  <span style='color:#8493a6'>#" + esc(event.seq) + "</span><br>" +
      "耗时 " + formatUs(event.dur_us) + " · " + esc(event.dev) + " · tid " + esc(event.tid) +
      " · " + esc(eventPhaseText(event)) +
      (event.layer >= 0 ? " · layer " + esc(event.layer) : "") +
      (requestOf(event) >= 0 ? " · req " + esc(requestOf(event)) : "") +
      (event.weight ? "<br>weight: " + esc(event.weight) : "") +
      (event.io ? "<br>io: " + esc(event.io) : "");
    const wrap = $("timelineWrap");
    const rect = wrap.getBoundingClientRect();
    tip.style.display = "block";
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    const scrollLeft = wrap.scrollLeft || 0;
    const scrollTop = wrap.scrollTop || 0;
    let left = clientX - rect.left + scrollLeft + 14;
    let top = clientY - rect.top + scrollTop + 14;
    if (left + width > scrollLeft + wrap.clientWidth - 8) {
      left = clientX - rect.left + scrollLeft - width - 14;
    }
    if (top + height > scrollTop + wrap.clientHeight - 8) {
      top = clientY - rect.top + scrollTop - height - 14;
    }
    tip.style.left = Math.max(4, left) + "px";
    tip.style.top = Math.max(4, top) + "px";
  }

  function findEventAt(clientX, clientY) {
    const layout = state.layout;
    if (!layout) return null;
    const rect = $("timeline").getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    if (x < layout.gutter || y < layout.axis) return null;
    const index = Math.floor((y - layout.axis) / layout.rowHeight);
    if (index < 0 || index >= layout.keys.length) return null;
    const key = layout.keys[index];
    const span = layout.end - layout.start;
    const plotWidth = Math.max(40, layout.width - layout.gutter - 10);
    const time = layout.start + ((x - layout.gutter) / plotWidth) * span;
    let best = null;
    for (const event of layout.visible) {
      if (rowKey(event) !== key) continue;
      const end = event.t_us + Math.max(1, event.dur_us || 0);
      if (time >= event.t_us - span * 0.002 && time <= end + span * 0.002) {
        if (!best || event.seq > best.seq) best = event;
      }
    }
    return best;
  }

  function showTrendTip(index, clientX, clientY) {
    const tip = $("trendTip");
    const g = state.trendGeom;
    if (!g || g.empty || index < 0) {
      tip.style.display = "none";
      return;
    }
    if (g.type === "token") {
      if (index >= g.bars.length) {
        tip.style.display = "none";
        return;
      }
      const token = g.bars[index].token;
      const splitText = token.hasSplit
        ? "<span style='color:#4ea1ff'>算子 " + formatUs(token.runUs) + "</span>" +
          " · <span style='color:#38c7c7'>传输 " + formatUs(token.xferUs) + "</span><br>" +
          "<span style='color:#ffb454'>等待 " + formatUs(token.waitUs) + "</span>" +
          " · 其他 " + formatUs(token.otherOpUs) +
          " · 空隙 " + formatUs(token.gapUs)
        : "算子累计 " + formatUs(token.totalUs) +
          " · cuda " + formatUs(token.cudaUs) +
          " · cpu " + formatUs(token.cpuUs);
      tip.innerHTML =
        "<b>" + esc(tokenLabel(token)) + "</b><br>" +
        "耗时 " + formatUs(token.latencyUs) + " · " + formatCount(token.count) + " ops<br>" +
        splitText +
        "<br><span style='color:#8190a3'>点击定位到该 token</span>";
      const wrap = $("trendWrap");
      const rect = wrap.getBoundingClientRect();
      tip.style.display = "block";
      const width = tip.offsetWidth;
      const height = tip.offsetHeight;
      let left = clientX - rect.left + 12;
      let top = clientY - rect.top + 12;
      if (left + width > wrap.clientWidth - 8) left = clientX - rect.left - width - 12;
      if (top + height > wrap.clientHeight - 8) top = clientY - rect.top - height - 12;
      tip.style.left = Math.max(4, left) + "px";
      tip.style.top = Math.max(4, top) + "px";
      return;
    }
    if (!g.buckets || index >= g.buckets.length) {
      tip.style.display = "none";
      return;
    }
    const bucket = g.buckets[index];
    const total = bucket.cuda + bucket.cpu + bucket.other;
    const ago = g.end - (g.start + (index + 0.5) * g.bucketUs);
    tip.innerHTML =
      "<b>T-" + formatAgo(ago) + "</b><br>" +
      "累计 " + formatUs(total) + " · 事件 " + bucket.count + "<br>" +
      "<span style='color:" + COLORS.gemm + "'>cuda " + formatUs(bucket.cuda) + "</span>" +
      " · <span style='color:" + COLORS.norm + "'>cpu " + formatUs(bucket.cpu) + "</span>" +
      (bucket.other ? " · other " + formatUs(bucket.other) : "");
    const wrap = $("trendWrap");
    const rect = wrap.getBoundingClientRect();
    tip.style.display = "block";
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    let left = clientX - rect.left + 12;
    let top = clientY - rect.top + 12;
    if (left + width > wrap.clientWidth - 8) left = clientX - rect.left - width - 12;
    if (top + height > wrap.clientHeight - 8) top = clientY - rect.top - height - 12;
    tip.style.left = Math.max(4, left) + "px";
    tip.style.top = Math.max(4, top) + "px";
  }

  function setFilter(text) {
    state.filter = text;
    state.lastSideDraw = 0;
    $("filter").value = text;
    scheduleDraw();
    drawTable();
  }

  function updateTokenChip() {
    const chip = $("tokenChip");
    if (state.tokenFilter == null) {
      chip.hidden = true;
      return;
    }
    chip.hidden = false;
    $("tokenChipValue").textContent =
      (state.tokenFilterReq >= 0 ? "R" + state.tokenFilterReq + " · " : "") +
      "T" + state.tokenFilter;
  }

  function setPhaseFilter(phase, reveal) {
    state.phaseFilter = phase;
    if (reveal || phase !== "decode") {
      state.tokenFilter = null;
      state.tokenFilterReq = null;
      state.frozenTokenKey = null;
    }
    $("phaseFilter").value = phase;
    updateTokenChip();
    state.lastSideDraw = 0;
    if (reveal) {
      state.windowMode = "time";
      state.windowUs = 0;
      $("window").value = "0";
      state.follow = true;
      $("follow").checked = true;
      state.paused = false;
      $("pause").checked = false;
      state.frozenEnd = 0;
      state.frozenTokenKey = null;
      state.lastSideDraw = 0;
    } else if (phase) {
      const currentView = computeView();
      const visible = filterEvents(state.events, currentView.start, currentView.end,
        { viewToken: currentView.tokenKey });
      if (!visible.length) {
        const matched = filterEvents(state.events, 0, Number.MAX_SAFE_INTEGER);
        if (matched.length) {
          let endUs = 0;
          for (const event of matched) {
            endUs = Math.max(endUs, event.t_us + (event.dur_us || 0));
          }
          state.follow = false;
          $("follow").checked = false;
          state.paused = false;
          $("pause").checked = false;
          state.frozenEnd = endUs + Math.max(
            state.windowUs > 0 ? state.windowUs * 0.1 : 0, 1000);
          state.lastSideDraw = 0;
          schedulePoll();
        }
      }
    }
    scheduleDraw();
    drawTable();
  }

  function selectToken(token, req) {
    const tokenEvents = filterEvents(state.events, 0, Number.MAX_SAFE_INTEGER,
      { ignoreTime: true, ignorePhase: true, ignoreToken: true })
      .filter((event) => phaseOf(event) === "decode" &&
        Number(event.token) === token && requestOf(event) === req);
    if (!tokenEvents.length) {
      return;
    }
    state.phaseFilter = "decode";
    $("phaseFilter").value = "decode";
    state.windowMode = "token";
    $("window").value = "token";
    state.tokenFilter = token;
    state.tokenFilterReq = req;
    state.frozenTokenKey = tokenKey(req, token);
    state.follow = false;
    $("follow").checked = false;
    state.paused = false;
    $("pause").checked = false;
    let endUs = 0;
    let slowest = null;
    for (const event of tokenEvents) {
      endUs = Math.max(endUs, event.t_us + (event.dur_us || 0));
      if (!slowest || event.dur_us > slowest.dur_us) {
        slowest = event;
      }
    }
    state.frozenEnd = endUs + Math.max(
      state.windowUs > 0 ? state.windowUs * 0.1 : 0, 1000);
    state.lastSideDraw = 0;
    updateTokenChip();
    schedulePoll();
    if (slowest) {
      selectEvent(slowest);
    }
    scheduleDraw();
    drawTable();
  }

  function resetFilters() {
    state.filter = "";
    state.deviceFilter = "";
    state.phaseFilter = "";
    state.tokenFilter = null;
    state.tokenFilterReq = null;
    state.frozenTokenKey = null;
    state.minDurUs = 0;
    state.focus = null;
    state.lastSideDraw = 0;
    $("filter").value = "";
    $("deviceFilter").value = "";
    $("phaseFilter").value = "";
    $("minDur").value = 0;
    updateTokenChip();
    scheduleDraw();
    drawTable();
  }

  function locateMatch() {
    const matched = state.events.length
      ? filterEvents(state.events, 0, Number.MAX_SAFE_INTEGER)
      : [];
    if (!matched.length) {
      setStatus("缓冲区没有匹配事件", "bad");
      return;
    }
    const last = matched[matched.length - 1];
    if (state.windowMode === "token" && phaseOf(last) === "decode" &&
        Number(last.token) >= 0) {
      state.tokenFilter = Number(last.token);
      state.tokenFilterReq = requestOf(last);
      state.frozenTokenKey = tokenKeyOf(last);
      updateTokenChip();
    }
    state.follow = false;
    $("follow").checked = false;
    state.paused = false;
    $("pause").checked = false;
    state.frozenEnd = last.t_us + Math.max(
      state.windowUs > 0 ? state.windowUs * 0.1 : 0,
      (last.dur_us || 0) * 2, 1000);
    state.lastSideDraw = 0;
    selectEvent(last);
    schedulePoll();
    scheduleDraw();
    drawTable();
  }

  async function toggleTracing() {
    const button = $("toggle");
    button.disabled = true;
    try {
      const response = await fetch("/api/op-trace/enabled", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !state.enabled }),
      });
      const data = await response.json();
      if (!data.supported) {
        state.supported = false;
        state.enabled = false;
        setStatus("原生库不支持", "bad");
      } else {
        state.enabled = !!data.enabled;
        state.idlePolls = 0;
        setStatus();
      }
    } catch (error) {
      setStatus("切换失败: " + error.message, "bad");
    } finally {
      button.disabled = !state.supported;
      schedulePoll();
      scheduleDraw();
    }
  }

  async function clearTrace() {
    try {
      await fetch("/api/op-trace/clear", { method: "POST" });
      state.events = [];
      state.afterSeq = 0;
      state.t0 = 0;
      state.selectedSeq = null;
      state.rate = 0;
      state.idlePolls = 0;
      state.rateTime = 0;
      state.rateSeq = 0;
      state.focus = null;
      state.deviceFilter = "";
      state.phaseFilter = "";
      state.tokenFilter = null;
      state.tokenFilterReq = null;
      state.frozenTokenKey = null;
      state.trendHover = -1;
      state.lastAgg = null;
      state.lastSideDraw = 0;
      $("deviceFilter").value = "";
      $("phaseFilter").value = "";
      updateTokenChip();
      renderDetail(null);
      drawTable();
      scheduleDraw();
      schedulePoll();
    } catch (error) {
      setStatus("清空失败: " + error.message, "bad");
    }
  }

  $("toggle").addEventListener("click", toggleTracing);
  $("clear").addEventListener("click", clearTrace);
  $("resetFilter").addEventListener("click", resetFilters);
  $("locateMatch").addEventListener("click", locateMatch);
  $("jumpMatch").addEventListener("click", locateMatch);
  $("window").addEventListener("change", (event) => {
    const value = event.target.value;
    state.lastSideDraw = 0;
    if (value === "token") {
      state.windowMode = "token";
      state.frozenTokenKey = state.tokenFilter != null
        ? tokenKey(state.tokenFilterReq, state.tokenFilter)
        : (state.follow ? null : (latestToken()?.key || null));
      state.frozenEnd = 0;
    } else {
      state.windowMode = "time";
      state.windowUs = Number(value);
      state.tokenFilter = null;
      state.tokenFilterReq = null;
      state.frozenTokenKey = null;
      state.frozenEnd = state.follow ? 0 : latestTime();
      updateTokenChip();
    }
    scheduleDraw();
    drawTable();
  });
  $("trendMode").addEventListener("change", (event) => {
    state.trendMode = event.target.value;
    scheduleDraw();
  });
  $("view").addEventListener("change", (event) => {
    state.view = event.target.value;
    scheduleDraw();
  });
  $("deviceFilter").addEventListener("change", (event) => {
    state.deviceFilter = event.target.value;
    state.lastSideDraw = 0;
    scheduleDraw();
    drawTable();
  });
  $("phaseFilter").addEventListener("change", (event) => {
    setPhaseFilter(event.target.value);
  });
  $("clearToken").addEventListener("click", () => {
    state.tokenFilter = null;
    state.tokenFilterReq = null;
    state.frozenTokenKey = null;
    if (state.windowMode === "token") {
      state.follow = true;
      $("follow").checked = true;
      state.paused = false;
      $("pause").checked = false;
      state.frozenEnd = 0;
      schedulePoll();
    }
    state.lastSideDraw = 0;
    updateTokenChip();
    scheduleDraw();
    drawTable();
  });
  $("phaseStats").addEventListener("click", (event) => {
    const box = event.target.closest("[data-phase]");
    if (!box) return;
    const phase = box.dataset.phase;
    setPhaseFilter(state.phaseFilter === phase ? "" : phase, true);
  });
  $("tokenList").addEventListener("click", (event) => {
    const row = event.target.closest(".token-row[data-token]");
    if (!row) return;
    selectToken(Number(row.dataset.token), Number(row.dataset.req));
  });
  $("minDur").addEventListener("input", (event) => {
    state.minDurUs = Math.max(0, Number(event.target.value) || 0);
    state.lastSideDraw = 0;
    scheduleDraw();
    drawTable();
  });
  $("filter").addEventListener("input", (event) => {
    state.filter = event.target.value;
    state.lastSideDraw = 0;
    scheduleDraw();
    drawTable();
  });
  $("follow").addEventListener("change", (event) => {
    state.follow = event.target.checked;
    if (!state.follow) {
      state.frozenEnd = latestTime();
      if (state.windowMode === "token") {
        state.frozenTokenKey = latestToken()?.key || null;
      }
    } else {
      state.tokenFilter = null;
      state.tokenFilterReq = null;
      state.frozenTokenKey = null;
      state.frozenEnd = 0;
      updateTokenChip();
      schedulePoll();
    }
    state.lastSideDraw = 0;
    scheduleDraw();
    drawTable();
  });
  $("pause").addEventListener("change", (event) => {
    state.paused = event.target.checked;
    if (state.paused) {
      state.frozenEnd = latestTime();
      if (state.windowMode === "token") {
        state.frozenTokenKey = latestToken()?.key || null;
      }
    } else {
      state.frozenEnd = 0;
      if (state.follow) state.frozenTokenKey = null;
      schedulePoll();
    }
    state.lastSideDraw = 0;
    scheduleDraw();
    drawTable();
  });

  $("hotTabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab[data-tab]");
    if (!tab) return;
    state.hotTab = tab.dataset.tab;
    state.focus = null;
    state.lastSideDraw = 0;
    scheduleDraw();
    drawTable();
  });
  $("clearFocus").addEventListener("click", () => {
    state.focus = null;
    state.lastSideDraw = 0;
    scheduleDraw();
    drawTable();
  });
  $("hotspots").addEventListener("click", (event) => {
    const row = event.target.closest(".hot-row[data-key]");
    if (!row) return;
    const key = row.dataset.key;
    const map = state.lastAgg && state.lastAgg.maps[state.hotTab];
    const entry = map && map.get(key);
    state.focus = { tab: state.hotTab, key };
    state.lastSideDraw = 0;
    if (entry && entry.slowest) {
      selectEvent(entry.slowest);
    } else {
      scheduleDraw();
      drawTable();
    }
  });
  $("formats").addEventListener("click", (event) => {
    const node = event.target.closest("[data-filter]");
    if (!node) return;
    setFilter(node.dataset.filter || "");
  });
  $("rows").addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-seq]");
    if (!row) return;
    const seq = Number(row.dataset.seq);
    const item = state.events.find((entry) => entry.seq === seq);
    if (item) selectEvent(item);
  });

  const timeline = $("timeline");
  timeline.addEventListener("mousemove", (event) => {
    const hit = findEventAt(event.clientX, event.clientY);
    showTooltip(hit, event.clientX, event.clientY);
  });
  timeline.addEventListener("mouseleave", () => showTooltip(null));
  timeline.addEventListener("click", (event) => {
    const hit = findEventAt(event.clientX, event.clientY);
    if (hit) selectEvent(hit);
  });

  const trendCanvas = $("trend");
  trendCanvas.addEventListener("mousemove", (event) => {
    const g = state.trendGeom;
    if (!g || g.empty) return;
    const rect = trendCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    if (x < g.pad.l || x > g.pad.l + g.plotWidth) {
      if (state.trendHover !== -1) {
        state.trendHover = -1;
        paintTrend();
      }
      $("trendTip").style.display = "none";
      return;
    }
    const count = g.type === "token" ? g.bars.length : g.bucketCount;
    const index = Math.max(0, Math.min(count - 1,
      Math.floor(((x - g.pad.l) / g.plotWidth) * count)));
    if (index !== state.trendHover) {
      state.trendHover = index;
      paintTrend();
    }
    showTrendTip(index, event.clientX, event.clientY);
  });
  trendCanvas.addEventListener("click", (event) => {
    const g = state.trendGeom;
    if (!g || g.type !== "token" || g.empty) return;
    const rect = trendCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    if (x < g.pad.l || x > g.pad.l + g.plotWidth) return;
    const index = Math.floor(((x - g.pad.l) / g.plotWidth) * g.bars.length);
    const bar = g.bars[Math.max(0, Math.min(g.bars.length - 1, index))];
    if (bar) selectToken(bar.token.token, bar.token.req);
  });
  trendCanvas.addEventListener("mouseleave", () => {
    state.trendHover = -1;
    paintTrend();
    $("trendTip").style.display = "none";
  });

  new ResizeObserver(() => scheduleDraw()).observe($("timelineWrap"));
  new ResizeObserver(() => scheduleDraw()).observe($("trendWrap"));

  setStatus("连接中");
  updateTokenChip();
  renderStats(null);
  poll();
})();
</script>
</body>
</html>
"""
