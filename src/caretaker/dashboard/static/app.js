/* Caretaker dashboard — vanilla JS, no external requests.
 * Memory text is untrusted: it is only ever inserted with textContent / text nodes. */
"use strict";

// ── Token handling ──────────────────────────────────────────────────────────
// `caretaker dashboard` opens /#token=…  The fragment never reaches the server
// or the Referer header; we keep it in sessionStorage and remove it from the URL.
const TOKEN_KEY = "caretaker-dashboard-token";
(function captureToken() {
  const m = location.hash.match(/token=([A-Za-z0-9_-]+)/);
  if (m) {
    try { sessionStorage.setItem(TOKEN_KEY, m[1]); } catch (_) { /* storage blocked */ }
    window.__caretakerToken = m[1];
    history.replaceState(null, "", location.pathname + "#/overview");
  }
})();
function token() {
  if (window.__caretakerToken) return window.__caretakerToken;
  try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch (_) { return ""; }
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", "X-Caretaker-Token": token(), ...(options.headers || {}) },
  });
  let body = null;
  try { body = await res.json(); } catch (_) { /* empty */ }
  if (!res.ok) {
    const err = new Error((body && body.error) || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return body;
}

// ── DOM helpers ─────────────────────────────────────────────────────────────
function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    // CSSOM, not a style attribute: allowed under the page's strict CSP.
    else if (k === "style") el.style.cssText = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  appendAll(el, children);
  return el;
}
function appendAll(el, children) {
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}
const SVG_NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// ── Formatting ──────────────────────────────────────────────────────────────
const nf = new Intl.NumberFormat();
const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
const fmtNum = (n) => (n >= 10000 ? compact.format(n) : nf.format(n || 0));
function fmtBytes(b) {
  if (!b) return "0 B";
  const u = ["B", "KB", "MB", "GB"]; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(b >= 10 || i === 0 ? 0 : 1)} ${u[i]}`;
}
function parseDate(v) {
  if (!v) return null;
  const d = new Date(String(v).includes("T") || String(v).includes("+") ? v : String(v).replace(" ", "T") + "Z");
  return isNaN(d) ? null : d;
}
function fmtWhen(v) {
  const d = parseDate(v);
  if (!d) return "—";
  const sec = (Date.now() - d.getTime()) / 1000;
  if (sec < 60 && sec > -60) return "just now";
  if (sec < 3600 && sec > 0) return `${Math.round(sec / 60)} min ago`;
  if (sec < 86400 && sec > 0) return `${Math.round(sec / 3600)} h ago`;
  if (sec < 7 * 86400 && sec > 0) return `${Math.round(sec / 86400)} d ago`;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}
const fmtDateTime = (v) => { const d = parseDate(v); return d ? d.toLocaleString() : "—"; };
const titleCase = (t) => String(t || "").toLowerCase().replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

const TEMPS = ["PRIORITY_HOT", "HOT", "WARM", "COLD"];
function tempBadge(t) {
  const cls = TEMPS.includes(t) ? `t-${t.toLowerCase().replace("_", "-")}` : "";
  return h("span", { class: "temp" }, h("span", { class: `dot ${cls}` }), titleCase(t || "unknown"));
}
const typeTag = (t) => h("span", { class: "tag" }, titleCase(t || "memory"));

// ── Tooltip ─────────────────────────────────────────────────────────────────
const tip = () => document.getElementById("tooltip");
function showTip(evt, title, line) {
  const t = tip();
  t.replaceChildren(h("strong", {}, title), line ? document.createTextNode(line) : "");
  t.hidden = false;
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const r = t.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  t.style.left = `${x}px`; t.style.top = `${y}px`;
}
const hideTip = () => { tip().hidden = true; };

// ── Charts ──────────────────────────────────────────────────────────────────
function tableView(caption, headers, rows) {
  return h("details", { class: "table-view" },
    h("summary", {}, "Show as table"),
    h("table", {},
      h("caption", { class: "visually-hidden" }, caption),
      h("thead", {}, h("tr", {}, headers.map((x, i) => h("th", { class: i ? "num" : null }, x)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c, i) => h("td", { class: i ? "num" : null }, c)))))));
}

function hbarChart(title, rows, unit, labelFn = titleCase) {
  if (!rows.length) return h("p", { class: "empty" }, "No data yet");
  const max = Math.max(...rows.map((r) => r.count), 1);
  const total = rows.reduce((a, r) => a + r.count, 0);
  const bars = h("div", { class: "hbars", role: "list", "aria-label": title },
    rows.map((r) => {
      const label = labelFn(r.key);
      const share = total ? Math.round((r.count / total) * 100) : 0;
      return h("div", {
        class: "hbar", role: "listitem", tabindex: "0",
        "aria-label": `${label}: ${r.count} ${unit} (${share}%)`,
        onmousemove: (e) => showTip(e, label, `${nf.format(r.count)} ${unit} · ${share}%`),
        onmouseleave: hideTip,
      },
        h("span", { class: "hbar-label", title: label }, label),
        h("span", { class: "hbar-track" },
          h("span", { class: "hbar-fill", style: `width:${(r.count / max) * 100}%` },
            h("span", { class: "hbar-value" }, nf.format(r.count)))));
    }));
  return h("div", {}, bars,
    tableView(title, ["Category", unit[0].toUpperCase() + unit.slice(1)], rows.map((r) => [labelFn(r.key), nf.format(r.count)])));
}

// Axis top for whole-number counts: ticks at 0, max/2, max are always integers.
function niceMax(v) {
  if (v <= 4) return 4;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 4, 6, 8, 10]) {
    const top = m * pow;
    if (top >= v && Number.isInteger(top / 2)) return top;
  }
  return 10 * pow;
}

function columnChart(title, points) {
  // Drawn at the container's real pixel width (and redrawn on resize) so text
  // and bar widths are true pixels, never stretched by viewBox scaling.
  const holder = h("div", { class: "col-chart" });
  let lastW = 0;
  const draw = () => {
    const W = Math.max(260, Math.round(holder.clientWidth));
    if (!holder.isConnected || W === lastW) return;
    lastW = W;
    holder.replaceChildren(drawColumns(title, points, W));
  };
  new ResizeObserver(draw).observe(holder);
  return h("div", {}, holder,
    tableView(title, ["Day", "New memories"], points.filter((p) => p.count).map((p) => [p.date, nf.format(p.count)])));
}

function drawColumns(title, points, W) {
  const H = 190, L = 30, R = 8, T = 18, B = 24;
  const innerW = W - L - R, innerH = H - T - B;
  const max = niceMax(Math.max(...points.map((p) => p.count), 0));
  const slot = innerW / points.length;
  const barW = Math.min(24, Math.max(4, slot - 4));
  const y = (v) => T + innerH - (v / max) * innerH;
  const svg = s("svg", { class: "chart-svg", width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });

  for (let i = 0; i <= 2; i++) {
    const v = (max / 2) * i, yy = y(v);
    svg.appendChild(s("line", { class: i === 0 ? "base-line" : "grid-line", x1: L, x2: W - R, y1: yy, y2: yy }));
    svg.appendChild(s("text", { class: "tick", x: L - 6, y: yy + 4, "text-anchor": "end" }, nf.format(v)));
  }
  let maxIdx = -1, maxVal = 0;
  points.forEach((p, i) => { if (p.count > maxVal) { maxVal = p.count; maxIdx = i; } });

  const band = s("rect", { class: "hover-band", x: 0, y: T, width: slot, height: innerH, visibility: "hidden" });
  svg.appendChild(band);
  points.forEach((p, i) => {
    const cx = L + slot * i + slot / 2, x = cx - barW / 2;
    if (p.count > 0) {
      const top = y(p.count), hgt = T + innerH - top, r = Math.min(4, hgt, barW / 2);
      // Rounded data-end (top), square at the baseline.
      const d = `M${x},${T + innerH} V${top + r} Q${x},${top} ${x + r},${top} H${x + barW - r} Q${x + barW},${top} ${x + barW},${top + r} V${T + innerH} Z`;
      svg.appendChild(s("path", { class: "col", d }));
    }
    // Weekly date labels, anchored on today so the last label never collides.
    if ((points.length - 1 - i) % 7 === 0) {
      const d = parseDate(p.date + "T00:00:00Z");
      svg.appendChild(s("text", { class: "tick", x: cx, y: H - 6, "text-anchor": "middle" },
        d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" })));
    }
    const hit = s("rect", { class: "hit", x: L + slot * i, y: T, width: slot, height: innerH });
    const label = parseDate(p.date + "T00:00:00Z").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
    hit.addEventListener("mousemove", (e) => {
      band.setAttribute("x", L + slot * i); band.setAttribute("visibility", "visible");
      showTip(e, label, `${nf.format(p.count)} new ${p.count === 1 ? "memory" : "memories"}`);
    });
    hit.addEventListener("mouseleave", () => { band.setAttribute("visibility", "hidden"); hideTip(); });
    svg.appendChild(hit);
  });
  if (maxIdx >= 0 && maxVal > 0) {
    svg.appendChild(s("text", { class: "cap-label", x: L + slot * maxIdx + slot / 2, y: y(maxVal) - 5, "text-anchor": "middle" }, nf.format(maxVal)));
  }
  return svg;
}

// ── Views ───────────────────────────────────────────────────────────────────
const view = () => document.getElementById("view");

// ── Live refresh state ──────────────────────────────────────────────────────
// Views re-fetch every REFRESH_MS. A background refresh only redraws when the
// data changed (or once a minute, so "5 min ago" labels stay current), and
// never while you're editing. `gen` makes sure only the newest request draws.
const REFRESH_MS = 10000;
const live = { gen: 0, lastSig: {}, lastRender: {}, lastOk: 0, lastAttempt: 0, busy: false, failed: false, stopped: false, detailDirty: false };

function shouldDraw(key, data, opts = {}) {
  if (opts.gen !== undefined && opts.gen !== live.gen) return false;   // superseded by a newer request
  const sig = JSON.stringify(data), now = Date.now();
  if (opts.background && live.lastSig[key] === sig && now - (live.lastRender[key] || 0) < 60000) return false;
  live.lastSig[key] = sig; live.lastRender[key] = now;
  return true;
}

// ── Project scope (sidebar switcher) ────────────────────────────────────────
// "" = all projects, "__global__" = memories with no project, else one project.
const PROJECT_KEY = "caretaker-dashboard-project";
let currentProject = (() => { try { return sessionStorage.getItem(PROJECT_KEY) || ""; } catch (_) { return ""; } })();
function withProject(path) {
  if (!currentProject) return path;
  return path + (path.includes("?") ? "&" : "?") + "project=" + encodeURIComponent(currentProject);
}
const scopedProject = () => (currentProject && currentProject !== "__global__" ? currentProject : "");

async function refreshProjectSwitcher() {
  const sel = document.getElementById("project-switcher");
  if (!sel) return;
  let list;
  try { list = await api("/api/projects"); } catch (_) { return; }
  const sig = JSON.stringify(list);
  if (sel.dataset.sig === sig) return;
  sel.dataset.sig = sig;
  const opts = [h("option", { value: "" }, "All projects"), h("option", { value: "__global__" }, "Global (no project)")];
  for (const p of list) if (p.project) opts.push(h("option", { value: p.project, title: p.project }, `${p.project} (${nf.format(p.count)})`));
  sel.replaceChildren(...opts);
  sel.value = currentProject;
  if (sel.value !== currentProject) { currentProject = ""; sel.value = ""; }
}

async function refreshNavCounts() {
  let s;
  try { s = await api("/api/summary"); } catch (_) { return; }
  const set = (key, n, label) => {
    const el = document.querySelector(`[data-count="${key}"]`);
    if (!el) return;
    el.hidden = !n;
    el.textContent = n > 999 ? "999+" : String(n || "");
    el.title = label;
  };
  set("review", s.replaced + s.stale, `${s.replaced} replaced · ${s.stale} not used lately`);
  set("privacy", s.privacy_review, `${s.privacy_review} waiting for privacy review`);
  updateDataNote(s.data_flow);
}

// Sidebar note on where memory text goes. It must never claim "on this device
// only" while summaries use a cloud model or cloud backup is switched on.
function updateDataNote(flow) {
  const el = document.getElementById("data-note");
  if (!el || !flow) return;
  const out = [];
  if (!flow.compression_on_device) out.push("Summaries via Anthropic API");
  if (flow.cloud_backup) out.push("Encrypted cloud backup on");
  const local = out.length === 0;
  const title = local ? "Stays on this device" : "Data leaves this device";
  const lines = local ? ["Nothing is sent to a server"] : out;
  el.dataset.state = local ? "local" : "leaves";
  el.querySelector(".dn-title").textContent = title;
  el.querySelector(".dn-sub").replaceChildren(...lines.map((t) => h("span", {}, t)));
  el.title = `${title}: ${lines.join(", ")}. Open Privacy for details.`;
  el.setAttribute("aria-label", el.title);
}

function setProject(value) {
  currentProject = value;
  try { sessionStorage.setItem(PROJECT_KEY, value); } catch (_) { /* storage blocked */ }
  memState.offset = 0;
  memState.checked.clear();
  memState.selected = null;
  ui.shown = {};
  live.lastSig = {};
  userRender();
}

// ── Shared UI building blocks ───────────────────────────────────────────────
// Per-page UI state that survives re-renders (sub-tabs, "show more" counts).
const ui = { shown: {}, review: "replaced", privacy: "queue", activity: "saves", setupApp: 0 };

const sensitivityBadge = (m) => m.sensitivity === "redacted" ? h("span", { class: "tag tag-redacted", title: "Sensitive text was removed" }, "redacted")
  : m.sensitivity === "flagged" ? h("span", { class: "tag tag-flagged", title: "Needs a privacy review" }, "review") : null;
const projectTag = (p) => (p ? h("span", { class: "tag tag-project", title: p }, p) : null);

function page(title, description, actions, ...body) {
  // replaceChildren() would turn null into the text "null", so drop empty parts.
  return [
    h("header", { class: "page-head" },
      h("div", { class: "page-title" }, h("h1", {}, title), description ? h("p", {}, description) : null),
      actions && [].concat(actions).filter(Boolean).length ? h("div", { class: "page-actions" }, actions) : null),
    ...body.flat().filter((x) => x !== null && x !== undefined && x !== false),
  ];
}

function card(title, sub, ...body) {
  return h("section", { class: "card" },
    title ? h("div", { class: "card-head" }, h("div", {}, h("h2", {}, title), sub ? h("p", { class: "sub" }, sub) : null)) : null,
    ...body);
}

function cardWithAction(title, sub, action, ...body) {
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("div", {}, h("h2", {}, title), sub ? h("p", { class: "sub" }, sub) : null), action || null),
    ...body);
}

function segmented(label, options, active, onChange) {
  return h("div", { class: "segmented", role: "tablist", "aria-label": label },
    options.map(([value, text, count]) => h("button", {
      class: "seg", role: "tab", type: "button", "aria-selected": String(value === active),
      onclick: () => { if (value !== active) onChange(value); },
    }, text, count !== undefined && count !== null ? h("span", { class: "seg-count" }, fmtNum(count)) : null)));
}

// Long lists show `step` items, then "Show more" — the page never becomes a wall.
function moreList(key, items, renderItem, step = 20, emptyText = "Nothing here") {
  if (!items.length) return h("p", { class: "empty" }, emptyText);
  const shown = ui.shown[key] || step;
  const rest = items.length - shown;
  return h("div", {},
    h("ul", { class: "list" }, items.slice(0, shown).map(renderItem)),
    rest > 0 ? h("div", { class: "more-row" }, h("button", {
      class: "btn", type: "button", onclick: () => { ui.shown[key] = shown + step; userRender(); },
    }, `Show ${Math.min(step, rest)} more · ${nf.format(rest)} left`)) : null);
}

const tableWrap = (table) => h("div", { class: "table-wrap" }, table);
const chip = (state, text) => h("span", { class: "chip" }, h("span", { class: `dot ${state}` }), text);

// ── Overview ────────────────────────────────────────────────────────────────
async function renderOverview(opts = {}) {
  const d = await api(withProject("/api/overview"));
  if (!shouldDraw("overview", d, opts)) return false;
  const t = d.totals;

  const kpi = (label, value, note, hero) => h("div", { class: `kpi${hero ? " hero" : ""}` },
    h("div", { class: "kpi-label" }, label), h("div", { class: "kpi-value" }, value), note ? h("div", { class: "kpi-note" }, note) : null);
  const kpis = h("div", { class: "kpis" },
    kpi("Active memories", fmtNum(t.active), `${nf.format(t.total)} saved in total`, true),
    kpi("Times recalled", fmtNum(t.recalls), "Given to an AI"),
    kpi("Waiting for a summary", fmtNum(t.awaiting_compression), "Made in the background"),
    kpi("Replaced", fmtNum(t.outdated), "By newer facts"),
    kpi("Archived", fmtNum(t.archived), "Never recalled"));

  const status = h("div", { class: "chips", "aria-label": "Where your data goes" },
    chip("good", "Stored on this device"),
    chip(d.privacy.compression_on_device ? "good" : "warn", d.privacy.compression_on_device ? "Summaries made on device" : "Summaries by Anthropic API"),
    chip(d.privacy.cloud_backup ? "warn" : "", d.privacy.cloud_backup ? "Encrypted cloud backup on" : "Cloud backup off"));

  const workflow = card("Memory workflow", "How memories move through Caretaker",
    h("div", { class: "workflow" }, d.workflow.map((w, i) => h("div", { class: "stage" },
      h("div", { class: "stage-name" }, h("span", { class: "stage-step" }, String(i + 1)), w.stage),
      h("div", { class: "stage-count" }, fmtNum(w.count)),
      h("div", { class: "stage-detail" }, w.detail)))));

  const top = card("Most recalled", "The memories AIs use most",
    d.top_recalled.length
      ? h("ul", { class: "list" }, d.top_recalled.map((m) => h("li", {},
          h("div", { class: "meta" }, typeTag(m.type), `${nf.format(m.retrieval_count)} recalls · last ${fmtWhen(m.last_used)}`),
          h("div", { class: "clamp-2" }, m.text))))
      : h("p", { class: "empty" }, "Nothing recalled yet"));

  const st = d.storage, mt = d.maintenance;
  const storage = card("Storage & maintenance", null, tableWrap(h("table", { class: "kv" }, h("tbody", {},
    h("tr", {}, h("td", {}, "Data folder"), h("td", { class: "mono" }, st.data_dir)),
    h("tr", {}, h("td", {}, "Config file"), h("td", { class: "mono" }, st.config_file)),
    h("tr", {}, h("td", {}, "Database"), h("td", {}, fmtBytes(st.database_bytes))),
    h("tr", {}, h("td", {}, "Search index"), h("td", {}, fmtBytes(st.vector_index_bytes))),
    h("tr", {}, h("td", {}, "Last maintenance"), h("td", {}, mt.last_run ? fmtWhen(mt.last_run.replace(" UTC", "Z").replace(" ", "T")) : "Not run yet")),
    h("tr", {}, h("td", {}, "Next maintenance"), h("td", {}, mt.next_run ? fmtDateTime(mt.next_run) : "Scheduler off"))))));

  view().replaceChildren(...page("Overview", "What Caretaker remembers and how it's being used.", status,
    kpis,
    h("div", { class: "section-gap" }, workflow),
    h("div", { class: "grid grid-2 section-gap" },
      card("New memories", "Last 30 days", columnChart("New memories per day, last 30 days", d.growth)),
      card("By type", "Active memories", hbarChart("Active memories by type", d.by_type, "memories"))),
    h("div", { class: "grid grid-2 section-gap" },
      card("By temperature", "Hot memories are recalled first", hbarChart("Active memories by temperature", d.by_temperature, "memories")),
      card("By AI app", "Which app saved them", hbarChart("Active memories by app", d.by_agent.slice(0, 8), "memories", (k) => k))),
    h("div", { class: "grid grid-main-side section-gap" }, top, storage)));
  return true;
}

// ── Memories ────────────────────────────────────────────────────────────────
const memState = { q: "", type: "", temperature: "", status: "ACTIVE", agent: "", sort: "updated", offset: 0, limit: window.matchMedia("(max-width: 1099px)").matches ? 25 : 50, selected: null, checked: new Set() };
const MEMORY_TYPES = ["PROJECT", "PREFERENCE", "PROBLEM", "DECISION", "LEARNING", "PERSONAL", "EMOTION", "CORRECTION"];

function bulkBar() {
  const n = memState.checked.size;
  if (!n) return null;
  const ids = [...memState.checked];
  const notice = h("span", { class: "notice", role: "status" });
  const run = async (action, value) => {
    notice.textContent = "Working…";
    try {
      await api("/api/memories/bulk", { method: "POST", body: JSON.stringify({ ids, action, value }) });
      memState.checked.clear();
      live.lastSig = {};
      await userRender();
    } catch (e) { notice.textContent = e.message; }
  };
  const project = h("input", { class: "field", placeholder: "Project name", "aria-label": "Move to project" });
  const type = h("select", { class: "field", "aria-label": "Set type" }, MEMORY_TYPES.map((t) => h("option", { value: t }, titleCase(t))));
  return h("div", { class: "bulk-bar", role: "region", "aria-label": "Bulk actions" },
    h("strong", {}, `${nf.format(n)} selected`),
    h("button", { class: "btn small danger", type: "button", onclick: () => run("archive") }, "Archive"),
    h("button", { class: "btn small", type: "button", onclick: () => run("restore") }, "Restore"),
    h("span", { class: "bulk-group" }, project, h("button", { class: "btn small", type: "button", onclick: () => run("set_project", project.value) }, "Move")),
    h("span", { class: "bulk-group" }, type, h("button", { class: "btn small", type: "button", onclick: () => run("set_type", type.value) }, "Set type")),
    h("button", { class: "btn small ghost", type: "button", onclick: () => { memState.checked.clear(); userRender(); } }, "Clear"),
    notice);
}

function selectMemory(id) {
  memState.selected = id;
  userRender();
}

async function renderMemories(opts = {}) {
  const params = new URLSearchParams();
  for (const k of ["q", "type", "temperature", "status", "agent", "sort", "offset", "limit"]) {
    if (memState[k] !== "" && memState[k] !== null) params.set(k, memState[k]);
  }
  if (currentProject) params.set("project", currentProject);
  const data = await api(`/api/memories?${params}`);
  let detailMem = null;
  if (memState.selected) {
    try { detailMem = await api(`/api/memories/${encodeURIComponent(memState.selected)}`); }
    catch (e) { if (e.status === 401) throw e; memState.selected = null; }
  }
  if (!shouldDraw("memories", [data, detailMem, [...memState.checked]], opts)) return false;
  if (!opts.background) live.detailDirty = false;

  const filterSelect = (key, label, options, fmt = titleCase) => h("select", {
    class: "field", "aria-label": label,
    onchange: (e) => { memState[key] = e.target.value; memState.offset = 0; userRender(); },
  }, h("option", { value: "" }, `All ${label.toLowerCase()}`),
    options.map((o) => { const opt = h("option", { value: o, title: o }, fmt(o)); if (memState[key] === o) opt.selected = true; return opt; }));

  let searchTimer;
  const search = h("input", {
    class: "field search", type: "search", placeholder: "Search memories…", "aria-label": "Search memories", value: memState.q,
    oninput: (e) => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => { memState.q = e.target.value; memState.offset = 0; userRender(); }, 250);
    },
  });
  const sort = h("select", { class: "field", "aria-label": "Sort",
    onchange: (e) => { memState.sort = e.target.value; memState.offset = 0; userRender(); } },
    [["updated", "Recently updated"], ["created", "Newest"], ["importance", "Most important"], ["recalled", "Most recalled"], ["last_used", "Recently used"]]
      .map(([v, l]) => { const o = h("option", { value: v }, l); if (memState.sort === v) o.selected = true; return o; }));
  const f = data.facets;
  const filtersActive = ["q", "type", "temperature", "agent"].some((k) => memState[k]) || memState.status !== "ACTIVE";
  const filters = h("div", { class: "filters" }, search,
    h("div", { class: "filter-row" },
      filterSelect("type", "Types", f.type), filterSelect("temperature", "Temperatures", f.temperature),
      filterSelect("status", "Statuses", f.status), filterSelect("agent", "Apps", f.agent, (x) => x), sort,
      filtersActive ? h("button", { class: "btn ghost", type: "button", onclick: () => {
        Object.assign(memState, { q: "", type: "", temperature: "", status: "ACTIVE", agent: "", offset: 0 }); userRender();
      } }, "Clear filters") : null));

  const checkbox = (m) => {
    const box = h("input", { type: "checkbox", "aria-label": "Select memory",
      onclick: (e) => e.stopPropagation(),
      onchange: (e) => { if (e.target.checked) memState.checked.add(m.id); else memState.checked.delete(m.id); userRender(); } });
    box.checked = memState.checked.has(m.id);
    return box;
  };
  const rows = data.items.map((m) => h("tr", {
    class: memState.selected === m.id ? "selected" : null, tabindex: "0", "aria-selected": String(memState.selected === m.id),
    onclick: () => selectMemory(m.id),
    onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectMemory(m.id); } },
  },
    h("td", {}, checkbox(m)),
    h("td", {},
      h("div", { class: "mem-text clamp-2" }, m.short || m.full),
      h("div", { class: "row-meta" }, typeTag(m.type), !currentProject ? projectTag(m.project) : null, sensitivityBadge(m)),
      h("div", { class: "row-sub" }, `${titleCase(m.temperature)} · ${fmtWhen(m.updated_at)}`)),
    h("td", { class: "c-hide-sm" }, tempBadge(m.temperature)),
    h("td", { class: "num c-hide-sm" }, nf.format(m.retrieval_count || 0)),
    h("td", { class: "c-hide-sm" }, fmtWhen(m.updated_at)),
    h("td", {}, h("span", { class: `status ${String(m.status).toLowerCase()}` }, titleCase(m.status)))));

  const allBox = h("input", { type: "checkbox", "aria-label": "Select all on this page", onchange: (e) => {
    for (const m of data.items) { if (e.target.checked) memState.checked.add(m.id); else memState.checked.delete(m.id); }
    userRender(); } });
  allBox.checked = data.items.length > 0 && data.items.every((m) => memState.checked.has(m.id));
  const table = data.items.length
    ? h("table", { class: "mem-table" },
        h("colgroup", {}, h("col", { class: "c-check" }), h("col", {}), h("col", { class: "c-temp c-hide-sm" }),
          h("col", { class: "c-rec c-hide-sm" }), h("col", { class: "c-upd c-hide-sm" }), h("col", { class: "c-status" })),
        h("thead", {}, h("tr", {}, h("th", {}, allBox), h("th", {}, "Memory"), h("th", { class: "c-hide-sm" }, "Temperature"),
          h("th", { class: "num c-hide-sm" }, "Recalled"), h("th", { class: "c-hide-sm" }, "Updated"), h("th", {}, "Status"))),
        h("tbody", {}, rows))
    : h("p", { class: "empty" }, filtersActive ? "No memories match these filters" : "No memories yet");

  const from = data.total ? data.offset + 1 : 0, to = data.offset + data.items.length;
  const pager = h("div", { class: "pane-foot" },
    h("span", { class: "muted small" }, `${nf.format(from)}–${nf.format(to)} of ${nf.format(data.total)}`),
    h("div", { class: "btn-row" },
      h("button", { class: "btn small", type: "button", disabled: data.offset === 0, onclick: () => { memState.offset = Math.max(0, data.offset - data.limit); userRender(); } }, "Previous"),
      h("button", { class: "btn small", type: "button", disabled: to >= data.total, onclick: () => { memState.offset = data.offset + data.limit; userRender(); } }, "Next")));

  const listPane = h("section", { class: "card pane", "aria-label": "Memory list" },
    h("div", { class: "pane-head" }, filters, bulkBar()),
    h("div", { class: "pane-body" }, table),
    pager);
  const detail = detailMem ? renderDetail(detailMem) : null;
  const focusSearch = document.activeElement && document.activeElement.type === "search";
  view().replaceChildren(...page("Memories", "Search, edit and archive what Caretaker remembers.", null,
    h("div", { class: `mem-layout${detail ? " with-detail" : ""}` }, listPane, detail),
    detail ? h("div", { class: "sheet-backdrop", onclick: () => { memState.selected = null; userRender(); } }) : null));
  if (focusSearch) { search.focus(); search.setSelectionRange(search.value.length, search.value.length); }
  const selectedRow = view().querySelector(".mem-table tr.selected");
  if (selectedRow && !opts.background) selectedRow.scrollIntoView({ block: "nearest" });
  return true;
}

let lastSaveNote = "";

function renderDetail(m) {
  const markDirty = () => { live.detailDirty = true; updateStatus(); };
  const close = () => {
    if (live.detailDirty && !confirm("Discard your unsaved changes?")) return;
    live.detailDirty = false; memState.selected = null; userRender();
  };

  const notice = h("p", { class: "notice", role: "status" });
  const short = h("textarea", { id: "f-short", class: "field", rows: "3", oninput: markDirty }); short.value = m.short || "";
  const full = h("textarea", { id: "f-full", class: "field full-text", rows: "7", oninput: markDirty }); full.value = m.full || "";
  const importance = h("input", { id: "f-imp", class: "field", type: "number", min: "0", max: "1", step: "0.05", value: String(m.importance ?? 0.5), oninput: markDirty });
  const project = h("input", { id: "f-project", class: "field", placeholder: "Global", value: m.project || "", oninput: markDirty });
  const type = h("select", { id: "f-type", class: "field", onchange: markDirty },
    MEMORY_TYPES.map((t) => { const o = h("option", { value: t }, titleCase(t)); if (t === m.type) o.selected = true; return o; }));

  const act = async (fn, okMsg) => {
    notice.className = "notice"; notice.textContent = "Saving…";
    try {
      lastSaveNote = "";
      await fn(); live.detailDirty = false; await renderMemories({ gen: ++live.gen });
      const n = document.querySelector(".detail .notice");
      if (n) { n.className = "notice ok"; n.textContent = okMsg + (lastSaveNote ? ` Privacy: ${lastSaveNote}.` : ""); }
    }
    catch (e) { notice.className = "notice err"; notice.textContent = e.message; }
  };
  const save = () => act(() => api(`/api/memories/${encodeURIComponent(m.id)}`, {
    method: "PATCH", body: JSON.stringify({ short: short.value, full: full.value, importance: Number(importance.value),
                                           project: project.value, type: type.value }),
  }).then((r) => { lastSaveNote = r.privacy_note || ""; }), "Saved. Search catches up at the next maintenance run.");
  const archived = m.status === "ARCHIVED";
  const toggle = () => act(() => api(`/api/memories/${encodeURIComponent(m.id)}/${archived ? "restore" : "archive"}`, { method: "POST" }),
    archived ? "Restored. It can be recalled again." : "Archived. AIs will no longer recall it.");
  const findings = [...new Set(m.privacy_findings.map((f) => `${f.label}: ${f.action}`))];

  return h("aside", { class: "card pane detail", "aria-label": "Memory details", onkeydown: (e) => { if (e.key === "Escape") close(); } },
    h("div", { class: "pane-head" },
      h("h2", {}, typeTag(m.type), h("span", {}, "Memory")),
      h("button", { class: "icon-btn", type: "button", "aria-label": "Close details", title: "Close", onclick: close },
        s("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }, s("path", { d: "M6 6l12 12M18 6L6 18" })))),
    h("div", { class: "pane-body" },
      h("label", { class: "lbl", for: "f-short" }, "Short summary (used for search)"), short,
      h("label", { class: "lbl", for: "f-full" }, "Full text"), full,
      h("div", { class: "field-row" },
        h("div", {}, h("label", { class: "lbl", for: "f-imp" }, "Importance"), importance),
        h("div", {}, h("label", { class: "lbl", for: "f-type" }, "Type"), type),
        h("div", {}, h("label", { class: "lbl", for: "f-project" }, "Project"), project)),
      h("dl", {},
        h("dt", {}, "Status"), h("dd", {}, titleCase(m.status)),
        h("dt", {}, "Temperature"), h("dd", {}, tempBadge(m.temperature)),
        h("dt", {}, "Recalled"), h("dd", {}, `${nf.format(m.retrieval_count || 0)} times · last ${fmtWhen(m.last_used)}`),
        h("dt", {}, "Saved by"), h("dd", {}, m.source_agent || "—"),
        h("dt", {}, "Created"), h("dd", {}, fmtDateTime(m.created_at)),
        h("dt", {}, "Updated"), h("dd", {}, fmtDateTime(m.updated_at)),
        m.keywords.length ? [h("dt", {}, "Keywords"), h("dd", { class: "keywords" }, m.keywords.map((k) => h("span", { class: "tag", title: k }, k)))] : null,
        h("dt", {}, "Privacy"), h("dd", {}, findings.length
          ? h("span", { class: "keywords" }, sensitivityBadge(m), findings.map((t) => h("span", { class: "tag", title: t }, t)))
          : "Nothing sensitive found"),
        h("dt", {}, "ID"), h("dd", { class: "mono" }, m.id))),
    h("div", { class: "pane-foot" },
      h("div", { class: "btn-row" },
        h("button", { class: "btn primary", type: "button", onclick: save }, "Save changes"),
        h("button", { class: `btn${archived ? "" : " danger"}`, type: "button", onclick: toggle }, archived ? "Restore" : "Archive")),
      notice));
}

// ── Preview ─────────────────────────────────────────────────────────────────
const previewState = { message: "", agent: "claude", result: null, error: "" };

async function renderPreview(opts = {}) {
  if (opts.background) return false;          // user-driven page: never auto-refresh
  const message = h("textarea", { class: "field", rows: "4", placeholder: "Type a message as you would to your AI…",
    "aria-label": "Message to preview" });
  message.value = previewState.message;
  const agent = h("select", { class: "field", "aria-label": "AI app" },
    [["claude", "Claude"], ["chatgpt", "ChatGPT"], ["gemini", "Gemini"], ["cursor", "Cursor"], ["copilot", "Copilot"], ["other", "Other"]]
      .map(([v, l]) => { const o = h("option", { value: v }, l); if (v === previewState.agent) o.selected = true; return o; }));
  const go = async () => {
    previewState.message = message.value; previewState.agent = agent.value; previewState.error = "";
    try {
      previewState.result = await api(`/api/preview?message=${encodeURIComponent(message.value)}&agent=${encodeURIComponent(agent.value)}&project=${encodeURIComponent(scopedProject())}`);
    } catch (e) { previewState.result = null; previewState.error = e.message; }
    message.blur();
    userRender();
  };
  message.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) go(); });

  const form = card("Message", scopedProject() ? `Project: ${scopedProject()}` : "All projects",
    h("div", { class: "preview-form" }, message),
    h("div", { class: "btn-row section-gap" }, agent, h("button", { class: "btn primary", type: "button", onclick: go }, "Preview"),
      h("span", { class: "muted small" }, "Ctrl+Enter")),
    previewState.error ? h("p", { class: "notice err section-gap" }, previewState.error) : null);

  const r = previewState.result;
  const why = r ? card("Why these memories", `${r.memories.length} picked`,
    moreList("preview-why", r.memories, (m) => h("li", {},
      h("div", { class: "meta" }, typeTag(m.type), projectTag(m.project), m.reason),
      h("div", { class: "clamp-2" }, m.text)), 8, "No memories would be included")) : null;
  const output = r
    ? card("What the AI receives", `About ${nf.format(r.approx_tokens)} tokens · up to ${nf.format(r.budget)} tokens of memories for a message like this`,
        r.note ? h("p", { class: "callout" }, r.note) : null,
        h("pre", { class: "code-block section-gap" }, r.text))
    : card("What the AI receives", null, h("p", { class: "empty" }, "Type a message and press Preview."));

  view().replaceChildren(...page("Preview", "See exactly what Caretaker would give an AI for a message. Nothing is saved or counted.", null,
    h("div", { class: "grid grid-main-side" }, output, h("div", { class: "stack" }, form, why))));
  return true;
}

// ── Review ──────────────────────────────────────────────────────────────────
async function renderReview(opts = {}) {
  const d = await api(withProject("/api/review"));
  if (!shouldDraw("review", [d, ui.review, ui.shown], opts)) return false;
  const act = async (path, body) => {
    await api(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });
    live.lastSig = {};
    userRender();
  };
  const tabs = segmented("Review lists", [
    ["replaced", "Replaced", d.replaced.length],
    ["stale", "Not used lately", d.stale.length],
  ], ui.review, (v) => { ui.review = v; userRender(); });

  let body;
  if (ui.review === "replaced") {
    body = cardWithAction("Replaced memories", "Older facts Caretaker replaced with newer ones, or merged duplicates. Undo if it got it wrong.", null,
      moreList("review-replaced", d.replaced, (m) => h("li", {},
        h("div", { class: "meta" }, typeTag(m.type), projectTag(m.project), `replaced ${fmtWhen(m.updated_at)}`),
        h("div", { class: "old-text clamp-2" }, m.short || m.full),
        m.replaced_by
          ? h("div", { class: "replaced-by" }, h("div", { class: "muted small" }, "Replaced by"), h("div", { class: "clamp-2" }, m.replaced_by.text))
          : h("div", { class: "muted small" }, "Merged as a duplicate"),
        h("div", { class: "item-actions" },
          h("button", { class: "btn small", type: "button", onclick: () => act(`/api/memories/${encodeURIComponent(m.id)}/restore`) }, "Undo — make active"),
          h("button", { class: "btn small danger", type: "button", onclick: () => act(`/api/memories/${encodeURIComponent(m.id)}/archive`) }, "Archive"))),
      10, "Nothing has been replaced"));
  } else {
    const ids = d.stale.map((m) => m.id);
    body = cardWithAction("Not used lately", `Active memories no AI has used in ${d.stale_days} days`,
      ids.length ? h("button", { class: "btn small danger", type: "button", onclick: () => {
        if (confirm(`Archive all ${ids.length} memories not used in ${d.stale_days} days? You can restore them later.`)) act("/api/memories/bulk", { ids, action: "archive" });
      } }, `Archive all ${nf.format(ids.length)}`) : null,
      moreList("review-stale", d.stale, (m) => h("li", {},
        h("div", { class: "meta" }, typeTag(m.type), projectTag(m.project), `last used ${fmtWhen(m.last_used || m.created_at)}`),
        h("div", { class: "clamp-2" }, m.short || m.full),
        h("div", { class: "item-actions" },
          h("button", { class: "btn small danger", type: "button", onclick: () => act(`/api/memories/${encodeURIComponent(m.id)}/archive`) }, "Archive"))),
      10, `Every memory was used in the last ${d.stale_days} days`));
  }
  view().replaceChildren(...page("Review", "Check Caretaker's automatic decisions and tidy up memories nobody uses.", tabs, body));
  return true;
}

// ── Privacy ─────────────────────────────────────────────────────────────────
const privacyState = { scan: null, notice: "" };
const ACTION_LABELS = { redact: "Remove (redact)", mask: "Keep last 4 (mask)", flag: "Keep, mark for review", allow: "Ignore" };
const CATEGORY_LABELS = {
  secret: ["Secrets", "API keys, tokens, passwords, private keys"],
  financial: ["Financial & ID numbers", "Cards, IBAN, SSN, Aadhaar, PAN"],
  contact: ["Contact details", "Email addresses, phone numbers"],
  personal: ["Personal topics", "Health, beliefs, sexuality, money"],
  custom: ["Your patterns", "The custom patterns below"],
};

async function renderPrivacy(opts = {}) {
  const d = await api("/api/privacy");
  if (!shouldDraw("privacy", [d, privacyState, ui.privacy, ui.shown], opts)) return false;
  if (!opts.background) live.detailDirty = false;
  const markDirty = () => { live.detailDirty = true; updateStatus(); };
  const after = async (fn, msg) => {
    try { await fn(); privacyState.notice = msg; } catch (e) { privacyState.notice = e.message; }
    live.detailDirty = false; live.lastSig = {}; userRender();
  };

  const L = d.leaves_device;
  const info = (state, title, text) => h("div", { class: "info" }, h("span", { class: `dot ${state}` }),
    h("div", {}, h("strong", {}, title), h("span", { class: "small" }, text)));
  const strip = h("div", { class: "info-strip" },
    info("good", "Stored on this device", d.data_dir),
    info(L.compression_on_device ? "good" : "warn", L.compression_on_device ? "Summaries made on device" : "Summaries by Anthropic's API",
      L.compression_on_device ? "No memory text is sent anywhere to be summarised." : `Memory text is sent to ${L.compression}.`),
    info(L.cloud_backup ? "warn" : "good", L.cloud_backup ? "Encrypted cloud backup on" : "Cloud backup off",
      L.cloud_backup ? "Encrypted on this device before upload to your Supabase." : "Memories exist only on this computer."));

  const tabs = segmented("Privacy sections", [
    ["queue", "Review queue", d.counts.flagged],
    ["scan", "Scan"],
    ["settings", "Settings"],
    ["redacted", "Redacted", d.counts.redacted],
  ], ui.privacy, (v) => {
    if (live.detailDirty && !confirm("Discard your unsaved settings?")) return;
    ui.privacy = v; live.detailDirty = false; privacyState.notice = ""; userRender();
  });
  const noticeEl = privacyState.notice ? h("p", { class: "notice ok", role: "status" }, privacyState.notice) : null;

  let body;
  if (ui.privacy === "queue") {
    body = card("Review queue", "Kept but marked for review — usually contact details or personal topics.", noticeEl,
      moreList("privacy-queue", d.queue, (m) => h("li", {},
        h("div", { class: "meta" }, typeTag(m.type), projectTag(m.project),
          [...new Set(m.privacy_findings.map((f) => f.label))].map((l) => h("span", { class: "tag tag-flagged" }, l))),
        h("div", { class: "clamp-3" }, m.full),
        h("div", { class: "item-actions" },
          h("button", { class: "btn small primary", type: "button", onclick: () => after(() => api(`/api/memories/${encodeURIComponent(m.id)}/privacy`, { method: "POST", body: JSON.stringify({ action: "redact" }) }), "Removed the sensitive part.") }, "Remove sensitive part"),
          h("button", { class: "btn small", type: "button", onclick: () => after(() => api(`/api/memories/${encodeURIComponent(m.id)}/privacy`, { method: "POST", body: JSON.stringify({ action: "not_sensitive" }) }), "Marked as fine.") }, "It's fine"),
          h("button", { class: "btn small danger", type: "button", onclick: () => after(() => api(`/api/memories/${encodeURIComponent(m.id)}/archive`, { method: "POST" }), "Archived.") }, "Archive"))),
      10, "Nothing waiting for review"));
  } else if (ui.privacy === "scan") {
    const scan = privacyState.scan;
    body = card("Scan your memories", "Checks every stored memory against your settings. Nothing changes until you apply.",
      h("div", { class: "btn-row" },
        h("button", { class: "btn", type: "button", onclick: () => after(async () => { privacyState.scan = await api("/api/privacy/scan", { method: "POST", body: "{}" }); }, "") }, scan ? "Scan again" : "Scan now"),
        scan && !scan.applied && scan.will_change ? h("button", { class: "btn primary", type: "button", onclick: () => after(async () => {
          privacyState.scan = await api("/api/privacy/scan", { method: "POST", body: JSON.stringify({ apply: true }) });
        }, "Applied. Older backups and exports may still contain the original text — delete them if you don't need them.") },
          `Apply to ${nf.format(scan.will_change)} memories`) : null),
      scan ? h("div", { class: "section-gap" },
        h("p", { class: "small" }, `${nf.format(scan.scanned)} scanned · ${nf.format(scan.with_findings)} with findings` + (scan.applied ? ` · ${nf.format(scan.changed)} changed` : "")),
        scan.by_kind.length ? tableWrap(h("table", {},
          h("thead", {}, h("tr", {}, h("th", {}, "Found"), h("th", { class: "num" }, "Memories"), h("th", {}, "What happens"))),
          h("tbody", {}, scan.by_kind.map((k) => h("tr", {}, h("td", {}, k.label), h("td", { class: "num" }, nf.format(k.count)), h("td", {}, ACTION_LABELS[k.action] || k.action)))))) : null) : null,
      noticeEl ? h("div", { class: "section-gap" }, noticeEl) : null);
  } else if (ui.privacy === "settings") {
    const selects = {};
    const rows = Object.entries(CATEGORY_LABELS).map(([cat, [name, hint]]) => {
      selects[cat] = h("select", { class: "field", "aria-label": name, onchange: markDirty },
        Object.entries(ACTION_LABELS).map(([v, l]) => { const o = h("option", { value: v }, l); if (d.policy[cat] === v) o.selected = true; return o; }));
      return h("div", { class: "setting-row" }, h("div", {}, h("strong", {}, name), h("div", { class: "muted small" }, hint)), selects[cat]);
    });
    const patternBox = h("div", {});
    const addPattern = (p = { pattern: "", label: "" }) => {
      const row = h("div", { class: "pattern-row" },
        h("input", { class: "field", placeholder: "Pattern, e.g. ACME-\\d+", value: p.pattern, "aria-label": "Pattern", oninput: markDirty }),
        h("input", { class: "field", placeholder: "Label", value: p.label, "aria-label": "Label", oninput: markDirty }),
        h("button", { class: "btn small", type: "button", onclick: () => { row.remove(); markDirty(); } }, "Remove"));
      patternBox.appendChild(row);
    };
    d.custom_patterns.forEach(addPattern);
    const allow = h("textarea", { class: "field", rows: "4", "aria-label": "Allowlist", placeholder: "One exact text per line, e.g. team@example.com", oninput: markDirty });
    allow.value = d.allowlist.join("\n");
    const save = () => after(() => api("/api/privacy/policy", { method: "PUT", body: JSON.stringify({
      ...Object.fromEntries(Object.entries(selects).map(([k, el]) => [k, el.value])),
      custom_patterns: [...patternBox.querySelectorAll(".pattern-row")].map((r) => {
        const [pat, lab] = r.querySelectorAll("input"); return { pattern: pat.value, label: lab.value };
      }).filter((x) => x.pattern),
      allowlist: allow.value.split("\n").map((x) => x.trim()).filter(Boolean),
    }) }), "Settings saved. They apply to new memories now; run a scan to apply them to existing ones.");
    body = h("div", { class: "grid grid-main-side" },
      card("When sensitive data is found", "Detection runs only on this device and is best-effort.", h("div", {}, rows)),
      h("div", { class: "stack" },
        card("Your patterns", "Extra things to treat as sensitive (regular expressions).", patternBox,
          h("button", { class: "btn small", type: "button", onclick: () => { addPattern(); markDirty(); } }, "Add pattern")),
        card("Never flag these", "Exact text that is fine, one per line.", allow),
        card(null, null, h("div", { class: "btn-row" }, h("button", { class: "btn primary", type: "button", onclick: save }, "Save settings"), noticeEl))));
  } else {
    body = card("Recently redacted", "Already cleaned — shown for reference.",
      moreList("privacy-redacted", d.redacted, (m) => h("li", {},
        h("div", { class: "meta" }, typeTag(m.type), projectTag(m.project), fmtWhen(m.updated_at)),
        h("div", { class: "clamp-2" }, m.short || m.full)), 10, "Nothing redacted yet"));
  }

  view().replaceChildren(...page("Privacy", "What leaves your device, and how Caretaker handles secrets and personal data.", null,
    strip, h("div", { class: "section-gap" }, tabs), h("div", { class: "section-gap" }, body)));
  return true;
}

// ── Activity ────────────────────────────────────────────────────────────────
async function renderActivity(opts = {}) {
  const d = await api(withProject("/api/activity"));
  if (!shouldDraw("activity", [d, ui.activity, ui.shown], opts)) return false;
  const tabs = segmented("Activity lists", [
    ["saves", "Recently saved", d.recent_saves.length],
    ["recalls", "Recently recalled", d.recently_recalled.length],
    ["maintenance", "Maintenance", d.maintenance_runs.length],
  ], ui.activity, (v) => { ui.activity = v; userRender(); });

  let body;
  if (ui.activity === "saves") {
    body = card("Recently saved", "Newest memories first", moreList("activity-saves", d.recent_saves, (m) => h("li", {},
      h("div", { class: "meta" }, typeTag(m.type), `${fmtWhen(m.created_at)} · via ${m.source_agent || "app"}`),
      h("div", { class: "clamp-2" }, m.text)), 10, "Nothing saved yet"));
  } else if (ui.activity === "recalls") {
    body = card("Recently recalled", "Memories recently given to an AI", moreList("activity-recalls", d.recently_recalled, (m) => h("li", {},
      h("div", { class: "meta" }, typeTag(m.type), `${fmtWhen(m.last_used)} · ${nf.format(m.retrieval_count || 0)} recalls`),
      h("div", { class: "clamp-2" }, m.text)), 10, "Nothing recalled yet"));
  } else {
    body = card("Maintenance runs", "Nightly clean-up: cooling down, archiving, merging duplicates, re-indexing",
      d.maintenance_runs.length ? tableWrap(h("table", {}, h("thead", {}, h("tr", {},
          h("th", {}, "Run"), h("th", { class: "num" }, "Cooled"), h("th", { class: "num" }, "Archived"),
          h("th", { class: "num" }, "Merged"), h("th", { class: "num" }, "Re-indexed"), h("th", { class: "num" }, "Took"))),
        h("tbody", {}, d.maintenance_runs.map((r) => {
          const v = r.values, pick = (...keys) => { for (const k of Object.keys(v)) if (keys.some((x) => k.toLowerCase().startsWith(x))) return v[k]; return "—"; };
          return h("tr", {}, h("td", {}, r.time), h("td", { class: "num" }, pick("decay")), h("td", { class: "num" }, pick("stale", "archiv")),
            h("td", { class: "num" }, pick("dedup", "merge", "duplicate")), h("td", { class: "num" }, pick("reindex")), h("td", { class: "num" }, pick("elapsed")));
        }))))
        : h("p", { class: "empty" }, "Maintenance hasn't run yet. It runs nightly, or with `caretaker maintenance`."));
  }
  view().replaceChildren(...page("Activity", "What Caretaker saved, what it gave to AIs, and nightly clean-up.", tabs, body));
  return true;
}

// ── Setup ───────────────────────────────────────────────────────────────────
async function renderSetup(opts = {}) {
  const d = await api("/api/setup");
  if (!shouldDraw("setup", [d, ui.setupApp], opts)) return false;
  const icon = { ok: ["good", "OK"], warn: ["warn", "Suggestion"], error: ["bad", "Problem"] };
  const problems = d.checks.filter((c) => c.status !== "ok").length;
  const checks = card("Health check", problems ? `${problems} suggestion${problems === 1 ? "" : "s"}` : "Everything looks good",
    h("ul", { class: "check-list" }, d.checks.map((c) => h("li", {},
      h("span", { class: `dot ${icon[c.status][0]}`, title: icon[c.status][1] }),
      h("div", {}, h("strong", {}, c.title), h("span", { class: "visually-hidden" }, ` (${icon[c.status][1]})`),
        h("div", { class: "muted small" }, c.detail), c.fix ? h("div", { class: "fix" }, c.fix) : null)))));

  const apps = card("AI apps on this computer", "Apps whose MCP config Caretaker found", d.clients.length
    ? h("ul", { class: "list" }, d.clients.map((c) => h("li", {},
        h("div", { class: "meta" }, h("strong", { style: "color: var(--text-primary)" }, c.app),
          c.configured === null ? h("span", { class: "status-pill" }, h("span", { class: "dot bad" }), c.note)
            : c.configured ? h("span", { class: "status-pill" }, h("span", { class: "dot good" }), "Caretaker added")
            : h("span", { class: "status-pill" }, h("span", { class: "dot warn" }), "Not added")),
        h("div", { class: "mono muted" }, c.path))))
    : h("p", { class: "empty" }, "No supported AI app config found"));

  const sn = d.snippets[Math.min(ui.setupApp, d.snippets.length - 1)];
  const text = sn.command || sn.json;
  const copyBtn = h("button", { class: "btn small", type: "button", onclick: (e) => {
    const btn = e.currentTarget;
    navigator.clipboard.writeText(text).then(() => { btn.textContent = "Copied"; setTimeout(() => { btn.textContent = "Copy"; }, 1500); },
      () => { btn.textContent = "Copy failed"; });
  } }, "Copy");
  const snippetTabs = segmented("AI app", d.snippets.map((x, i) => [i, x.app]), ui.setupApp, (v) => { ui.setupApp = v; userRender(); });
  const snippets = cardWithAction("Add Caretaker to an AI app", "Paste into the app's MCP config, then fully restart the app.", copyBtn,
    snippetTabs,
    h("p", { class: "muted small section-gap" }, `Where: ${sn.where}`),
    h("pre", { class: "code-block section-gap" }, text));

  view().replaceChildren(...page("Setup", "Check your installation and connect Caretaker to your AI apps.", null,
    h("div", { class: "grid grid-main-side" }, checks, h("div", { class: "stack" }, snippets, apps))));
  return true;
}

// ── Router ──────────────────────────────────────────────────────────────────
const ROUTES = {
  overview: renderOverview, memories: renderMemories, preview: renderPreview, review: renderReview,
  privacy: renderPrivacy, activity: renderActivity, setup: renderSetup,
};

function showError(e) {
  hideTip();
  if (e.status === 401) {
    view().replaceChildren(...page("Open the dashboard from Caretaker", "For your privacy this page only works through the link printed by the command below.", null,
      card(null, null, h("pre", { class: "code-block" }, "caretaker dashboard"))));
    return;
  }
  view().replaceChildren(...page("Something went wrong", e.message, null));
}

let currentRender = renderOverview;

function userRender() {
  return currentRender({ gen: ++live.gen }).then(markOk, (e) => markFailed(e, false));
}

let returningFromCancelledNav = false;

function route() {
  const name = (location.hash.match(/^#\/(\w+)/) || [])[1] || "overview";
  if (returningFromCancelledNav) { returningFromCancelledNav = false; return; }
  if (live.detailDirty && !confirm("Discard your unsaved changes?")) {
    returningFromCancelledNav = true;   // the history.back() below fires hashchange once more
    history.back();
    return;
  }
  currentRender = ROUTES[name] || renderOverview;
  document.querySelectorAll(".nav a").forEach((a) => {
    if (a.dataset.route === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  const active = document.querySelector(".nav a[aria-current]");
  if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
  hideTip();
  window.scrollTo(0, 0);
  ui.shown = {};
  live.detailDirty = false;
  userRender();
}

function markOk() { live.lastOk = Date.now(); live.failed = false; updateStatus(); refreshProjectSwitcher(); refreshNavCounts(); }
function markFailed(e, background) {
  if (e.status === 401) { live.stopped = true; showError(e); updateStatus(); return; }
  live.failed = true;
  updateStatus();
  if (!background) showError(e);
}

// Typing or unsaved edits pause background refresh; so does hovering a chart (keeps the tooltip).
function isEditing() {
  const a = document.activeElement;
  return live.detailDirty || !!(a && view().contains(a) && /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName));
}
const isHoveringChart = () => !!document.querySelector(".hbars:hover, .col-chart:hover");

async function backgroundRefresh() {
  if (live.busy || live.stopped) return;
  live.busy = true;
  live.lastAttempt = Date.now();
  const gen = live.gen;
  const y = window.scrollY;
  const open = [...view().querySelectorAll("details")].map((d) => d.open);
  const paneScroll = [...view().querySelectorAll(".pane-body")].map((p) => p.scrollTop);
  try {
    if (await currentRender({ background: true, gen })) {
      view().querySelectorAll("details").forEach((d, i) => { if (open[i]) d.open = true; });
      view().querySelectorAll(".pane-body").forEach((p, i) => { if (paneScroll[i]) p.scrollTop = paneScroll[i]; });
      window.scrollTo(0, y);
    }
    markOk();
  } catch (e) {
    markFailed(e, true);
  } finally {
    live.busy = false;
  }
}

function ago(t) {
  const sec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (sec < 5) return "just now";
  if (sec < 60) return `${sec}s ago`;
  return `${Math.round(sec / 60)} min ago`;
}

function updateStatus() {
  const el = document.getElementById("live-status");
  if (!el) return;
  let state = "live", label = live.lastOk ? `Live · updated ${ago(live.lastOk)}` : "Connecting…";
  if (live.stopped) { state = "off"; label = "Disconnected"; }
  else if (live.failed) { state = "error"; label = "Can't reach Caretaker · retrying"; }
  else if (isEditing()) { state = "paused"; label = "Paused while you edit"; }
  el.dataset.state = state;
  el.querySelector(".live-text").textContent = label;
  el.title = label;
}

function manualRefresh() {
  if (live.detailDirty && !confirm("Discard your unsaved changes and refresh?")) return;
  const btn = document.getElementById("refresh-btn");
  btn.classList.add("spinning");
  live.lastSig = {};
  live.detailDirty = false;
  if (document.activeElement && view().contains(document.activeElement)) document.activeElement.blur();
  backgroundRefresh().finally(() => setTimeout(() => btn.classList.remove("spinning"), 300));
}

setInterval(() => {
  updateStatus();
  if (document.hidden || live.stopped || isEditing() || isHoveringChart()) return;
  if (Date.now() - live.lastAttempt >= REFRESH_MS) backgroundRefresh();
}, 1000);
document.addEventListener("visibilitychange", () => { if (!document.hidden && !isEditing()) backgroundRefresh(); });
window.addEventListener("hashchange", route);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && memState.selected && currentRender === renderMemories) {
    const closeBtn = document.querySelector(".detail .pane-head .icon-btn");
    if (closeBtn) closeBtn.click();
  }
});
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refresh-btn").addEventListener("click", manualRefresh);
  document.getElementById("project-switcher").addEventListener("change", (e) => setProject(e.target.value));
  refreshProjectSwitcher();
  refreshNavCounts();
  route();
});
