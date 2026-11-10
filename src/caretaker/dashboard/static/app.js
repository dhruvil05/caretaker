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

function card(title, sub, ...body) {
  return h("section", { class: "card" }, h("h2", {}, title), sub ? h("p", { class: "sub" }, sub) : null, ...body);
}

async function renderOverview(opts = {}) {
  const d = await api("/api/overview");
  if (!shouldDraw("overview", d, opts)) return false;
  const t = d.totals;
  const privacy = h("div", { class: "privacy" },
    h("span", { class: "pill" }, h("span", { class: "dot good" }), "Stored on this device"),
    h("span", { class: "pill" }, h("span", { class: `dot ${d.privacy.compression_on_device ? "good" : "warn"}` }),
      d.privacy.compression_on_device ? "Summaries made on this device" : "Summaries made by Anthropic API"),
    h("span", { class: "pill" }, h("span", { class: `dot ${d.privacy.cloud_backup ? "good" : ""}` }),
      d.privacy.cloud_backup ? "Encrypted cloud backup on" : "Cloud backup off"));

  const hero = card("Active memories", null,
    h("div", { class: "hero-value" }, fmtNum(t.active)),
    h("div", { class: "hero-label" }, `${nf.format(t.total)} saved in total`),
    privacy);

  const tile = (label, value, note) => h("div", { class: "tile" },
    h("div", { class: "tile-label" }, label), h("div", { class: "tile-value" }, value), note ? h("div", { class: "tile-note" }, note) : null);
  const tiles = card("At a glance", null, h("div", { class: "tiles" },
    tile("Times recalled", fmtNum(t.recalls), "Memories given to an agent"),
    tile("Waiting for a summary", fmtNum(t.awaiting_compression), "Compressed in the background"),
    tile("Archived", fmtNum(t.archived), "Never recalled; can be restored"),
    tile("Replaced", fmtNum(t.outdated), "Superseded by newer facts")));

  const workflow = card("Memory workflow", "How memories move through Caretaker",
    h("div", { class: "workflow" }, d.workflow.map((w, i) => h("div", { class: "stage" },
      h("div", { class: "stage-name" }, h("span", { class: "stage-step" }, String(i + 1)), w.stage),
      h("div", { class: "stage-count" }, fmtNum(w.count)),
      h("div", { class: "stage-detail" }, w.detail)))));

  const top = card("Most recalled", "Memories agents use most",
    d.top_recalled.length
      ? h("ul", { class: "feed" }, d.top_recalled.map((m) => h("li", {},
          h("div", { class: "meta" }, typeTag(m.type), `${nf.format(m.retrieval_count)} recalls · last ${fmtWhen(m.last_used)}`),
          h("div", { class: "clamp" }, m.text))))
      : h("p", { class: "empty" }, "Nothing recalled yet"));

  const st = d.storage, mt = d.maintenance;
  const storage = card("Storage & maintenance", null,
    h("table", {}, h("tbody", {},
      h("tr", {}, h("td", { class: "muted" }, "Data folder"), h("td", { class: "mono" }, st.data_dir)),
      h("tr", {}, h("td", { class: "muted" }, "Config file"), h("td", { class: "mono" }, st.config_file)),
      h("tr", {}, h("td", { class: "muted" }, "Database"), h("td", {}, fmtBytes(st.database_bytes))),
      h("tr", {}, h("td", { class: "muted" }, "Search index"), h("td", {}, fmtBytes(st.vector_index_bytes))),
      h("tr", {}, h("td", { class: "muted" }, "Last maintenance"), h("td", {}, mt.last_run ? fmtWhen(mt.last_run.replace(" UTC", "Z").replace(" ", "T")) : "Not run yet")),
      h("tr", {}, h("td", { class: "muted" }, "Next maintenance"), h("td", {}, mt.next_run ? fmtDateTime(mt.next_run) : "Scheduler off")))));

  view().replaceChildren(
    h("div", { class: "grid grid-hero" }, hero, tiles),
    h("div", { class: "section" }, workflow),
    h("div", { class: "grid grid-2 section" },
      card("New memories", "Last 30 days", columnChart("New memories per day, last 30 days", d.growth)),
      card("Active memories by type", null, hbarChart("Active memories by type", d.by_type, "memories"))),
    h("div", { class: "grid grid-2 section" },
      card("By temperature", "Hot memories are recalled first", hbarChart("Active memories by temperature", d.by_temperature, "memories")),
      card("By agent", "Which agent saved them", hbarChart("Active memories by agent", d.by_agent, "memories", (k) => k))),
    h("div", { class: "grid grid-2 section" }, top, storage));
  return true;
}

// Memories view state survives re-renders of the detail panel.
const memState = { q: "", type: "", temperature: "", status: "ACTIVE", agent: "", sort: "updated", offset: 0, limit: 50, selected: null };

async function renderMemories(opts = {}) {
  const params = new URLSearchParams();
  for (const k of ["q", "type", "temperature", "status", "agent", "sort", "offset", "limit"]) {
    if (memState[k] !== "" && memState[k] !== null) params.set(k, memState[k]);
  }
  const data = await api(`/api/memories?${params}`);
  let detailMem = null;
  if (memState.selected) {
    try { detailMem = await api(`/api/memories/${encodeURIComponent(memState.selected)}`); }
    catch (e) { if (e.status === 401) throw e; memState.selected = null; }
  }
  if (!shouldDraw("memories", [data, detailMem], opts)) return false;
  if (!opts.background) live.detailDirty = false;

  const select = (key, label, options, fmt = titleCase) => h("select", {
    class: "field", "aria-label": label,
    onchange: (e) => { memState[key] = e.target.value; memState.offset = 0; userRender(); },
  }, h("option", { value: "" }, `All ${label.toLowerCase()}`),
    options.map((o) => { const opt = h("option", { value: o }, fmt(o)); if (memState[key] === o) opt.selected = true; return opt; }));

  let searchTimer;
  const search = h("input", {
    class: "field", type: "search", placeholder: "Search memories…", "aria-label": "Search memories", value: memState.q,
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
  const filters = h("div", { class: "filters" }, search,
    select("type", "Types", f.type), select("temperature", "Temperatures", f.temperature),
    select("status", "Statuses", f.status), select("agent", "Agents", f.agent, (x) => x), sort);

  const rows = data.items.map((m) => h("tr", {
    class: `clickable${memState.selected === m.id ? " selected" : ""}`, tabindex: "0",
    onclick: () => { memState.selected = m.id; userRender(); },
    onkeydown: (e) => { if (e.key === "Enter") { memState.selected = m.id; userRender(); } },
  },
    h("td", {}, typeTag(m.type)),
    h("td", { class: "text-cell" }, h("div", { class: "clamp" }, m.short || m.full)),
    h("td", { class: "hide-sm" }, tempBadge(m.temperature)),
    h("td", { class: "num hide-sm hide-detail" }, (m.importance ?? 0).toFixed(2)),
    h("td", { class: "num hide-sm hide-detail" }, nf.format(m.retrieval_count || 0)),
    h("td", { class: "hide-sm hide-detail" }, fmtWhen(m.updated_at)),
    h("td", {}, h("span", { class: `status ${String(m.status).toLowerCase()}` }, titleCase(m.status)))));

  const table = data.items.length
    ? h("table", {}, h("thead", {}, h("tr", {},
        h("th", {}, "Type"), h("th", {}, "Memory"), h("th", { class: "hide-sm" }, "Temperature"),
        h("th", { class: "num hide-sm hide-detail" }, "Importance"), h("th", { class: "num hide-sm hide-detail" }, "Recalled"),
        h("th", { class: "hide-sm hide-detail" }, "Updated"), h("th", {}, "Status"))), h("tbody", {}, rows))
    : h("p", { class: "empty" }, "No memories match these filters");

  const from = data.total ? data.offset + 1 : 0, to = data.offset + data.items.length;
  const pager = h("div", { class: "pager" }, `${nf.format(from)}–${nf.format(to)} of ${nf.format(data.total)}`,
    h("div", { class: "btn-row", style: "margin:0" },
      h("button", { class: "btn", disabled: data.offset === 0, onclick: () => { memState.offset = Math.max(0, data.offset - data.limit); userRender(); } }, "Previous"),
      h("button", { class: "btn", disabled: to >= data.total, onclick: () => { memState.offset = data.offset + data.limit; userRender(); } }, "Next")));

  const listCard = h("section", { class: "card" }, filters, table, pager);
  const detail = detailMem ? renderDetail(detailMem) : null;
  const focusSearch = document.activeElement && document.activeElement.type === "search";
  view().replaceChildren(h("div", { class: `layout-split${detail ? "" : " no-detail"}` }, listCard, detail));
  if (focusSearch) { search.focus(); search.setSelectionRange(search.value.length, search.value.length); }
  return true;
}

function renderDetail(m) {
  const markDirty = () => { live.detailDirty = true; updateStatus(); };

  const notice = h("p", { class: "notice", role: "status" });
  const short = h("textarea", { id: "f-short", "aria-label": "Short summary", oninput: markDirty }); short.value = m.short || "";
  const full = h("textarea", { id: "f-full", class: "full", "aria-label": "Full text", oninput: markDirty }); full.value = m.full || "";
  const importance = h("input", { id: "f-imp", class: "field", type: "number", min: "0", max: "1", step: "0.05", value: String(m.importance ?? 0.5), oninput: markDirty });

  const act = async (fn, okMsg) => {
    notice.className = "notice"; notice.textContent = "Saving…";
    try { await fn(); await renderMemories({ gen: ++live.gen }); const n = document.querySelector(".detail .notice"); if (n) { n.className = "notice ok"; n.textContent = okMsg; } }
    catch (e) { notice.className = "notice err"; notice.textContent = e.message; }
  };
  const save = () => act(() => api(`/api/memories/${encodeURIComponent(m.id)}`, {
    method: "PATCH", body: JSON.stringify({ short: short.value, full: full.value, importance: Number(importance.value) }),
  }), "Saved. Search results catch up at the next maintenance run.");
  const archived = m.status === "ARCHIVED";
  const toggle = () => act(() => api(`/api/memories/${encodeURIComponent(m.id)}/${archived ? "restore" : "archive"}`, { method: "POST" }),
    archived ? "Restored. It can be recalled again." : "Archived. Agents will no longer recall it.");

  return h("aside", { class: "card detail", "aria-label": "Memory details" },
    h("h2", {}, typeTag(m.type), titleCase(m.subtype || m.type)),
    h("dl", {},
      h("dt", {}, "Status"), h("dd", {}, titleCase(m.status)),
      h("dt", {}, "Temperature"), h("dd", {}, tempBadge(m.temperature)),
      h("dt", {}, "Recalled"), h("dd", {}, `${nf.format(m.retrieval_count || 0)} times · last ${fmtWhen(m.last_used)}`),
      h("dt", {}, "Saved by"), h("dd", {}, m.source_agent || "—"),
      h("dt", {}, "Created"), h("dd", {}, fmtDateTime(m.created_at)),
      h("dt", {}, "Updated"), h("dd", {}, fmtDateTime(m.updated_at)),
      m.keywords.length ? [h("dt", {}, "Keywords"), h("dd", { class: "keywords" }, m.keywords.map((k) => h("span", { class: "tag" }, k)))] : null,
      h("dt", {}, "ID"), h("dd", { class: "mono" }, m.id)),
    h("label", { for: "f-short" }, "Short summary (used for search)"), short,
    h("label", { for: "f-full" }, "Full text"), full,
    h("label", { for: "f-imp" }, "Importance (0–1)"), importance,
    h("div", { class: "btn-row" },
      h("button", { class: "btn primary", onclick: save }, "Save changes"),
      h("button", { class: `btn${archived ? "" : " danger"}`, onclick: toggle }, archived ? "Restore" : "Archive"),
      h("button", { class: "btn", onclick: () => { memState.selected = null; userRender(); } }, "Close")),
    notice);
}

async function renderActivity(opts = {}) {
  const d = await api("/api/activity");
  if (!shouldDraw("activity", d, opts)) return false;
  const feed = (items, metaFn) => items.length
    ? h("ul", { class: "feed" }, items.map((m) => h("li", {}, h("div", { class: "meta" }, metaFn(m)), h("div", { class: "clamp" }, m.text))))
    : h("p", { class: "empty" }, "Nothing yet");

  const runs = d.maintenance_runs.length
    ? h("table", {}, h("thead", {}, h("tr", {},
        h("th", {}, "Run"), h("th", { class: "num" }, "Decayed"), h("th", { class: "num" }, "Archived"),
        h("th", { class: "num" }, "Merged"), h("th", { class: "num" }, "Reindexed"), h("th", { class: "num hide-sm" }, "Took"))),
      h("tbody", {}, d.maintenance_runs.map((r) => {
        const v = r.values, pick = (...keys) => { for (const k of Object.keys(v)) if (keys.some((x) => k.toLowerCase().startsWith(x))) return v[k]; return "—"; };
        return h("tr", {}, h("td", {}, r.time), h("td", { class: "num" }, pick("decay")), h("td", { class: "num" }, pick("stale", "archiv")),
          h("td", { class: "num" }, pick("dedup", "merge")), h("td", { class: "num" }, pick("reindex")), h("td", { class: "num hide-sm" }, pick("elapsed")));
      })))
    : h("p", { class: "empty" }, "Maintenance hasn't run yet. It runs nightly, or with `caretaker maintenance`.");

  view().replaceChildren(
    h("div", { class: "grid grid-2" },
      card("Recently saved", "Newest memories first", feed(d.recent_saves, (m) => [typeTag(m.type), `${fmtWhen(m.created_at)} · via ${m.source_agent || "agent"}`])),
      card("Recently recalled", "Memories recently given to an agent", feed(d.recently_recalled, (m) => [typeTag(m.type), `${fmtWhen(m.last_used)} · ${nf.format(m.retrieval_count || 0)} recalls`]))),
    h("div", { class: "section" }, card("Maintenance runs", "Nightly clean-up: decay, archiving, merging duplicates, reindexing", runs)));
  return true;
}

// ── Router ──────────────────────────────────────────────────────────────────
const ROUTES = { overview: renderOverview, memories: renderMemories, activity: renderActivity };

function showError(e) {
  hideTip();
  if (e.status === 401) {
    view().replaceChildren(h("section", { class: "card" },
      h("h2", {}, "Open the dashboard from Caretaker"),
      h("p", { class: "muted" }, "For your privacy this page only works through the link printed by:"),
      h("pre", { class: "mono" }, "caretaker dashboard")));
    return;
  }
  view().replaceChildren(h("section", { class: "card" }, h("h2", {}, "Something went wrong"), h("p", { class: "muted" }, e.message)));
}

let currentRender = renderOverview;

function userRender() {
  return currentRender({ gen: ++live.gen }).then(markOk, (e) => markFailed(e, false));
}

function route() {
  const name = (location.hash.match(/^#\/(\w+)/) || [])[1] || "overview";
  currentRender = ROUTES[name] || renderOverview;
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.route === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  hideTip();
  window.scrollTo(0, 0);
  live.detailDirty = false;
  userRender();
}

function markOk() { live.lastOk = Date.now(); live.failed = false; updateStatus(); }
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
  try {
    if (await currentRender({ background: true, gen })) {
      view().querySelectorAll("details").forEach((d, i) => { if (open[i]) d.open = true; });
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
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refresh-btn").addEventListener("click", manualRefresh);
  route();
});
