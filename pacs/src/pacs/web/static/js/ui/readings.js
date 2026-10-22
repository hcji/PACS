/** 表型读数卡片：当前大脑**最近一次**被观察得到的分析者读数，以雷达图为主呈现。
 *
 * 这份读数锚定的是**大脑**，不是你此刻打开的那段历史对话 —— 因此它放在表型观察里，
 * 与「对话过程」并列而不重复：换了大脑、或刚跑完一轮观察，这里的读数都会跟着变，
 * 而不管你在历史列表里点开了哪一条。
 *
 * **卡片里没有任何 θ。** 读数只由提问与观察得到（提问者提问 → 大脑作答 → 分析者读对话），
 * 扰动参数只作为模拟大脑的提示词输入、不进入这条链路。因此这张卡不显示、也不比对
 * 扰动与读数的关系 —— 那种「读数是否与扰动相符」的判断属于离线分析，不属于观察界面。
 *
 * 卡片头部只放三个动作（重新观察 / 查看对话过程 / 查看这段对话）与一行状态，
 * 两个 LLM 的过程本身没有任何常显内容 —— 要看得点按钮进抽屉。
 *
 * 数据来自 `GET /api/brain-report`：只用这一个大脑自己的数据，不做任何群体统计。
 */

import { $, esc, html } from "../core/dom.js";
import { renderKpi, renderTable } from "./widgets.js";
import { dimColor, pct } from "../core/format.js";
import { dimLabel, dimsAll, isPhenotypeDim } from "../core/store.js";

export function template() {
  return `
  <div class="card" id="rd-card">
    <div class="card-head"><h2>当前大脑的表型读数</h2>
      <div class="actions">
        <span class="note" id="rd-meta"></span>
        <label class="note" for="rd-max-turns"
               style="display:inline-flex;align-items:center;gap:6px">
          轮数上限
          <input id="rd-max-turns" type="number" min="35" max="500" step="1"
                 style="width:72px" title="基础 35 轮走完后，若某方面不足 5 个问题就继续补问，到这里为止">
        </label>
        <span class="note" id="rd-cost"></span>
        <button class="btn ghost sm" id="rd-open" hidden>查看这段对话</button>
        <button class="btn ghost sm" id="rd-toggle-probe" aria-expanded="false">查看对话过程</button>
        <button class="btn sm" id="rd-reobserve">重新观察</button>
      </div></div>
    <div id="rd-body" hidden>
      <p class="note" id="rd-hint" style="margin:0 0 var(--sp-3)">
        得分方向：100 = 与同龄典型发展无异，0 = 严重偏离。读数 = 提问者提问 → 模拟大脑作答 →
        分析者判读对话，这条链路上没有任何扰动参数。
      </p>
      <div id="rd-stale"></div>
      <div class="grid g3" id="rd-kpi"></div>
      <div id="rd-notes"></div>
      <div id="rd-conclusion"></div>
      <div class="grid g2" style="margin-top:var(--sp-3)">
        <div id="rd-radar"></div>
        <div class="scroll sm"><table id="rd-dims"></table></div>
      </div>
      <details style="margin-top:var(--sp-3)">
        <summary>逐维量化记录 · 分析者叙述与证据引文</summary>
        <div id="rd-dim-details"></div>
        <h3>总体叙述</h3><p class="note" id="rd-narrative"></p>
        <div id="rd-evidence"></div>
      </details>
    </div>
    <p class="note" id="rd-empty" hidden>
      当前大脑还没有被观察过，因此没有读数 —— 点右上角「重新观察」立即读一次
      （进入本界面时也会自动读一次）。
    </p>
  </div>`;
}

let currentSessionId = "";
let onOpenSession = null;

/** 动作回调由视图注入：卡片只负责摆按钮，不关心观察是怎么跑的。 */
export function bind({ onOpenSession: open, onReobserve, onToggleProbe,
                       onMaxTurnsChange } = {}) {
  onOpenSession = open || null;
  $("rd-max-turns").addEventListener("change", () => {
    // 基础轮数（每通道 ≥5 个问题）是协议的硬要求，因此上限不能低于它 —— 输入框允许
    // 输小值但这里会钳回去，并如实告知，而不是静默按一个更小的数跑。
    const raw = parseInt($("rd-max-turns").value, 10);
    const value = Number.isFinite(raw) ? Math.max(TURN_FLOOR, Math.min(500, raw)) : "";
    if (value !== raw) $("rd-max-turns").value = value === "" ? "" : value;
    onMaxTurnsChange?.(value);
  });
  $("rd-open").addEventListener("click", () => {
    if (currentSessionId && onOpenSession) onOpenSession(currentSessionId);
  });
  $("rd-reobserve").addEventListener("click", (event) => onReobserve?.(event.currentTarget));
  $("rd-toggle-probe").addEventListener("click", () => onToggleProbe?.());
}

/** 一行状态：观察进行到哪一步、或失败了。下一次 `render()` 会把它换回读数时间。 */
export function setMeta(text) {
  const node = $("rd-meta");
  if (node) node.textContent = text || "";
}

/** 抽屉开关的按钮文案与展开态（按钮在本卡片头部，状态由调用方掌握）。 */
export function setDrawerState(open) {
  const button = $("rd-toggle-probe");
  button.setAttribute("aria-expanded", String(open));
  button.textContent = open ? "收起对话过程" : "查看对话过程";
}

//: 轮数上限的下界 = 基础轮数（协议要求每通道 ≥5 个问题）。
const TURN_FLOOR = 35;

/** 把轮数协议（服务端 `dialogue` 口径）显示出来：上限输入框 + 成本两段口径。 */
export function setTurnBudget({ nTurns, maxTurns, textLimit } = {}) {
  const node = $("rd-max-turns");
  if (node && maxTurns) node.value = maxTurns;
  const base = nTurns || TURN_FLOOR;
  const cap = maxTurns || base;
  const limit = textLimit ? ` · 每轮 ≤${textLimit} 字` : "";
  html("rd-cost", `基础 ${base} 轮 ≈ ${base * 2 + 1} 次调用；`
    + `补问至上限 ${cap} 轮 ≈ ${cap * 2 + 1} 次调用${limit}`);
}

export function reset() {
  currentSessionId = "";
  $("rd-body").hidden = true;
  $("rd-empty").hidden = false;
  $("rd-open").hidden = true;
  setMeta("");
  html("rd-stale", "");
  html("rd-notes", "");
  html("rd-kpi", "");
  html("rd-conclusion", "");
  html("rd-radar", "");
  html("rd-evidence", "");
  html("rd-dim-details", "");
}

/** 读数单元格的取值：只接受真数值。`null`（未探查）与空串一律是「没有读数」——
 *  不能顺手当 0：0 分在这个分制里是「严重偏离」，两种含义必须分开。 */
function valueOf(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  const num = Number(raw);
  return Number.isFinite(num) ? Math.max(0, Math.min(1, num)) : null;
}

/** 雷达图：一具心智的读数一眼看完（数值另有一张表逐维列出）；轴数随表型维度注册表变化。
 *
 * 轴端数值用协议的分制（0–100），内部仍按 0..1 存（见 `dims`）。
 *
 * 数值标在**轴端**而不是顶点上：各维读数全部偏低时（例如一具全面受损的心智），
 * 各顶点会挤在圆心附近，标在顶点上的数字会互相压字。轴端标签彼此分散，任何读数分布下都可读。
 *
 * **缺读数的轴照画、照标注，只是不参与多边形**：分析者偶尔只给出部分维度，此时整块拒画
 * 等于把已有读数也藏起来。缺的那一维按「没有读数」处理（轴虚线、标 `—（无读数）`），
 * 而不是当 0 分 —— 那会把「没问过/没读出」画成「严重偏离」。
 */
function drawRadar(dims) {
  const order = dimsAll().length ? dimsAll() : Object.keys(dims || {});
  const n = order.length || 1;
  const W = 400;
  const H = 330;
  const cx = 200;
  const cy = 152;
  const R = 104;
  const angleOf = (i) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const at = (i, value) => [
    cx + R * value * Math.cos(angleOf(i)),
    cy + R * value * Math.sin(angleOf(i)),
  ];

  const values = order.map((dim) => valueOf(dims?.[dim]));
  const missing = values.filter((v) => v === null).length;

  let svg = `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" role="img" `
    + `aria-label="当前大脑的 ${n} 维表型读数雷达图">`;
  // 参考环：0.25 / 0.5 / 0.75 / 1.0（最外圈即「与同龄典型无异」）
  for (let ring = 1; ring <= 4; ring += 1) {
    const r = ring / 4;
    const points = order.map((_, i) => at(i, r).map((v) => v.toFixed(1)).join(",")).join(" ");
    svg += `<polygon points="${points}" fill="none" stroke="#2a3441"/>`;
  }
  // 读数多边形：只连有读数的顶点（缺读数的那一维不画顶点，也不当 0 分）
  const scored = order.map((_, i) => i).filter((i) => values[i] !== null);
  if (scored.length >= 3) {
    const points = scored.map((i) => at(i, values[i]).map((v) => v.toFixed(1)).join(",")).join(" ");
    svg += `<polygon points="${points}" fill="var(--accent2)" fill-opacity="0.15"`
      + ` stroke="var(--accent2)" stroke-width="2"/>`;
  }
  // 轴 + 轴端标签（维度名 + 读数，两行，永不互相压字）
  order.forEach((dim, i) => {
    const [ex, ey] = at(i, 1);
    const [lx, ly] = at(i, 1.24);
    const value = values[i];
    const has = value !== null;
    const color = has ? dimColor(value) : "#4a5866";
    const [px, py] = at(i, has ? value : 0);
    svg += `<line x1="${cx}" y1="${cy}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" `
      + `stroke="${has ? "#2a3441" : "#33404d"}"${has ? "" : ' stroke-dasharray="3 3"'}/>`;
    if (has) svg += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="3" fill="${color}"/>`;
    svg += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" `
      + `dominant-baseline="middle">`
      + `<tspan x="${lx.toFixed(1)}" fill="#8b98a5" font-size="10.5">${esc(dimLabel(dim))}</tspan>`
      + `<tspan x="${lx.toFixed(1)}" dy="13" fill="${color}" font-size="11" `
      + `font-family="monospace">${has ? Math.round(value * 100) : "—"}</tspan></text>`;
  });
  // 外圈刻度说明
  svg += `<text x="${cx}" y="${H - 10}" fill="#5f6b78" font-size="9.5" `
    + `text-anchor="middle">外圈 = 100（与同龄典型无异）`
    + `${missing ? `　虚线轴 = 没有读数（${missing} 项），不是 0 分` : ""}</text>`;
  svg += `</svg>`;
  return svg;
}

/**
 * 逐维量化记录（协议阶段 10 的五项）：行为得分 / 置信度 / 偏离基线 / 支持判断的观察 /
 * 任务级测量。分析者没给记录（例如旧会话或只有扁平读数）时整块不渲染 —— 不编造字段。
 */
function renderDimDetails(details) {
  const entries = Object.entries(details).filter(([, d]) => d && typeof d === "object");
  if (!entries.length) {
    html("rd-dim-details", "");
    return;
  }
  html("rd-dim-details", `<h3>逐维量化记录</h3>` + entries.map(([dim, d]) => {
    const bits = [];
    if (d.score !== undefined) bits.push(`得分 <b class="mono">${Math.round(d.score)}</b> / 100`);
    if (d.confidence !== undefined) bits.push(`置信度 <span class="mono">${pct(d.confidence)}</span>`);
    if (d.deviation !== undefined) {
      bits.push(`偏离基线 <span class="mono">${d.deviation > 0 ? "+" : ""}${d.deviation}</span>`);
    }
    const observations = (d.observations || [])
      .map((o) => `<li>${esc(o)}</li>`).join("");
    const measurements = (d.task_measurements || [])
      .map((m) => `<li><span class="note">${esc(m.task || "任务")}</span>：${esc(m.measurement || "")}</li>`)
      .join("");
    return `<div class="kv" style="margin-top:var(--sp-2)">
        <div class="nm" style="font-weight:600">${esc(dimLabel(dim))}
          <span class="mono note">${esc(dim)}</span></div>
        <div class="note">${bits.join(" · ")}</div>
        ${observations ? `<div class="note">行为观察：</div><ul class="lines">${observations}</ul>` : ""}
        ${measurements ? `<div class="note">任务级测量：</div><ul class="lines">${measurements}</ul>` : ""}
      </div>`;
  }).join(""));
}

/** 表型印象徽标：`present` 为 null 时如实说「未给出」，不替分析者下结论。 */
const LIKENESS_LEVEL_ZH = { none: "无", mild: "轻度", moderate: "中度", marked: "明显" };
//: 徽章配色沿用既有语义类（见 components.css）：无 = 通过，轻度/中度 = 薄弱，明显 = 未达
const LIKENESS_LEVEL_CLASS = {
  none: "b-pass", mild: "b-weak", moderate: "b-weak", marked: "b-fail",
};

/**
 * 探查结论卡片：分析者对这段问答的病例报告式总结（约 350 字）+ 结构化表型印象。
 *
 * **这不是诊断**，卡片里也如实写着这一点：结论只描述对话里出现过的表现与程度，
 * 被试是虚拟心智。旧会话（或分析者没给出结论）时整块不渲染 —— 不编造结论。
 *
 * 字数与告警都来自分析者输出的解析结果（`conclusion_chars` / `conclusion_notes`）：
 * 越出约定字数、或出现诊断式用词，都照实显示，而不是替分析者修饰。
 */
export function renderConclusion(readings) {
  return renderConclusionInto("rd-conclusion", readings);
}

/** 同上，但渲染到指定容器（访谈视图复用它 —— 两条链路的结论必须长得一样）。 */
export function renderConclusionInto(containerId, readings) {
  const text = String(readings?.conclusion || "").trim();
  if (!text) {
    html(containerId, "");
    return false;
  }
  const likeness = readings.asd_likeness || {};
  const level = String(likeness.level || "unknown");
  const levelZh = LIKENESS_LEVEL_ZH[level] || level;
  let badge = '<span class="badge b-gap">表型印象：未给出</span>';
  if (likeness.present !== null && likeness.present !== undefined) {
    const verdict = likeness.present ? "表现出孤独症样表型特征" : "未表现出孤独症样表型特征";
    badge = `<span class="badge ${LIKENESS_LEVEL_CLASS[level] || "b-gap"}">`
      + `${esc(verdict)} · 程度 ${esc(levelZh)}</span>`;
  }
  const basis = (likeness.basis || []).filter(Boolean);
  const notes = (readings.conclusion_notes || []).filter(Boolean);
  const chars = readings.conclusion_chars || text.length;
  const target = readings.conclusion_target_chars || 350;

  html(containerId, `<div class="card" style="margin-top:var(--sp-3)">
    <div class="card-head"><h3 style="margin:0">探查结论</h3>
      <div class="actions">${badge}
        <span class="note">${chars} 字 / 约 ${target} 字</span></div></div>
    <p style="margin:0">${esc(text)}</p>
    ${basis.length ? `<p class="note" style="margin:var(--sp-2) 0 0">判定依据：${basis.map(esc).join("；")}</p>` : ""}
    ${notes.length ? `<div class="warnbox" style="margin-top:var(--sp-2)">
      <ul class="lines" style="margin:0">
        ${notes.map((n) => `<li>${esc(n)}</li>`).join("")}
      </ul></div>` : ""}
    <p class="note" style="margin:var(--sp-2) 0 0">
      行为学表型印象，不是临床诊断；对象为虚拟被试（仿真产物）。
    </p>
  </div>`);
  return true;
}

/** 用 `/api/brain-report` 的载荷填充。`report.readings` 为空时显示可操作的空态。 */
export function render(report) {
  const readings = report?.readings;
  if (!readings) {
    reset();
    return false;
  }
  $("rd-empty").hidden = true;
  $("rd-body").hidden = false;
  currentSessionId = readings.session_id || "";

  // 空闲态的那一行：读数时间 + 它来自哪次对话（一次观察的「出处」）
  setMeta([readings.ts ? `读数时间 ${readings.ts}` : "", readings.session_id]
    .filter(Boolean).join(" · "));
  $("rd-open").hidden = !readings.session_id;

  // 这份读数不是最近一次跑出来的那一份（最近那次分析者没给出读数）→ 明说，
  // 免得被当成「刚跑出来的结果」。只有一行。
  const stale = readings.newest_attempt;
  html("rd-stale", stale
    ? `<p class="note" style="margin:0 0 var(--sp-3)">最近一次观察（会话
        ${esc(stale.session_id || "—")}）没有产出读数：${esc(stale.error || "分析者没有给出逐维读数")}
        —— 以下是此前最近一次可用读数。</p>`
    : "");

  const score = (value) => (value === null ? "—" : `${Math.round(value * 100)}`);
  const mean = valueOf(readings.mean_reading);
  renderKpi("rd-kpi", [
    ["平均得分", mean === null ? "—" : `${score(mean)} / 100`],
    ["分析者置信度", readings.confidence === null || readings.confidence === undefined
      ? "—" : pct(readings.confidence)],
    ["观察轮数", readings.n_turns_run ?? readings.n_brain_turns ?? "—"],
    ["分析者", readings.model_id || "—"],
  ]);

  const dims = readings.dims || {};
  const entries = Object.entries(dims)
    .map(([dim, raw]) => [dim, valueOf(raw)])
    .filter(([, value]) => value !== null);
  // 一个维度都没有 ≠ 这份读数用了别的维度：**分析者根本没产出内容**（调用失败/空回答）。
  // 这两种情形必须分开说，否则界面会指着一份空读数说「这是另一套维度口径，请重新观察」，
  // 把「模型没吐出东西」演成「口径不匹配」，用户无从下手。
  const noReading = !entries.length;
  const foreignDims = !noReading && !entries.some(([dim]) => isPhenotypeDim(dim));
  const incomplete = entries.length < dimsAll().length;   // 有表型维度没读到
  // 分析者自己的说明（以及解析问题）只在**读数不完整或有告警**时显示一行：
  // 完整读数下它只是免责性质的套话，不必常显。
  const notes = String(readings.notes || "").trim();
  html("rd-notes", notes && (incomplete || readings.error)
    ? `<div class="warnbox" style="margin-bottom:var(--sp-3)">${esc(notes)}</div>`
    : "");
  if (noReading) {
    html("rd-radar", `<div class="errbox"><b>这次观察没有产出读数</b>`
      + (readings.error ? `<br>分析者输出问题：${esc(readings.error)}` : "")
      + `<br><span class="note">会话已存下（${esc(readings.session_id || "—")}）。
        常见原因：输出额度太小（<span class="mono">max_tokens</span> 被截断成半截 JSON）、
        分析者是推理型模型（额度被思维链吃掉）。调大
        <span class="mono">max_tokens</span> 或换非推理模型后点「重新观察」。</span></div>`);
  } else if (foreignDims) {
    // 确实是另一套维度口径（例如上传量表的逐题读数）：不画雷达图，得分仍列在下表
    html("rd-radar", `<p class="note">这份读数用的是<b>另一套维度口径</b>`
      + `（${entries.map(([dim]) => esc(dimLabel(dim) || dim)).join("、")}），`
      + `与当前表型维度对不上，故不画雷达图 —— 得分仍列在右侧表里。</p>`);
  } else {
    html("rd-radar", drawRadar(dims));
  }
  renderTable("rd-dims", ["维度", "得分（0–100）", "分布"],
    scoreRows(dims, entries, foreignDims),
    { emptyText: "这条反馈没有给出可用的维度读数" });

  renderConclusion(readings);
  renderDimDetails(readings.dim_details || {});

  $("rd-narrative").textContent = readings.narrative || "（无）";

  const evidence = readings.evidence || [];
  html("rd-evidence", evidence.length
    ? `<h3>证据引文</h3><ul class="lines">${evidence.map((e) => `<li><span class="mono note">${esc(dimLabel(e.dim) || e.dim)}</span>：<q>${esc(e.quote)}</q><br><span class="note">${esc(e.reading)}</span></li>`).join("")}</ul>`
    : "");
  return true;
}

/** 得分表：**表型维度按注册表顺序全列**（缺读数的行照样在，得分显示 `—`），
 *  其余键（对照通道、上传量表的逐题读数）跟在其后。表与雷达图对得上，是同一批通道。 */
function scoreRows(dims, entries, foreignDims) {
  const extras = entries.filter(([dim]) => !isPhenotypeDim(dim));
  const rows = foreignDims
    ? extras
    : [...dimsAll().map((dim) => [dim, valueOf(dims[dim])]), ...extras];
  return rows.map(([dim, value]) => {
    const has = value !== null;
    const color = has ? dimColor(value) : "var(--muted)";
    // 对照通道与表型维度分列说明：它会被评分、会落库，但不是雷达图上的一条轴
    const tag = isPhenotypeDim(dim) ? esc(dim) : `${esc(dim)} · 对照通道`;
    const width = has ? value * 100 : 0;
    return `<tr><td>${esc(dimLabel(dim))} <span class="mono note">${tag}</span></td>
      <td class="mono" style="color:${color}">${has ? Math.round(value * 100) : "—"}</td>
      <td><div class="bar" style="margin:0"><i style="width:${width}%;background:${color}"></i></div></td></tr>`;
  });
}
