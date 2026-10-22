/** 内置条目库面板（③ 表型观察的一个区块）。
 *
 * 数据来自 `/api/examiner/bank`（由 AsdKB 总结生成，441 条）。
 * 交互要点：
 * - 筛选条件写进用户首选项，切换界面/刷新后保留；
 * - 支持分页（接口本身就是 limit/offset），默认 50 条一页，避免一次渲染 441 行；
 * - 「仅儿童面向探查条目」是一等筛选：只有这些条目能用于对话提问；
 * - 每条可展开明细（施测方式、计分规则、关联症状与标准、映射依据）。
 */

import { $, esc, html, delegate } from "../core/dom.js";
import { api, query } from "../core/api.js";
import { store, update, dimLabel, dimsAll, primitives, primNames } from "../core/store.js";
import { prefs } from "../core/prefs.js";
import { status, empty } from "../ui/widgets.js";

const PAGE_SIZE = 50;

let offset = 0;

export function template() {
  return `
  <div class="card">
    <div class="card-head"><h2>内置条目库 · 来源 AsdKB</h2>
      <div class="actions">
        <button class="btn ghost sm" id="btn-bank-rules">映射规则与来源</button>
      </div></div>
    <p class="hint">
      来源 AsdKB 知识库精选；量表均为知情者报告 / 临床评定 / 成人自评，<b>无儿童直接施测条目</b>，
      故可探查条目分两种形态：直接提问、或按条目设定的情境探查（先描述场景再问「这种情况你会怎么处理」）；病史与医学事实类条目不可探查，只作观察要点。
    </p>
    <p class="hint">
      这些条目是提问者的<b>参考材料</b>：默认那一轮观察里，提问者按观察目标<b>自行设计</b>问题与情境，
      只在合适时才借用这里的条目；勾选条目后点「以所选条目开始对话观察」，则是你指定题目、提问者照着问。
    </p>
    <div class="grid g3" style="margin-top:var(--sp-4)">
      <div><label for="bank-filter-probe-mode">探查方式</label><select id="bank-filter-probe-mode"></select></div>
      <div><label for="bank-filter-dim">表型维度</label><select id="bank-filter-dim"></select></div>
      <div><label for="bank-filter-prim">可扰动基础过程</label><select id="bank-filter-prim"></select></div>
      <div><label for="bank-filter-admin">施测方式</label><select id="bank-filter-admin"></select></div>
      <div><label for="bank-filter-scale">量表</label><select id="bank-filter-scale"></select></div>
      <div><label for="bank-filter-age">年龄（月龄，留空=不限）</label><input id="bank-filter-age" type="number" min="0" placeholder="36"></div>
      <div><label for="bank-filter-q">关键词</label><input id="bank-filter-q" placeholder="如：假装 / 目光 / 积木"></div>
    </div>
    <div class="row" style="margin-top:var(--sp-3)">
      <label class="inline"><input type="checkbox" id="bank-filter-probe"> 仅儿童面向探查语句</label>
      <button class="btn ghost sm" id="btn-bank-search">应用筛选</button>
      <button class="btn ghost sm" id="btn-bank-reset">重置</button>
      <span class="note" id="bank-count"></span>
      <span class="status" id="bank-status"></span>
    </div>
    <div class="row" style="margin-top:var(--sp-3)">
      <span class="note">已选 <b id="bank-selected-n">0</b> 条</span>
      <button class="btn ghost sm" id="btn-bank-clear">清空勾选</button>
      <button class="btn sm" id="btn-bank-dialogue">以所选条目开始对话观察</button>
    </div>

    <div class="scroll" style="margin-top:var(--sp-3);max-height:520px"><table id="bank-table"></table></div>
    <div class="row" style="margin-top:var(--sp-3)">
      <button class="btn ghost sm" id="btn-bank-prev">上一页</button>
      <button class="btn ghost sm" id="btn-bank-next">下一页</button>
      <span class="note" id="bank-page"></span>
    </div>
    <div id="bank-rules" hidden style="margin-top:var(--sp-4)"></div>
  </div>`;
}

export function bind({ onStartDialogue, onUseAsQuestion }) {
  const filters = prefs.bankFilters();
  const select = (id, options, value) => {
    html(id, options.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join(""));
    $(id).value = value || "";
  };
  select("bank-filter-dim", [["", "全部"], ...dimsAll().map((d) => [d, dimLabel(d)])], filters.dim);
  select("bank-filter-prim", [["", "全部"], ...primitives().map((p) => [p.id, `${p.name_zh} · ${p.id}`])], filters.primitive);
  select("bank-filter-admin", [["", "全部"], ...Object.entries(store.bank.adminZh || {})], filters.admin);
  select("bank-filter-probe-mode", [
    ["", "全部"], ["question", "直接提问"], ["scenario", "情境式探查"], ["none", "不可探查（仅观察要点）"],
  ], filters.probeMode);
  select("bank-filter-scale", [["", "全部"]], filters.sourceId);
  $("bank-filter-age").value = filters.age || "";
  $("bank-filter-q").value = filters.q || "";
  $("bank-filter-probe").checked = Boolean(filters.probeOnly);

  $("btn-bank-search").addEventListener("click", () => { offset = 0; load(); });
  $("bank-filter-q").addEventListener("keydown", (event) => {
    if (event.key === "Enter") { offset = 0; load(); }
  });
  $("btn-bank-reset").addEventListener("click", () => {
    prefs.setBankFilters({
      dim: "", primitive: "", admin: "", probeMode: "", sourceId: "", age: "", q: "", probeOnly: false,
    });
    ["bank-filter-dim", "bank-filter-prim", "bank-filter-admin", "bank-filter-probe-mode",
   "bank-filter-scale", "bank-filter-age", "bank-filter-q"]
      .forEach((id) => { $(id).value = ""; });
    $("bank-filter-probe").checked = false;
    offset = 0;
    load();
  });
  $("btn-bank-clear").addEventListener("click", () => {
    store.bankSelected.clear();
    render();
  });
  $("btn-bank-prev").addEventListener("click", () => {
    offset = Math.max(0, offset - PAGE_SIZE);
    load();
  });
  $("btn-bank-next").addEventListener("click", () => {
    if (offset + PAGE_SIZE < store.bank.total) { offset += PAGE_SIZE; load(); }
  });
  $("btn-bank-dialogue").addEventListener("click", () => onStartDialogue(true));
  $("btn-bank-rules").addEventListener("click", () => {
    const box = $("bank-rules");
    if (box.hidden) renderRules();
    else box.hidden = true;
  });

  delegate($("bank-table"), "input[data-bank]", "change", (event, input) => {
    if (input.checked) store.bankSelected.add(input.dataset.bank);
    else store.bankSelected.delete(input.dataset.bank);
    $("bank-selected-n").textContent = store.bankSelected.size;
  });
  delegate($("bank-table"), "button[data-use]", "click", (event, button) => {
    const item = store.bank.items.find((x) => x.key === button.dataset.use);
    if (item) onUseAsQuestion?.(item.probe_zh || item.text_zh);
  });
  delegate($("bank-table"), "button[data-detail]", "click", (event, button) => {
    const row = $(`bank-detail-${button.dataset.detail}`);
    if (row) row.hidden = !row.hidden;
  });
}

export async function refresh() {
  await load();
}

async function load() {
  status("bank-status", "载入中…", "busy");
  try {
    const params = {
      dim: $("bank-filter-dim").value,
      primitive: $("bank-filter-prim").value,
      admin: $("bank-filter-admin").value,
      probe_mode: $("bank-filter-probe-mode").value,
      source_id: $("bank-filter-scale").value,
      age: $("bank-filter-age").value.trim(),
      q: $("bank-filter-q").value.trim(),
      probe_ready: $("bank-filter-probe").checked,
      limit: PAGE_SIZE,
      offset,
    };
    const data = await api.get(`/api/examiner/bank${query(params)}`);
    update({ bank: { ...store.bank, total: data.total, items: data.items, stats: data.stats, rules: data.mapping_rules, meta: data.meta, adminZh: data.admin_zh, dims: data.dims } }, "bank");
    prefs.setBankFilters({
      dim: params.dim, primitive: params.primitive, admin: params.admin,
      probeMode: params.probe_mode, sourceId: params.source_id,
      age: params.age, q: params.q, probeOnly: params.probe_ready,
    });
    fillScales(data.stats.by_scale);
    render();
    status("bank-status", `条目库共 ${data.stats.items_total} 条，当前筛选出 ${data.total} 条`, "ok");
  } catch (error) {
    status("bank-status", `失败：${error.message}`, "err");
  }
}

function fillScales(byScale) {
  const current = $("bank-filter-scale").value;
  const names = {};
  store.bank.items.forEach((item) => { names[item.source_id] = item.source_name; });
  html("bank-filter-scale", `<option value="">全部</option>`
    + Object.entries(byScale || {}).sort((a, b) => a[0].localeCompare(b[0]))
      .map(([sid, count]) => `<option value="${esc(sid)}">${esc(names[sid] || sid)}（${count}）</option>`).join(""));
  $("bank-filter-scale").value = current;
}

function render() {
  const items = store.bank.items;
  const stats = store.bank.stats || {};
  $("bank-count").textContent = stats.items_total
    ? `库内 ${stats.items_total} 条（其中 ${stats.probe_ready} 条可探查：直接提问 ${stats.by_probe_mode?.question ?? 0} · 情境式 ${stats.by_probe_mode?.scenario ?? 0}）`
      + ` · 当前筛选 ${store.bank.total} 条 · 本页 ${items.length} 条`
    : `本页 ${items.length} 条`;
  $("bank-selected-n").textContent = store.bankSelected.size;
  const from = store.bank.total ? offset + 1 : 0;
  $("bank-page").textContent = `第 ${from}–${offset + items.length} 条 / 共 ${store.bank.total} 条`;
  $("btn-bank-prev").disabled = offset === 0;
  $("btn-bank-next").disabled = offset + PAGE_SIZE >= store.bank.total;

  if (!items.length) {
    empty("bank-table", { title: "没有符合条件的条目", hint: "请放宽维度 / 施测方式 / 关键词等筛选条件" });
    return;
  }
  html("bank-table", `<thead><tr><th style="width:34px"></th><th>条目 / 儿童面向探查语句</th><th>施测方式</th>
      <th>适用年龄</th><th>表型维度</th><th>基础过程</th><th>来源与出处</th><th></th></tr></thead><tbody>`
    + items.map(row).join("") + `</tbody>`);
}

function row(item) {
  const dims = Object.keys(item.dims || {}).map(dimLabel).join("、") || "—";
  const primIds = Object.keys(item.primitives || {});
  const prims = primNames(primIds).join("、") || "—";
  const age = item.age_field_broken
    ? `<span title="源数据把作答时长写进了 age 字段">${esc(String(item.age_hint || "").slice(0, 10))}</span>`
    : esc(String(item.age_hint || "—").slice(0, 16));
  const selected = store.bankSelected.has(item.key) ? " checked" : "";
  const modeLabel = { question: "直接提问", scenario: "情境式", none: "仅观察" }[item.probe_mode] || item.probe_mode;
  const probe = item.probe_zh
    ? item.probe_zh.split("\n").map((line, i) =>
        `<div class="note" style="color:var(--accent2);margin-top:${i ? 2 : 3}px">${esc(line)}</div>`).join("")
    : `<div class="note" style="margin-top:3px">${esc((item.observation || "").slice(0, 56))}</div>`;
  return `<tr>
      <td><input type="checkbox" data-bank="${esc(item.key)}"${selected} aria-label="选择 ${esc(item.key)}"></td>
      <td><div>${esc(item.text_zh)}</div>${probe}</td>
      <td><span class="badge ${item.probe_mode === "scenario" ? "b-pass" : "b-gap"}">${esc(modeLabel)}</span>
        <div class="note" style="margin-top:3px">${esc(store.bank.adminZh?.[item.admin] || item.admin)}</div></td>
      <td class="mono note">${age}</td>
      <td class="note">${esc(dims)}</td>
      <td class="note" title="${esc(primIds.join("、") || "—")}">${esc(prims)}</td>
      <td class="note">${esc(item.source_name)}<br><span class="mono">${esc(item.source_ref || "")}</span></td>
      <td><button class="tiny" data-detail="${esc(item.key)}">明细</button>
        ${item.probe_zh ? `<div style="margin-top:4px"><button class="tiny" data-use="${esc(item.key)}">用作提问</button></div>` : ""}</td>
    </tr>
    <tr id="bank-detail-${esc(item.key)}" hidden><td></td><td colspan="7">${detail(item)}</td></tr>`;
}

function detail(item) {
  const parts = [];
  if (item.probe_zh) {
    const label = item.probe_mode === "scenario" ? "情境式探查（本项目重写）" : "直接提问（本项目重写）";
    parts.push([label, esc(item.probe_zh).replace(/\n/g, "<br>")]);
    if ((item.scenario || {}).observe) parts.push(["观察目标", esc(item.scenario.observe)]);
    if (item.probe_mode === "scenario") {
      parts.push(["施测方式", "先呈现情境，再问「这种情况你会怎么处理」，按儿童的回答追问"]);
    }
  }
  if (item.observation) parts.push(["观察要点", esc(item.observation)]);
  parts.push(["原条目（源库原文）", esc(item.text_zh)]);
  if (item.options?.length) {
    parts.push(["选项与计分", item.options.map((o) => `${esc(o.content)}${o.score !== null && o.score !== undefined ? `（${esc(o.score)} 分）` : ""}`).join("　")]);
  }
  if (item.scoring_rule) parts.push(["原量表计分规则", esc(item.scoring_rule)]);
  parts.push(["作答人（源数据）", esc(item.respondent || "—")]);
  if (item.symptoms_derived?.length) {
    parts.push(["关联症状（按词语重叠派生）", item.symptoms_derived
      .map((s) => `${esc(s.name_zh)} <span class="mono note">（命中「${esc(s.matched)}」）</span>`).join("<br>")]);
  }
  if (item.criteria?.length) {
    parts.push(["关联诊断标准条目", item.criteria
      .map((c) => `<span class="mono note">${esc(c.id)}</span> ${esc(c.text_zh)}`).join("<br>")]);
  }
  if (item.symptoms_source_scale_coverage) {
    parts.push(["源数据症状关联", esc(item.symptoms_source_scale_coverage.note)]);
  }
  if (item.rationale) parts.push(["量表维度先验理由", esc(item.rationale)]);
  parts.push(["映射依据", `<span class="mono">${esc(JSON.stringify(item.mapping_basis || {}))}</span>`]);
  return `<div style="padding:var(--sp-2) 0">`
    + parts.map(([k, v]) => `<div style="margin-bottom:7px"><span class="note">${esc(k)}：</span><br>${v}</div>`).join("")
    + `</div>`;
}

function renderRules() {
  const rules = store.bank.rules || {};
  const meta = store.bank.meta || {};
  const source = meta.source || {};
  html("bank-rules", `
    <div class="warnbox"><b>来源</b>：${esc(source.name || "")}（${esc(source.kind || "")}）
      <br><span class="mono note">${esc(source.path || "")}</span>
      <br><span class="note">sha256[:16]：${esc(Object.entries(source.sha256_16 || {}).map(([k, v]) => `${k}=${v}`).join("　"))}</span>
      <br><span class="note">生成时间 ${esc(meta.generated_at || "")} · 生成器 <span class="mono">${esc(meta.generator || "")}</span></span></div>
    <h3>维度判定优先级</h3>
    <p class="note">${esc(rules.dim_priority_note || "")}（权重 ${esc(String(rules.dim_weight))}）</p>
    <h3>关键词 → 表型维度</h3>
    <div class="scroll sm"><table><thead><tr><th class="mono">维度</th><th>关键词</th></tr></thead><tbody>
      ${Object.entries(rules.dim_keywords || {}).map(([dim, keywords]) =>
        `<tr><td>${esc(dimLabel(dim))}<br><span class="mono note">${esc(dim)}</span></td><td class="note">${esc((keywords || []).join("、"))}</td></tr>`).join("")}
    </tbody></table></div>
    <h3>量表 → 维度先验（关键词未命中时兜底）</h3>
    <div class="scroll md"><table><thead><tr><th class="mono">量表</th><th>维度与权重</th><th>理由</th></tr></thead><tbody>
      ${Object.entries(rules.scale_dims || {}).map(([sid, dims]) =>
        `<tr><td class="mono">${esc(sid)}</td><td class="mono">${esc(Object.entries(dims).map(([k, v]) => `${k}=${v}`).join(", "))}</td><td class="note">${esc((rules.scale_dims_rationale || {})[sid] || "")}</td></tr>`).join("")}
    </tbody></table></div>
    <h3>关键词 → 可扰动基础过程</h3>
    <div class="scroll sm"><table><thead><tr><th class="mono">基础过程</th><th>关键词</th><th>依据</th></tr></thead><tbody>
      ${Object.entries(rules.primitive_keywords || {}).map(([pid, rule]) =>
        `<tr><td class="mono">${esc(pid)}</td><td class="note">${esc((rule.keywords || []).join("、"))}</td><td class="note">${esc(rule.why || "")}</td></tr>`).join("")}
    </tbody></table></div>
    <h3>症状分类 → 维度</h3>
    <table><thead><tr><th class="mono">AsdKB 分类</th><th>维度</th></tr></thead><tbody>
      ${Object.entries(rules.symptom_category_dims || {}).map(([category, dims]) =>
        `<tr><td class="mono note">${esc(category)}</td><td>${esc((dims || []).map(dimLabel).join("、"))}</td></tr>`).join("")}
    </tbody></table>
    <h3>标准条目域判定</h3><p class="note">${esc(rules.criteria_domain_rule || "")}</p>
    <h3>施测方式与源数据缺陷</h3>
    <p class="note">施测方式：${esc(Object.entries(rules.scale_admin || {}).map(([k, v]) => `${k}=${v}`).join("　"))}</p>
    <p class="note">age 字段被污染的量表：${esc(JSON.stringify(rules.broken_age_fields || {}))}</p>
    <h3>生成说明</h3>
    <ul class="lines">${(meta.notes || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`);
  $("bank-rules").hidden = false;
}
