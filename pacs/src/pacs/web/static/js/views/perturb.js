/** ② 扰动注入：系统采样 / 手工调整 → 提示词构造过程 → 注入并重新初始化。
 *
 * 交互要点（面向使用习惯）：
 * - 左右两栏把「系统生成」和「手工调整」并列，二者都写同一份 θ 状态，不会互相覆盖；
 * - 注入前显示将被施加的扰动向量与被削弱的过程，避免「不知道改了什么」；
 * - 提示词构造过程按步骤展开，默认折叠原文片段，避免一屏堆满长文本；
 * - 注入前不需要先点「预览」也能直接注入（预览只是审计选项）。
 */

import { $, esc, html, withBusy } from "../core/dom.js";
import { api } from "../core/api.js";
import { store, primitives, primName } from "../core/store.js";
import { pollJob } from "../core/job.js";
import { toast } from "../ui/toast.js";
import { renderKpi, status, progress } from "../ui/widgets.js";
import { dimColor } from "../core/format.js";

export const id = "perturb";
export const index = "②";
export const label = "扰动注入";

const GEN_MODES = [
  ["random_single", "随机单因素（一个基础过程被削弱）"],
  ["random_pair", "随机组合（两个过程同时削弱）"],
  ["random_lhs", "随机多维（分层采样）"],
  ["generic", "通用退化对照（非特异）"],
  ["intact", "完整基线（不扰动）"],
  ["single", "指定过程（用下方的过程与剂量）"],
];

/** 名称输入框当前是为哪个 θ 自动填的（见 syncBrainName）。 */
let namedForTheta = null;

export function template() {
  return `
  <div class="card">
    <div class="card-head"><h2>扰动参数 θ</h2>
      <div class="actions">
        <button class="btn ghost sm" id="btn-preview-prompt">预览提示词构造过程</button>
        <button class="btn" id="btn-inject">注入并重新初始化</button>
      </div></div>
    <p class="hint">注入后该 θ 入库并成为当前大脑，随后按其架构重新初始化心智。</p>
    <div class="grid g2">
      <div>
        <h3 style="margin-top:0">系统采样生成</h3>
        <div class="grid g2">
          <div><label for="gen-mode">生成方式</label><select id="gen-mode"></select></div>
          <div><label for="gen-seed">随机种子</label><input id="gen-seed" type="number" value="42"></div>
          <div><label for="gen-dose">组合剂量</label><input id="gen-dose" type="number" step="0.1" min="0.2" max="1" value="0.2"></div>
          <div><label for="gen-generic">通用退化强度</label><input id="gen-generic" type="number" step="0.1" min="0" max="0.8" value="0.5"></div>
        </div>
        <div class="row" style="margin-top:var(--sp-3)">
          <button class="btn ghost" id="btn-generate">生成并载入</button>
          <span class="note">确定性：相同随机种子产生同一 θ</span>
        </div>
      </div>
      <div>
        <h3 style="margin-top:0">手工调整</h3>
        <div><label for="adj-generic">通用退化（非特异对照）</label>
          <input type="range" id="adj-generic" min="0" max="0.8" step="0.1" value="0">
          <span class="note mono" id="adj-generic-val">0.0</span></div>
        <div class="scroll md" style="margin-top:var(--sp-3)"><table id="adj-table"></table></div>
        <div class="row" style="margin-top:var(--sp-2)">
          <button class="btn ghost sm" id="btn-reset-sliders">全部复位为完整</button>
          <span class="status" id="inject-status"></span>
        </div>
      </div>
    </div>
    <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100"><i id="inject-bar"></i></div>
  </div>

  <div class="card" id="theta-card" hidden>
    <div class="card-head"><h2>当前 θ</h2>
      <div class="actions"><button class="btn ghost sm" id="btn-save-theta">保存到大脑库</button></div></div>
    <div class="grid g3" id="theta-kpi"></div>
    <h3>被削弱的过程</h3>
    <div class="scroll sm"><table id="theta-table"></table></div>
  </div>

  <div class="card" id="steps-card" hidden>
    <div class="card-head"><h2>提示词构造过程</h2>
      <span class="note">各步原文按顺序拼接即最终提示词 —— 界面所见即实际发送</span></div>
    <div id="steps-box"></div>
    <h3>最终系统提示词</h3>
    <pre id="prompt-full"></pre>
  </div>

  <div class="card" id="report-card" hidden>
    <h2>心智的自我报告</h2>
    <div class="grid g3" id="brain-kpi"></div>
    <pre id="brain-report" style="margin-top:var(--sp-3)"></pre>
  </div>

  <div class="card" id="plugin-hint-card">
    <h2>下一步</h2>
    <p class="hint" style="margin-bottom:0">
      注入完成后到「③ 表型观察」用对话探针或直接提问观察该大脑的行为表型 ——
      它最近一次的读数就在那一屏的「当前大脑的表型读数」卡里。
    </p>
  </div>`;
}

export function bind() {
  $("gen-mode").innerHTML = GEN_MODES.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join("");
  renderSliders();

  $("adj-generic").addEventListener("input", () => {
    $("adj-generic-val").textContent = Number($("adj-generic").value).toFixed(1);
    renderThetaCard(buildSpec(), "手工调整");
  });
  $("btn-reset-sliders").addEventListener("click", () => {
    setValues({}, 0);
    renderThetaCard(buildSpec(), "手工调整");
  });
  $("btn-generate").addEventListener("click", (event) => withBusy(event.currentTarget, "生成中…", async () => {
    try {
      const data = await api.post("/api/perturb/generate", {
        mode: $("gen-mode").value,
        seed: parseInt($("gen-seed").value, 10),
        dose: parseFloat($("gen-dose").value),
        generic_level: parseFloat($("gen-generic").value),
      });
      setValues(data.values, data.generic_degradation);
      renderThetaCard(data, "系统采样");
      status("inject-status", `已生成 ${data.theta_id} · ${data.describe}`, "ok");
    } catch (error) {
      status("inject-status", `失败：${error.message}`, "err");
    }
  }));

  $("btn-preview-prompt").addEventListener("click", (event) => withBusy(event.currentTarget, "生成中…", async () => {
    try {
      const spec = buildSpec();
      const data = await api.post("/api/prompt", spec);
      renderSteps(data);
      renderThetaCard({ theta_id: data.theta_id, describe: data.describe, ...spec }, "手工调整");
      status("inject-status", `已生成构造过程（未调用模型）· θ=${data.theta_id}`, "ok");
    } catch (error) {
      status("inject-status", `失败：${error.message}`, "err");
    }
  }));

  $("btn-inject").addEventListener("click", (event) => withBusy(event.currentTarget, "注入中…", async () => {
    const body = buildSpec();
    body.name = $("brain-name")?.value.trim() || defaultBrainName();
    try {
      const data = await api.post("/api/inject", body);
      renderSteps(data);
      renderThetaCard({ ...data.brain, theta_id: data.theta_id, describe: data.describe }, "注入");
      status("inject-status", `已注入「${data.brain.name}」，等待自我报告…`);
      progress("inject-bar", 0.3);
      const job = await pollJob(data.job_id, {
        onTick: (tick) => progress("inject-bar", tick.total ? tick.progress / tick.total : 0.5),
      });
      progress("inject-bar", 1);
      if (job.status === "error") {
        status("inject-status", `失败：${job.message}`, "err");
        return;
      }
      renderSelfReport(job.result);
      status("inject-status", job.result.ok ? "初始化成功" : "未检出 SELF_STATE: OK（不必然是失败）",
        job.result.ok ? "ok" : "");
      toast.ok(`已注入并初始化：${data.brain.name}`);
      const { refreshBrains } = await import("../ui/sidebar.js");
      await refreshBrains();
      $("brain-name").value = "";
    } catch (error) {
      progress("inject-bar", 0);
      status("inject-status", `失败：${error.message}`, "err");
    }
  }));

  $("btn-save-theta").addEventListener("click", async () => {
    const name = $("brain-name")?.value.trim() || defaultBrainName();
    try {
      const data = await api.post("/api/brains/save", { name, ...buildSpec() });
      toast.ok(`已保存 θ=${data.brain.theta_id}`);
      const { refreshBrains } = await import("../ui/sidebar.js");
      await refreshBrains();
    } catch (error) {
      toast.error(error.message);
    }
  });
}

export async function refresh() {
  renderSliders();
  applyActiveBrain();
}

/** 供侧栏切换大脑后调用：把新的提示词与自我报告回填。 */
export function onBrainChanged(payload) {
  if (payload?.steps) renderSteps(payload);
  applyActiveBrain(payload?.brain);
}

/** 让滑杆与 θ 卡对准**当前大脑**。
 *
 * 为什么要在进入本界面时做这一步：滑杆本身只是「待注入的草稿」，若进入时不回填，
 * 就会出现「侧栏写着有扰动、② 却说完整心智」这种自相矛盾的画面 —— 同一件事在两处说法不同。
 * 回填后 θ 卡标为「当前大脑」；用户一动滑杆，来源就切回「手工调整」，语义仍然清楚。 */
function applyActiveBrain(brain = null) {
  const current = brain || store.activeBrain;
  if (!current) {
    renderThetaCard(buildSpec(), "手工调整");
    return;
  }
  setValues(current.values || {}, current.generic_degradation || 0);
  renderThetaCard({ ...current, theta_id: current.theta_id || "" }, "当前大脑");
}

/* ---------- 内部 ---------- */

function renderSliders() {
  html("adj-table", `<thead><tr><th>基础过程</th><th class="mono">id</th><th style="width:150px">完整度</th><th style="width:52px">值</th></tr></thead><tbody>`
    + primitives().map((p) => `<tr>
        <td>${esc(p.name_zh)}</td><td class="mono note">${esc(p.id)}</td>
        <td><input type="range" data-pid="${esc(p.id)}" min="0.2" max="1" step="0.1" value="1"
             aria-label="${esc(p.name_zh)} 完整度"></td>
        <td class="mono" id="pv-${esc(p.id)}">1.0</td></tr>`).join("")
    + `</tbody>`);
  $("#adj-table").querySelectorAll("input[type=range]").forEach((slider) => {
    slider.addEventListener("input", () => {
      $(`pv-${slider.dataset.pid}`).textContent = Number(slider.value).toFixed(1);
      renderThetaCard(buildSpec(), "手工调整");
    });
  });
}

function currentValues() {
  const values = {};
  $("#adj-table").querySelectorAll("input[type=range]").forEach((slider) => {
    const value = Number(slider.value);
    if (value < 0.999) values[slider.dataset.pid] = value;
  });
  return values;
}

function currentGeneric() {
  return Number($("adj-generic").value) || 0;
}

function buildSpec() {
  return { kind: "custom", values: currentValues(), generic_degradation: currentGeneric() };
}

function setValues(values, generic) {
  $("#adj-table").querySelectorAll("input[type=range]").forEach((slider) => {
    const value = values?.[slider.dataset.pid] !== undefined ? Number(values[slider.dataset.pid]) : 1;
    slider.value = value;
    $(`pv-${slider.dataset.pid}`).textContent = value.toFixed(1);
  });
  $("adj-generic").value = generic || 0;
  $("adj-generic-val").textContent = Number(generic || 0).toFixed(1);
}

/** 默认大脑名：用中文过程名而不是 id，便于在侧栏里辨识。 */
function defaultBrainName() {
  const values = currentValues();
  // 与后端 describe() 同一套措辞与位数（中文过程名 + 两位完整度），名称只是它的可编辑副本
  const parts = Object.keys(values).map((pid) => `${primName(pid)} ${Number(values[pid]).toFixed(2)}`);
  if (currentGeneric() > 0) parts.push(`通用退化 ${currentGeneric().toFixed(2)}`);
  const name = parts.length ? parts.join("、") : "完整心智";
  return name.length > 60 ? `${name.slice(0, 60)}…` : name;
}

function renderThetaCard(data, source) {
  const values = data.values || {};
  const impaired = Object.entries(values).filter(([, v]) => Number(v) < 0.999);

  renderKpi("theta-kpi", [
    ["θ", data.theta_id || "—", ""],
    ["被削弱的过程", impaired.length, impaired.length ? "" : "完整心智（无扰动）"],
    ["通用退化", Number(data.generic_degradation || 0).toFixed(2), "非特异对照通道"],
    ["来源", source, ""],
  ]);

  const rows = primitives()
    .map((p) => ({ p, value: values[p.id] !== undefined ? Number(values[p.id]) : 1 }))
    .filter((row) => row.value < 0.999);
  if (rows.length) {
    html("theta-table", `<thead><tr><th>基础过程</th><th class="mono">id</th><th>完整度</th></tr></thead><tbody>`
      + rows.map(({ p, value }) => `<tr><td>${esc(p.name_zh)}</td><td class="mono note">${esc(p.id)}</td>
          <td class="mono" style="color:${dimColor(value)}">${value.toFixed(2)}</td></tr>`).join("")
      + `</tbody>`);
  } else {
    html("theta-table", `<tbody><tr><td class="empty-cell">无被削弱的过程（完整心智）</td></tr></tbody>`);
  }

  $("#theta-card").hidden = !data.theta_id;
  syncBrainName(data.theta_id || "");
}

/** 名称输入框是「大脑名」的输入位，默认跟随当前 θ 的中文描述。
 *
 * 为什么不能只在它空着的时候填：② 里的 θ 是会被换掉的（「生成并载入」、侧栏切换大脑），
 * 而名称框里的旧名字不会自己消失 —— 于是注入出来的新大脑会顶着上一个大脑的名字入库，
 * 大脑库里出现两条名字几乎一样、θ 却完全不同的记录。所以按 θ 记账：θ 变了就重填，
 * θ 没变则保留用户手写的名字。滑杆草稿（没有 θ id）不参与，免得用户拖滑杆时名字被改写。
 */
function syncBrainName(thetaId) {
  const field = $("brain-name");
  if (!field || !thetaId || namedForTheta === thetaId) return;
  namedForTheta = thetaId;
  field.value = defaultBrainName();
}

/** 构造过程中各行的列：中文列名 + 固定顺序（基础过程在前，原始 id 退居第二列备查）。 */
const STEP_COLUMNS = {
  name_zh: "基础过程",
  id: "id",
  family: "家族",
  computational_role: "计算职责",
  impairment_signature: "削弱后的表现",
  level: "完整度",
  phrase: "程度描述",
  impaired: "是否被削弱",
};
const STEP_COLUMN_ORDER = Object.keys(STEP_COLUMNS);

/** 单元格取值：家族与布尔值要换成界面语言，其余原样。 */
function stepCell(column, value) {
  if (typeof value === "boolean") return value ? "是" : "否";
  if (column === "family") return store.ontology?.families?.[value] || value;
  return value;
}

function renderSteps(payload) {
  const steps = payload.steps || [];
  html("steps-box", steps.map((step) => {
    let body;
    if (step.kind === "rows" && (step.rows || []).length) {
      const present = Object.keys(step.rows[0]);
      const columns = STEP_COLUMN_ORDER.filter((c) => present.includes(c))
        .concat(present.filter((c) => !STEP_COLUMN_ORDER.includes(c)));
      body = `<div class="scroll md"><table><thead><tr>${columns.map((c) => `<th>${esc(STEP_COLUMNS[c] || c)}</th>`).join("")}</tr></thead><tbody>`
        + step.rows.map((row) => `<tr>${columns.map((c) => {
          const value = row[c];
          const style = c === "level" && typeof value === "number"
            ? ` class="mono" style="color:${dimColor(value)}"` : (c === "id" || c === "level" ? ' class="mono note"' : ' class="note"');
          return `<td${style}>${esc(stepCell(c, value))}</td>`;
        }).join("")}</tr>`).join("") + `</tbody></table></div>`;
    } else if (step.kind === "lines" && (step.lines || []).length) {
      body = `<ul class="lines">${step.lines.map((line) => `<li>${esc(line)}</li>`).join("")}</ul>`;
    } else {
      body = `<pre style="max-height:170px">${esc(step.text)}</pre>`;
    }
    return `<div class="step">
      <div class="hd"><b>${esc(step.title)}</b><span class="badge b-gap">${esc(step.kind)}</span></div>
      <div class="why">为什么需要这一步：${esc(step.why)}</div>
      ${body}
      <details><summary>查看该步在最终提示词里的原文片段</summary><pre>${esc(step.text)}</pre></details>
    </div>`;
  }).join(""));
  $("prompt-full").textContent = payload.prompt || "";
  $("#steps-card").hidden = false;
  $("#steps-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSelfReport(result) {
  if (!result) return;
  $("#report-card").hidden = false;
  $("brain-report").textContent = result.raw || result.self_report || "(空)";
  renderKpi("brain-kpi", [
    ["初始化标记", result.ok ? "SELF_STATE: OK" : "未检出", result.ok ? "心智已按架构上线" : "检查提示词或换模型"],
    ["模型", result.model_id || "—", ""],
    ["θ", result.theta_id || "—", store.activeBrain?.name || ""],
    ["架构指纹", result.fingerprint || "—", ""],
  ]);
}
