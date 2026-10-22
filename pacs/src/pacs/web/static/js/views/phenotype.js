/** ③ 表型观察：只保留**对话相关**的功能，且默认只显示三块内容。
 *
 * 默认可见：
 * - 当前大脑的表型读数：该大脑最近一次被观察得到的读数，以**雷达图**为主
 *   （锚定的是大脑，不随你打开哪段历史而变）；
 * - 历史对话：提问者 LLM 与模拟大脑的往来记录，可回看；
 * - 直接提问（访谈）：由**你**提问，同一条链路回答，用于自定义追问。
 *
 * 两个 LLM 的行为（提问者逐轮提问／设定情境 → 模拟大脑作答 → 分析者形成读数）
 * **自动运行**：进入本界面时，若当前大脑还没有读数就跑一次，之后每次换大脑同理。
 * 这段过程默认**收起**，点「查看对话过程」展开 —— 抽屉里**只有对话本身**，
 * 没有参数、也没有别的信息。
 *
 * **观测链路里没有扰动参数**（本界面最重要的一条约定）：
 * - 扰动只作为模拟大脑的提示词输入，提问者与分析者都拿不到；
 * - 表型只由提问与观察得到，读数不附带任何 θ 基准真值，因此界面上不存在
 *   「读数与扰动是否相符」这类比对（那是**测量之外**的离线分析，用 CLI）。
 *
 * 界面约定（按用户要求）：表型观察里**不再有任何群体相关功能**（群体表型运行、表型剖面、
 * 表型景观均已移出）；机制分析界面也**暂时**移出。要跑受试集合层面的分析，用 CLI。
 *
 * 测量通道只剩一条：提问者 LLM → 模拟大脑 → 分析者 LLM。强制选择行为电池连同它的
 * 「定量通道」界面一并删除，因此这里没有题目数、正确率、题量分布这类概念。
 * 内置条目库与自定义条目上传是进阶工具，同样默认收起，点按钮展开。
 */

import { $, esc, html, withBusy } from "../core/dom.js";
import { api } from "../core/api.js";
import { store, update, dimLabel, dimsAll, scoredChannels } from "../core/store.js";
import { prefs } from "../core/prefs.js";
import { pollJob } from "../core/job.js";
import { toast } from "../ui/toast.js";
import { status, renderTable } from "../ui/widgets.js";
import * as bank from "./bank.js";
import * as interview from "./interview.js";
import * as readings from "../ui/readings.js";

export const id = "phenotype";
export const index = "③";
export const label = "表型观察";

let autoRunning = false;   // 观察进行中：避免换大脑/重进界面时并发跑同一件事
let autoFailed = false;    // 自动观察失败过：不再自动重试，只提示手动重试

export function template() {
  return `
  <div id="readings-panel"></div>

  <div id="probe-drawer" hidden>
    <div class="card" id="dlg-chat-card">
      <div class="card-head"><h2>对话过程</h2>
        <div class="actions"><span class="note" id="dlg-chat-meta"></span></div></div>
      <div id="dlg-problems"></div>
      <div class="chat" id="dlg-chat"></div>
    </div>
  </div>

  <div class="card" id="dlg-list-card" hidden>
    <div class="card-head"><h2>历史对话</h2>
      <div class="actions"><span class="note" id="dlg-list-count"></span></div></div>
    <p class="hint">
      <b>当前大脑</b>的观察记录（别的 θ 的对话不在这里）；每一条都可以点开，在上方展开它的完整问答。
      换大脑或重新初始化后，这里会跟着清空 —— 读数与历史锚定的是同一具心智。
    </p>
    <div class="scroll md"><table id="dlg-list"></table></div>
  </div>

  <div id="interview-panel"></div>

  <div class="card">
    <div class="card-head"><h2>进阶工具</h2></div>
    <p class="hint">
      内置条目库是提问者的<b>参考材料</b>（不是必须照问的题目清单：提问者会按观察目标<b>自己出题</b>），
      也可勾选指定条目，让提问者就着这些条目提一轮；自定义条目集用于上传你自己的量表。
      两者都属于按需工具，默认收起。
    </p>
    <div class="row">
      <button class="btn ghost sm" id="btn-toggle-bank" aria-expanded="false">浏览内置条目库</button>
      <button class="btn ghost sm" id="btn-toggle-upload" aria-expanded="false">上传自定义条目集</button>
      <span class="note" id="adv-status"></span>
    </div>
  </div>

  <div id="bank-panel" hidden></div>

  <div id="upload-panel" hidden>
    <div class="card">
      <div class="card-head"><h2>自定义条目集上传</h2>
        <div class="actions"><button class="btn ghost sm" id="btn-upload-close">收起</button></div></div>
      <p class="hint">
        上传 JSON 条目集。条目一律由<b>分析者 LLM 依据对话表现逐条评分</b>（表型全部由 LLM 读出），
        可点「用它观察一次」叠加到一轮观察上。条目里若带旧的
        <span class="mono">key</span> / <span class="mono">primitive_sensitivity</span>
        （定量管线的字段），只作出处信息保留并给出提示 —— 那套按正确率算分的管线已随行为电池删除。
      </p>
      <div class="grid g2">
        <div>
          <label for="set-name">条目集名称</label><input id="set-name" placeholder="例如：自编社交情境问卷 v1">
          <label style="margin-top:var(--sp-2)" for="set-file">选择文件（.json）</label>
          <input type="file" id="set-file" accept=".json,.txt">
          <label style="margin-top:var(--sp-2)" for="set-text">或粘贴 JSON</label>
          <textarea id="set-text" rows="7"></textarea>
          <div class="row" style="margin-top:var(--sp-2)">
            <button class="btn ghost sm" id="btn-set-upload">校验并上传</button>
            <button class="btn ghost sm" id="btn-set-template">填入模板</button>
            <span class="status" id="set-status"></span>
          </div>
        </div>
        <div>
          <h3 style="margin-top:0">已上传条目集</h3>
          <div class="scroll md"><table id="set-table"></table></div>
        </div>
      </div>
      <div id="set-problems"></div>
    </div>
  </div>`;
}

export function bind() {
  html("interview-panel", interview.template());
  interview.bind({ onNeedBrain: () => { location.hash = "#/perturb"; } });
  html("readings-panel", readings.template());
  readings.bind({
    onOpenSession: (sessionId) => openSession(sessionId),
    onReobserve: (button) => withBusy(button, "观察中…", () => observe({ reveal: false })),
    onToggleProbe: () => setDrawer($("probe-drawer").hidden),
    // 轮数上限是**这一台机器上的偏好**（不写进服务端配置）：它只影响下一次观察，
    // 而且每次改动都会重算成本提示 —— 一具心智一次观察是 71 次起步的调用，不能默默涨。
    onMaxTurnsChange: (value) => {
      if (value) prefs.set("maxTurns", value);
      renderTurnBudget();
    },
  });
  html("bank-panel", bank.template());
  bank.bind({
    onStartDialogue: (useSelected) => { observe({ useSelected, reveal: true }); },
    onUseAsQuestion: (text) => interview.insertQuestion(text),
  });
  setDrawer(Boolean(prefs.get("probeDrawerOpen")), { persist: false });
  mountAdvancedTools();
  mountUploadControls();
  renderChat([], "");
}

export async function refresh() {
  await Promise.all([bank.refresh(), refreshItemSets()]);
  await refreshSessions();
  interview.syncBrainContext();
  renderTurnBudget();
  await ensureReading();
}

/** 把「基础轮数 / 上限 / 字数上限」与成本写到读数卡的头部（成本必须可见）。 */
function renderTurnBudget() {
  const protocol = store.boot?.dialogue || {};
  readings.setTurnBudget({
    nTurns: protocol.default_n_turns,
    maxTurns: prefs.get("maxTurns") || protocol.max_n_turns,
    textLimit: protocol.text_limit,
  });
}

export function onBrainChanged() {
  interview.syncBrainContext();
  // 读数、历史列表、对话抽屉三处都锚定同一具大脑：换大脑时一起清，
  // 否则抽屉里还留着上一具心智的问答、历史里还列着别的会话，读起来像是同一段。
  autoFailed = false;
  readings.reset();
  renderChat([], "");
  refreshSessions().catch(() => {});
  ensureReading().catch((error) => readings.setMeta(`读数读取失败：${error.message}`));
}

/* ---------- 两个 LLM 的观察：自动跑、界面收起 ---------- */

/** 抽屉（对话过程）默认收起，点读数卡上的按钮展开，状态记在偏好里。 */
function setDrawer(open, { persist = true } = {}) {
  $("probe-drawer").hidden = !open;
  readings.setDrawerState(open);
  if (persist) prefs.set("probeDrawerOpen", open);
}

/** 进入界面时保证「当前大脑有一份读数」：没有就让两个 LLM 自动跑一次。 */
async function ensureReading() {
  const report = await refreshReadings();
  if (report.readings) {
    // 读数是从哪段对话读出来的：把那一段预载进收起的抽屉，点开即见
    if (report.readings.session_id) {
      await openSession(report.readings.session_id, { quiet: true, reveal: false }).catch(() => {});
    }
    return;
  }
  if (autoFailed) {
    readings.setMeta("上次自动观察失败 —— 点「重新观察」重试");
    return;
  }
  await autoObserve();
}

/** 自动观察：与手点「重新观察」走同一条链路，只是不需要人去按。 */
async function autoObserve() {
  if (!store.activeBrain) {
    readings.setMeta("还没有当前大脑 —— 先去 ② 扰动注入里注入或选一个大脑");
    return;
  }
  await observe({ reveal: false, auto: true });
}

/**
 * 跑一轮观察。自动与手动共用这一个函数，避免两条链路的参数悄悄分叉。
 *
 * 请求体只有「观察目标」、（可选的）选题来源与**轮数上限** —— **没有扰动、没有基础轮数、
 * 也没有探查方式**：基础轮数（每通道 ≥5 个问题）与探查形态由服务端的观察协议给定
 * （`/api/boot` 的 `dialogue` 字段），于是每一具心智都被同一套协议观察，读数之间才可比。
 * 唯一由界面给的是「补问到这里为止」的上限（成本旋钮，不改变协议下限）。
 */
async function observe({ reveal = false, auto = false, useSelected = false, setId = "" } = {}) {
  if (autoRunning) return;
  // 观察目标 = 计分通道（含对照通道）：漏掉它，Control 1 就没有测量依据
  // 轮数上限随偏好下发（基础轮数与字数上限由服务端协议给定）；下限由服务端校验兜底
  const body = { goals: scoredChannels(), max_turns: prefs.get("maxTurns") || undefined };
  if (setId) body.set_id = setId;
  if (useSelected && store.bankSelected.size) {
    body.bank_keys = [...store.bankSelected];
    const blocked = store.bank.items.filter((i) => store.bankSelected.has(i.key) && !i.probe_zh);
    if (blocked.length) {
      readings.setMeta(`所选条目中 ${blocked.length} 条没有儿童面向探查语句（${blocked.slice(0, 3).map((b) => b.key).join("、")}${blocked.length > 3 ? " 等" : ""}），只能作为观察要点。`);
      return;
    }
  } else {
    body.use_bank = true;
  }

  autoRunning = true;
  autoFailed = false;
  if (reveal) setDrawer(true);
  renderChat([], "准备第一个探查语句…");
  readings.setMeta("正在观察…");
  try {
    const start = await api.post("/api/dialogue/start", body);
    readings.setMeta(`正在观察 · 预计 ${start.estimated_calls}–${start.estimated_calls_max} 次调用`);
    const job = await pollJob(start.job_id, {
      onTick: (tick) => {
        if (tick.transcript) renderChat(tick.transcript);
        readings.setMeta(tick.total
          ? `正在观察 · ${tick.progress || 0}/${tick.total} 步`
          : "正在观察…");
      },
    });
    if (job.status === "error") {
      autoFailed = auto;
      readings.setMeta(`观察失败：${job.message}${auto ? " —— 可点「重新观察」重试" : ""}`);
      toast.error(`观察失败：${job.message}`);
      return;
    }
    renderChat(job.transcript || []);
    await Promise.all([refreshReadings(), refreshSessions()]);
    toast.ok(`观察完成 · 会话 ${job.result?.session?.session_id || ""}`);
  } catch (error) {
    autoFailed = auto;
    readings.setMeta(`观察失败：${error.message}${auto ? " —— 可点「重新观察」重试" : ""}`);
    toast.error(`观察失败：${error.message}`);
  } finally {
    autoRunning = false;
  }
}

/** 取当前大脑最近一次读数（不依赖任何群体运行，也不随打开哪段历史而变）。 */
async function refreshReadings() {
  const report = await api.get("/api/brain-report");
  readings.render(report);
  return report;
}

/* ---------- 进阶工具（默认折叠）---------- */

function mountAdvancedTools() {
  const toggle = (buttonId, panelId, prefKey, label) => {
    const button = $(buttonId);
    const panel = $(panelId);
    panel.hidden = !prefs.get(prefKey);
    button.setAttribute("aria-expanded", String(!panel.hidden));
    button.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      button.setAttribute("aria-expanded", String(!panel.hidden));
      prefs.set(prefKey, !panel.hidden);
      status("adv-status", panel.hidden ? `${label}已收起` : `已展开${label}`);
    });
  };
  toggle("btn-toggle-bank", "bank-panel", "advancedOpenBank", "内置条目库");
  toggle("btn-toggle-upload", "upload-panel", "advancedOpenUpload", "自定义条目集");
  $("btn-upload-close").addEventListener("click", () => $("btn-toggle-upload").click());
}

/* ---------- 对话过程 ---------- */

function renderChat(turns, placeholder) {
  const box = $("dlg-chat");
  const turnNo = (t) => (t.turn ?? t.turn_index ?? "?");
  // 提问者每轮自报的探查方面（`FOCUS`）：自行出题的轮次没有条目可查，
  // 这一行是该轮归属到哪个通道的唯一依据，必须让人看得见。
  const focusTag = (t) => (t.role === "examiner" && t.focus
    ? ` · 探查：${esc(dimLabel(t.focus) || t.focus)}` : "");
  let markup = (turns || []).map((t) => `<div class="bub ${t.role === "examiner" ? "ex" : "br"}">
      <span class="who">${t.role === "examiner" ? "提问者 LLM" : "模拟大脑"} · 第 ${esc(turnNo(t))} 轮${focusTag(t)}</span>${esc(t.content)}</div>`).join("");
  if (placeholder) markup += `<div class="bub ex wait"><span class="who">提问者 LLM</span>${esc(placeholder)}</div>`;
  if (!markup) markup = `<div class="empty">还没有轮次</div>`;
  html(box, markup);
  box.scrollTop = box.scrollHeight;
}

async function refreshSessions() {
  // 列表锚定**当前大脑**（与服务端同一套归属规则）：换大脑、或重新初始化之后，
  // 这里必须立刻清空 —— 否则会出现「读数说这具心智还没被观察过，历史里却列着
  // 别的几十段对话」。`brain_id` 即使为空也要显式带上（服务端据此判定「只看向这具大脑」）。
  const scoped = `brain_id=${encodeURIComponent(store.activeBrain?.brain_id || "")}`;
  const data = await api.get(`/api/dialogue/sessions?${scoped}`);
  $("dlg-list-card").hidden = false;
  const shown = data.sessions.length;
  const total = data.total ?? shown;
  $("dlg-list-count").textContent = total > shown
    ? `最近 ${shown} 条 · 共 ${total} 条`
    : `共 ${total} 条`;
  renderTable("dlg-list",
    ["session", "轮数", "表型读数", "提示", "时间", ""],
    data.sessions.map((s) => `<tr>
      <td class="mono">${esc(s.session_id)}</td>
      <td class="mono">${esc(s.n_turns)}${s.n_turns_run ? ` / 上限 ${esc(s.max_turns ?? "—")}` : ""}</td>
      <td>${readingCell(s)}</td>
      <td>${(s.problems || []).length
        ? `<span class="badge b-weak" title="${esc(s.problems.join(" / "))}">${(s.problems || []).length} 条</span>`
        : '<span class="note">—</span>'}</td>
      <td class="note mono">${esc(s.created_at)}</td>
      <td><button class="tiny" data-session="${esc(s.session_id)}">打开</button></td></tr>`),
    { emptyText: store.activeBrain
      ? "这具大脑还没有被观察过 —— 点读数卡右上角的「重新观察」跑一轮"
      : "还没有当前大脑 —— 先去 ② 扰动注入里注入或选一个大脑" });
  $("#dlg-list").querySelectorAll("button[data-session]").forEach((button) => {
    button.addEventListener("click", () => openSession(button.dataset.session));
  });
}

/** 打开一段对话：展开收起的抽屉，显示它的逐轮问答。
 *
 * `reveal: false` 是「预载」用的：把读数来源那段对话先渲染进抽屉，但**不**替用户把抽屉拉开 ——
 * 否则「默认收起、按钮唤起」就形同虚设（每次进界面预载都会自动展开）。
 */
/** 这段对话的读数状态：有读数 / 无读数（跑过但分析者没吐出东西）/ 没跑过分析。
 *
 * 「有反馈记录」不等于「有读数」：分析者调用失败也会落一条空反馈。把那种情形显示成
 * 「有读数」，用户会以为这一段已经量出来了 —— 与读数卡上的说明自相矛盾。
 */
function readingCell(session) {
  if (session.has_reading) return '<span class="badge b-pass">有</span>';
  if (session.has_feedback) {
    return '<span class="badge b-weak" title="分析者没有给出可用的逐维读数（查看对话过程可见原因）">无读数</span>';
  }
  return '<span class="note">未评分</span>';
}

async function openSession(sessionId, { quiet = false, reveal = true } = {}) {
  try {
    const data = await api.get(`/api/dialogue?session_id=${encodeURIComponent(sessionId)}`);
    if (reveal) setDrawer(true);
    renderChat(data.turns, "");
    renderSessionProblems((data.session || {}).problems, data.session || {});
    if (!quiet) toast.ok(`已载入对话 ${sessionId}`);
  } catch (error) {
    if (!quiet) toast.error(error.message);
    else throw error;
  }
}

/** 这段对话自身的告警（覆盖缺口、提问者没产出可解析问题等）。
 *
 * 为什么必须显示：覆盖不到的方面，分析者对它的读数没有证据支撑，只能给保守值；
 * 界面若不显示这条，用户会把「没问过」误读成「问过且正常」。
 */
function renderSessionProblems(problems, session) {
  const list = problems || [];
  const goals = session.goals || [];
  $("dlg-chat-meta").textContent = goals.length
    ? `覆盖 ${goals.length - coverageGapSize(list, goals)}/${goals.length} 个方面`
    : "";
  html("dlg-problems", list.length
    ? `<div class="errbox" style="margin:var(--sp-3) var(--sp-4) 0"><b>这段对话有 ${list.length} 条提示</b>
        <ul>${list.map((p) => `<li>${esc(p)}</li>`).join("")}</ul></div>`
    : "");
}

/** 从告警文本里取「未覆盖方面数」，取不到就报 0（不猜）。 */
function coverageGapSize(problems, goals) {
  for (const text of problems) {
    const match = /覆盖\s*(\d+)\/(\d+)\s*个方面/.exec(text);
    if (match) return Number(match[2]) - Number(match[1]);
  }
  return 0;
}

/* ---------- 自定义条目集 ---------- */

function mountUploadControls() {
  $("btn-set-template").addEventListener("click", () => {
    $("set-text").value = JSON.stringify({
      name: "自编社交情境问卷 v1",
      items: [
        { id: "q1", battery_dim: "自编维度·分享与轮流", prompt: "自由作答：描述一次和小朋友轮流玩的经历（由分析者 LLM 依据对话表现评分）",
          options: ["（文字作答）"] },
      ],
    }, null, 2);
    status("set-status", "已填入模板（开放式条目 → 由分析者 LLM 逐条评分）");
  });
  $("set-file").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    $("set-text").value = await file.text();
    if (!$("set-name").value) $("set-name").value = file.name.replace(/\.(json|csv|txt)$/i, "");
    status("set-status", `已读取文件 ${file.name}，点「校验并上传」`);
  });
  $("btn-set-upload").addEventListener("click", (event) => withBusy(event.currentTarget, "校验中…", async () => {
    html("set-problems", "");
    try {
      const data = await api.post("/api/itemsets", { name: $("set-name").value.trim(), text: $("set-text").value });
      if (!data.valid) {
        html("set-problems", `<div class="errbox" style="margin-top:var(--sp-3)"><b>校验未通过</b><ul>${data.problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>（未保存）</div>`);
        status("set-status", "校验未通过", "err");
        return;
      }
      status("set-status", `已上传 ${data.n_items} 条 · 由分析者 LLM 逐条评分`
        + "（可点「用它观察一次」叠加到一轮观察）", "ok");
      for (const notice of data.item_set.notices || []) toast.ok(notice);
      toast.ok("条目集已上传");
      await refreshItemSets();
    } catch (error) {
      status("set-status", `失败：${error.message}`, "err");
    }
  }));
}

async function refreshItemSets() {
  const data = await api.get("/api/itemsets");
  update({ itemSets: data.item_sets || [] }, "itemSets");
  renderTable("set-table", ["名称", "模式", "条目", "维度", ""],
    store.itemSets.map((set) => {
      const payload = set.payload || {};
      return `<tr><td>${esc(set.name)}</td>
        <td><span class="badge ${set.scoring_mode === "llm" ? "b-pass" : "b-weak"}">${esc(set.scoring_mode)}</span>
          <div class="note" style="margin-top:3px">${esc(payload.mode_reason || "")}</div></td>
        <td class="mono">${(payload.items || []).length}</td>
        <td class="note">${esc((payload.dims || []).map((d) => dimLabel(d.dim)).join("、"))}</td>
        <td>${set.scoring_mode === "llm"
          ? `<button class="tiny" data-observe="${esc(set.set_id)}">用它观察一次</button>`
          : ""}
          <button class="tiny" data-del="${esc(set.set_id)}">删除</button></td></tr>`;
    }), { emptyText: "还没有上传条目集" });
  $("#set-table").querySelectorAll("button[data-observe]").forEach((button) => {
    button.addEventListener("click", () => withBusy(button, "观察中…", () => observe({ setId: button.dataset.observe, reveal: true })));
  });
  $("#set-table").querySelectorAll("button[data-del]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (button.dataset.confirm !== "1") {
        button.dataset.confirm = "1";
        button.textContent = "确认删除？";
        return;
      }
      await api.post("/api/itemsets/delete", { set_id: button.dataset.del });
      toast.ok("已删除条目集");
      await refreshItemSets();
    });
  });
}
