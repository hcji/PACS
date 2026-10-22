/** ③ 表型观察 · 直接提问（自定义访谈）。
 *
 * 与「对话探针」的分工：对话探针由提问者 LLM 按条目库自动出题；
 * 这里由**用户本人**针对当前大脑自由提问，逐轮实时作答，用于验证特定假设或追问。
 * 会话与对话探针共用后端的会话/轮次/反馈存储，因此同样可以回看、也可以交给分析者 LLM 打分。
 */

import { $, esc, html, withBusy } from "../core/dom.js";
import { api } from "../core/api.js";
import { store, update } from "../core/store.js";
import { pollJob } from "../core/job.js";
import { toast } from "../ui/toast.js";
import * as readings from "../ui/readings.js";
import { status, empty } from "../ui/widgets.js";
import { dimColor, pct } from "../core/format.js";

export function template() {
  return `
  <div class="card">
    <div class="card-head"><h2>直接提问（自定义访谈）</h2>
      <div class="actions">
        <button class="btn ghost sm" id="btn-itv-new">新建访谈</button>
        <button class="btn ghost sm" id="btn-itv-analyze">交给分析者评分</button>
        <button class="btn ghost sm" id="btn-itv-list">历史访谈</button>
      </div></div>
    <p class="hint">
      由你本人针对<b>当前大脑</b>自由提问，大脑按其认知架构逐轮作答。
      与对话探针共用同一套记录，因此同样可以回看、也可以交给分析者 LLM 形成表型反馈。
    </p>
    <div class="grid g3">
      <div class="kpi"><b class="sm" id="itv-brain">—</b><span>受问大脑</span></div>
      <div class="kpi"><b class="sm" id="itv-session">未开始</b><span>会话 · 轮数</span></div>
      <div>
        <label class="inline"><input type="checkbox" id="itv-with-report" checked> 开始前先做提示词初始化并显示自我报告</label>
      </div>
    </div>
    <div class="row" style="margin-top:var(--sp-3)">
      <span class="status" id="itv-status"></span>
    </div>
    <div class="chat" id="itv-chat" style="margin-top:var(--sp-3);min-height:120px"></div>
    <div style="margin-top:var(--sp-3)">
      <label for="itv-input">你的问题（Enter 发送，Shift+Enter 换行）</label>
      <textarea id="itv-input" rows="2" placeholder="例如：如果有小朋友抢你的玩具，你会怎么办？"></textarea>
      <div class="row" style="margin-top:var(--sp-2)">
        <button class="btn" id="btn-itv-send">发送</button>
        <button class="btn ghost sm" id="btn-itv-clear">清空输入</button>
        <span class="note">也可以从「内置条目库」的条目里点「用作提问」把它填进来</span>
      </div>
    </div>
    <div id="itv-fb" style="margin-top:var(--sp-4)"></div>
    <div id="itv-history" hidden style="margin-top:var(--sp-4)">
      <h3>历史访谈</h3>
      <div class="scroll md"><table id="itv-history-table"></table></div>
    </div>
  </div>`;
}

export function bind({ onNeedBrain }) {
  $("btn-itv-new").addEventListener("click", () => startInterview(onNeedBrain));
  $("btn-itv-send").addEventListener("click", () => send(onNeedBrain));
  $("btn-itv-clear").addEventListener("click", () => { $("itv-input").value = ""; });
  $("btn-itv-list").addEventListener("click", () => refreshHistory());
  $("btn-itv-analyze").addEventListener("click", () => analyze());
  $("itv-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(onNeedBrain);
    }
  });
  render();
}

/** 大脑变化后的上下文同步：受问大脑标签 + 旧会话归属提示。 */
export function syncBrainContext() {
  render();
  const session = store.interview;
  if (session?.sessionId && session.brainId && session.brainId !== store.activeBrain?.brain_id) {
    status("itv-status", "当前大脑已切换：这段访谈属于先前的大脑，如需对新大脑提问请「新建访谈」", "err");
  }
}

/** 把一段现成的话术填进提问框（条目库的「用作提问」按钮会调用）。 */
export function insertQuestion(text) {
  const input = $("itv-input");
  if (!input) return;
  input.value = text;
  input.focus();
  toast.info("已填入提问框，可直接发送或修改");
}

function render() {
  const brain = store.activeBrain;
  $("itv-brain").textContent = brain ? (brain.name || brain.theta_id) : "无当前大脑";
  const session = store.interview || {};
  $("itv-session").textContent = session.sessionId
    ? `${session.sessionId.slice(-8)} · ${(session.turns || []).length} 条消息`
    : "未开始";
  renderChat(session.turns || []);
}

function renderChat(turns) {
  const box = $("itv-chat");
  if (!turns.length) {
    html(box, `<div class="empty"><b>还没有对话</b>先「新建访谈」，然后输入你的问题</div>`);
    return;
  }
  html(box, turns.map((turn) => {
    // 三种角色：user=你提问（左侧气泡）、brain=大脑回应（右侧）、init=大脑的初始化自我报告
    const mine = turn.role === "user";
    const isInit = turn.role === "init";
    const who = mine ? "你（提问者）" : "模拟大脑";
    const meta = isInit ? " · 初始化自我报告" : (mine ? ` · 第 ${turn.turn ?? "?"} 轮` : ` · 回应第 ${turn.turn ?? "?"} 轮`);
    return `<div class="bub ${mine ? "ex" : "br"}${isInit ? " wait" : ""}">
      <span class="who">${who}${meta}</span>${esc(turn.content)}</div>`;
  }).join(""));
  box.scrollTop = box.scrollHeight;
}

async function startInterview(onNeedBrain) {
  const brain = store.activeBrain;
  if (!brain) {
    toast.warn("还没有当前大脑：请先在「扰动注入」里注入或保存一个大脑");
    onNeedBrain?.();
    return;
  }
  await withBusy("btn-itv-new", "启动中…", async () => {
    try {
      const data = await api.post("/api/interview/start", {
        brain_id: brain.brain_id,
        include_self_report: $("itv-with-report").checked,
      });
      const turns = [];
      if (data.init?.raw) {
        turns.push({ role: "init", content: data.init.raw, turn: 0 });
      }
      update({
        interview: {
          sessionId: data.session_id, turns, modelId: data.model_id, init: data.init,
          brainId: data.brain?.brain_id,
        },
      }, "interview");
      render();
      html("itv-fb", "");
      status("itv-status", `已开始访谈（${data.model_id}）· 可以提问了`, "ok");
    } catch (error) {
      status("itv-status", `失败：${error.message}`, "err");
    }
  });
}

async function send(onNeedBrain) {
  const input = $("itv-input");
  const question = input.value.trim();
  if (!question) return;
  if (!store.interview?.sessionId) {
    await startInterview(onNeedBrain);
    if (!store.interview?.sessionId) return;
  }
  const session = store.interview;
  const optimistic = session.turns.concat([{ role: "user", content: question, turn: (session.turns.length + 1) / 2 }]);
  update({ interview: { ...session, turns: optimistic, pending: question } }, "interview");
  render();
  html("itv-chat", $("itv-chat").innerHTML
    + `<div class="bub br wait"><span class="who">模拟大脑</span>正在作答…</div>`);
  input.value = "";

  try {
    const data = await api.post("/api/interview/ask", { session_id: session.sessionId, question });
    const turns = store.interview.turns.concat([{ role: "brain", content: data.answer, turn: data.turn }]);
    update({ interview: { ...store.interview, turns, pending: null } }, "interview");
    render();
    status("itv-status", `已收到回应 · ${data.model_id}`, "ok");
  } catch (error) {
    // 失败时把问题保留在输入框里，避免用户重打一遍
    update({ interview: { ...store.interview, turns: store.interview.turns.slice(0, -1), pending: null } }, "interview");
    render();
    input.value = question;
    status("itv-status", `失败：${error.message}`, "err");
  }
}

async function analyze() {
  const session = store.interview;
  if (!session?.sessionId) {
    toast.warn("还没有可评分的访谈：先提问几轮");
    return;
  }
  await withBusy("btn-itv-analyze", "分析中…", async () => {
    status("itv-status", "正在由分析者 LLM 评估这段对话…", "busy");
    try {
      const start = await api.post("/api/dialogue/analyze", { session_id: session.sessionId });
      const job = await pollJob(start.job_id);
      if (job.status === "error") {
        status("itv-status", `失败：${job.message}`, "err");
        return;
      }
      renderFeedback((job.result || {}).feedback);
      status("itv-status", "表型反馈已生成", "ok");
      await refreshHistory();
    } catch (error) {
      status("itv-status", `失败：${error.message}`, "err");
    }
  });
}

function renderFeedback(feedback) {
  if (!feedback) {
    html("itv-fb", "");
    return;
  }
  const dims = feedback.dims || {};
  html("itv-fb", `
    <h3>表型反馈（分析者 LLM，盲法）</h3>
    ${feedback.error ? `<div class="errbox" style="margin-bottom:var(--sp-3)">分析者输出解析有问题：${esc(feedback.error)}</div>` : ""}
    <div class="grid g3">
      <div class="kpi"><b>${feedback.mean_reading == null ? "—" : pct(feedback.mean_reading)}</b><span>平均读数 · 1.0=典型 · 0.0=严重受损</span></div>
      <div class="kpi"><b class="sm">${esc(feedback.model_id || "—")}</b><span>分析者模型 · ${feedback.n_brain_turns} 轮作答</span></div>
      <div class="kpi"><b class="sm">${esc(feedback.session_id || "")}</b><span>会话 id</span></div>
    </div>
    <div class="scroll sm" style="margin-top:var(--sp-3)"><table>
      <thead><tr><th>维度</th><th>读数</th><th>分布</th></tr></thead><tbody>
      ${Object.entries(dims).map(([dim, value]) => {
        const color = dimColor(value);
        return `<tr><td>${esc(dim)}</td><td class="mono" style="color:${color}">${pct(value)}</td>
          <td><div class="bar" style="margin:0"><i style="width:${Math.max(0, Math.min(1, Number(value))) * 100}%;background:${color}"></i></div></td></tr>`;
      }).join("")}
      </tbody></table></div>
    <p class="note" style="margin-top:var(--sp-3)">${esc(feedback.narrative || "")}</p>
    <div id="itv-conclusion"></div>`);
  // 探查结论与表型观察里的那一段是同一份数据、同一个渲染函数 —— 两条链路各写一遍
  // 迟早会显示得不一样（例如字数口径或免责声明只在其中一边改）
  readings.renderConclusionInto("itv-conclusion", feedback);
}

async function refreshHistory() {
  try {
    const data = await api.get("/api/interview/sessions");
    const box = $("itv-history");
    box.hidden = false;
    html("itv-history-table", `<thead><tr><th class="mono">会话</th>
        <th>轮数</th><th>反馈</th><th>时间</th><th></th></tr></thead><tbody>`
      + (data.sessions.length ? data.sessions.map((s) => `<tr>
          <td class="mono">${esc(s.session_id)}</td>
          <td class="mono">${esc(s.n_turns)}</td>
          <td>${s.has_reading
            ? '<span class="badge b-pass">有读数</span>'
            : (s.has_feedback
              ? '<span class="badge b-weak" title="分析者没有给出可用的逐维读数">无读数</span>'
              : '<span class="note">未评分</span>')}</td>
          <td class="note mono">${esc(s.created_at)}</td>
          <td><button class="tiny" data-itv="${esc(s.session_id)}">打开</button></td></tr>`).join("")
        : '<tr><td colspan="5" class="empty-cell">还没有访谈记录</td></tr>')
      + `</tbody>`);
    $("itv-history-table").querySelectorAll("button[data-itv]").forEach((button) => {
      button.addEventListener("click", () => openInterview(button.dataset.itv));
    });
  } catch (error) {
    toast.error(error.message);
  }
}

async function openInterview(sessionId) {
  try {
    const data = await api.get(`/api/dialogue?session_id=${encodeURIComponent(sessionId)}`);
    const turns = (data.turns || []).map((t) => ({ role: t.role, content: t.content, turn: t.turn ?? t.turn_index }));
    update({
      interview: {
        sessionId, turns, modelId: (data.meta || {}).brain_model,
        brainId: (data.meta || {}).brain_id,
      },
    }, "interview");
    render();
    renderFeedback(data.feedback);
    status("itv-status", `已载入访谈 ${sessionId}，可继续提问`, "ok");
  } catch (error) {
    toast.error(error.message);
  }
}
