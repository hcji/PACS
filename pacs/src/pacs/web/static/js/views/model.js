/** ① 模型设置：两套 LLM 接入 + 运行环境。
 *
 * 大脑库已移到左侧栏（切换大脑是全局上下文操作），这一屏只保留「配置」与「环境状态」，
 * 避免同一件事有两个入口。
 */

import { $, esc, html, withBusy } from "../core/dom.js";
import { api } from "../core/api.js";
import { prefs } from "../core/prefs.js";
import { store, update } from "../core/store.js";
import { toast } from "../ui/toast.js";
import { renderKpi, status } from "../ui/widgets.js";

export const id = "model";
export const index = "①";
export const label = "模型设置";

export function template() {
  return `
  <div class="card">
    <div class="card-head"><h2>LLM 接入配置</h2>
      <span class="note">两套独立配置；Key 留空表示保持不变，且永不回传</span></div>
    <div class="grid g2">
      <div>
        <h3 style="margin-top:0">模拟大脑（被扰动的人工心智）</h3>
        <div class="grid g2">
          <div><label for="cfg-provider">提供方</label>
            <select id="cfg-provider">
              <option value="mock">mock —— 离线规则式心智</option>
              <option value="openai">openai —— OpenAI 兼容端点</option>
            </select></div>
          <div><label for="cfg-model">模型名</label><input id="cfg-model" placeholder="gpt-4o-mini"></div>
          <div><label for="cfg-base">API Base URL</label><input id="cfg-base" placeholder="https://api.openai.com/v1"></div>
          <div><label for="cfg-key">API Key</label><input id="cfg-key" type="password" placeholder="留空则不修改" autocomplete="off"></div>
        </div>
      </div>
      <div>
        <h3 style="margin-top:0">提问者 / 分析者（独立第二套接入）</h3>
        <div class="grid g2">
          <div><label for="cfg-eprovider">提供方</label>
            <select id="cfg-eprovider">
              <option value="mock">mock —— 离线规则式</option>
              <option value="openai">openai —— OpenAI 兼容端点</option>
            </select></div>
          <div><label for="cfg-emodel">模型名</label><input id="cfg-emodel" placeholder="gpt-4o-mini"></div>
          <div><label for="cfg-ebase">API Base URL</label><input id="cfg-ebase" placeholder="https://api.openai.com/v1"></div>
          <div><label for="cfg-ekey">API Key</label><input id="cfg-ekey" type="password" placeholder="留空则不修改" autocomplete="off"></div>
          <div><label for="cfg-emax">输出额度 max_tokens</label>
            <input id="cfg-emax" type="number" min="64" max="32000" step="64" placeholder="2048"></div>
        </div>
      </div>
    </div>
    <div class="row" style="margin-top:var(--sp-4)">
      <div style="min-width:140px"><label for="cfg-temp">温度</label><input id="cfg-temp" type="number" step="0.1" min="0" max="2"></div>
      <div style="min-width:160px"><label for="cfg-max">输出额度 max_tokens</label>
        <input id="cfg-max" type="number" min="64" max="32000" step="64" placeholder="2048"></div>
      <div style="padding-top:18px"><button class="btn" id="btn-save-cfg">保存配置</button></div>
      <label class="inline" style="margin-top:18px"><input type="checkbox" id="cfg-persist"> 写入 config.json（含 Key，谨慎）</label>
      <span class="status" id="cfg-status"></span>
    </div>
  </div>

  <div class="card">
    <div class="card-head"><h2>运行环境</h2>
      <div class="actions"><button class="btn ghost sm" id="btn-refresh-env">刷新</button></div></div>
    <div class="grid g3" id="env-kpi"></div>
    <div id="env-notes" style="margin-top:var(--sp-4)"></div>
  </div>`;
}

export function bind() {
  $("btn-save-cfg").addEventListener("click", (event) => withBusy(event.currentTarget, "保存中…", async () => {
    const body = {
      provider: $("cfg-provider").value,
      model: $("cfg-model").value,
      base_url: $("cfg-base").value,
      temperature: parseFloat($("cfg-temp").value),
      max_tokens: parseInt($("cfg-max").value, 10) || undefined,
      persist: true,
      persist_key: $("cfg-persist").checked,
      examiner: {
        provider: $("cfg-eprovider").value,
        model: $("cfg-emodel").value,
        base_url: $("cfg-ebase").value,
        max_tokens: parseInt($("cfg-emax").value, 10) || undefined,
      },
    };
    if ($("cfg-key").value) body.api_key = $("cfg-key").value;
    if ($("cfg-ekey").value) body.examiner.api_key = $("cfg-ekey").value;
    try {
      const data = await api.post("/api/config", body);
      $("cfg-key").value = "";
      $("cfg-ekey").value = "";
      update({ config: data.config }, "config");
      const ready = Boolean(data.config.configured?.ready);
      // 「离线模式」这个标记跟着真实状态走：配好了就不再是离线，没配好就还是
      prefs.set("offlineAcknowledged", !ready);
      const brainKey = data.config.api_key_set ? "已配置" : "未配置";
      // 提问者/分析者没自己填 Key，但同端点时可以用大脑那把 —— 状态里如实说「共用」，
      // 不要写成「未配置 · 可真实调用」这种自相矛盾的一行
      const own = Boolean(data.config.examiner?.api_key_set);
      const examinerKey = own ? "已配置" : (ready ? "共用大脑的 Key" : "未配置");
      status("cfg-status", `已保存 · 大脑 Key ${brainKey} · 分析者 Key ${examinerKey}`
        + (ready ? " · 可真实调用" : " · 尚不具备真实调用条件"), "ok");
      toast.ok("配置已保存");
      renderEnvironment();
    } catch (error) {
      status("cfg-status", `失败：${error.message}`, "err");
    }
  }));

  $("btn-refresh-env").addEventListener("click", () => refresh());
}

export async function refresh() {
  const [config, boot] = await Promise.all([api.get("/api/config"), api.get("/api/boot")]);
  update({ config, boot }, "config");
  $("cfg-provider").value = config.provider;
  $("cfg-model").value = config.model || "";
  $("cfg-base").value = config.base_url || "";
  $("cfg-temp").value = config.temperature;
  $("cfg-max").value = config.max_tokens ?? "";
  const examiner = config.examiner || {};
  $("cfg-eprovider").value = examiner.provider || "mock";
  $("cfg-emodel").value = examiner.model || "";
  $("cfg-ebase").value = examiner.base_url || "";
  $("cfg-emax").value = examiner.max_tokens ?? "";
  renderEnvironment();
}

/** 启动初始化状态码 → 可读文本（界面不直接暴露 ready/pending 这类内部码）。 */
const INIT_STATUS = {
  ready: "已完成",
  running: "进行中",
  pending: "等待中",
  error: "失败",
  skipped: "已跳过",
};

function renderEnvironment() {
  const config = store.config || {};
  const boot = store.boot || {};
  const init = boot.boot || {};
  const examiner = config.examiner || {};
  const brain = store.activeBrain;

  renderKpi("env-kpi", [
    ["模拟大脑", `${config.provider || "—"} / ${config.model || "—"}`,
      config.configured?.brain ? "可真实调用" : "当前为 mock 或未配置 Key"],
    ["提问者 / 分析者", `${examiner.provider || "—"} / ${examiner.model || "—"}`,
      examiner.api_key_set ? "Key 已配置" : "未配置（跟随模拟大脑）"],
    ["输出额度", `大脑 ${config.max_tokens ?? "—"} · 分析者 ${examiner.max_tokens ?? "—"}`,
      "推理型模型会把额度花在思维链上；太小=正文为空（读数空白）"],
    ["当前大脑", brain ? (brain.name || brain.theta_id) : "无",
      brain ? (Object.keys(brain.impaired || {}).length ? "含扰动" : "完整心智") : "请在左侧保存或注入"],
    ["启动初始化", init.ok ? "SELF_STATE: OK" : INIT_STATUS[boot.boot?.status] || "—",
      init.fingerprint ? `架构指纹 ${init.fingerprint}` : boot.boot?.message || ""],
    ["心智年龄", `${config.age_months ?? 36} 月（固定）`,
      "常量口径，不可配置：参照线一旦随参数漂移，各次读数就不可比"],
    ["数据目录", "", config.db_path || ""],
  ]);

  const notes = [];
  if (!config.configured?.ready) {
    const missing = [];
    if (!config.configured?.brain) missing.push("模拟大脑");
    if (!config.configured?.examiner) missing.push("提问者 / 分析者");
    notes.push("当前不具备真实 LLM 调用条件"
      + (missing.length ? `（缺 Key 或仍是 mock：${missing.join("、")}）` : "")
      + "：可在本屏填入 Key，或继续以 mock 模式验证管线。");
    notes.push("注意：API Key **不落盘**（只留在本次运行的内存里）。要重启后仍然有效，"
      + "请勾选「写入 config.json」，或改用环境变量 PACS_LLM_API_KEY / "
      + "PACS_EXAMINER_LLM_API_KEY —— 否则重启后本屏会再次提示未配置。");
  } else if (config.examiner && !config.examiner.api_key_set) {
    notes.push("提问者 / 分析者未单独填 Key，将与模拟大脑**共用同一把**"
      + "（同厂商、同 base_url）；若要接到另一家，请单独填它的 Key。");
  }
  if (init.raw) {
    notes.push(`启动自我报告：${String(init.raw).replace(/\s+/g, " ").slice(0, 160)}`);
  }
  html("env-notes", notes.map((n) => `<div class="infobox" style="margin-top:6px">${esc(n)}</div>`).join(""));
}
