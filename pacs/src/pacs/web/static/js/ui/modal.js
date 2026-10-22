/** 配置门禁弹窗（规则 1：打开界面时要求配置 LLM API）。
 *
 * 行为约定（面向使用习惯）：
 * - 打开时按 `需要配置` 判定自动弹出；已配置或已确认离线则不再打扰；
 * - 两套 LLM 分组呈现（模拟大脑 / 提问者·分析者），Key 留空＝不修改；
 * - 提供显式「离线 mock 模式」，并把该选择持久化，避免每次刷新都弹；
 * - Esc 关闭（仅在已确认离线时允许），焦点自动落到第一个输入框。
 */

import { $, esc, html, withBusy } from "../core/dom.js";
import { api } from "../core/api.js";
import { store, update } from "../core/store.js";
import { prefs } from "../core/prefs.js";
import { toast } from "./toast.js";

export function gateNeeded(boot) {
  return Boolean(boot?.gate?.requires_setup) && !prefs.get("offlineAcknowledged");
}

export function openGate() {
  const config = store.config || {};
  const examiner = config.examiner || {};
  $("gate-b-provider").value = config.provider || "mock";
  $("gate-b-model").value = config.model || "";
  $("gate-b-base").value = config.base_url || "";
  $("gate-e-provider").value = examiner.provider || "mock";
  $("gate-e-model").value = examiner.model || "";
  $("gate-e-base").value = examiner.base_url || "";

  const reasons = store.boot?.gate?.reasons || [];
  html("gate-reasons", reasons.length
    ? `<div class="warnbox">当前缺失：${esc(reasons.join("；"))}</div>` : "");

  $("gate").classList.add("show");
  document.body.style.overflow = "hidden";
  setTimeout(() => $("gate-b-key")?.focus(), 30);
}

export function closeGate() {
  $("gate").classList.remove("show");
  document.body.style.overflow = "";
  // 广播一次「配置可能变了」：顶部那条离线提示按当前配置决定显隐，需要重新算一遍
  update({}, "config");
}

export function mountGate({ onEntered }) {
  $("gate-save").addEventListener("click", (event) => withBusy(event.currentTarget, "保存中…", async () => {
    const body = {
      provider: $("gate-b-provider").value,
      model: $("gate-b-model").value,
      base_url: $("gate-b-base").value,
      persist: true,
      persist_key: $("gate-persist").checked,
      examiner: {
        provider: $("gate-e-provider").value,
        model: $("gate-e-model").value,
        base_url: $("gate-e-base").value,
      },
    };
    if ($("gate-b-key").value) body.api_key = $("gate-b-key").value;
    if ($("gate-e-key").value) body.examiner.api_key = $("gate-e-key").value;
    try {
      const data = await api.post("/api/config", body);
      $("gate-b-key").value = "";
      $("gate-e-key").value = "";
      update({ config: data.config }, "config");
      const ready = data.config.configured?.ready;
      prefs.set("offlineAcknowledged", !ready);
      toast.ok(ready ? "配置已保存" : "已保存（尚未具备真实调用条件）");
      closeGate();
      await onEntered?.();
    } catch (error) {
      toast.error(error.message);
    }
  }));

  $("gate-offline").addEventListener("click", (event) => withBusy(event.currentTarget, "切换中…", async () => {
    try {
      const data = await api.post("/api/config", {
        provider: "mock", model: "gpt-4o-mini", examiner: { provider: "mock" },
      });
      update({ config: data.config }, "config");
      prefs.set("offlineAcknowledged", true);
      closeGate();
      await onEntered?.();
    } catch (error) {
      toast.error(error.message);
    }
  }));

  $("gate-close").addEventListener("click", () => {
    // 只有已经确认过离线模式时才允许关闭，否则门禁形同虚设
    if (prefs.get("offlineAcknowledged")) closeGate();
    else toast.warn("请先保存真实 API 配置，或选择「离线 mock 模式进入」");
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && $("gate").classList.contains("show")) {
      if (prefs.get("offlineAcknowledged")) closeGate();
    }
  });
}
