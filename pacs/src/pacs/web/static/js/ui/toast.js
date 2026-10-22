/** 全局提示（右上角堆叠）。
 *
 * 为什么要有它：原来每个卡片各有一个状态行，用户切屏或滚动后就看不到反馈了。
 * 跨屏的重要结果（大脑已切换、配置已保存、任务失败）走 toast，卡片内只留表单级提示。
 */

import { el } from "../core/dom.js";

const DURATION = { ok: 3200, info: 3600, warn: 5000, err: 8000 };

let host = null;

function container() {
  if (!host) {
    host = el("div", { class: "toasts", role: "status", "aria-live": "polite" });
    document.body.append(host);
  }
  return host;
}

export function toast(message, kind = "info") {
  if (!message) return;
  const node = el("div", { class: `toast ${kind}`, text: String(message) });
  node.addEventListener("click", () => node.remove());
  container().append(node);
  setTimeout(() => node.remove(), DURATION[kind] || DURATION.info);
  return node;
}

toast.ok = (message) => toast(message, "ok");
toast.success = (message) => toast(message, "ok");
toast.info = (message) => toast(message, "info");
toast.warn = (message) => toast(message, "warn");
toast.error = (message) => toast(message, "err");
