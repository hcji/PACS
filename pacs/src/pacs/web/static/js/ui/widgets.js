/** 可复用展示组件：KPI 网格、状态行、空态、进度条、表格骨架。
 *
 * 都是「纯渲染 + 少量 DOM 句柄」的小件，视图只负责给数据。
 */

import { $, esc, html } from "../core/dom.js";

/**
 * KPI 网格。
 * @param {string} targetId 容器 id
 * @param {Array<[label, value, sublabel?, options?]>} rows
 *   options.focus=true 时高亮该卡（用于「当前大脑」这类上下文）
 */
export function renderKpi(targetId, rows) {
  const markup = rows
    .map(([label, value, sub, options = {}]) => {
      const small = String(value ?? "—").length > 8 ? " sm" : "";
      return `<div class="kpi${options.focus ? " is-focus" : ""}">
        <b class="${small.trim()}">${esc(value ?? "—")}</b>
        <span>${esc(label)}${sub ? ` · ${esc(sub)}` : ""}</span>
      </div>`;
    })
    .join("");
  html(targetId, markup);
}

export function status(targetId, message = "", kind = "") {
  const node = $(targetId);
  if (!node) return;
  node.textContent = message;
  node.className = `status${kind ? ` ${kind}` : ""}`;
  node.setAttribute("aria-live", "polite");
}

/** 空态：统一文案结构（说明 + 下一步该做什么），比「暂无数据」有用。 */
export function empty(containerId, { title = "暂无数据", hint = "", action } = {}) {
  const node = $(containerId);
  if (!node) return;
  node.innerHTML = `<div class="empty"><b>${esc(title)}</b>${esc(hint)}</div>`;
  if (action) {
    const button = document.createElement("button");
    button.className = "btn ghost sm";
    button.textContent = action.label;
    button.addEventListener("click", action.onClick);
    node.querySelector(".empty").append(button);
  }
}

/** 表格骨架：`columns` 为表头文本；`rows` 为已经转义好的行 HTML 数组。 */
export function renderTable(targetId, columns, rows, { emptyText = "（无数据）" } = {}) {
  const head = columns.map((c) => `<th>${esc(c)}</th>`).join("");
  const body = rows.length ? rows.join("") : `<tr><td class="empty-cell" colspan="${columns.length}">${esc(emptyText)}</td></tr>`;
  html(targetId, `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`);
}

/** 进度条：同时更新宽度与 aria 属性（读屏可感知）。 */
export function progress(targetId, value) {
  const node = $(targetId);
  if (!node) return;
  const clamped = Math.max(0, Math.min(1, Number(value) || 0));
  node.style.width = `${(clamped * 100).toFixed(1)}%`;
  node.parentElement?.setAttribute("aria-valuenow", String(Math.round(clamped * 100)));
}
