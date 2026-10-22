/** 侧栏：主导航 + 当前大脑 + 大脑库（切换/重命名/删除/保存）。
 *
 * 为什么导航与大脑库放在一起：它们都是「当前会话上下文」的控制项 —— 用户要先知道
 * 自己在看哪个大脑，再决定去哪一屏操作。放进侧栏后四个界面共享同一上下文。
 */

import { $, esc, html, delegate } from "../core/dom.js";
import { api } from "../core/api.js";
import {
  store, subscribe, brainLabel, update, perturbationIds, perturbationText,
} from "../core/store.js";
import { toast } from "./toast.js";

export const SCREENS = [
  { id: "model", index: "①", label: "模型设置", purpose: "配置两套 LLM 接入与当前大脑的提示词上下文" },
  { id: "perturb", index: "②", label: "扰动注入", purpose: "生成或调整 θ，查看提示词构造过程并下发" },
  { id: "phenotype", index: "③", label: "表型观察", purpose: "观察当前大脑：读数（雷达图）/ 历史对话 / 你直接提问" },
];

let editingBrain = null;
let confirmingDelete = null;

export function renderNav(current) {
  html("nav", SCREENS.map((screen) => `
    <button class="nav-item" role="tab" data-screen="${screen.id}"
            aria-selected="${screen.id === current}"
            aria-controls="screen-${screen.id}">
      <span class="idx">${screen.index}</span><span>${esc(screen.label)}</span>
    </button>`).join(""));
  $("nav").setAttribute("role", "tablist");
  $("nav").setAttribute("aria-label", "主导航");
}

function renderSidebar() {
  const brain = store.activeBrain;
  if (brain) {
    const perturb = perturbationText(brain.impaired, brain.generic_degradation);
    html("side-current", `
      <div class="nm" style="font-size:var(--fs-md);font-weight:600;color:var(--accent2);margin-bottom:4px">${esc(brainLabel(brain))}</div>
      <div class="note mono">θ ${esc(brain.theta_id)}</div>
      <div class="note" title="${esc(perturbationIds(brain.impaired, brain.generic_degradation))}">扰动 ${esc(perturb || "无（完整心智）")}</div>
      <div class="note">来源 ${esc(brain.source || "")} · ${esc(brain.updated_at || "")}</div>`);
  } else {
    html("side-current", `<div class="note">无当前大脑：请注入扰动或保存一个 θ</div>`);
  }

  html("brain-list", store.brains.length
    ? store.brains.map((item) => {
      const active = store.activeBrain && item.brain_id === store.activeBrain.brain_id;
      const perturb = perturbationText(item.impaired, item.generic_degradation);
      const editing = editingBrain === item.brain_id;
      const confirming = confirmingDelete === item.brain_id;
      const name = editing
        ? `<input class="inline-edit" data-edit-for="${esc(item.brain_id)}" value="${esc(item.name)}">`
        : `<span class="nm" style="display:block">${active ? "● " : ""}${esc(item.name)}</span>`;
      const actions = editing
        ? `<button class="tiny" data-act="rename-save" data-id="${esc(item.brain_id)}">保存</button>
           <button class="tiny" data-act="rename-cancel">取消</button>`
        : confirming
          ? `<span class="note">确认删除？</span>
             <button class="tiny" data-act="delete-yes" data-id="${esc(item.brain_id)}">确认</button>
             <button class="tiny" data-act="delete-cancel">取消</button>`
          : `<button class="tiny" data-act="activate" data-id="${esc(item.brain_id)}">切换</button>
             <button class="tiny" data-act="rename" data-id="${esc(item.brain_id)}">重命名</button>
             <button class="tiny" data-act="delete" data-id="${esc(item.brain_id)}">删除</button>`;
      return `<div class="brain-item" data-brain="${esc(item.brain_id)}" style="border:1px solid ${active ? "var(--accent2)" : "var(--line)"};border-radius:var(--r-md);padding:7px 9px;margin-bottom:6px;background:var(--panel2);cursor:pointer">
          ${name}
          <span class="note mono" style="display:block;font-size:var(--fs-xs);word-break:break-all">${esc(item.theta_id)}</span>
          <span class="note" style="display:block;font-size:var(--fs-xs)" title="${esc(perturbationIds(item.impaired, item.generic_degradation))}">${esc(perturb || "完整心智")}</span>
          <span class="row tight" style="margin-top:5px">${actions}</span>
        </div>`;
    }).join("")
    : `<div class="note">库中无条目</div>`);

  const topbar = $("topbar-brain");
  if (topbar) topbar.textContent = brain ? brainLabel(brain) : "无";
}

export function mountSidebar({ onBrainChanged }) {
  const sidebar = $("sidebar");
  const nav = $("nav");
  nav.addEventListener("click", (event) => {
    const item = event.target.closest(".nav-item");
    if (item) location.hash = `#/${item.dataset.screen}`;
  });

  delegate(sidebar, "button[data-act]", "click", async (event, button) => {
    event.stopPropagation();
    await brainAction(button.dataset.act, button.dataset.id, onBrainChanged);
  });

  delegate(sidebar, ".brain-item", "click", async (event, card) => {
    if (event.target.closest("button") || event.target.closest("input")) return;
    if (store.activeBrain && card.dataset.brain === store.activeBrain.brain_id) return;
    await brainAction("activate", card.dataset.brain, onBrainChanged);
  });

  $("btn-sidebar-save").addEventListener("click", async () => {
    const name = $("brain-name").value.trim();
    if (!name) return toast.warn("请先为当前 θ 命名");
    try {
      const data = await api.post("/api/brains/save", { name });
      toast.ok(`已保存 θ=${data.brain.theta_id}`);
      $("brain-name").value = "";
      await refreshBrains();
    } catch (error) {
      toast.error(error.message);
    }
  });

  subscribe((reason) => {
    if (reason === "brains" || reason === "activeBrain") renderSidebar();
  });
}

async function brainAction(act, brainId, onBrainChanged) {
  const button = (label) => toast.info(label);
  try {
    if (act === "activate") {
      button("正在切换大脑并重新初始化…");
      const data = await api.post("/api/brains/activate", { brain_id: brainId });
      onBrainChanged?.(data);                    // 让②扰动注入视图刷新提示词与自我报告
      const job = await (await import("../core/job.js")).pollJob(data.job_id, {
        onTick: (tick) => tick.message && $("side-current")?.setAttribute("title", tick.message),
      });
      toast.ok(job.status === "done" ? "已切换大脑" : `切换完成：${job.message}`);
    } else if (act === "rename") {
      editingBrain = brainId; confirmingDelete = null; renderSidebar();
      const input = document.querySelector(`[data-edit-for="${brainId}"]`);
      input?.focus(); input?.select();
      return;
    } else if (act === "rename-cancel") {
      editingBrain = null; renderSidebar(); return;
    } else if (act === "rename-save") {
      const input = document.querySelector(`[data-edit-for="${brainId}"]`);
      const name = input?.value.trim();
      if (!name) return toast.warn("名称不能为空");
      await api.post("/api/brains/rename", { brain_id: brainId, name });
      editingBrain = null;
      toast.ok(`已重命名为「${name}」`);
    } else if (act === "delete") {
      confirmingDelete = brainId; editingBrain = null; renderSidebar(); return;
    } else if (act === "delete-cancel") {
      confirmingDelete = null; renderSidebar(); return;
    } else if (act === "delete-yes") {
      await api.post("/api/brains/delete", { brain_id: brainId });
      confirmingDelete = null;
      toast.ok("已删除（表型数据未受影响）");
    }
    await refreshBrains();
    onBrainChanged?.(null);
  } catch (error) {
    toast.error(error.message);
  }
}

export async function refreshBrains() {
  const data = await api.get("/api/brains");
  update({ brains: data.brains || [], activeBrain: data.active }, "brains");
  return data;
}

export function setSidebarOpen(open) {
  $("sidebar").classList.toggle("open", open);
  $("btn-sidebar").setAttribute("aria-expanded", String(open));
}
