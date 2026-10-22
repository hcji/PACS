/** 应用入口：启动流程 + 哈希路由 + 视图挂载。
 *
 * 路由约定：`#/<screen>`（model / perturb / phenotype）。
 * 好处（面向使用习惯）：刷新留在原界面、浏览器前进后退可用、可直接分享某屏链接。
 *
 * 挂载策略：每个界面在首次进入时才渲染，之后只做 `refresh()`；
 * 离开时保留 DOM（切回很快），但脏数据由各视图的 refresh 纠正。
 */

import { $ } from "./core/dom.js";
import { api } from "./core/api.js";
import { store, subscribe, update } from "./core/store.js";
import { prefs } from "./core/prefs.js";
import { toast } from "./ui/toast.js";
import * as sidebar from "./ui/sidebar.js";
import * as gate from "./ui/modal.js";

import * as modelView from "./views/model.js";
import * as perturbView from "./views/perturb.js";
import * as phenotypeView from "./views/phenotype.js";

const VIEWS = {
  model: modelView,
  perturb: perturbView,
  phenotype: phenotypeView,
};

const mounted = new Set();
/** 导航序号：并发导航（例如用户在加载中又点了另一个界面）时，
 *  只允许最后一次导航改动 DOM，否则先发起的调用可能在后面把界面切回去。 */
let navToken = 0;

function screenFromHash() {
  const raw = location.hash.replace(/^#\/?/, "").split("?")[0];
  return VIEWS[raw] ? raw : null;
}

function rootOf(screen) {
  return $(`screen-${screen}`);
}

/** 首次进入某界面：建容器、渲染模板、绑定事件。
 *  先登记再渲染，避免并发导航导致同一个界面被挂载两次（会产生重复 id）。 */
function mount(screen) {
  if (mounted.has(screen)) return;
  mounted.add(screen);
  const view = VIEWS[screen];
  const root = document.createElement("section");
  root.id = `screen-${screen}`;
  root.className = "screen";
  root.setAttribute("role", "tabpanel");
  root.setAttribute("aria-labelledby", `nav-${screen}`);
  // 页面头：标题 + 目的 + 主操作（各界面统一结构）
  root.innerHTML = `
    <div class="page-head">
      <div>
        <h2 class="title"><span class="mono" style="color:var(--accent)">${escIndex(view.index)}</span> ${view.label}</h2>
        <div class="purpose">${purposeOf(screen)}</div>
      </div>
    </div>
    <div data-view-body></div>`;
  root.querySelector("[data-view-body]").innerHTML = view.template();
  $("view").append(root);
  view.bind();
}

function escIndex(index) {
  return String(index || "");
}

function purposeOf(screen) {
  return sidebar.SCREENS.find((s) => s.id === screen)?.purpose || "";
}

async function navigate(screen, { replace = false } = {}) {
  const target = VIEWS[screen] ? screen : "model";
  const token = ++navToken;
  mount(target);
  for (const id of Object.keys(VIEWS)) {
    const root = rootOf(id);
    if (root) root.hidden = id !== target;
  }
  sidebar.renderNav(target);
  prefs.set("screen", target);
  $("topbar-screen").textContent = VIEWS[target].label;
  sidebar.setSidebarOpen(false);
  window.scrollTo({ top: 0, behavior: replace ? "auto" : "smooth" });

  try {
    await VIEWS[target].refresh?.();
  } catch (error) {
    console.error(`[${target}] 加载失败`, error);
    if (token === navToken) toast.error(`${VIEWS[target].label}加载失败：${error.message}`);
  }
  // 期间用户可能已经切到别的界面 —— 那时不再覆盖状态
  if (token === navToken) $("topbar-screen").textContent = VIEWS[target].label;
}

/** 大脑发生变化：通知所有已挂载的视图（各自决定要不要重算/清空）。 */
function broadcastBrainChanged(payload) {
  for (const screen of mounted) {
    VIEWS[screen].onBrainChanged?.(payload);
  }
}

/** 顶部常驻提示：当前这批读数**不是** LLM 涌现的（mock / 未配置 Key）。
 *
 * 判定只看**现在的**配置（`boot.gate.requires_setup` = 实际不可真实调用），
 * 不看用户当初是否点过「离线 mock 模式进入」：那个标记是历史，不是现状 ——
 * 按它显示会出现「已经填好 Key 了，横幅还在说我在用 mock」。
 */
function renderOfflineBanner() {
  const host = $("banner-host");
  if (!host) return;
  // 以**实时配置**为准（`/api/config` 在每次保存后都会回传最新状态）；
  // 配置还没拉到时退回启动时的 gate 判定。
  const config = store.config;
  const ready = config
    ? Boolean(config.configured?.ready)
    : !store.boot?.gate?.requires_setup;
  const gateOpen = Boolean($("gate")?.classList.contains("show"));
  let banner = host.querySelector(".banner");
  if (ready || gateOpen) {
    banner?.remove();
    return;
  }
  if (!banner) {
    banner = document.createElement("div");
    banner.className = "banner";
    host.append(banner);
  }
  const missing = [];
  if (config) {
    if (!config.configured?.brain) missing.push("模拟大脑尚未配置可用的 API");
    if (!config.configured?.examiner) missing.push("提问者 / 分析者尚未配置可用的 API");
  } else {
    missing.push(...(store.boot?.gate?.reasons || []));
  }
  const why = missing.length ? `（${missing.join("；")}）` : "";
  banner.textContent = "当前不是真实 LLM：表型结构由规则式 mock 构造，不是 LLM 涌现的。"
    + `真实结论请在「① 模型设置」里配置真实 API${why}。`;
}

async function bootstrap() {
  // 1) 上下文：本体 / 配置 / 启动状态
  // 注意：本项目不做任何「构造属性 → 测量维度」的映射展示（预设对应已被明令禁止），
  // 因此这里不再拉取任何映射端点。
  const [ontology, boot] = await Promise.all([
    api.get("/api/ontology"),
    api.get("/api/boot"),
  ]);
  update({ ontology, boot, goals: boot.goals || [], config: boot.config || null }, "bootstrap");

  // 2) 离线模式常驻提示：**按当前配置实时判定**，而不是按「用户曾经点过什么」
  renderOfflineBanner();
  // 配置一变（在 ① 里保存真实 API、或在门禁里切回 mock）就要重算 ——
  // 只在启动时算一次的话，配好 API 之后那条「当前为离线 mock 模式」会一直挂在那儿。
  subscribe(() => renderOfflineBanner());

  // 3) 大脑库与侧栏（先挂载交互与订阅，再拉数据）
  sidebar.renderNav(screenFromHash() || prefs.get("screen") || "model");
  sidebar.mountSidebar({ onBrainChanged: broadcastBrainChanged });
  await sidebar.refreshBrains();

  // 4) 路由
  window.addEventListener("hashchange", () => navigate(screenFromHash() || "model"));
  $("btn-sidebar").addEventListener("click", () => {
    sidebar.setSidebarOpen(!$("sidebar").classList.contains("open"));
  });

  // 5) 门禁
  gate.mountGate({ onEntered: () => navigate(screenFromHash() || "model", { replace: true }) });

  const initial = screenFromHash() || prefs.get("screen") || "model";
  if (!screenFromHash()) history.replaceState(null, "", `#/${initial}`);
  await navigate(initial, { replace: true });

  if (gate.gateNeeded(boot)) gate.openGate();
}

// 视图在需要时可以请求「通知其它视图大脑已变」
window.addEventListener("pacs:brain-changed", (event) => broadcastBrainChanged(event.detail));

bootstrap().catch((error) => {
  document.body.insertAdjacentHTML("afterbegin", `<div class="warnbox" style="margin:16px">初始化失败：${error.message}</div>`);
});
