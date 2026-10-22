/** 全局状态：唯一数据源 + 订阅通知。
 *
 * 约定：视图**不直接**互相调用；改状态 → 订阅者（侧栏、顶栏、当前视图）自动更新。
 * 这样「切换大脑」这类事件只需要改一次 store，所有与会话上下文相关的界面自会跟上。
 */

const listeners = new Set();

export const store = {
  /** 会话级只读上下文（启动时拉取一次） */
  ready: false,
  boot: null,           // /api/boot 原始载荷
  ontology: null,       // /api/ontology
  config: null,         // /api/config
  goals: [],            // 观察目标维度（来自 boot）

  /** 可变的会话状态 */
  brains: [],
  activeBrain: null,
  itemSets: [],

  /** 视图态（各视图自己的临时数据，放这里便于订阅与调试） */
  theta: null,
  promptSteps: null,
  bank: { total: 0, items: [], stats: null, rules: null, meta: null, adminZh: {}, dims: {} },
  bankSelected: new Set(),
};

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** 提交一批变更并广播。`patch` 里的键直接并入 store。 */
export function update(patch, reason = "") {
  Object.assign(store, patch);
  for (const listener of listeners) {
    try {
      listener(reason, store);
    } catch (error) {
      console.error("[store] 订阅者报错", error);
    }
  }
}

/** 维度字面量：来自 /api/ontology 的 phenotype_dims，避免前端写死。
 *
 * 对照通道（general_cognition）不在这份词表里，但它是计分通道 —— 读数表里会出现它，
 * 因此也要有中文名，否则界面会露出英文 id。
 */
export function dimLabel(dim) {
  const raw = store.ontology?.phenotype_dims?.[dim]
    || (store.ontology?.control_channel?.id === dim ? store.ontology?.control_channel?.label : "");
  if (!raw) return dim;
  return String(raw).replace(/^[①②③④⑤⑥⑦⑧⑨⑩]\s*/, "");
}

/** 这个通道是不是表型维度（对照通道不是：它不进雷达图）。 */
export function isPhenotypeDim(dim) {
  return Object.prototype.hasOwnProperty.call(store.ontology?.phenotype_dims || {}, dim);
}

export function dimsAll() {
  return Object.keys(store.ontology?.phenotype_dims || {});
}

/** 观察/评分的通道 = 6 个表型维度 + 对照通道（它不是表型维度，但必须被问到、被评分）。 */
export function scoredChannels() {
  const dims = dimsAll();
  const control = store.ontology?.control_channel?.id;
  return control && !dims.includes(control) ? [...dims, control] : dims;
}


export function primitives() {
  return store.ontology?.primitives || [];
}

export function primName(id) {
  return primitives().find((p) => p.id === id)?.name_zh || id;
}

/** 一组 primitive id → 中文名列表（原始 id 由调用方放进 title 备查）。 */
export function primNames(ids) {
  return (ids || []).map((id) => primName(id));
}

/** 扰动描述：**中文过程名 + 两位完整度**，与 ② 扰动注入、④ 机制分析、CLI 同一套措辞。
 *
 * 界面主语言是中文过程名（`perturb.js` 的 θ 表也是这个约定），因此这里不输出
 * `information_seeking_under_uncertainty=0.20` 这类英文 id；需要追溯时用
 * `perturbationIds()` 放进 `title` 提示。 */
export function perturbationText(impaired, genericDegradation = 0) {
  const parts = Object.entries(impaired || {})
    .map(([id, value]) => `${primName(id)} ${Number(value).toFixed(2)}`);
  const generic = Number(genericDegradation || 0);
  if (generic > 0) parts.push(`通用退化 ${generic.toFixed(2)}`);
  return parts.join("、");
}

/** 同一批扰动的 id 形式 —— 只用于 title 提示，保持可追溯但不占界面版面。 */
export function perturbationIds(impaired, genericDegradation = 0) {
  const parts = Object.entries(impaired || {})
    .map(([id, value]) => `${id}=${Number(value).toFixed(2)}`);
  const generic = Number(genericDegradation || 0);
  if (generic > 0) parts.push(`generic_degradation=${generic.toFixed(2)}`);
  return parts.join("、") || "intact";
}

export function brainLabel(brain) {
  return brain ? brain.name || brain.theta_id : "无";
}
