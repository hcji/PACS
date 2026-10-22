/** 展示格式化：数值、分档颜色、徽章。集中一处，避免各视图口径不一致。 */

/** 保留小数的展示；缺失值统一显示「—」，不用 0 冒充。 */
export function pct(value, digits = 2) {
  return value === null || value === undefined || value === "" ? "—" : Number(value).toFixed(digits);
}

/** 表型得分 → CSS 变量色（与热图分档一致）。 */
export function dimColor(value) {
  if (value === null || value === undefined) return "var(--muted)";
  const v = Number(value);
  if (v < 0.45) return "var(--bad)";
  if (v < 0.70) return "var(--warn)";
  if (v < 0.85) return "#c9c26a";
  return "var(--good)";
}

const PASS = /pass|convergent|additive|numeric|入选|在范围内/;
const FAIL = /fail|divergent|superadditive|subadditive/;
const WEAK = /weak|partial|insufficient|n\/a|llm|缺/;

/** 判定词 → 徽章样式（判定来自后端统计，前端只做配色）。 */
function verdictClass(value) {
  const text = String(value ?? "");
  if (PASS.test(text)) return "b-pass";
  if (FAIL.test(text)) return "b-fail";
  if (WEAK.test(text)) return "b-weak";
  return "b-gap";
}

export function badge(value) {
  const text = String(value ?? "");
  return `<span class="badge ${verdictClass(text)}">${escapeBadge(text)}</span>`;
}

function escapeBadge(text) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
