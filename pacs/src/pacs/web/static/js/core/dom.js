/** DOM 与事件小工具。所有模块只通过这里操作 DOM，避免各处重复造轮子。 */

/** 按 id 取元素。
 *
 * 容错：同时接受 `"adj-table"` 与 `"#adj-table"` 两种写法 —— 视图代码里两种风格都会出现，
 * 若只认一种，另一种会静默返回 null，然后在 `null.querySelectorAll(...)` 处才炸出来，
 * 排查成本很高。视图内的 id 全局唯一，故这里直接用 document 查找。
 */
export const $ = (id) => document.getElementById(String(id).replace(/^#/, ""));

/** 创建元素：el("div", {class:"card"}, [child, "text"]) */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** HTML 转义。凡是把外部数据拼进 innerHTML，必须先过这里。 */
export function esc(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/** 事件委托：在容器上按选择器绑定，避免为每一行动态绑定。 */
export function delegate(root, selector, type, handler) {
  root.addEventListener(type, (event) => {
    const target = event.target.closest(selector);
    if (target && root.contains(target)) handler(event, target);
  });
}

/** 写 HTML 到元素。 */
export function html(target, markup) {
  const node = typeof target === "string" ? $(target) : target;
  if (node) node.innerHTML = markup;
  return node;
}

/** 显示/隐藏。 */
export function show(target, visible = true) {
  const node = typeof target === "string" ? $(target) : target;
  if (node) node.style.display = visible ? "" : "none";
  return node;
}

/** 按钮忙碌态：禁用 + 文案替换，避免重复提交（长任务里这是最容易出的体验问题）。 */
export async function withBusy(button, label, task) {
  const node = typeof button === "string" ? $(button) : button;
  if (!node) return task();
  const original = node.textContent;
  node.disabled = true;
  node.dataset.busy = "1";
  if (label) node.textContent = label;
  try {
    return await task();
  } finally {
    node.disabled = false;
    delete node.dataset.busy;
    node.textContent = original;
  }
}
