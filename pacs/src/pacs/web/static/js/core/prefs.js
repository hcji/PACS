/** 用户首选项：localStorage 持久化。
 *
 * 目的（面向使用习惯）：刷新 / 重开浏览器后，用户上次停在哪一屏、用的什么轮数上限、
 * 筛选了什么条目、离线模式是否已确认，都不该重来一遍。
 * 只持久化「界面偏好」，不写任何密钥（Key 始终只在服务端内存/配置里）。
 */

const KEY = "pacs.prefs.v1";

const DEFAULTS = {
  screen: "model",          // 上次所在界面
  offlineAcknowledged: false, // 是否已显式选择过「离线 mock 模式」
  sidebarOpen: false,       // 窄屏抽屉是否展开
  // ③ 的轮数上限（对话探针：基础轮数走完后按「每通道 ≥5 个问题」补问的上限）。
  // 心智年龄**不是**偏好：它是固定常量 36 个月，界面不再提供入口。
  maxTurns: 100,
  probeDrawerOpen: false,   // ③ 的「对话过程」抽屉是否展开（默认收起）
  bankFilters: { dim: "", primitive: "", admin: "", probeMode: "", age: "", q: "", probeOnly: false },
  advancedOpenBank: false,  // 进阶工具（内置条目库）是否展开
  advancedOpenUpload: false,// 进阶工具（自定义条目上传）是否展开
};

let cache = null;

function readDisk() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "{}");
  } catch {
    return {};
  }
}

function load() {
  if (cache) return cache;
  const raw = readDisk();
  cache = { ...DEFAULTS, ...raw, bankFilters: { ...DEFAULTS.bankFilters, ...(raw.bankFilters || {}) } };
  return cache;
}

/** 写盘前先合并磁盘上的现值，只覆盖本次变更的键。
 *
 * 为什么不能直接写内存副本：页面里可能有尚未完成的异步流程（例如启动时那次
 * `navigate()` 还在等网络），它持有的旧副本一旦写盘，就会把用户刚做的选择（例如
 * 「已确认离线模式」）覆盖掉。逐键覆盖可以避免这类竞态。
 */
function persist(delta) {
  try {
    const merged = { ...readDisk(), ...delta };
    localStorage.setItem(KEY, JSON.stringify(merged));
  } catch {
    /* 隐私模式或配额满：偏好丢失不影响功能，静默降级 */
  }
}

export const prefs = {
  all: () => ({ ...load() }),
  get: (key) => load()[key],
  set(key, value) {
    cache = { ...load(), [key]: value };
    persist({ [key]: value });
    return value;
  },
  patch(partial) {
    cache = { ...load(), ...partial };
    persist(partial);
    return cache;
  },
  bankFilters: () => ({ ...load().bankFilters }),
  setBankFilters(partial) {
    const next = { ...load().bankFilters, ...partial };
    cache = { ...load(), bankFilters: next };
    persist({ bankFilters: next });
    return next;
  },
  reset() {
    cache = { ...DEFAULTS };
    persist(DEFAULTS);
  },
};
