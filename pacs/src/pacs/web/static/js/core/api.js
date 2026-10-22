/** HTTP 层：统一请求、统一错误信封、统一 toast 反馈。
 *
 * 服务端契约：成功 `{ok:true, ...}`；失败 `{ok:false, error:"..."}` 且带 HTTP 状态码。
 * 这里把它收敛成两种行为：成功直接返回数据；失败抛 Error（消息可直接展示给用户），
 * 需要时再自动弹一条 toast —— 视图不必各自处理。
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function request(path, { method = "GET", body, timeoutMs = 120000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: JSON_HEADERS,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timer);
    throw new ApiError(
      error.name === "AbortError" ? `请求超时（${Math.round(timeoutMs / 1000)} 秒）` : `网络错误：${error.message}`,
      0, null,
    );
  }
  clearTimeout(timer);

  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(`响应不是 JSON（HTTP ${response.status}）`, response.status, null);
  }
  if (!response.ok || data.ok === false) {
    throw new ApiError(data.error || `HTTP ${response.status}`, response.status, data);
  }
  return data;
}

export const api = {
  get: (path, options) => request(path, options),
  post: (path, body, options) => request(path, { ...options, method: "POST", body }),
};

/** 查询串构造：丢掉空值，避免出现 `?dim=&age=` 这类噪音参数。 */
export function query(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "" || value === false) continue;
    search.set(key, value === true ? "1" : String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}
