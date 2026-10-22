/** 后台任务：统一轮询与进度回调。
 *
 * 服务端所有耗时动作（注入/初始化、定量运行、对话探针、分析者重打分）都返回 `job_id`，
 * 由 `GET /api/job` 提供 `{status, progress, total, message, transcript?, result?}`。
 * 界面只需要「进度回调 + 拿到最终结果」，因此把轮询收敛成一个函数。
 */

import { api } from "./api.js";

/** 轮询间隔：任务通常 <1s 完成，故前几秒用更短的间隔，之后放宽以省请求。 */
const FAST_MS = 400;
const SLOW_MS = 1200;
const FAST_TICKS = 10;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function pollJob(jobId, { onTick, signal, timeoutMs = 30 * 60 * 1000 } = {}) {
  const startedAt = Date.now();
  for (let tick = 0; ; tick += 1) {
    if (signal?.aborted) throw new DOMException("已取消", "AbortError");
    const { job } = await api.get(`/api/job?job_id=${encodeURIComponent(jobId)}`);
    onTick?.(job);
    if (job.status === "done" || job.status === "error") return job;
    if (Date.now() - startedAt > timeoutMs) {
      throw new Error(`任务超时（已等待 ${Math.round(timeoutMs / 60000)} 分钟）`);
    }
    await sleep(tick < FAST_TICKS ? FAST_MS : SLOW_MS);
  }
}

/** 进度比例（0~1），total 未知时返回 0。 */
export function ratio(job) {
  return job?.total ? Math.max(0, Math.min(1, job.progress / job.total)) : 0;
}
