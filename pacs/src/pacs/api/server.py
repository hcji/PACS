"""本地 Web 服务（标准库 http.server，零第三方依赖）。

为可视化界面提供 JSON API：
- 只读：本体 / 映射 / 配置 / 运行记录 / 表型 / 分析 / Atlas
- 动作：保存配置、预览提示词、初始化大脑、启动扰动运行、导出、冻结
- 耗时动作（运行 / 初始化）走后台线程 + 进度轮询，界面不阻塞

安全：
- 默认只绑定 127.0.0.1（本机）；
- **API Key 永不出现在任何 GET 响应里**（只回报「是否已配置」）；
- 下载接口做路径穿越防护，只允许项目目录内的文件。
"""

from __future__ import annotations

import json
import mimetypes
import random
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..agents.dialogue import (
    ABSOLUTE_MAX_TURNS,
    DEFAULT_N_TURNS,
    MAX_N_TURNS,
    DialogueSession,
    analyze_dialogue,
    new_session_id,
    run_dialogue,
)
from ..agents.virtual_agent import VirtualAgent
from ..analysis.brain_report import build_brain_report, feedback_has_reading
from ..phenotype import (
    CONTROL_DIM,
    CONTROL_DIM_ZH,
    DIM_ANCHORS,
    PHENOTYPE_DIMS,
    SCORED_CHANNELS,
)
from ..scales import parse_item_set
from ..brain import examiner_bank as eb
from ..brain.architecture import build_brain_prompt, build_brain_prompt_steps, default_goals
from ..brain.perturb import (
    DEFAULT_DOSE_LEVELS,
    compose,
    from_values,
    generic_degradation_control,
    intact,
    sample_design,
    sample_lhs,
    single,
)
from .. import (
    DEFAULT_AGE_MONTHS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TEXT_LIMIT,
    MAX_TEXT_LIMIT,
    QUESTIONS_PER_CHANNEL,
)
from ..config import Settings
from ..freeze import build_freeze, save_freeze, verify_freeze
from ..llm.provider import ChatMessage, LLMError, get_examiner_provider, get_provider
from ..llm.registry import profile_for
from ..ontology.assays import ASSAYS, primitives_without_assay
from ..ontology.primitives import CANDIDATES, FAMILIES, PRIMITIVES, validate_ontology
from ..ontology.systems import SYSTEMS
from ..report import build_atlas, render_atlas
from ..store.db import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 后台任务表：job_id -> 进度状态（进程内，够用）
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 3600

#: 启动自动初始化的状态（规则 3：打开界面时「正常大脑」应当已经上线）。
BOOT: dict[str, Any] = {
    "status": "pending",
    "message": "尚未开始初始化",
    "active_brain": None,
    "init": None,
}
BOOT_LOCK = threading.Lock()

#: 数据库里记录「当前大脑」的键
META_ACTIVE_BRAIN = "active_brain_id"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _cleanup_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with JOBS_LOCK:
        for key in [k for k, v in JOBS.items() if v.get("created", 0) < cutoff]:
            JOBS.pop(key, None)


def _job_update(job_id: str, **fields: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {"created": time.time()})
        job.update(fields)


def _job_get(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def _perturbation_from_spec(spec: dict) -> Any:
    """把界面传来的扰动描述转成 `Perturbation`。"""
    kind = spec.get("kind", "intact")
    if kind == "intact":
        return intact()
    if kind == "generic":
        level = spec.get("generic_level", spec.get("dose", 0.5))
        return generic_degradation_control(float(level))
    if kind == "single":
        return single(str(spec["primitive"]), float(spec.get("dose", 0.4)))
    if kind in ("custom", "values"):
        values = {k: float(v) for k, v in (spec.get("values") or {}).items()}
        generic = float(spec.get("generic_degradation") or 0.0)
        if not values and generic <= 0:
            return intact()
        return from_values(
            values, design="custom", generic_degradation=generic, note="界面自定义扰动"
        )
    raise ValueError(f"未知扰动类型：{kind}")


def _designs_from_spec(spec: dict) -> list:
    mode = spec.get("mode", "single")
    doses = tuple(float(d) for d in (spec.get("doses") or DEFAULT_DOSE_LEVELS))
    seed = int(spec.get("seed", 42))
    max_pairs = spec.get("max_pairs")
    if mode == "intact":
        return [intact(seed=seed)]
    if mode == "all":
        designs = sample_design("single", doses=doses, seed=seed)
        designs += sample_design("pair", pair_dose=float(spec.get("pair_dose", 0.2)), seed=seed,
                                 max_pairs=max_pairs)
        for rep in range(5):   # 对照条件重复 5 次并取平均，避免单次噪声决定 Control 1 判定
            designs.append(
                generic_degradation_control(
                    float(spec.get("generic_level", 0.5)), seed=seed + rep
                )
            )
        return designs
    if mode == "pair":
        return sample_design("pair", pair_dose=float(spec.get("pair_dose", 0.2)), seed=seed,
                             max_pairs=max_pairs)
    if mode == "lhs":
        return sample_design("lhs", seed=seed, n_samples=int(spec.get("n_samples", 24)))
    return sample_design("single", doses=doses, seed=seed)


def _apply_limit(designs: list, limit: int | None) -> list:
    """按上限裁剪设计，但**保持代表性**，而不是从前面硬截断。

    为什么不能直接 `designs[:limit]`：设计列表的前段全是单因素扰动，硬截断会
    ① 丢掉全部组合扰动（Control 2 / 交互效应无从评估）；② 丢掉通用退化对照
    （Control 1 变成「缺失」）。用户为了控成本设了 limit，却在不知情的情况下
    失去了两个关键对照，Atlas 会报「incomplete」。
    因此这里：先保留完整心智基线与通用退化对照，其余按确定性等距抽样。
    """
    if not limit or limit >= len(designs):
        return designs
    keep: list = []
    rest: list = []
    for d in designs:
        if d.is_intact or d.generic_degradation > 0:
            keep.append(d)
        else:
            rest.append(d)
    budget = max(0, limit - len(keep))
    if budget == 0:
        return keep
    if len(rest) <= budget:
        return keep + rest
    stride = len(rest) / budget
    sampled = [rest[min(len(rest) - 1, int(i * stride))] for i in range(budget)]
    # 去重（等距抽样可能撞到同一下标）
    seen: set[str] = set()
    unique: list = []
    for d in sampled:
        if d.theta_id not in seen:
            seen.add(d.theta_id)
            unique.append(d)
    return keep + unique


#: 「历史对话」一次列出的条数上限。群体运行一次就会产生几十到上百段对话。
DEFAULT_SESSION_LIMIT = 200
def _text_limit_of(body: dict) -> int:
    """问题与回答的字数上限（默认 100；超出只告警，不截断）。

    显式传了非法值（例如 0）要**报错**，不能静默换成默认值 —— 那会让人以为限制生效了。
    只有「没传」或传 `None` 才用默认。
    """
    raw = body.get("text_limit")
    limit = DEFAULT_TEXT_LIMIT if raw is None else int(raw)
    if not (1 <= limit <= MAX_TEXT_LIMIT):
        raise ValueError(f"字数上限必须在 1..{MAX_TEXT_LIMIT} 之间，收到 {limit}")
    return limit


def _build_atlas_from_store(
    store: Store,
    *,
    run_id: str | None = None,
    model_id: str | None = None,
    endpoint: str = "social_communication_pragmatics",
    threshold: float = 0.70,
    dose: float | None = None,
):
    profiles = store.load_profiles(run_id=run_id, model_id=model_id)
    if not profiles:
        return None
    return _build_atlas(profiles, endpoint=endpoint, threshold=threshold, dose=dose)


def _build_atlas(
    profiles: list,
    *,
    endpoint: str = "social_communication_pragmatics",
    threshold: float = 0.70,
    dose: float | None = None,
):
    model_profiles = [
        profile_for(*(m.split(":", 1) if ":" in m else (m, m)))
        for m in sorted({p.model_id for p in profiles})
    ]
    return build_atlas(
        profiles,
        model_version=DEFAULT_MODEL_VERSION,
        endpoint_dim=endpoint,
        model_profiles=model_profiles,
        threshold=threshold,
        dose=dose,
    )


class Conflict(RuntimeError):
    """请求的前置条件不满足（对应 HTTP 409）—— 例如当前大脑还没被测量过。"""


# ---------------------------------------------------------------------------
# 大脑库（保存 / 切换）与启动自动初始化
# ---------------------------------------------------------------------------


def _boot_update(**fields: Any) -> None:
    with BOOT_LOCK:
        BOOT.update(fields)


def _boot_get() -> dict:
    with BOOT_LOCK:
        return dict(BOOT)


def _new_brain_id() -> str:
    return f"brain_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def brain_payload(name: str, perturbation, *, source: str = "user", note: str = "") -> dict:
    """大脑库记录：θ 本体 + 可读描述（界面直接渲染，避免前端重复推导）。"""
    return {
        "name": name,
        "source": source,
        "note": note,
        "describe": perturbation.describe(),
        "design": perturbation.design,
        "values": {k: round(float(v), 4) for k, v in perturbation.values.items()},
        "generic_degradation": round(float(perturbation.generic_degradation), 4),
        "theta_id": perturbation.theta_id,
        "impaired": perturbation.active_perturbations(),
    }


def perturbation_of_brain(record: dict):
    """从大脑记录还原 `Perturbation`（未列出的 primitive 视为完整）。"""
    payload = record.get("payload") or {}
    return from_values(
        {k: float(v) for k, v in (payload.get("values") or {}).items()},
        design=str(payload.get("design") or "custom"),
        generic_degradation=float(payload.get("generic_degradation") or 0.0),
        note=str(payload.get("note") or record.get("name") or ""),
    )


def _brain_summary(record: dict | None) -> dict | None:
    """给界面用的大脑摘要（不含 15 维完整 θ，避免每个响应都很大）。"""
    if not record:
        return None
    payload = record.get("payload") or {}
    return {
        "brain_id": record.get("brain_id"),
        "name": record.get("name"),
        "source": record.get("source"),
        "theta_id": record.get("theta_id"),
        "describe": payload.get("describe", ""),
        "impaired": payload.get("impaired") or {},
        "generic_degradation": payload.get("generic_degradation", 0.0),
        "values": payload.get("values") or {},
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }


def _sessions_of_brain(store, record: dict | None) -> list[dict]:
    """挑出**属于这具大脑**的对话会话。

    归属规则（两种都要认）：
    - `brain_id` 相同 —— 大脑库里的一次记录（界面上的观察走的都是这条）；
    - `brain_id` 为空且 `theta_id` 相同 —— 命令行留下、尚未绑定大脑库的会话。
      θ 相同即同一具心智，把它算进来才不至于让 CLI 跑过的那段对话「消失」。

    `record` 为空（还没有当前大脑）时返回空列表：**没有主体就没有「它的历史」**。
    历史对话与读数卡锚定的是同一个主体（当前大脑），两处口径必须一致 ——
    否则会出现「读数说这具心智没被观察过，历史列表里却列着别的几十段对话」。
    """
    if not record:
        return []
    return store.list_brain_sessions(
        record.get("brain_id") or "", record.get("theta_id") or "", limit=500
    )


def register_brain(
    store: Store,
    name: str,
    perturbation,
    *,
    source: str = "user",
    note: str = "",
    activate: bool = True,
) -> dict:
    """保存（或按 θ 去重更新）+ 可选设为当前大脑。"""
    brain_id = store.save_brain(
        _new_brain_id(), name, perturbation.theta_id,
        brain_payload(name, perturbation, source=source, note=note), source=source,
    )
    if activate:
        store.set_meta(META_ACTIVE_BRAIN, brain_id)
    store.audit("brain_save", f"{brain_id} {name} {perturbation.theta_id}")
    return store.load_brain(brain_id) or {}


def _launch_auto_init(settings: Settings, store: Store) -> threading.Thread:
    """启动时自动用提示词初始化一个**正常大脑**（规则 3），结果缓存进 `BOOT`。

    为什么启动就用完整心智：每次会话都从一个有明确定义的基线开始，「被扰动的大脑」
    必须由用户在扰动注入界面显式注入 —— 开机默认就该是正常大脑。
    """

    def worker() -> None:
        try:
            _boot_update(status="running", message="正在用提示词初始化模拟大脑（完整心智）…")
            perturbation = intact()
            provider = get_provider(settings.llm)
            agent = VirtualAgent(
                provider, perturbation,
                model_id=f"{provider.name}:{provider.model}",
            )
            result = agent.initialize()
            record = register_brain(
                store, "完整心智（启动基线）", perturbation,
                source="intact", note="启动时自动初始化的正常大脑", activate=True,
            )
            store.audit("boot_init", f"{record.get('brain_id')} ok={result.ok}")
            _boot_update(
                status="ready",
                message="正常大脑已初始化" if result.ok else "初始化未返回 SELF_STATE: OK（不必然是失败）",
                active_brain=_brain_summary(record),
                init={
                    "ok": result.ok,
                    "self_report": result.self_report,
                    "raw": result.raw,
                    "theta_id": result.theta_id,
                    "model_id": result.model_id,
                    "fingerprint": agent.fingerprint,
                    "ts": result.ts,
                },
            )
        except Exception as exc:      # 未配置 Key / 网络不通都走这里，界面据此提示去配置
            _boot_update(
                status="error",
                message=f"{type(exc).__name__}: {exc}",
                init=None,
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# 请求处理
# ---------------------------------------------------------------------------


class PacsHandler(BaseHTTPRequestHandler):
    server_version = "PACS/0.1"
    # 用 HTTP/1.1 以便保持连接：默认的 HTTP/1.0 会在每个响应后关闭连接，浏览器并发请求
    # 复用连接时就会偶发 `Failed to fetch`（表现为「网络错误」，但服务端其实一切正常）。
    protocol_version = "HTTP/1.1"
    settings: Settings          # 由 serve() 注入
    store: Store                # 由 serve() 注入

    # ---------- 输出助手 ----------

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _send_file(self, path: Path, *, as_attachment: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self._send_error_json(f"文件不存在：{path.name}", status=404)
            return
        ctype, _ = mimetypes.guess_type(str(path))
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        # 本地工具：一律不缓存。否则改了前端资源后浏览器仍用旧副本，
        # 表现为「代码已改但界面没变」，排查成本很高。
        self.send_header("Cache-Control", "no-store, must-revalidate")
        if as_attachment:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{urllib.parse.quote(path.name)}"'
            )
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc

    def log_message(self, fmt: str, *args: Any) -> None:  # 静默（避免刷屏）
        return

    # ---------- 路由 ----------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        # 保留空值参数：`?brain_id=`（当前大脑为空或还没建库）与「不带参数」（取全库）
        # 是两种不同意图，默认丢空值会让界面无法表达前者。
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        try:
            if path in ("/", "/index.html"):
                self._send_file(WEB_DIR / "index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path[len("/static/") :])
                return
            if path == "/api/ontology":
                self._send_json(self._ontology_payload())
                return
                return
            if path == "/api/config":
                self._send_json(self._config_payload())
                return
            if path == "/api/boot":
                self._send_json(self._boot_payload())
                return
            if path == "/api/brains":
                self._send_json({
                    "ok": True,
                    "active": _brain_summary(self._active_brain()),
                    "brains": [_brain_summary(b) for b in self.store.list_brains()],
                    "primitives": [
                        {"id": p.id, "name_zh": p.name_zh, "family": p.family}
                        for p in PRIMITIVES
                    ],
                })
                return
            if path == "/api/itemsets":
                self._send_json({"ok": True, "item_sets": self.store.list_item_sets()})
                return
            if path == "/api/examiner/bank":
                self._send_json(self._examiner_bank_payload(query))
                return
            if path == "/api/interview/sessions":
                mode = (query.get("mode") or ["interview"])[0]
                scoped = "brain_id" in query or "scoped" in query
                pool = (
                    _sessions_of_brain(self.store, self._active_brain()) if scoped
                    else self.store.list_dialogue_sessions(limit=200)
                )
                sessions = [
                    s for s in pool
                    if (s["payload"] or {}).get("mode", "probe") == mode
                ]
                self._send_json({
                    "ok": True,
                    "sessions": [
                        {
                            "session_id": s["session_id"], "theta_id": s["theta_id"],
                            "brain_id": s["brain_id"], "examiner_model": s["examiner_model"],
                            "n_turns": s["n_turns"], "created_at": s["created_at"],
                            "n_turns_run": (s["payload"] or {}).get("n_turns_run"),
                            "max_turns": (s["payload"] or {}).get("max_turns"),
                            "mode": (s["payload"] or {}).get("mode", "probe"),
                            "has_feedback": self.store.load_feedback(s["session_id"]) is not None,
                            "has_reading": feedback_has_reading(
                                self.store.load_feedback(s["session_id"])
                            ),
                        }
                        for s in sessions
                    ],
                })
                return
            if path == "/api/dialogue/sessions":
                # 群体运行一次就会产生几十段对话，因此这里既给「最近 N 条」也给总数，
                # 界面才不会让人以为只有列出来这些。
                limit = int((query.get("limit") or [DEFAULT_SESSION_LIMIT])[0])
                limit = max(1, min(limit, 500))
                # 界面传 `brain_id`（可空）表示「只看当前大脑的观察史」：换大脑或重新初始化后，
                # 列表应当立刻清空，而不是继续列着别的几十段对话。不传参数 = 取全局
                # （CLI / 脚本要看全库时用）。
                scoped = "brain_id" in query
                if scoped:
                    pool = _sessions_of_brain(self.store, self._active_brain())
                    total = len(pool)
                    sessions = pool[:limit]
                else:
                    sessions = self.store.list_dialogue_sessions(limit=limit)
                    total = self.store.counts().get("dialogues", len(sessions))
                self._send_json({
                    "ok": True,
                    "total": total,
                    "limit": limit,
                    "sessions": [
                        {
                            "session_id": s["session_id"],
                            "theta_id": s["theta_id"],
                            "brain_id": s["brain_id"],
                            "brain_model": s["brain_model"],
                            "examiner_model": s["examiner_model"],
                            "n_turns": s["n_turns"],
                            "n_turns_run": (s["payload"] or {}).get("n_turns_run"),
                            "max_turns": (s["payload"] or {}).get("max_turns"),
                            "text_limit": (s["payload"] or {}).get("text_limit"),
                            "created_at": s["created_at"],
                            "goals": (s["payload"] or {}).get("goals") or [],
                            "problems": (s["payload"] or {}).get("problems") or [],
                            "has_feedback": self.store.load_feedback(s["session_id"]) is not None,
                            # 有反馈记录 ≠ 有读数：分析者调用失败也会落一条空反馈。
                            # 历史列表据此显示「有读数 / 无读数」，与读数卡的判据同一个。
                            "has_reading": feedback_has_reading(
                                self.store.load_feedback(s["session_id"])
                            ),
                        }
                        for s in sessions
                    ],
                })
                return
            if path == "/api/dialogue":
                self._send_json(self._dialogue_payload((query.get("session_id") or [""])[0]))
                return
            if path == "/api/runs":
                self._send_json({"ok": True, "runs": self.store.list_runs(),
                                 "counts": self.store.counts()})
                return
            if path == "/api/profiles":
                run_id = (query.get("run_id") or [None])[0]
                model_id = (query.get("model_id") or [None])[0]
                profiles = self.store.load_profiles(run_id=run_id, model_id=model_id)
                self._send_json({
                    "ok": True,
                    "n": len(profiles),
                    "profiles": [p.to_dict() for p in profiles],
                })
                return
            if path == "/api/brain-report":
                self._send_json(self._brain_report_payload(query))
                return
            if path == "/api/analysis":
                self._send_json(self._analysis_payload(query))
                return
            if path == "/api/job":
                job = _job_get((query.get("job_id") or [""])[0])
                if job is None:
                    self._send_error_json("任务不存在或已过期", status=404)
                    return
                self._send_json({"ok": True, "job": job})
                return
                return
            if path == "/api/download":
                self._handle_download((query.get("path") or [""])[0])
                return
            self._send_error_json(f"未知路径：{path}", status=404)
        except Conflict as exc:                    # 前置条件不满足 → 409
            self._send_error_json(str(exc), status=409)
        except (ValueError, KeyError) as exc:      # 参数问题属客户端错误，不报 500
            self._send_error_json(f"{type(exc).__name__}: {exc}", status=400)
        except Exception as exc:  # 统一兜底，避免界面白屏
            self._send_error_json(f"{type(exc).__name__}: {exc}", status=500)

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_error_json(str(exc), status=400)
            return
        try:
            if path == "/api/config":
                self._send_json(self._save_config(body))
                return
            if path == "/api/prompt":
                perturbation = _perturbation_from_spec(body)
                self._send_json({
                    "ok": True,
                    "theta_id": perturbation.theta_id,
                    "describe": perturbation.describe(),
                    "prompt": build_brain_prompt(perturbation),
                    # 提示词构造过程：界面据此做「prompt 是怎么长出来的」分步可视化。
                    # 与上面 prompt 同源（prompt = 各步 text 拼接），因此不会漂移。
                    "steps": build_brain_prompt_steps(perturbation),
                })
                return
            if path == "/api/perturb/generate":
                self._send_json(self._generate_perturbation(body))
                return
            if path == "/api/inject":
                self._send_json(self._start_inject(body), status=202)
                return
            if path == "/api/brains/save":
                self._send_json(self._save_brain(body))
                return
            if path == "/api/brains/activate":
                self._send_json(self._activate_brain(body))
                return
            if path == "/api/brains/rename":
                self._send_json(self._rename_brain(body))
                return
            if path == "/api/brains/delete":
                self._send_json(self._delete_brain(body))
                return
            if path == "/api/interview/start":
                self._send_json(self._start_interview(body))
                return
            if path == "/api/interview/ask":
                self._send_json(self._interview_ask(body))
                return
            if path == "/api/dialogue/start":
                self._send_json(self._start_dialogue(body), status=202)
                return
            if path == "/api/dialogue/analyze":
                self._send_json(self._start_dialogue_analysis(body), status=202)
                return
            if path == "/api/itemsets":
                self._send_json(self._upload_item_set(body))
                return
            if path == "/api/itemsets/delete":
                set_id = str(body.get("set_id") or "")
                if not set_id:
                    raise ValueError("缺少 set_id")
                self._send_json({"ok": True, "deleted": self.store.delete_item_set(set_id)})
                return
            if path == "/api/init-brain":
                self._send_json(self._start_init_brain(body), status=202)
                return
            if path == "/api/freeze":
                self._send_json(self._do_freeze(body))
                return
            if path == "/api/atlas-md":
                self._send_json(self._save_atlas_md(body))
                return
            self._send_error_json(f"未知路径：{path}", status=404)
        except (ValueError, KeyError) as exc:
            self._send_error_json(f"{type(exc).__name__}: {exc}", status=400)
        except LLMError as exc:
            self._send_error_json(f"LLM 调用失败：{exc}", status=502)
        except Exception as exc:
            self._send_error_json(f"{type(exc).__name__}: {exc}", status=500)

    # ---------- 静态文件 ----------

    def _serve_static(self, relative: str) -> None:
        """静态资源根目录为 `web/static/`（与 URL 前缀 `/static/...` 对应）。"""
        base = (WEB_DIR / "static").resolve()
        target = (base / relative).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            self._send_error_json("非法路径", status=403)
            return
        self._send_file(target)

    def _handle_download(self, raw_path: str) -> None:
        if not raw_path:
            self._send_error_json("缺少 path 参数", status=400)
            return
        target = Path(raw_path).resolve()
        # 路径穿越防护：只允许项目目录内的文件
        try:
            target.relative_to(self.settings.base_dir.resolve())
        except ValueError:
            self._send_error_json("只允许下载项目目录内的文件", status=403)
            return
        self._send_file(target, as_attachment=True)

    # ---------- 载荷 ----------

    def _ontology_payload(self) -> dict:
        return {
            "ok": True,
            "families": FAMILIES,
            "primitives": [
                {
                    "id": p.id,
                    "name_zh": p.name_zh,
                    "family": p.family,
                    "definition": p.definition,
                    "computational_role": p.computational_role,
                    "impairment_signature": p.impairment_signature,
                    "dose_levels": list(p.dose_levels),
                }
                for p in PRIMITIVES
            ],
            "candidates": [{"id": c.id, "name_zh": c.name_zh} for c in CANDIDATES],
            "systems": [
                {
                    "id": s.id,
                    "name_zh": s.name_zh,
                    "members": list(s.members),
                    "emergence_rule": s.emergence_rule,
                    "falsifiable_expectation": s.falsifiable_expectation,
                }
                for s in SYSTEMS
            ],
            # 探查登记：只说明「用什么探查、哪些是对照、dose 档位」，
            # **不含任何维度声称**（构造属性与测量维度之间不允许有预设对应）
            "assays": [
                {
                    "primitive_id": a.primitive_id,
                    "primary_probe": a.primary_probe.key,
                    "primary_probe_text": a.primary_probe.text,
                    "weight": a.sensitivity_weight,
                    "n_controls": a.n_controls,
                    "dose_levels_count": len(a.dose_levels),
                    "expected_direction": a.expected_direction,
                }
                for a in ASSAYS
            ],
            # 表型维度（6 维）是观察目标与雷达图的词表；对照通道单独给出，它不是表型维度
            "phenotype_dims": PHENOTYPE_DIMS,
            "dim_anchors": DIM_ANCHORS,
            "control_channel": {"id": CONTROL_DIM, "label": CONTROL_DIM_ZH},
            "checks": {
                "ontology_problems": validate_ontology(),
                "without_assay": list(primitives_without_assay()),
            },
            "dose_levels": list(DEFAULT_DOSE_LEVELS),
        }

    def _config_payload(self) -> dict:
        """注意：**不返回任何 API Key**，只回报是否已配置。"""
        llm = self.settings.llm
        examiner = self.settings.examiner
        return {
            "ok": True,
            # —— 模拟大脑（保留原有键名，向后兼容）——
            "provider": llm.provider,
            "model": llm.model,
            "base_url": llm.base_url,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
            "api_key_set": bool(llm.api_key),
            # —— 提问者 / 分析者：另外的 LLM（独立第二套配置）——
            "examiner": {
                "provider": examiner.provider,
                "model": examiner.model,
                "base_url": examiner.base_url,
                "temperature": examiner.temperature,
                "max_tokens": examiner.max_tokens,
                "api_key_set": bool(examiner.api_key),
            },
            "configured": {
                "brain": self.settings.brain_ready(),
                "examiner": self.settings.examiner_ready(),
                "ready": self.settings.is_configured(),
            },
            "base_dir": str(self.settings.base_dir),
            "db_path": str(self.settings.db_path),
            # 年龄是固定常量：回给界面一个**只读**值（不是可改配置），界面别再放输入框
            "age_months": DEFAULT_AGE_MONTHS,
            "age_months_fixed": True,
        }

    def _save_config(self, body: dict) -> dict:
        provider = body.get("provider")
        if provider:
            if provider not in ("mock", "openai"):
                raise ValueError(f"未知 provider：{provider}")
            self.settings.llm.provider = provider
        for key, caster in (
            ("model", str),
            ("base_url", str),
            ("api_key", str),
            ("temperature", float),
            # 输出额度：真实模型（尤其推理型）太小会让正文为空 —— 这是要能改的
            ("max_tokens", int),
        ):
            if body.get(key) not in (None, ""):
                setattr(self.settings.llm, key, caster(body[key]))

        # 第二套 API（提问者 / 分析者）：独立子对象，可与模拟大脑不同厂商/模型
        examiner_body = body.get("examiner")
        if isinstance(examiner_body, dict):
            examiner_provider = examiner_body.get("provider")
            if examiner_provider:
                if examiner_provider not in ("mock", "openai"):
                    raise ValueError(f"未知 examiner provider：{examiner_provider}")
                self.settings.examiner.provider = examiner_provider
            for key, caster in (
                ("model", str),
                ("base_url", str),
                ("api_key", str),
                ("temperature", float),
                ("max_tokens", int),
            ):
                if examiner_body.get(key) not in (None, ""):
                    setattr(self.settings.examiner, key, caster(examiner_body[key]))

        # 心智年龄不可配置：请求里带了 age_months 也一律忽略（见 DEFAULT_AGE_MONTHS）
        if body.get("persist"):
            # 默认不把 Key 写盘（BYOK）；显式要求时才写
            self.settings.save(include_key=bool(body.get("persist_key")))
        self.settings.ensure_dirs()
        return {"ok": True, "config": self._config_payload(),
                "persisted": bool(body.get("persist"))}

    # ---------- 观察者内置题库（来自 resources 分析生成）----------

    def _examiner_bank_payload(self, query: dict) -> dict:
        """浏览内置条目库：统计 + 筛选后的条目。

        条目库由 `tools/build_examiner_bank.py` 从 `resources/asdkb/`（AsdKB 知识库精选）
        生成后随包分发，运行时不再依赖 resources 目录。
        """
        def one(key: str, default: str = "") -> str:
            return (query.get(key) or [default])[0]

        page = eb.list_items(
            dim=one("dim"),
            probe_mode=one("probe_mode"),
            primitive=one("primitive"),
            admin=one("admin"),
            source_id=one("source_id"),
            probe_ready=one("probe_ready") in ("1", "true", "yes"),
            age=int(one("age")) if one("age").isdigit() else None,
            query=one("q"),
            limit=int(one("limit", "50") or 50),
            offset=int(one("offset", "0") or 0),
        )
        return {
            "ok": True,
            "stats": eb.bank_stats(),
            "meta": eb.source_meta(),
            "mapping_rules": eb.mapping_rules(),
            "admin_zh": eb.ADMIN_ZH,
            "dims": eb.dims_map(),
            "probe_mode_notes": eb.mapping_rules().get("probe_mode_notes", {}),
            "total": page["total"],
            "items": page["items"],
        }


    # ---------- 大脑库 / 启动状态 ----------

    # ---------- 机制分析（限定到当前大脑所属的那次运行）----------

    def _brain_report_payload(self, query: dict) -> dict:
        """单个大脑的机制报告（本体与映射 + 它自己的读数）。

        与 `/api/analysis` 的分工：那个是**受试集合**层面的推断（需要一次运行），
        这个是**这一个大脑**层面的报告，不依赖任何运行，也不含任何群体统计。
        """
        brain_id = (query.get("brain_id") or [""])[0]
        record = self.store.load_brain(brain_id) if brain_id else self._active_brain()
        if record is None:
            raise Conflict("还没有当前大脑。请先在左侧栏注入或保存一个大脑。")
        return build_brain_report(
            self.store, record, perturbation=perturbation_of_brain(record)
        )

    def _analysis_payload(self, query: dict) -> dict:
        """只输出**与当前大脑有关**的分析结果。

        默认范围 = 当前大脑被测量过的那次运行（最近的）。这样「机制分析」里的每一个数字
        都来自包含这个大脑的那批受试，而不是库里任意历史的静态汇总。
        显式给 `run_id` 时以它为准（供脚本/复现使用）。
        """

        def one(key: str, default: str = "") -> str:
            return (query.get(key) or [default])[0]

        endpoint = one("endpoint", "social_communication_pragmatics")
        threshold = float(one("threshold", "0.70") or 0.70)
        dose = float(one("dose")) if one("dose") else None
        run_id = one("run_id")
        model_id = one("model_id") or None

        record = self.store.load_brain(one("brain_id")) if one("brain_id") else self._active_brain()
        focus = _brain_summary(record) if record else None
        scope: dict[str, Any] = {
            "mode": "explicit_run" if run_id else "current_brain",
            "brain": focus, "run_id": run_id or None, "model_id": model_id,
            "n_profiles": 0, "runs_measuring_brain": [],
        }

        if run_id:
            profiles = self.store.load_profiles(run_id=run_id, model_id=model_id)
        else:
            if record is None:
                raise Conflict("还没有当前大脑。请先在左侧栏注入或保存一个大脑。")
            runs = self.store.runs_for_theta(record["theta_id"])
            scope["runs_measuring_brain"] = runs
            if not runs:
                raise Conflict(
                    f"当前大脑（{record.get('name')}）还没有被测量过，"
                    "因此没有与它相关的机制分析结果。"
                    "请到「③ 表型观察 → 群体表型运行」跑一次包含该大脑的扰动设计。"
                )
            scope["run_id"] = runs[0]["run_id"]
            scope["model_id"] = runs[0]["model_id"]
            scope["n_profiles"] = runs[0]["n_profiles"]
            scope["instrument"] = runs[0].get("instrument")
            profiles = self.store.load_profiles(run_id=scope["run_id"])

        if not profiles:
            raise Conflict("该范围内没有表型数据。")
        scope["n_profiles"] = len(profiles)
        # 读数来源以实际剖面为准（显式指定 run_id 的分支也要有，界面据此说明读数由什么测出）
        scope["instrument"] = profiles[0].instrument

        atlas = _build_atlas(profiles, endpoint=endpoint, threshold=threshold, dose=dose)
        focus_position = self._focus_position(atlas, profiles, focus)
        scope["focus_in_run"] = bool(focus_position and focus_position["in_run"])
        return {
            "ok": True,
            "atlas": atlas.to_dict(),
            "markdown": render_atlas(atlas),
            "scope": scope,
            "focus": focus_position,
        }

    @staticmethod
    def _focus_position(atlas, profiles: list, focus: dict | None) -> dict | None:
        """当前大脑在这次分析里的定位：它自己测到的表型、匹配的候选机制、所属亚型。"""
        if not focus:
            return None
        theta_id = focus.get("theta_id")
        impaired = set(focus.get("impaired") or {})
        profile = next((p for p in profiles if p.theta_id == theta_id), None)

        candidates = (atlas.convergence.candidates if atlas.convergence else []) or []
        matches: list[dict] = []
        for cand in candidates:
            members = set(cand.members)
            if members == impaired:
                matches.append({"members": list(cand.members), "dose": cand.dose, "exact": True})
            elif impaired and impaired < members:
                matches.append({"members": list(cand.members), "dose": cand.dose, "exact": False})

        subtype = next(
            (s for s in atlas.subtypes if theta_id in s.member_theta_ids), None
        )
        return {
            "theta_id": theta_id,
            "name": focus.get("name"),
            "impaired": sorted(impaired),
            "in_run": profile is not None,
            "dims": profile.dims if profile else None,
            "reference": profile.note if profile else "",
            "parse_rate": profile.parse_rate if profile else None,
            "instrument": profile.instrument if profile else None,
            "confidence": profile.confidence if profile else None,
            "n_items": profile.n_items if profile else None,
            "n_parsed": profile.n_parsed if profile else None,
            "generic_degradation": profile.generic_degradation if profile else None,
            "candidate_matches": matches,
            "subtype": (
                {"subtype_id": subtype.subtype_id, "size": subtype.size,
                 "deviant_dims": subtype.deviant_dims,
                 "impaired_primitives": subtype.impaired_primitives}
                if subtype else None
            ),
        }


    def _active_brain(self) -> dict | None:
        """当前大脑：元数据里记录的 id → 不存在则退化为库中最近更新的一条。"""
        brain_id = self.store.get_meta(META_ACTIVE_BRAIN)
        if brain_id:
            record = self.store.load_brain(brain_id)
            if record:
                return record
        brains = self.store.list_brains()
        return brains[0] if brains else None

    def _boot_payload(self) -> dict:
        """界面打开时的第一份数据：配置门禁状态 + 启动自动初始化的正常大脑。"""
        boot = _boot_get()
        reasons = []
        if not self.settings.brain_ready():
            reasons.append("模拟大脑尚未配置可用的 LLM API")
        if not self.settings.examiner_ready():
            reasons.append("提问者 / 分析者尚未配置可用的 LLM API")
        return {
            "ok": True,
            "configured": {
                "brain": self.settings.brain_ready(),
                "examiner": self.settings.examiner_ready(),
                "ready": self.settings.is_configured(),
            },
            "gate": {
                "requires_setup": not self.settings.is_configured(),
                "reasons": reasons,
            },
            "brain_llm": {
                "provider": self.settings.llm.provider,
                "model": self.settings.llm.model,
            },
            "examiner_llm": {
                "provider": self.settings.examiner.provider,
                "model": self.settings.examiner.model,
            },
            "boot": boot,
            "active_brain": _brain_summary(self._active_brain()),
            "goals": default_goals(),
            "dialogue": {
                "default_n_turns": DEFAULT_N_TURNS,
                "max_n_turns": MAX_N_TURNS,
                "questions_per_channel": QUESTIONS_PER_CHANNEL,
                "text_limit": DEFAULT_TEXT_LIMIT,
                "age_months": DEFAULT_AGE_MONTHS,
            },
            "counts": self.store.counts(),
        }

    def _save_brain(self, body: dict) -> dict:
        """把 θ 存进大脑库。未给 values 时保存**当前大脑**的 θ（即「另存/重命名」）。"""
        name = str(body.get("name") or "").strip()
        if not name:
            raise ValueError("请给这个大脑起一个名字")
        if body.get("values"):
            perturbation = _perturbation_from_spec({"kind": "custom", "values": body["values"],
                                                     "dose": body.get("generic_degradation")})
        else:
            record = self._active_brain()
            if record is None:
                raise ValueError("库中还没有大脑可保存，请先注入扰动")
            perturbation = perturbation_of_brain(record)
        saved = register_brain(
            self.store, name, perturbation,
            source=str(body.get("source") or "user"),
            note=str(body.get("note") or ""),
            activate=bool(body.get("activate", True)),
        )
        return {"ok": True, "brain": _brain_summary(saved)}

    def _activate_brain(self, body: dict) -> dict:
        brain_id = str(body.get("brain_id") or "")
        if not brain_id:
            raise ValueError("缺少 brain_id")
        record = self.store.load_brain(brain_id)
        if record is None:
            raise ValueError(f"大脑不存在：{brain_id}")
        perturbation = perturbation_of_brain(record)
        self.store.set_meta(META_ACTIVE_BRAIN, brain_id)
        self.store.audit("brain_activate", f"{brain_id} {record.get('name')}")
        out = self._start_init_job(perturbation, brain_id=brain_id,
                                  label=f"切换到「{record.get('name')}」并重新初始化")
        out["brain"] = _brain_summary(record)
        return out

    def _rename_brain(self, body: dict) -> dict:
        brain_id = str(body.get("brain_id") or "")
        name = str(body.get("name") or "").strip()
        if not brain_id or not name:
            raise ValueError("需要 brain_id 与新的名字")
        if not self.store.rename_brain(brain_id, name):
            raise ValueError(f"大脑不存在：{brain_id}")
        return {"ok": True, "brain": _brain_summary(self.store.load_brain(brain_id))}

    def _delete_brain(self, body: dict) -> dict:
        brain_id = str(body.get("brain_id") or "")
        if not brain_id:
            raise ValueError("缺少 brain_id")
        if self.store.get_meta(META_ACTIVE_BRAIN) == brain_id:
            self.store.set_meta(META_ACTIVE_BRAIN, "")
        deleted = self.store.delete_brain(brain_id)
        self.store.audit("brain_delete", brain_id)
        return {"ok": True, "deleted": deleted}

    # ---------- 扰动注入（系统生成 / 用户调整 → 生成提示词）----------

    def _generate_perturbation(self, body: dict) -> dict:
        """系统生成一个扰动 θ（规则 4 的「系统生成」入口）。确定性：同 seed 同结果。"""
        mode = str(body.get("mode") or "random_single")
        seed = int(body.get("seed", 42))
        rng = random.Random(seed)
        primitives = [p.id for p in PRIMITIVES]
        doses = [d for d in DEFAULT_DOSE_LEVELS if d < 0.999]

        if mode == "intact":
            perturbation = intact(seed=seed, note="系统生成：完整基线")
        elif mode == "generic":
            level = float(body.get("generic_level", 0.5))
            perturbation = generic_degradation_control(level, seed=seed)
        elif mode == "single":
            perturbation = single(str(body["primitive"]), float(body.get("dose", 0.4)), seed=seed)
        elif mode == "random_pair":
            pair = rng.sample(primitives, 2)
            dose = float(body.get("dose", 0.2))
            perturbation = compose(
                single(pair[0], dose, seed=seed), single(pair[1], dose, seed=seed),
                seed=seed, note=f"系统生成：组合 {pair[0]} + {pair[1]} @ {dose:.2f}",
            )
        elif mode == "random_lhs":
            n_samples = max(2, min(64, int(body.get("n_samples", 16))))
            index = int(body.get("index", 0)) % n_samples
            perturbation = sample_lhs(seed=seed, n_samples=n_samples)[index]
        elif mode == "random_single":
            perturbation = single(
                rng.choice(primitives), rng.choice(doses), seed=seed,
                note="系统生成：随机单因素",
            )
        else:
            raise ValueError(f"未知生成模式：{mode}")

        return {
            "ok": True,
            "mode": mode,
            "theta_id": perturbation.theta_id,
            "describe": perturbation.describe(),
            "design": perturbation.design,
            "values": {k: round(float(v), 4) for k, v in perturbation.values.items()},
            "generic_degradation": round(float(perturbation.generic_degradation), 4),
            "impaired": perturbation.active_perturbations(),
        }

    def _start_inject(self, body: dict) -> dict:
        """注入扰动：生成 θ → 入大脑库 → 设为当前 → 重新初始化被扰动的大脑。"""
        perturbation = _perturbation_from_spec(body)
        # 默认大脑名不带设计标签（「显著性分配 0.30」而不是「[自定义] 显著性分配 0.30」）：
        # 这个名字会出现在侧栏、下拉与报告里，标签在那儿只是噪声。
        name = str(body.get("name") or "").strip() or perturbation.describe(include_design=False)
        name = name[:80]
        record = register_brain(
            self.store, name, perturbation,
            source=str(body.get("source") or "user"),
            note="界面注入的扰动",
            activate=True,
        )
        out = self._start_init_job(
            perturbation, brain_id=record.get("brain_id") or "",
            label=f"注入扰动并重新初始化：{perturbation.describe()[:60]}",
        )
        out["brain"] = _brain_summary(record)
        return out

    def _start_init_brain(self, body: dict) -> dict:
        """按界面给的扰动描述跑一次「提示词初始化」，回传自我报告与指纹。

        与 `/api/inject` 的区别：**不保存也不激活大脑** —— 它只让用户看一眼这份提示词
        把模型初始化成什么样，因此不需要 θ 入库。
        """
        perturbation = _perturbation_from_spec(body)
        return self._start_init_job(
            perturbation, brain_id="",
            label=f"初始化：{perturbation.describe()[:60]}",
            text_limit=_text_limit_of(body),
        )

    def _start_init_job(self, perturbation, *, brain_id: str, label: str,
                        text_limit: int = DEFAULT_TEXT_LIMIT) -> dict:
        """后台把这份提示词下发给模型做初始化，并把自我报告回传界面。"""
        job_id = f"init_{int(time.time() * 1000)}"
        _job_update(job_id, status="running", kind="init_brain", progress=0, total=1,
                    brain_id=brain_id, message=label)
        settings = self.settings

        def worker() -> None:
            try:
                provider = get_provider(settings.llm)
                agent = VirtualAgent(
                    provider, perturbation,
                    model_id=f"{provider.name}:{provider.model}",
                    text_limit=text_limit,
                )
                result = agent.initialize()
                _job_update(
                    job_id,
                    status="done",
                    progress=1,
                    result={
                        "ok": result.ok,
                        "self_report": result.self_report,
                        "raw": result.raw,
                        "theta_id": result.theta_id,
                        "model_id": result.model_id,
                        "fingerprint": agent.fingerprint,
                        "brain_id": brain_id,
                    },
                    message="初始化完成" if result.ok else "初始化未返回 SELF_STATE: OK",
                )
            except Exception as exc:
                _job_update(job_id, status="error", message=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc()[-1500:])

        threading.Thread(target=worker, daemon=True).start()
        return {
            "ok": True,
            "job_id": job_id,
            "brain_id": brain_id,
            "theta_id": perturbation.theta_id,
            "describe": perturbation.describe(),
            # 注入的同时把提示词与构造过程一并回传，界面无需再请求一次
            "prompt": build_brain_prompt(perturbation),
            "steps": build_brain_prompt_steps(perturbation),
            "text_limit": DEFAULT_TEXT_LIMIT,
            "age_months": DEFAULT_AGE_MONTHS,
        }

    # ---------- 表型观察：用户直接提问（自定义访谈）----------

    def _interview_brain(self, brain_id: str) -> tuple[dict, Any]:
        record = self.store.load_brain(brain_id) if brain_id else self._active_brain()
        if record is None:
            raise ValueError("还没有可提问的大脑，请先在「扰动注入」里注入扰动或保存一个大脑")
        return record, perturbation_of_brain(record)

    def _start_interview(self, body: dict) -> dict:
        """开一段由**用户本人**提问的访谈：大脑按当前架构上线，之后逐轮由用户提问。"""
        record, perturbation = self._interview_brain(str(body.get("brain_id") or ""))
        provider = get_provider(self.settings.llm)
        agent = VirtualAgent(
            provider, perturbation,
            model_id=f"{provider.name}:{provider.model}",
            text_limit=_text_limit_of(body),
        )
        session_id = new_session_id()
        turns: list[dict] = []
        init = None
        if body.get("include_self_report", True):
            # 先做一次提示词初始化，把自我报告作为第一轮呈现给用户 —— 这也是「大脑已上线」的证据
            result = agent.initialize()
            init = {
                "ok": result.ok, "self_report": result.self_report, "raw": result.raw,
                "model_id": result.model_id, "theta_id": result.theta_id,
                "fingerprint": agent.fingerprint,
            }
        session = DialogueSession(
            session_id=session_id,
            theta_id=perturbation.theta_id,
            brain_id=record.get("brain_id") or "",
            brain_model=agent.model_id,
            examiner_model="user（用户直接提问）",
            age_months=DEFAULT_AGE_MONTHS,
            n_turns=0,
            goals=(),
            seed=int(body.get("seed", 42)),
            mode="interview",
        )
        session.probe_mode = "manual"
        self.store.save_dialogue_session(session_id, session.to_dict())
        for index, turn in enumerate(turns):
            self.store.save_dialogue_turn(session_id, index, turn["role"], turn["content"], turn)
        self.store.audit("interview_start", f"{session_id} {record.get('brain_id')}")
        return {
            "ok": True, "session_id": session_id,
            "brain": _brain_summary(record), "init": init,
            "turns": turns, "model_id": agent.model_id,
        }

    def _interview_ask(self, body: dict) -> dict:
        """用户提一个问题 → 大脑按当前架构作答。逐轮落盘，随时可回看与交给分析者。"""
        session_id = str(body.get("session_id") or "")
        question = str(body.get("question") or "").strip()
        if not session_id:
            raise ValueError("缺少 session_id（请先开始一段访谈）")
        if not question:
            raise ValueError("问题不能为空")
        records = [s for s in self.store.list_dialogue_sessions(limit=1000)
                   if s["session_id"] == session_id]
        if not records:
            raise ValueError(f"访谈不存在：{session_id}")
        payload = records[0].get("payload") or {}
        if payload.get("mode", "probe") != "interview":
            raise ValueError("这不是用户访谈会话")
        record = self.store.load_brain(records[0].get("brain_id") or "") or self._active_brain()
        if record is None:
            raise ValueError("访谈对应的大脑已被删除，无法继续提问")
        perturbation = perturbation_of_brain(record)

        history = self._interview_history(session_id)
        history.append(ChatMessage("user", question))
        provider = get_provider(self.settings.llm)
        agent = VirtualAgent(
            provider, perturbation, model_id=f"{provider.name}:{provider.model}",
        )
        answer = agent.reply(history, turn=len(history), stage="interview")

        index = len(self.store.load_dialogue_turns(session_id))
        self.store.save_dialogue_turn(session_id, index, "user", question, {"from": "user"})
        self.store.save_dialogue_turn(session_id, index + 1, "brain", answer, {})
        payload["n_turns"] = int(payload.get("n_turns") or 0) + 1
        payload["turns"] = payload.get("turns", []) + [
            {"turn": payload["n_turns"], "role": "user", "content": question},
            {"turn": payload["n_turns"], "role": "brain", "content": answer},
        ]
        self.store.save_dialogue_session(session_id, payload)
        return {
            "ok": True, "session_id": session_id, "answer": answer,
            "turn": payload["n_turns"], "model_id": agent.model_id,
        }

    def _interview_history(self, session_id: str) -> list:
        """把已落盘的轮次还原成给大脑的消息历史（人类提问 → user，大脑回应 → assistant）。"""
        history: list = []
        for turn in self.store.load_dialogue_turns(session_id):
            role = turn.get("role")
            if role in ("user", "examiner"):
                history.append(ChatMessage("user", turn.get("content") or ""))
            elif role == "brain":
                history.append(ChatMessage("assistant", turn.get("content") or ""))
        return history

    # ---------- 表型观察：对话探针 ----------

    def _start_dialogue(self, body: dict) -> dict:
        """对话探针：提问者 LLM 提问 → 模拟大脑作答 → 分析者 LLM 形成表型反馈。

        逐轮写进任务状态（`job.transcript`），界面轮询即可做逐轮可视化 —— 不需要 SSE。
        """
        n_turns = int(body.get("n_turns", DEFAULT_N_TURNS))
        max_turns = int(body.get("max_turns") or MAX_N_TURNS)
        text_limit = _text_limit_of(body)
        # 轮数的完整校验（含「上限不小于基础轮数」「不超过硬顶」）交给 run_dialogue 做，
        # 这里只挡住明显非法的值，避免把一次额度不小的运行开起来才发现参数错。
        if not (1 <= n_turns <= ABSOLUTE_MAX_TURNS):
            raise ValueError(f"轮数必须在 1..{ABSOLUTE_MAX_TURNS} 之间")
        if not (n_turns <= max_turns <= ABSOLUTE_MAX_TURNS):
            raise ValueError(
                f"轮数上限必须在 {n_turns}..{ABSOLUTE_MAX_TURNS} 之间（且不小于基础轮数），"
                f"收到 {max_turns}"
            )
        goals = tuple(str(g) for g in (body.get("goals") or ())) or tuple(SCORED_CHANNELS)

        brain_id = str(body.get("brain_id") or "")
        record = self.store.load_brain(brain_id) if brain_id else self._active_brain()
        if record is None:
            raise ValueError("还没有可对话的大脑，请先在「扰动注入」里注入扰动或保存一个大脑")
        perturbation = perturbation_of_brain(record)
        seed = int(body.get("seed", 42))

        # 内置题库：`bank_keys` 指定题目；否则按年龄+目标自动挑（use_bank=False 可关闭）
        bank_keys = [str(k) for k in (body.get("bank_keys") or [])]
        if bank_keys:
            probe_items = eb.items_by_keys(bank_keys)
            if len(probe_items) != len(bank_keys):
                found = {i["key"] for i in probe_items}
                raise ValueError(
                    "以下条目不在内置条目库中：" + "、".join(k for k in bank_keys if k not in found)
                )
            # 只有带「儿童面向探查语句」的条目才能用于提问：其余条目（临床评定构念、
            # 成人自评措辞等）本来就不可询问儿童，硬拿它们提问只会产出空问题或怪句子。
            blocked = [i["key"] for i in probe_items if not i.get("probe_zh")]
            if blocked:
                raise ValueError(
                    "以下条目没有儿童面向探查语句，不能用于对话提问（它们只能作为观察要点）："
                    + "、".join(blocked)
                    + "。请在「内置条目库」里改选带「儿童面向探查语句」的条目。"
                )
        elif bool(body.get("use_bank", True)):
            probe_items = eb.dialogue_probes(
                goals=goals, n=n_turns, seed=seed,
                mode=str(body.get("probe_mode") or "auto"),
            )
        else:
            probe_items = []

        # 上传量表：llm 模式由分析者按对话逐条打分，并把量表的维度作为观察目标
        set_id = str(body.get("set_id") or "")
        set_record = next((s for s in self.store.list_item_sets() if s["set_id"] == set_id), None)
        scale_items = None
        scale_goals: tuple[tuple[str, str], ...] = ()
        if set_record:
            payload = set_record.get("payload") or {}
            scale_goals = tuple((d["dim"], d.get("label") or d["dim"]) for d in payload.get("dims") or [])
            if set_record.get("scoring_mode") == "llm":
                scale_items = payload.get("items") or []

        # 成本两段口径：基础轮数是**至少**会发生的调用数（提问 + 作答各一次 + 分析者一次），
        # 上限是配额补满时可能达到的最坏情况 —— 界面与 CLI 都按这两行提示，别只报最小值。
        expected_calls = n_turns * 2 + 1
        expected_calls_max = max_turns * 2 + 1
        session_id = new_session_id()
        job_id = f"dlg_{int(time.time() * 1000)}"
        _job_update(
            job_id, status="running", kind="dialogue", progress=0, total=expected_calls_max,
            transcript=[], session_id=session_id, theta_id=perturbation.theta_id,
            message=f"开始对话：基础 {n_turns} 轮，每通道不足 "
                    f"{QUESTIONS_PER_CHANNEL} 个问题时补问到 {max_turns} 轮为止"
                    f"（预计 {expected_calls}–{expected_calls_max} 次调用）",
        )
        settings = self.settings
        store = self.store

        def worker() -> None:
            session = None
            try:
                brain_provider = get_provider(settings.llm)
                examiner = get_examiner_provider(settings)
                agent = VirtualAgent(
                    brain_provider, perturbation,
                    model_id=f"{brain_provider.name}:{brain_provider.model}",
                    seed=seed, text_limit=text_limit,
                )
                collected: list[dict] = []

                def on_turn(turn: dict) -> None:
                    collected.append(turn)
                    who = "提问者" if turn.get("role") == "examiner" else "大脑"
                    _job_update(
                        job_id, transcript=list(collected), progress=len(collected),
                        message=f"第 {turn.get('turn')} 轮 · {who}：{turn.get('content','')[:48]}",
                    )

                session = run_dialogue(
                    agent, examiner, probe_items=probe_items, n_turns=n_turns,
                    max_turns=max_turns, text_limit=text_limit,
                    probe_mode=str(body.get("probe_mode") or "auto"),
                    goals=goals,
                    seed=seed, session_id=session_id, brain_id=record.get("brain_id") or "",
                    on_turn=on_turn,
                )
                store.save_dialogue_session(session.session_id, session.to_dict())
                for index, turn in enumerate(session.turns):
                    store.save_dialogue_turn(
                        session.session_id, index, turn.get("role", ""), turn.get("content", ""), turn
                    )
                store.audit("dialogue_done", f"{session.session_id} turns={session.n_brain_turns}")
                _job_update(
                    job_id, progress=min(session.n_turns_run * 2, expected_calls_max),
                    message=f"对话完成（{session.n_turns_run} 轮），"
                            "正在由分析者 LLM 形成表型反馈 …",
                )
            except Exception as exc:
                _job_update(job_id, status="error", message=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc()[-1500:])
                return

            # 分析失败不应丢掉已经发生的对话
            try:
                analyst = get_examiner_provider(settings)
                feedback = analyze_dialogue(
                    analyst, session,
                    goals=scale_goals or None,
                    scale_items=scale_items,
                    rubric_items=eb.observation_rubric(goals=goals, limit=12, seed=seed),
                )
                store.save_feedback(session.session_id, session.theta_id,
                                    feedback.get("model_id", ""), feedback)
                _job_update(
                    job_id, status="done", progress=expected_calls_max,
                    result={"session": session.to_dict(), "feedback": feedback},
                    # 进度条走的是「最坏情况」的刻度（否则补问会把进度顶过 100%），
                    # 因此这里把**实际**轮数与调用数写进消息，别让人以为真跑了 201 步
                    message=f"对话与表型反馈完成（实际 {session.n_turns_run} 轮 / 上限 "
                            f"{max_turns}，约 {session.n_turns_run * 2 + 1} 次调用）",
                )
            except Exception as exc:
                _job_update(
                    job_id, status="done", progress=expected_calls_max,
                    result={"session": session.to_dict(), "feedback": None},
                    message=f"对话已完成，但表型反馈生成失败：{type(exc).__name__}: {exc}",
                )

        threading.Thread(target=worker, daemon=True).start()
        return {
            "ok": True, "job_id": job_id, "session_id": session_id,
            "theta_id": perturbation.theta_id, "n_turns": n_turns,
            "max_turns": max_turns, "text_limit": text_limit,
            "estimated_calls": expected_calls, "estimated_calls_max": expected_calls_max,
            "brain": _brain_summary(record),
        }

    def _start_dialogue_analysis(self, body: dict) -> dict:
        """对一段已有对话重新形成表型反馈（例如换用上传量表再打分一次）。"""
        session_id = str(body.get("session_id") or "")
        if not session_id:
            raise ValueError("缺少 session_id")
        session_record = next(
            (s for s in self.store.list_dialogue_sessions(limit=1000)
             if s["session_id"] == session_id), None,
        )
        if session_record is None:
            raise ValueError(f"对话不存在：{session_id}")
        payload = session_record.get("payload") or {}
        turns = self.store.load_dialogue_turns(session_id)
        if not turns:
            raise ValueError("这段对话没有轮次记录")

        set_id = str(body.get("set_id") or "")
        set_record = next((s for s in self.store.list_item_sets() if s["set_id"] == set_id), None)
        scale_items = None
        scale_goals: tuple[tuple[str, str], ...] = ()
        if set_record:
            sp = set_record.get("payload") or {}
            scale_goals = tuple((d["dim"], d.get("label") or d["dim"]) for d in sp.get("dims") or [])
            if set_record.get("scoring_mode") == "llm":
                scale_items = sp.get("items") or []

        job_id = f"anl_{int(time.time() * 1000)}"
        _job_update(job_id, status="running", kind="dialogue_analysis", progress=0, total=1,
                    message="分析者 LLM 正在对这段对话形成表型反馈 …")
        settings = self.settings
        store = self.store

        def worker() -> None:
            from ..agents.dialogue import DialogueSession

            session = DialogueSession.from_dict(payload)
            if not session.turns and turns:
                session.turns = [
                    {"turn": t.get("turn_index"), "role": t.get("role"), "content": t.get("content")}
                    for t in turns
                ]
            try:
                analyst = get_examiner_provider(settings)
                session_goals = tuple((payload or {}).get("goals") or ()) or tuple(SCORED_CHANNELS)
                feedback = analyze_dialogue(
                    analyst, session,
                    goals=scale_goals or None,
                    scale_items=scale_items,
                    rubric_items=eb.observation_rubric(goals=session_goals, limit=12),
                )
                store.save_feedback(session_id, session.theta_id, feedback.get("model_id", ""), feedback)
                _job_update(job_id, status="done", progress=1,
                            result={"session": session.to_dict(), "feedback": feedback},
                            message="表型反馈已更新")
            except Exception as exc:
                _job_update(job_id, status="error", message=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc()[-1500:])

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "job_id": job_id, "session_id": session_id}

    def _dialogue_payload(self, session_id: str) -> dict:
        if not session_id:
            raise ValueError("缺少 session_id")
        record = next(
            (s for s in self.store.list_dialogue_sessions(limit=1000)
             if s["session_id"] == session_id), None,
        )
        if record is None:
            raise ValueError(f"对话不存在：{session_id}")
        return {
            "ok": True,
            "session": record.get("payload") or {},
            "meta": {k: v for k, v in record.items() if k != "payload"},
            # 优先用会话记录里的轮次（字段是 turn），与实时可视化一致；
            # 没有时才回落到 dialogue_turns 表（字段是 turn_index）。
            "turns": (record.get("payload") or {}).get("turns")
                     or self.store.load_dialogue_turns(session_id),
            "feedback": self.store.load_feedback(session_id),
        }

    # ---------- 用户上传的量表 ----------

    def _upload_item_set(self, body: dict) -> dict:
        """校验并保存一份上传的量表；校验失败只回报问题清单，不落盘。"""
        name = str(body.get("name") or "").strip()
        payload = body.get("payload")
        if payload is None and body.get("text"):
            try:
                payload = json.loads(str(body["text"]))
            except json.JSONDecodeError as exc:
                raise ValueError(f"上传内容不是合法 JSON：{exc}") from exc
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"上传内容不是合法 JSON：{exc}") from exc
        if payload is None:
            raise ValueError("缺少 payload（量表内容）")

        set_id, normalized, problems = parse_item_set(payload, name=name)
        if problems:
            return {"ok": True, "valid": False, "problems": problems, "set_id": set_id}
        self.store.save_item_set(set_id, normalized["name"], normalized["scoring_mode"], normalized)
        self.store.audit("itemset_upload", f"{set_id} {normalized['name']} {normalized['scoring_mode']}")
        return {"ok": True, "valid": True, "problems": [], "item_set": normalized,
                "n_items": len(normalized["items"])}

    def _do_freeze(self, body: dict) -> dict:
        run_id = body.get("run_id")
        endpoint = body.get("endpoint", "social_communication_pragmatics")
        profiles = self.store.load_profiles(run_id=run_id)
        if not profiles:
            raise ValueError("没有数据可供冻结（需先运行设计）。")
        atlas = _build_atlas_from_store(self.store, run_id=run_id, endpoint=endpoint,
                                       threshold=float(body.get("threshold", 0.70)))
        snapshot = build_freeze(
            perturbations=[
                {"theta_id": p.theta_id, "design": p.design, "theta": p.theta,
                 "generic_degradation": p.generic_degradation}
                for p in profiles
            ],
            convergence=atlas.convergence if atlas else None,
            counterfactuals=atlas.counterfactuals if atlas else None,
            model_version=DEFAULT_MODEL_VERSION,
        )
        problems = verify_freeze(snapshot)
        out_dir = save_freeze(snapshot, self.settings.exports_dir)
        self.store.save_freeze(snapshot.freeze_id, snapshot.to_dict())
        self.store.audit("ui_freeze", snapshot.freeze_id)
        return {
            "ok": True,
            "freeze_id": snapshot.freeze_id,
            "directory": str(out_dir),
            "digests": snapshot.digests,
            "problems": problems,
            "notes": snapshot.notes,
            "human_predictions": snapshot.objects.get("human_predictions", {}),
        }

    def _save_atlas_md(self, body: dict) -> dict:
        run_id = body.get("run_id")
        endpoint = body.get("endpoint", "social_communication_pragmatics")
        atlas = _build_atlas_from_store(self.store, run_id=run_id, endpoint=endpoint,
                                       threshold=float(body.get("threshold", 0.70)))
        if atlas is None:
            raise ValueError("没有数据，无法生成 Atlas。")
        md_path = self.settings.exports_dir / "atlas.md"
        json_path = self.settings.exports_dir / "atlas.json"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_atlas(atlas), encoding="utf-8")
        json_path.write_text(
            json.dumps(atlas.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.store.save_export("atlas", md_path, {"endpoint": endpoint})
        return {"ok": True, "markdown_path": str(md_path), "json_path": str(json_path)}


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------


def serve(
    settings: Settings,
    host: str = "127.0.0.1",
    port: int = 8770,
    *,
    open_browser: bool = False,
    auto_init: bool = True,
) -> None:
    """启动本地 Web 服务（阻塞）。

    `auto_init=True` 时，启动即在后台用提示词初始化一个**正常大脑**（规则 3），
    结果由 `GET /api/boot` 提供给界面；CI / 测试可传 `auto_init=False` 保持确定性。
    """
    settings.ensure_dirs()
    store = Store(settings.db_path)
    PacsHandler.settings = settings
    PacsHandler.store = store
    _cleanup_jobs()

    if auto_init:
        _launch_auto_init(settings, store)
    else:
        _boot_update(status="skipped", message="已按参数跳过启动自动初始化")

    httpd = ThreadingHTTPServer((host, port), PacsHandler)
    url = f"http://{host}:{port}/"
    print("=" * 74)
    print("PACS 可视化界面已启动（四界面：模型设置 / 扰动注入 / 表型观察 / 机制分析）")
    print("=" * 74)
    print(f"  地址      ：{url}")
    print(f"  模拟大脑  ：{settings.llm.provider} / {settings.llm.model}"
          f"（Key {'已配置' if settings.llm.api_key else '未配置'}）")
    print(f"  提问/分析 ：{settings.examiner.provider} / {settings.examiner.model}"
          f"（Key {'已配置' if settings.examiner.api_key else '未配置'}）")
    print(f"  数据目录  ：{settings.data_dir}")
    print()
    print("  提示：默认 provider=mock，不需要 Key 即可在界面上试跑全流程。")
    print("        要接入真实 LLM：在界面「① 模型设置」里填 Key，或设置环境变量")
    print("        PACS_LLM_API_KEY / PACS_EXAMINER_LLM_API_KEY 后重启本服务。")
    print()
    print("  按 Ctrl+C 停止。")
    print("=" * 74)
    if open_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
        store.close()
