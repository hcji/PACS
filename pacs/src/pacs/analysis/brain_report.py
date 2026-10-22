"""单个大脑的机制报告：本体与探查来源 + 该大脑自己的读数。

**范围约定**：这份报告只用「这一个大脑」的数据 —— 它的 θ、它在条目库里的条目归属、
以及它自己那些对话留下的分析者读数。**不做任何群体分析**（特异性、剂量-反应、
机制异质性、亚型、跨模型收敛都是受试集合层面的推断，不在本模块里）。

报告里出现的关联有两类，都**不是**测量维度与构造属性之间的对应关系：

- **基础过程 → 高阶构念**：`ontology/systems.py` 的组合定义（哪些 primitive 组合出这个能力）；
- **基础过程 → 题库条目的维度标签**：条目库里逐条目的 `dims` 归属汇总而来。这是**题目设计
  元数据**（这条观察量表条目覆盖哪些方面），只说明「这个问题问的是哪方面」，不构成
  「该机制应当体现在哪个测量维度」的期望。

**红线**：本报告不提供、也不再存在「primitive → Model 1 维度」的映射（`dim_map_targets`
已随该链一并删除），也不提供任何「机制 → 测量维度」的预期表。测量维度读数一律由分析者
LLM 从对话得出，见 `phenotype/dims.py` 与 `analysis/specificity.py`。

另外附上该过程在**条目库里的探查条目**（它现在是怎么被测的）与这些条目派生出的
症状/标准关联 —— 后者来自 AsdKB 的逐题症状索引，是派生映射，已显式标注。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from ..agents.measure import readings_from_feedback
from ..brain import examiner_bank as eb
from ..brain.perturb import Perturbation
from ..llm.parse import reading_dims
from ..ontology.assays import ASSAYS
from ..ontology.primitives import PRIMITIVES
from ..ontology.systems import SYSTEMS, systems_of

if TYPE_CHECKING:  # 只用于类型标注：运行时不导入，避免 agents ← 本模块的环形导入
    from ..phenotype import PhenotypeProfile

#: 单个过程最多列出的探查条目数（条目库按 primitive 匹配可能有几十条）
MAX_PROBES_PER_PRIMITIVE = 8

_PRIMITIVE_BY_ID = {p.id: p for p in PRIMITIVES}
_SYSTEM_BY_ID = {s.id: s for s in SYSTEMS}
_ASSAY_BY_ID = {a.primitive_id: a for a in ASSAYS}


def _probe_brief(item: dict) -> dict:
    """给界面看的探查条目摘要（不带整条 payload）。"""
    scenario = item.get("scenario") or {}
    return {
        "key": item["key"],
        "text_zh": item.get("text_zh") or "",
        "probe_zh": item.get("probe_zh") or "",
        "probe_mode": item.get("probe_mode") or "",
        "admin": item.get("admin") or "",
        "admin_zh": eb.ADMIN_ZH.get(item.get("admin") or "", item.get("admin") or ""),
        # 该条目在题库里的维度标签（题目设计元数据，不是测量期望）
        "dim_labels": sorted((item.get("dims") or {}).keys()),
        "source_name": item.get("source_name") or "",
        "observation": item.get("observation") or "",
        "observe": scenario.get("observe") or "",
    }


def _collect_for_primitive(primitive_id: str) -> dict:
    """把一个 primitive 的映射与探查来源收齐（不依赖任何运行）。"""
    primitive = _PRIMITIVE_BY_ID.get(primitive_id)
    items = [i for i in eb.load_bank()["items"]
             if primitive_id in (i.get("primitives") or {})]
    probes = [i for i in items if i.get("probe_zh")]
    # 探查条目优先：它们才是这个心智现在真被怎么测的；观察要点只是在观察提纲里当锚点
    probes.sort(key=lambda i: (-float((i.get("primitives") or {}).get(primitive_id) or 0.0),
                               i["key"]))
    dim_weights: dict[str, float] = {}
    for item in items:
        for dim, weight in (item.get("dims") or {}).items():
            dim_weights[dim] = dim_weights.get(dim, 0.0) + float(weight)
    symptoms: dict[str, dict] = {}
    for item in items:
        for sym in item.get("symptoms_derived") or []:
            symptoms.setdefault(sym["id"], {
                "id": sym["id"], "name_zh": sym.get("name_zh", ""),
                "category": sym.get("category", ""),
            })
    criteria: dict[str, dict] = {}
    for item in items:
        for crit in item.get("criteria") or []:
            criteria.setdefault(crit["id"], {
                "id": crit["id"], "text_zh": crit.get("text_zh", ""),
                "domain": crit.get("domain", ""),
            })
    assay = _ASSAY_BY_ID.get(primitive_id)
    return {
        "id": primitive_id,
        "name_zh": getattr(primitive, "name_zh", primitive_id),
        "name_en": getattr(primitive, "name_en", ""),
        "family": getattr(primitive, "family", ""),
        "family_zh": getattr(primitive, "family_zh", ""),
        "definition": getattr(primitive, "definition", ""),
        "computational_role": getattr(primitive, "computational_role", ""),
        "impairment_signature": getattr(primitive, "impairment_signature", ""),
        "systems": [
            {
                "id": sid,
                "name_zh": _SYSTEM_BY_ID[sid].name_zh if sid in _SYSTEM_BY_ID else sid,
                "emergence_rule": _SYSTEM_BY_ID[sid].emergence_rule if sid in _SYSTEM_BY_ID else "",
                "members": list(_SYSTEM_BY_ID[sid].members) if sid in _SYSTEM_BY_ID else [],
            }
            for sid in systems_of(primitive_id)
        ],
        # 题库条目层面的维度标签汇总：题目设计元数据（这些条目覆盖哪些方面），
        # **不是**「该机制应在哪个测量维度上体现」的期望 —— 故键名显式写明来源。
        "bank_item_dim_labels": [
            {"dim": dim, "weight": round(weight, 3)}
            for dim, weight in sorted(dim_weights.items(), key=lambda kv: -kv[1])
        ],
        "assay": (
            {"probe_key": assay.primary_probe.key,
             "probe_text": assay.primary_probe.text,
             "n_controls": assay.n_controls,
             "dose_levels": list(assay.dose_levels),
             "expected_direction": assay.expected_direction}
            if assay else None
        ),
        "items_total": len(items),
        "probes_total": len(probes),
        "probes": [_probe_brief(i) for i in probes[:MAX_PROBES_PER_PRIMITIVE]],
        "symptoms": sorted(symptoms.values(), key=lambda s: s["id"]),
        "criteria": sorted(criteria.values(), key=lambda c: c["id"]),
    }


def feedback_has_reading(feedback: "dict | None") -> bool:
    """这份反馈里有没有**可用的逐维读数**（至少一维给出数值，且不是结构字段）。

    只判断「有没有反馈记录」是不够的：分析者调用失败时也会落一条反馈（`dims` 空、
    `error` 写明原因）。那种记录若被当成「最近一次读数」，读数卡就成了一张空卡 ——
    而同一具心智此前明明有可用读数，看上去像「这具心智测不出东西」。历史列表同样据此
    区分「有读数」与「跑过但没读出东西」，两处用同一个判据。

    键还得**像测量通道**：`confidence` 这类结构字段不算有读数
    （见 `llm/parse.py` 的 `reading_dims`）。
    """
    if not feedback:
        return False
    return any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in reading_dims(feedback.get("dims")).values()
    )


def _latest_feedback(
    store, theta_id: str, record_brain_id: str = ""
) -> "tuple[dict | None, list[dict], dict | None]":
    """该大脑的对话会话，以及最近一次**可用**的表型反馈。

    三件事分开返回：`latest` 是最近一份**有读数**的反馈（界面据此画雷达图），`briefs` 是
    历史列表，`newest` 是最近一次**尝试**（哪怕它失败了）。最近那次失败必须原样说出来，
    否则用户会把眼前这份读数当成刚跑出来的结果。

    一份可用读数都没有时退回 `newest`（哪怕它是失败的）：界面据此说明**为什么**没有读数，
    而不是把「跑过但分析者没吐出东西」显示成「还没被观察过」。

    排序与界面「历史对话」用同一条归属规则、同一个排序（同秒决胜靠 rowid）：两处若各算
    一遍，「最近一次读数」与「历史列表的第一条」就会指向不同的会话。
    """
    sessions = store.list_brain_sessions(
        str(record_brain_id or ""), str(theta_id or ""), limit=500
    )
    briefs = []
    usable: dict | None = None
    newest: dict | None = None
    for session in sessions:
        feedback = store.load_feedback(session["session_id"])
        briefs.append({
            "session_id": session["session_id"],
            "mode": (session.get("payload") or {}).get("mode") or "probe",
            "n_turns": session.get("n_turns"),
            "brain_model": session.get("brain_model"),
            "examiner_model": session.get("examiner_model"),
            "created_at": session.get("created_at"),
            "has_feedback": feedback is not None,
            "problems": (session.get("payload") or {}).get("problems") or [],
        })
        if feedback is not None and newest is None:
            newest = feedback
        if usable is None and feedback_has_reading(feedback):
            usable = feedback
    return (usable or newest), briefs, newest


def _mean_of(dims: dict) -> "float | None":
    """一组读数的平均值（没有数值就返回 None —— 不拿 0 或旧值顶替）。"""
    values = [
        float(value) for value in dims.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return round(sum(values) / len(values), 4) if values else None


def _readings_payload(feedback: dict | None, newest: dict | None = None) -> dict | None:
    if not feedback:
        return None
    # 眼前这份读数**不是**最近一次跑出来的那一份：把最近那次尝试失败的原因也带出去，
    # 界面用一行说明它（不说的话，这份读数看起来就像刚跑出来的）。
    stale_of = None
    if newest is not None and newest.get("session_id") != feedback.get("session_id"):
        stale_of = {
            "session_id": newest.get("session_id"),
            "ts": newest.get("ts"),
            "error": newest.get("error") or "没有给出可用的逐维读数",
        }
    return {
        "session_id": feedback.get("session_id"),
        "model_id": feedback.get("model_id"),
        # 只把**像测量通道的键**交给界面：结构字段（confidence/deviation…）不是维度，
        # 旧记录里混进来的那种键一律不带出去。
        "dims": reading_dims(feedback.get("dims")),
        "dim_details": feedback.get("dim_details") or {},
        # 平均值由**过滤后的**读数现算：旧记录里的那个值是从假维度（例如 confidence）
        # 算出来的，直接透传会让卡片一边说「没有读数」、一边显示「平均 80 分」。
        "mean_reading": _mean_of(reading_dims(feedback.get("dims"))),
        "confidence": feedback.get("confidence"),
        "n_brain_turns": feedback.get("n_brain_turns"),
        "narrative": feedback.get("narrative") or "",
        "evidence": feedback.get("evidence") or [],
        "notes": feedback.get("notes") or "",
        "error": feedback.get("error"),
        # 这份读数来自哪一次尝试、最近一次尝试有没有失败（失败时界面必须说出来）
        "newest_attempt": stale_of,
        "probe_keys": feedback.get("probe_keys") or [],
        "rubric_keys": feedback.get("rubric_keys") or [],
        "ts": feedback.get("ts"),
        # 探查结论（约 350 字的病例报告式总结）+ 结构化表型印象：界面的读数卡直接从
        # 这份载荷取（它只认白名单里的键，漏了这几个键结论就永远显示不出来）。
        "conclusion": feedback.get("conclusion") or "",
        "conclusion_chars": feedback.get("conclusion_chars") or 0,
        "conclusion_target_chars": feedback.get("conclusion_target_chars"),
        "conclusion_notes": feedback.get("conclusion_notes") or [],
        "asd_likeness": feedback.get("asd_likeness")
        or {"present": None, "level": "unknown", "basis": []},
        # 轮数与配额：界面据此显示「实际轮数 / 上限」与「哪些方面问得不够」
        "n_turns": feedback.get("n_turns"),
        "n_turns_run": feedback.get("n_turns_run"),
        "max_turns": feedback.get("max_turns"),
        "question_counts": feedback.get("question_counts") or {},
        "under_quota_channels": feedback.get("under_quota_channels") or [],
    }


def profile_from_feedback(
    feedback: dict | None,
    *,
    record: dict,
    perturbation: Perturbation,
) -> "PhenotypeProfile | None":
    """把「该大脑最近一次对话的分析者读数」整理成一份表型剖面。

    为什么需要它：界面里的表型来自**对话**，不再来自批量设计运行，因此库里不会有该 θ 的
    `PhenotypeProfile`；而单脑报告与下游统计都按剖面结构读数据，所以这里把读数补成一份剖面
    —— 它和量产剖面的字段完全一致，只是 `instrument="dialogue"`。
    """
    from .. import DEFAULT_AGE_MONTHS
    from ..phenotype import PhenotypeProfile

    if not feedback:
        return None
    dims = readings_from_feedback(feedback)
    if not any(v is not None for v in dims.values()):
        return None
    answered = int(feedback.get("n_brain_turns") or 0)
    return PhenotypeProfile(
        theta_id=str(record.get("theta_id") or perturbation.theta_id),
        model_id=str(feedback.get("model_id") or ""),
        run_id="",
        age_months=DEFAULT_AGE_MONTHS,
        theta={k: float(v) for k, v in (perturbation.values or {}).items()},
        generic_degradation=float(perturbation.generic_degradation),
        dims=dims,
        # n_items 记该会话**实际发起**的轮数（含补问轮；旧记录里没有这个键时退回基础轮数），
        # n_parsed 记实际完成轮数 → parse_rate 即完成率且恒 ≤ 1
        n_items=max(1, int(feedback.get("n_turns_run") or feedback.get("n_turns") or answered)),
        n_parsed=answered,
        design=perturbation.design,
        note=f"LLM 涌现读数：会话 {feedback.get('session_id') or '—'}",
        instrument="dialogue",
        confidence=(
            None if feedback.get("confidence") is None else float(feedback["confidence"])
        ),
    )


def build_brain_report(
    store,
    record: dict,
    *,
    perturbation: Perturbation,
    all_primitives: Iterable[str] | None = None,
) -> dict[str, Any]:
    """组装单个大脑的机制报告。

    `record` 是 `store.load_brain()` 的记录；`perturbation` 由调用方还原（服务端已有该逻辑），
    避免这里再依赖一次 θ 的编码方式。
    """
    payload = record.get("payload") or {}
    impaired = dict(payload.get("impaired") or {})
    generic = float(payload.get("generic_degradation") or 0.0)
    theta: dict[str, float] = {k: float(v) for k, v in (perturbation.values or {}).items()}

    feedback, sessions, newest = _latest_feedback(
        store, record.get("theta_id") or "", record.get("brain_id") or ""
    )
    readings = _readings_payload(feedback, newest)
    profile = profile_from_feedback(feedback, record=record, perturbation=perturbation)

    # 受损过程：只列真正被削弱的（完整度 < 1）；完整心智则没有可映射的对象。
    active = sorted(
        (pid for pid, value in theta.items() if value < 0.999),
        key=lambda pid: (theta[pid], pid),
    )
    primitives = [_collect_for_primitive(pid) for pid in active]
    for entry in primitives:
        entry["value"] = theta.get(entry["id"])

    reference = [
        {
            "id": pid,
            "name_zh": getattr(_PRIMITIVE_BY_ID.get(pid), "name_zh", pid),
            "family_zh": getattr(_PRIMITIVE_BY_ID.get(pid), "family_zh", ""),
            "value": theta.get(pid),
        }
        for pid in sorted(theta, key=lambda p: theta[p])
    ]

    return {
        "ok": True,
        "brain": {
            "brain_id": record.get("brain_id"),
            "name": record.get("name"),
            "source": record.get("source"),
            "theta_id": record.get("theta_id"),
            "describe": payload.get("describe", ""),
            "impaired": impaired,
            "generic_degradation": generic,
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        },
        "intact": not active and generic <= 0,
        "readings": readings,
        # 与该大脑读数一致的剖面（CLI/API 的统计与展示直接用它）
        "profile": profile.to_dict() if profile else None,
        "sessions": sessions,
        "primitives": primitives,
        "reference": reference,
        "notes": {
            "scope": "只使用这一个大脑的数据；群体层面的推断（特异性/剂量-反应/机制异质性/"
                     "亚型/跨模型收敛）不在此报告中。",
            "symptoms_basis": "症状与标准关联取自 AsdKB 的逐题症状索引，属派生映射"
                              "（源数据为量表级覆盖，无法逐题路由），已显式标注。",
        },
    }
