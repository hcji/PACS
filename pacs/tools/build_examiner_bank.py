"""从 `D:\\ASD Agent\\resources\\asdkb` 生成观察者的**内置条目库 + 映射关系**（可复现）。

用法：
    python tools/build_examiner_bank.py
    python tools/build_examiner_bank.py --kb "D:\\ASD Agent\\resources\\asdkb"
    python tools/build_examiner_bank.py --check           # 只校验与打印，不写文件

**唯一来源**：AsdKB 孤独症知识库精选（`https://w3id.org/asdkb`）的四个 JSON：

| 文件 | 内容 | 在本库中的角色 |
|---|---|---|
| `tools.json` | 20 套筛查量表（14 套有逐题，共 441 题）+ 元信息与计分规则 | 条目主体 |
| `symptoms.json` | 65 条症状（含分类） | 症状层 + 逐题关联的派生依据 |
| `diagnostic_criteria.json` | 43 条 DSM/ICD 行为标准条目 | 条目的标准关联 |
| `treatments.json` | 28 条循证干预（EBP） | 干预层参考（非题目） |

**两条必须写清楚的源数据事实**（它们直接决定了本库的形态）：

1. **没有一套量表是儿童直接施测的** —— 9 套知情者报告、2 套临床评定、2 套成人自评、1 套分段。
   因此儿童面向探查语句**全部由本项目人工重写**（`probe_basis="curated"`），
   未重写的条目只作观察要点，不会被问给孩子。
2. **逐题症状关联在源数据里是量表级覆盖，不可用于逐题路由** —— 每条症状被 117–319 道题
   提及（中位 176），单个量表内部只有 3–10 种不同的症状集合。因此逐题症状关联由本脚本
   按词语重叠**派生**（`symptoms_derived`），并把源数据这一事实连同统计值写进每条记录。

产出：`src/pacs/data/examiner_bank.json`（随包分发，运行时不依赖 `resources/`）。
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import examiner_bank_curated as C  # noqa: E402
from pacs.phenotype import CONTROL_DIM_ZH, PHENOTYPE_DIMS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB = Path(r"D:\ASD Agent\resources\asdkb")
OUT_PATH = ROOT / "src" / "pacs" / "data" / "examiner_bank.json"

ADMIN_ZH = {
    "caregiver_report": "知情者报告",
    "clinician_rating": "临床评定",
    "self_report": "自评（成人版）",
    "observation": "主试者现场任务",
    "split": "分段（知情者报告 + 现场任务）",
}

#: 维度中文名直接取自表型维度注册表（**不在这里另写一份**，否则两份会漂移）；
#: 对照通道不是表型维度，单独标注。
DIM_ZH = {**PHENOTYPE_DIMS, "general_cognition": CONTROL_DIM_ZH}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# 映射
# ---------------------------------------------------------------------------


def dims_for(text: str, scale_id: str) -> tuple[dict[str, float], str, list[str]]:
    """条目文本 → 表型维度。

    优先逐题关键词；未命中任何维度时回落到该量表的维度先验（并标注 basis）。
    返回 `(dims, basis, matched_terms)`，命中词一并记录以便复核。
    """
    hits: dict[str, float] = {}
    matched: list[str] = []
    for dim in C.DIM_PRIORITY:
        for keyword in C.DIM_KEYWORDS.get(dim, ()):
            if keyword in text:
                hits[dim] = C.DIM_WEIGHT
                matched.append(keyword)
                break
    if hits:
        return hits, "keyword", matched
    prior = C.SCALE_DIMS.get(scale_id) or [("social_cognition", 1.0)]
    return dict(prior), "scale", []


def scenario_for(text: str, key: str) -> dict[str, str] | None:
    """条目 → 情境式探查（设定场景 → 问「这种情况你会怎么处理」）。

    优先用人工撰写的情境；否则按行为域关键词套模板；两者都没有（或属于病史/医学事实）
    则返回 None —— 不硬造场景，因为一个引不出目标行为的情境比没有情境更糟。
    """
    if key in C.CURATED_SCENARIOS:
        entry = dict(C.CURATED_SCENARIOS[key])
        entry["basis"] = "curated"
        entry["category"] = "curated"
        return entry
    if any(word in text for word in C.NON_ELICITABLE_KEYWORDS):
        return None
    for category in C.SCENARIO_CATEGORIES:
        if any(word in text for word in category["keywords"]):
            return {
                "category": category["id"],
                "category_zh": category["label_zh"],
                "scenario": category["scenario"],
                "ask": category["ask"],
                "observe": category["observe"],
                "basis": "template",
            }
    return None


def render_scenario_text(scenario: dict[str, str]) -> str:
    """情境式探查的呈现文本：先给情境，再给「这种情况你会怎么处理」。"""
    ask = scenario.get("ask") or "这种情况你会怎么处理？"
    return f"情境：{scenario['scenario']}\n提问：{ask}"


def render_probe_text(question: str, scenario: dict[str, str] | None) -> tuple[str, str]:
    """返回 `(首选呈现文本, 探查方式)`。

    首选规则：**人工撰写的问句优先**（比模板更贴合该条目），其次才是模板情境。
    两种形态都会写进条目（`question_zh` / `scenario_zh`），界面与对话探针可按需切换，
    首选只是默认值。呈现文本统一进 `probe_zh`，让下游只有一个入口。
    """
    if question:
        return question, "question"
    if scenario:
        return render_scenario_text(scenario), "scenario"
    return "", "none"


def primitives_for(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pid, (keywords, _why) in C.PRIMITIVE_KEYWORDS.items():
        if any(k in text for k in keywords):
            out[pid] = C.PRIMITIVE_WEIGHT
    return out


def match_symptoms(text: str, symptoms: list[dict]) -> list[dict]:
    """题目文本与症状名的**词语重叠**匹配（显式标注为派生，非源数据的逐题标注）。"""
    hits: list[dict] = []
    for symptom in symptoms:
        name = str(symptom.get("name_zh") or "").strip()
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", name):
            if token in text:
                hits.append({
                    "id": symptom["id"],
                    "name_zh": name,
                    "category": normalize_category(symptom.get("category")),
                    "matched": token,
                })
                break
    return hits[:4]


def match_criteria(text: str, criteria: list[dict]) -> list[dict]:
    hits: list[dict] = []
    for criterion in criteria:
        body = str(criterion.get("text_zh") or "")
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", body):
            if token in text:
                hits.append({"id": criterion["id"], "text_zh": body[:70], "matched": token})
                break
    return hits[:3]


def first_value(raw: object) -> str:
    """源数据里若干字段存的是「字符串化的列表」（如 `"['EBP']"`），统一取首值。"""
    if isinstance(raw, list):
        return str(raw[0]).strip() if raw else ""
    text = str(raw or "").strip()
    match = re.fullmatch(r"\[\s*['\"](.+?)['\"]\s*\]", text)
    return match.group(1).strip() if match else text


def normalize_category(raw: object) -> str:
    """症状分类：源数据里 RRB 被拼成了两个字符串（多一个尾字母 r），此处归一。"""
    value = first_value(raw)
    if value.startswith("restrictive_repetitive"):
        return "restrictive_repetitive_and_stereotyped_behaviors"
    return value


def age_range(age_text: str) -> tuple[int | None, int | None]:
    """从 `age` 字段里抠出月龄区间（只认「岁 / 个月 / 月」写法）。"""
    text = str(age_text or "")
    low = high = None
    m_months = re.search(r"(\d{1,2})\s*[-~至]\s*(\d{1,2})\s*个?月", text)
    if m_months:
        low, high = int(m_months.group(1)), int(m_months.group(2))
    m_years = re.search(r"(\d{1,2})\s*[-~至]\s*(\d{1,2})\s*岁", text)
    if m_years:
        y_low, y_high = int(m_years.group(1)), int(m_years.group(2))
        low = y_low * 12 if low is None else low
        high = y_high * 12 + 11 if high is None else high
    elif low is None:
        m_single = re.search(r"(\d{1,2})\s*岁", text)
        if m_single:
            low, high = int(m_single.group(1)) * 12, None
    return low, high


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------


def load_kb(kb_dir: Path) -> dict:
    def read(name: str):
        return json.loads((kb_dir / name).read_text(encoding="utf-8"))

    return {
        "tools": read("tools.json"),
        "symptoms": read("symptoms.json"),
        "criteria": read("diagnostic_criteria.json"),
        "treatments": read("treatments.json"),
        "sha256": {
            name: _sha256(kb_dir / name)
            for name in ("tools.json", "symptoms.json",
                         "diagnostic_criteria.json", "treatments.json")
        },
    }


def build_items(kb: dict) -> list[dict]:
    items: list[dict] = []
    for tool in kb["tools"]:
        tid = tool["id"]
        scale_admin = C.SCALE_ADMIN.get(tid, "caregiver_report")
        low, high = age_range(tool.get("age"))
        covers_36 = bool(low is not None and low <= 36 and (high is None or high >= 36))
        for idx, question in enumerate(tool.get("questions") or []):
            text = re.sub(r"\s+", " ", str(question.get("content") or "")).strip()
            text = re.sub(r"^(?:\d{1,3}|[ivx]{1,4})\s*[.．]\s*", "", text)
            if not text:
                continue
            # CHAT 整表在 SCALE_ADMIN 里标为 split（说明该表分段），逐题再落到具体方式：
            # A 段（0–8）为知情者报告，B 段（9–13）为主试者现场任务脚本。
            if tid == "tools5":
                admin = "observation" if idx >= 9 else "caregiver_report"
            else:
                admin = scale_admin
            key = f"{tid}:{idx}"
            # 注意命名：question 是循环里的源题对象，这里用 curated_question 避免遮蔽
            curated_question = C.CURATED_PROBES_ASDKB.get(key)
            scenario = scenario_for(text, key)
            probe_text, probe_mode = render_probe_text(curated_question, scenario)
            dims, dim_basis, matched = dims_for(text, tid)
            primitives = primitives_for(text)
            items.append({
                "key": f"{tid}:{idx}",
                "source_id": tid,
                "source_name": tool.get("name_zh") or tid,
                "source_en": tool.get("name_en") or "",
                "source_ref": f"{tool.get('name_zh')} 第 {idx + 1} 题",
                "source_no": idx + 1,
                "text_zh": text,
                "options": [
                    {"content": re.sub(r"\s+", " ", str(o.get("content") or "")).strip(),
                     "score": o.get("score")}
                    for o in (question.get("options") or [])
                ],
                "scoring_rule": (tool.get("rule") or "")[:400],
                "screening_boundary": tool.get("screening_boundary") or [],
                "respondent": tool.get("user") or "",
                "age_hint": tool.get("age") or "",
                "age_months_min": low,
                "age_months_max": high,
                "covers_36_months": covers_36,
                "age_field_broken": tid in C.BROKEN_AGE_FIELDS,
                "admin": admin,
                "dims": dims,
                "dims_basis": dim_basis,
                "dims_matched_terms": matched,
                "scale_dims": dict(C.SCALE_DIMS.get(tid) or {}),
                "primitives": primitives,
                "symptoms_source_scale_coverage": {
                    "n": len(question.get("symptoms") or []),
                    "note": "源数据的逐题症状关联是量表级覆盖（每条症状被 117–319 题提及），"
                            "不可用于逐题路由，仅作出处记录。",
                },
                "symptoms_derived": match_symptoms(text, kb["symptoms"]),
                "criteria": match_criteria(text, kb["criteria"]),
                "probe_zh": probe_text,
                "probe_mode": probe_mode,
                "probe_basis": (
                    "curated" if curated_question else (scenario["basis"] if scenario else "none")
                ),
                "question_zh": curated_question,
                "scenario": scenario,
                "scenario_zh": render_scenario_text(scenario) if scenario else "",
                "observation": None if probe_text else observation_note(admin, text),
                "mapping_basis": {
                    "dims": dim_basis,
                    "primitives": "keyword" if primitives else "none",
                    "probe": probe_mode if probe_text else "none",
                    "symptoms": "derived_keyword_overlap",
                },
                "rationale": C.SCALE_DIMS_RATIONALE.get(tid, ""),
            })
    return items


def observation_note(admin: str, text: str) -> str:
    if admin == "clinician_rating":
        return f"临床评定条目（不可直接询问儿童）：由观察者依据对话表现评定「{text[:24]}」。"
    if admin == "self_report":
        return f"自评条目（措辞为第一人称成人版）：不适用于幼儿，仅作术语参考 —— {text[:24]}"
    if admin == "observation":
        return "主试者可现场布置的任务：观察儿童的实际表现。"
    return f"知情者报告条目：作为观察要点与知情者访谈提纲，不询问儿童 —— {text[:24]}"


def _is_rrb_criterion(criterion: dict) -> bool:
    """源数据把 D29–D42（受限重复行为）错标成了社交互动分类 —— 按 id 段与条文判定更可靠。"""
    text = str(criterion.get("text_zh") or "")
    number = int(re.sub(r"\D", "", str(criterion.get("id") or "0")) or 0)
    if number >= 29:
        return True
    return any(k in text for k in ("刻板", "重复", "仪式", "受限", "感觉输入", "旋转", "嗅闻"))


def build_reference_layers(kb: dict) -> dict:
    """症状 / 标准条目 / 循证干预三个参考层的结构化总结。"""
    symptoms = []
    for symptom in kb["symptoms"]:
        category = normalize_category(symptom.get("category"))
        symptoms.append({
            "id": symptom["id"],
            "name_zh": str(symptom.get("name_zh") or "").strip(),
            "name_en": str(symptom.get("name_en") or "").strip(),
            "category": category,
            "dims": C.SYMPTOM_CATEGORY_DIMS.get(category, []),
            "mentioned_in_questions": len(symptom.get("mentioned_in_questions") or []),
        })
    criteria = [
        {
            "id": c["id"],
            "text_zh": str(c.get("text_zh") or "").strip(),
            "category": first_value(c.get("category")),
            "domain": "rrb" if _is_rrb_criterion(c) else "social_communication",
        }
        for c in kb["criteria"]
    ]
    treatments = [
        {
            "id": t["id"],
            "name_zh": str(t.get("name_zh") or "").strip(),
            "name_en": str(t.get("name_en") or "").strip(),
            "type": first_value(t.get("type")),
            "introduction": str(t.get("introduction") or "").strip(),
        }
        for t in kb["treatments"]
    ]
    return {"symptoms": symptoms, "criteria": criteria, "treatments": treatments}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="生成观察者内置条目库（唯一来源：AsdKB）")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB, help="asdkb 目录")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--check", action="store_true", help="只校验与统计，不写文件")
    args = parser.parse_args()

    kb_dir: Path = args.kb
    needed = ("tools.json", "symptoms.json", "diagnostic_criteria.json", "treatments.json")
    missing = [name for name in needed if not (kb_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"AsdKB 目录不完整：{kb_dir}\n缺少：{'、'.join(missing)}\n"
            "请用 --kb <目录> 指定正确的 asdkb 路径。"
        )

    kb = load_kb(kb_dir)
    items = build_items(kb)
    reference = build_reference_layers(kb)

    # 校验：源数据里有多少题就必须有多少条，不允许静默丢项
    source_items = sum(len(t.get("questions") or []) for t in kb["tools"])
    if len(items) != source_items:
        raise SystemExit(f"条目数 {len(items)} ≠ 源数据题数 {source_items}，存在被丢弃的条目")
    if not items:
        raise SystemExit("没有生成任何条目")

    facets: dict[str, collections.Counter] = {
        "by_admin": collections.Counter(i["admin"] for i in items),
        "by_dim": collections.Counter(d for i in items for d in i["dims"]),
        "by_primitive": collections.Counter(p for i in items for p in i["primitives"]),
        "by_scale": collections.Counter(i["source_id"] for i in items),
        "by_dims_basis": collections.Counter(i["dims_basis"] for i in items),
    }
    ages = [i["age_months_min"] for i in items if i["age_months_min"] is not None]

    stats = {
        "items_total": len(items),
        "probe_ready": sum(1 for i in items if i["probe_zh"]),
        "curated_probes": sum(1 for i in items if i["probe_basis"] == "curated"),
        "observation_only": sum(1 for i in items if not i["probe_zh"]),
        "by_probe_mode": {
            mode: sum(1 for i in items if i["probe_mode"] == mode)
            for mode in ("question", "scenario", "none")
        },
        "by_probe_basis": {
            basis: sum(1 for i in items if i["probe_basis"] == basis)
            for basis in ("curated", "template", "none")
        },
        "scenario_ready": sum(1 for i in items if i.get("scenario")),
        "with_primitives": sum(1 for i in items if i["primitives"]),
        "with_symptoms": sum(1 for i in items if i["symptoms_derived"]),
        "with_criteria": sum(1 for i in items if i["criteria"]),
        "covers_36_months": sum(1 for i in items if i["covers_36_months"]),
        "scales": len(kb["tools"]),
        "scales_with_items": sum(1 for t in kb["tools"] if t.get("questions")),
        "metadata_only_scales": list(C.METADATA_ONLY_SCALES),
        "symptoms": len(reference["symptoms"]),
        "diagnostic_criteria": len(reference["criteria"]),
        "treatments": len(reference["treatments"]),
        "age_range_months": [min(ages), max(ages)] if ages else [None, None],
        "by_admin": dict(facets["by_admin"]),
        "by_admin_zh": {ADMIN_ZH.get(k, k): v for k, v in facets["by_admin"].items()},
        "by_dim": dict(facets["by_dim"]),
        "by_primitive": dict(facets["by_primitive"]),
        "by_scale": dict(facets["by_scale"]),
        "by_dims_basis": dict(facets["by_dims_basis"]),
    }

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "generator": "tools/build_examiner_bank.py",
            "curated_tables": "tools/examiner_bank_curated.py",
            "purpose": "观察者内置条目库：仅由 AsdKB 知识库内容总结生成；"
                       "供「③ 表型观察」浏览、选用，并给分析者作评分锚点。",
            "source": {
                "id": "ASDKB",
                "name": "AsdKB 孤独症知识库精选",
                "kind": "公开本体知识库（https://w3id.org/asdkb），双语 zh/en",
                "path": str(kb_dir),
                "sha256_16": kb["sha256"],
                "stats": {
                    "scales": stats["scales"],
                    "scales_with_items": stats["scales_with_items"],
                    "metadata_only_scales": stats["metadata_only_scales"],
                    "items": stats["items_total"],
                    "symptoms": stats["symptoms"],
                    "diagnostic_criteria": stats["diagnostic_criteria"],
                    "treatments": stats["treatments"],
                },
            },
            "notes": [
                "唯一来源为 AsdKB 的四个 JSON；本文件随包分发，运行时不读取 resources 目录。",
                "AsdKB 的 20 套量表**没有一套是儿童直接施测的**（知情者报告 9 / 临床评定 2 / "
                "成人自评 2 / 分段 1）。因此儿童面向探查语句全部由本项目人工重写，"
                "标注 probe_basis=curated；其余条目只作观察要点。",
                "逐题症状关联在源数据里是**量表级覆盖**（每条症状被 117–319 题提及，中位 176），"
                "无法用于逐题路由；symptoms_derived 是按词语重叠派生的逐题关联，已显式标注。",
                "诊断标准条目 D29–D42 在源数据里被错标为社交互动分类，本库按 id 段与条文内容"
                "重新判定为受限重复行为域（见 mapping_rules.criteria_domain_rule）。",
                "六套量表（DBD-ES / GADS / GARS-2 / KADI / PDDST-II / STAT）源库仅有元信息、"
                "无逐题数据 —— 属源数据特性，非抽取遗漏。",
            ],
        },
        "dims": DIM_ZH,
        "mapping_rules": {
            "scale_dims": {k: dict(v) for k, v in C.SCALE_DIMS.items()},
            "scale_dims_rationale": C.SCALE_DIMS_RATIONALE,
            "dim_keywords": {k: list(v) for k, v in C.DIM_KEYWORDS.items()},
            "dim_weight": C.DIM_WEIGHT,
            "dim_priority": list(C.DIM_PRIORITY),
            "dim_priority_note": C.DIM_PRIORITY_NOTE,
            "primitive_keywords": {
                pid: {"keywords": list(kw), "why": why}
                for pid, (kw, why) in C.PRIMITIVE_KEYWORDS.items()
            },
            "symptom_category_dims": C.SYMPTOM_CATEGORY_DIMS,
            "criteria_domain_rule": "id 序号 ≥29，或条文含刻板/重复/仪式/受限/感觉输入等词 → "
                                    "受限重复行为域；其余为社交沟通域。",
            "scale_admin": C.SCALE_ADMIN,
            "broken_age_fields": C.BROKEN_AGE_FIELDS,
            "probe_mode_notes": C.PROBE_MODE_NOTES,
            "scenario_categories": [
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in category.items()}
                for category in C.SCENARIO_CATEGORIES
            ],
            "non_elicitable_keywords": list(C.NON_ELICITABLE_KEYWORDS),
            "scenario_note": "情境式探查是本项目的派生化工作：AsdKB 条目为知情者报告/临床评定，"
                             "不能直接对儿童提问，但多数行为可通过布置情境引出。"
                             "只有确实对应可布置行为域的条目才生成情境（probe_basis=template）；"
                             "病史与医学事实类条目不生成，保持 observation-only。",
        },
        "stats": stats,
        "items": items,
        **reference,
    }

    print("=" * 74)
    print("内置条目库生成（唯一来源：AsdKB）")
    print("=" * 74)
    print(f"  来源目录 : {kb_dir}")
    print(f"  量表     : {stats['scales']} 套（{stats['scales_with_items']} 套有逐题，"
          f"{len(stats['metadata_only_scales'])} 套仅元信息）")
    print(f"  条目     : {stats['items_total']}（校验：与源数据题数一致）")
    print(f"  可探查条目       : {stats['probe_ready']}"
          f"（直接提问 {stats['by_probe_mode']['question']} · 情境式 {stats['by_probe_mode']['scenario']}）")
    print(f"  探查来源         : 人工撰写 {stats['by_probe_basis']['curated']}"
          f" · 模板派生 {stats['by_probe_basis']['template']}")
    print(f"  仅作观察要点     : {stats['observation_only']}")
    print(f"  施测方式         : {stats['by_admin_zh']}")
    print(f"  维度归属         : {stats['by_dim']}")
    print(f"  维度判定依据     : {stats['by_dims_basis']}")
    print(f"  参考层           : 症状 {stats['symptoms']} · 标准条目 "
          f"{stats['diagnostic_criteria']} · 循证干预 {stats['treatments']}")
    print(f"  覆盖 36 月龄     : {stats['covers_36_months']} 条")
    print("\n  抽查（覆盖 36 月龄且有儿童面向语句）：")
    shown = 0
    for item in items:
        if item["covers_36_months"] and item["probe_zh"] and shown < 4:
            print(f"    {item['source_ref']}：{item['text_zh'][:34]}")
            print(f"      → {item['probe_zh']}")
            print(f"      → 维度 {list(item['dims'])} 基础过程 {list(item['primitives'])}")
            shown += 1

    if args.check:
        print("\n（--check：未写文件）")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  已写出：{args.out}（{args.out.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
