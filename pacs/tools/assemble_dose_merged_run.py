"""装配多档合并运行目录（把若干个批次的 `cells/` 合成一条完整剂量阶梯）。

## 为什么需要「装配」这一步

`scale_single.py --aggregate-only` 只读**一个** `--out` 下的 `cells/`。要让所有档位落在
同一张剂量-反应曲线、同一份逐档特异性表里，就必须把它们放进同一个目录。而已经跑过的档位
（0.4/0.8 来自更早的批次，0.2/0.6 来自上一次补充）**不重跑**：重跑既多花钱，又会让
「同一批测量」的前提消失（同一个 θ 换一次采样，读数的误差结构就换了）。所以只能从已完成的
运行目录里**复制**。

## 两条硬约束

1. **源目录只读**。所有写入只发生在 `--out` 里，源目录一个字节都不改。
2. **剂量对不上就整体失败**。装配最危险的事故是「混进了不该混的记录却看不出来」：
   `scale_single.py` 的 `--resume` 冲突检查只比 `n_turns/max_turns/max_tokens`，
   **不比剂量**（见 `scale_single.py` 的 `run_condition`），所以一个按默认剂量跑出来的批次
   会被静默复用，报告照样生成、数字照样好看。因此这里逐份记录做三重校验：

   - `parameters.active` 恰好一个 primitive，且与文件所在条件目录名一致；
   - 剂量落在**该来源目录自己的**允许档位内（每个来源都要显式声明它该有哪些档）；
   - `theta_id` 与 `single(primitive, dose, seed).theta_id` 的重算结果一致
     （θ id 由取值确定性派生，重算得上才说明这份记录真的对应它自称的剂量）。

   任一份不符即失败退出，**一个文件都不复制**。

## 来源怎么声明

`--source` 可重复，每条形如 `路径|档位,档位|来源性质|备注`（备注可省）：

```
--source "…/scale_single_dose09_supplement|0.9|new"              # 本次新跑
--source "…/scale_single_dose_supplement|0.2,0.6|new"            # 更早的补充批次
--source "…/scale_single|0.4,0.8|reused"                         # 更早批次，整批复用数据
--source "…/scale_single_dose0408_supplement|0.4,0.8|new|其中 r0–r2 是复制进来的既有测量"
```

第三段 `new` / `reused` 只影响 PROVENANCE 里的措辞（哪些是本次新跑、哪些是复用测量），
不参与校验。第四段备注自由文本，用来记录**同一个目录里混合了新旧测量**这类无法用
单一 `origin` 表达的情况 —— 一个目录里既有复用又有新跑时，说它是 `new` 会过度声称，
说它是 `reused` 又会把新数据说成旧的，所以必须靠备注讲清楚。

路径里含 `:`（Windows 盘符），所以字段分隔符用 `|`。

## 用法

```bash
# 只看会复制什么、会报什么错
python tools/assemble_dose_merged_run.py --dry-run

# 装配（默认三个来源 → 0.2/0.4/0.6/0.8/0.9 五档）
python tools/assemble_dose_merged_run.py

# 然后聚合（--doses 必须给全，否则特异性会按默认档位算）
python tools/scale_single.py --aggregate-only \
    --primitives agent_state_updating,belief_updating,rule_updating,self_other_distinction,social_reward_weighting \
    --doses 0.2,0.4,0.6,0.8,0.9 --no-generic \
    --out data/runs/scale_single_dose_merged_5dose
```

只复制 `subjects/*.json`：`fig1_reliability.load_condition_records` 只 glob
`cells/<id>/subjects/*.json`，`_finish` 也不读 `condition.json`，所以合并目录不需要
条件级文件 —— 写一份看着像工具产物、实际不是的 `condition.json` 只会误导下游。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pacs import SIMULATION_NOTICE  # noqa: E402
from pacs.brain.perturb import single  # noqa: E402

RUNS = _ROOT / "data" / "runs"

#: 本次补充批次跑的 5 个 primitive（与剂量阶梯的 5 个机制一一对应）
DEFAULT_PRIMITIVES: tuple[str, ...] = (
    "agent_state_updating",
    "belief_updating",
    "rule_updating",
    "self_other_distinction",
    "social_reward_weighting",
)

#: 默认来源：三段式的 `路径|档位|new|reused`。顺序 = 复制顺序（不影响聚合）。
DEFAULT_SOURCES: tuple[str, ...] = (
    f"{RUNS / 'scale_single_dose09_supplement'}|0.9|new",
    f"{RUNS / 'scale_single_dose_supplement'}|0.2,0.6|new",
    f"{RUNS / 'scale_single'}|0.4,0.8|reused",
)

DEFAULT_OUT = RUNS / "scale_single_dose_merged_5dose"

DISCLAIMER = "用途声明：本工具为方法学研究材料，不用于医学诊断、不替代专业评估。"


def parse_source(spec: str) -> dict:
    """解析一条 `路径|档位,档位|来源性质|备注`（备注可省）。"""
    parts = str(spec).split("|")
    if len(parts) < 2:
        raise SystemExit(f"✗ --source 格式应为 `路径|档位,档位|new|reused|备注`，收到 {spec!r}")
    path = Path(parts[0].strip())
    doses = tuple(
        float(chunk) for chunk in parts[1].replace("，", ",").split(",") if chunk.strip()
    )
    if not doses:
        raise SystemExit(f"✗ --source 没有给出档位：{spec!r}")
    origin = (parts[2].strip() if len(parts) > 2 and parts[2].strip() else "new")
    if origin not in ("new", "reused"):
        raise SystemExit(f"✗ --source 的来源性质只能是 new / reused，收到 {origin!r}")
    note = parts[3].strip() if len(parts) > 3 else ""
    return {"dir": path, "doses": tuple(sorted(doses, reverse=True)), "origin": origin,
            "note": note, "label": path.name}


# ---------------------------------------------------------------------------
# 扫描与校验
# ---------------------------------------------------------------------------


def _active_dose(record: dict) -> tuple[str, float] | None:
    """单因素记录被削弱的 (primitive, 剂量)；非单因素返回 None。"""
    params = record.get("parameters") or {}
    if float(params.get("generic_degradation") or 0.0) > 0:
        return None
    active = params.get("active") or {}
    if len(active) != 1:
        return None
    pid, dose = next(iter(active.items()))
    return str(pid), round(float(dose), 4)


def scan(
    run_dir: Path,
    primitives: tuple[str, ...],
    allowed_doses: tuple[float, ...],
    *,
    label: str,
    seed: int,
) -> tuple[list[dict], list[str]]:
    """扫一个运行目录，返回（可用条目, 问题清单）。只读，不写任何东西。"""
    entries: list[dict] = []
    problems: list[str] = []
    allowed = {round(float(d), 4) for d in allowed_doses}

    for pid in primitives:
        subjects = run_dir / "cells" / f"single_{pid}" / "subjects"
        if not subjects.is_dir():
            problems.append(f"[{label}] 缺条件目录：{subjects}")
            continue
        for path in sorted(subjects.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"[{label}] {path.name}：读不出（{type(exc).__name__}）")
                continue

            key = _active_dose(record)
            if key is None:
                problems.append(f"[{label}] {path.name}：不是单因素扰动记录（跳过）")
                continue
            got_pid, dose = key

            if got_pid != pid:
                problems.append(
                    f"[{label}] {path.name}：削弱的 primitive 是 {got_pid}，"
                    f"却在 single_{pid} 条件目录里"
                )
                continue
            if round(dose, 4) not in allowed:
                problems.append(
                    f"[{label}] {path.name}：剂量 {dose:g} 不在本来源声明的档位 "
                    f"{sorted(allowed)} 内 —— 这正是「按默认剂量跑出来的批次被混进来」的征兆"
                )
                continue

            theta_id = str((record.get("parameters") or {}).get("theta_id") or "")
            expected_id = single(got_pid, float(dose), seed=seed).theta_id
            if theta_id != expected_id:
                problems.append(
                    f"[{label}] {path.name}：θ 指纹对不上（记录 {theta_id or '缺失'}，"
                    f"按 single({got_pid}, {dose:g}, seed={seed}) 重算应为 {expected_id}）"
                )
                continue

            entries.append({
                "pid": got_pid,
                "dose": round(dose, 4),
                "theta_id": theta_id,
                "replicate": int((record.get("condition") or {}).get("replicate") or 0),
                "path": path,
                "label": label,
                "origin": None,      # 由 plan() 填
            })

    return entries, problems


def plan(
    sources: list[dict],
    primitives: tuple[str, ...],
    *,
    seed: int,
) -> tuple[list[dict], list[str]]:
    """所有来源合起来的装配计划；同时检查同名文件冲突。"""
    entries: list[dict] = []
    problems: list[str] = []
    for source in sources:
        found, found_problems = scan(source["dir"], primitives, source["doses"],
                                    label=source["label"], seed=seed)
        for entry in found:
            entry["origin"] = source["origin"]
        entries.extend(found)
        problems.extend(found_problems)

    # 目标位置同名 = 两个来源对同一 (primitive, θ, 重复) 各有一份，合并后必然丢一份。
    seen: dict[tuple[str, str], dict] = {}
    for entry in entries:
        key = (entry["pid"], entry["path"].name)
        if key in seen:
            problems.append(
                f"目标同名冲突：{entry['path']} 与 {seen[key]['path']} 都会落到 "
                f"cells/single_{entry['pid']}/subjects/{entry['path'].name}"
            )
        seen[key] = entry

    return entries, problems


# ---------------------------------------------------------------------------
# 复制与出处
# ---------------------------------------------------------------------------


def execute(entries: list[dict], out_dir: Path, *, force: bool) -> dict[Path, int]:
    """把条目复制到 out_dir。返回 {目标目录: 份数}。"""
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise SystemExit(
            f"✗ 目标目录已存在且非空：{out_dir}\n"
            "  确认要重写请加 `--force`（它只影响 --out，源目录始终只读）。"
        )

    counts: dict[Path, int] = {}
    for entry in entries:
        target_dir = out_dir / "cells" / f"single_{entry['pid']}" / "subjects"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["path"], target_dir / entry["path"].name)
        counts[target_dir] = counts.get(target_dir, 0) + 1
    return counts


def write_provenance(
    out_dir: Path,
    entries: list[dict],
    *,
    sources: list[dict],
    seed: int,
) -> Path:
    """写 PROVENANCE.md：逐档构成、θ 指纹、逐份来源，以及**哪些是复用数据**。"""
    by_cell: dict[tuple[str, float], list[dict]] = {}
    for entry in entries:
        by_cell.setdefault((entry["pid"], entry["dose"]), []).append(entry)

    doses = sorted({e["dose"] for e in entries}, reverse=True)

    lines: list[str] = []
    lines.append("# 多档合并目录 · 装配出处")
    lines.append("")
    lines.append("本目录**不是一次运行跑出来的**，是把若干个批次的 `cells/*/subjects/*.json`")
    lines.append("复制到一起后由 `scale_single.py --aggregate-only` 重新聚合的产物。")
    lines.append("")
    lines.append(f"- 档位：{', '.join(f'{d:g}' for d in doses)}（共 {len(doses)} 档）")
    lines.append(f"- θ 重算种子：{seed}")
    lines.append(f"- 记录总数：{len(entries)}")
    lines.append("")
    lines.append("## 来源批次")
    lines.append("")
    lines.append("| 来源目录 | 声明档位 | 性质 | 份数 | 备注 |")
    lines.append("|---|---|---|---|---|")
    for source in sources:
        n = sum(1 for e in entries if e["label"] == source["label"])
        lines.append(
            f"| `{source['dir']}` | {', '.join(f'{d:g}' for d in source['doses'])} | "
            f"{'本次新跑' if source['origin'] == 'new' else '**复用数据**'} | {n} | "
            f"{source.get('note') or '—'} |"
        )
    lines.append("")
    lines.append("**复制是只读的**：所有源目录在装配前后逐字节未变。")
    lines.append("")
    lines.append("## 逐档构成")
    lines.append("")
    lines.append("| primitive | 剂量 | 份数 | θ 指纹 | 重复号 | 来源 |")
    lines.append("|---|---|---|---|---|---|")
    for (pid, dose) in sorted(by_cell, key=lambda k: (k[0], -k[1])):
        bucket = by_cell[(pid, dose)]
        lines.append(
            f"| {pid} | {dose:g} | {len(bucket)} | "
            f"{'、'.join(sorted({e['theta_id'] for e in bucket}))} | "
            f"{'、'.join(str(r) for r in sorted({e['replicate'] for e in bucket}))} | "
            f"{'、'.join(sorted({e['label'] for e in bucket}))} |"
        )
    lines.append("")
    reused_doses = sorted({e["dose"] for e in entries if e["origin"] == "reused"}, reverse=True)
    new_doses = sorted({e["dose"] for e in entries if e["origin"] == "new"}, reverse=True)
    lines.append("## 引用这份数据时必须一并说明的两件事")
    lines.append("")
    if reused_doses:
        lines.append(
            f"1. **{', '.join(f'{d:g}' for d in reused_doses)} 档是复用数据**，与 "
            f"{', '.join(f'{d:g}' for d in new_doses)} 不是同一批测量：它们同源、协议逐项一致"
            "（题池逐位相同），但误差结构不同源，阶梯上这些档的重复间离散度不能与"
            "新跑的档位直接比。")
    else:
        lines.append("1. 全部档位都来自本目录各来源批次，没有跨批次复用。")
        lines.append("2. 每份记录的完整出处（源文件绝对路径）如下，供逐条追溯。")
    if any(source.get("note") for source in sources):
        lines.append("3. 部分来源批次的目录内部**混合了新旧测量**（见上表「备注」列）："
                     "这些目录里的部分重复是更早批次的复制件、部分是本批新跑，"
                     "不能用单一来源标签概括。")
    lines.append("")
    lines.append("## 逐份记录出处")
    lines.append("")
    lines.append("| primitive | 剂量 | 重复 | θ 指纹 | 来源 | 源文件 |")
    lines.append("|---|---|---|---|---|---|")
    for entry in sorted(entries, key=lambda e: (e["pid"], -e["dose"], e["replicate"])):
        lines.append(
            f"| {entry['pid']} | {entry['dose']:g} | {entry['replicate']} | "
            f"{entry['theta_id']} | {entry['label']} | `{entry['path']}` |"
        )
    lines.append("")
    lines.append(SIMULATION_NOTICE)
    lines.append("")
    lines.append(DISCLAIMER)
    lines.append("")

    path = out_dir / "PROVENANCE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assemble_dose_merged_run",
        description="把若干批次装配成一个完整剂量阶梯的合并运行目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tools/assemble_dose_merged_run.py --dry-run\n"
            "  python tools/assemble_dose_merged_run.py\n"
            "  python tools/assemble_dose_merged_run.py --force \\\n"
            '    --source "…/scale_single_dose09_supplement|0.9|new" \\\n'
            '    --source "…/scale_single|0.4,0.8|reused" \\\n'
            "    --out data/runs/scale_single_dose_merged        # 复现旧的 4 档版本\n"
        ),
    )
    parser.add_argument("--source", action="append", default=None,
                        help="可重复；`路径|档位,档位|new|reused`")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"合并目录（默认 {DEFAULT_OUT.name}）")
    parser.add_argument("--primitives", default=",".join(DEFAULT_PRIMITIVES),
                        help="只装配这些 primitive（逗号分隔）")
    parser.add_argument("--seed", type=int, default=42,
                        help="重算 θ 指纹用的种子（默认 42，与批次协议一致）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划与校验结果，不写任何文件")
    parser.add_argument("--force", action="store_true", help="目标目录非空时仍重写（源目录始终只读）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    primitives = tuple(p.strip() for p in str(args.primitives).split(",") if p.strip())
    sources = [parse_source(s) for s in (args.source or list(DEFAULT_SOURCES))]
    out_dir = Path(args.out)
    all_doses = sorted({d for s in sources for d in s["doses"]}, reverse=True)

    print("=" * 96)
    print(f"多档合并目录装配（{', '.join(f'{d:g}' for d in all_doses)}）")
    print("=" * 96)
    for source in sources:
        flag = "新跑" if source["origin"] == "new" else "复用"
        print(f"  [{flag}] {source['dir']}　档位 {[f'{d:g}' for d in source['doses']]}")
    print(f"目标目录：{out_dir}")
    print(f"primitive：{len(primitives)} 个　θ 重算种子：{args.seed}")
    print()

    entries, problems = plan(sources, primitives, seed=args.seed)

    # ---- 逐校验项显式报出来（不做静默通过）----
    print("校验：")
    print("  ✓ 源目录只读：本脚本只在 --out 下写入")
    print("  ✓ 逐份记录检查了 ①active 与条件目录一致 ②剂量在该来源声明的档位内 "
          "③θ 指纹可由 single(...) 重算复现")
    if problems:
        print()
        print(f"✗ {len(problems)} 项校验未通过，**不复制任何文件**：")
        for note in problems[:40]:
            print(f"    {note}")
        if len(problems) > 40:
            print(f"    ……另有 {len(problems) - 40} 项")
        return 2
    print("  ✓ 全部记录通过三重校验")
    print()

    by_dose: dict[float, int] = {}
    for entry in entries:
        by_dose[entry["dose"]] = by_dose.get(entry["dose"], 0) + 1
    print(f"计划复制 {len(entries)} 份记录：")
    for dose in sorted(by_dose, reverse=True):
        n_new = sum(1 for e in entries if e["dose"] == dose and e["origin"] == "new")
        n_re = sum(1 for e in entries if e["dose"] == dose and e["origin"] == "reused")
        detail = f"（新跑 {n_new} / 复用 {n_re}）" if n_new and n_re else (
            "（复用）" if n_re else "（新跑）")
        print(f"    剂量 {dose:g} → {by_dose[dose]:>3} 份 {detail}")
    print()

    if args.dry_run:
        print("（--dry-run：未写入任何文件）")
        print()
        print(SIMULATION_NOTICE)
        print(DISCLAIMER)
        return 0

    counts = execute(entries, out_dir, force=args.force)
    for target in sorted(counts, key=lambda p: str(p)):
        print(f"    {target.relative_to(out_dir)}：{counts[target]} 份")
    provenance = write_provenance(out_dir, entries, sources=sources, seed=args.seed)
    print()
    print(f"✓ 装配完成：{out_dir}")
    print(f"  出处说明：{provenance}")
    print()
    print("下一步（--doses 必须给全，否则特异性会按默认档位算）：")
    print("  python tools/scale_single.py --aggregate-only \\")
    print(f"      --primitives {','.join(primitives)} \\")
    print(f"      --doses {','.join(f'{d:g}' for d in all_doses)} --no-generic \\")
    print(f'      --out "{out_dir}"')
    print()
    print(SIMULATION_NOTICE)
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
