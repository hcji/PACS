"""`智能体介绍.md` §6 附录（Supplementary Tables 1–2）与代码的对应关系守卫。

为什么需要这个测试：附录里的表是**文献稿**（操作性定义 + 理论依据 + PMID 由论文侧提供），
而代码里的 `PRIMITIVES` / `PHENOTYPE_DIMS` 是**可执行设定**。两边一旦漂移 —— 改了 id、
挪了家族、加了通道、删了某一行 —— 文档会继续声称「31/31 一一对应、顺序一致」，
而读者无从发现。这里把「一一对应」变成断言，改代码忘改文档（或反过来）都会立刻失败。

顺带守住两条口径：
- 每一行都必须有 PMID（文献稿不能只留标题）；
- 对照通道出现在表 2 里，但**必须**显式标注它不在 Supplementary Table 2 内 ——
  它（`general_cognition`）是本项目为阴性预测新增的，不是论文表里的第 7 维。
"""

from __future__ import annotations

import re
from pathlib import Path

from pacs.ontology.primitives import FAMILIES, PRIMITIVES
from pacs.phenotype import CONTROL_DIM, DIM_ANCHORS, PHENOTYPE_DIMS, SCORED_CHANNELS

DOC = Path(__file__).resolve().parents[1] / "智能体介绍.md"

#: 表 1 的行首：`| A 信息处理 | \`salience_assignment\` | 显著性分配 | …`
_ROW1 = re.compile(r"^\|\s*([^|]+?)\s*\|\s*`([a-z0-9_]+)`\s*\|\s*([^|]*?)\s*\|(.*)\|\s*$")
#: 表 2 的行首：`| 1 | Social cognition（社会认知） | \`social_cognition\` | …`
_ROW2 = re.compile(r"^\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*`([a-z0-9_]+)`\s*\|(.*)\|\s*$")


def _lines() -> list[str]:
    return DOC.read_text(encoding="utf-8").splitlines()


def _table1() -> list[tuple[str, str, str, str]]:
    """返回 `[(家族, id, 中文名, 其余各列原文)]`，按文档顺序。"""
    rows: list[tuple[str, str, str, str]] = []
    inside = False
    for line in _lines():
        if line.startswith("| 家族 | 基础过程 |"):
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith("|"):
            break
        if set(line) <= set("|-: "):          # 分隔行
            continue
        match = _ROW1.match(line)
        assert match, f"表 1 行无法解析：{line[:60]}"
        rows.append((match.group(1), match.group(2), match.group(3), match.group(4)))
    return rows


def _table2() -> list[tuple[str, str, str, str]]:
    """返回 `[(编号列, 维度列, 通道 id, 其余各列原文)]`，按文档顺序。"""
    rows: list[tuple[str, str, str, str]] = []
    inside = False
    for line in _lines():
        if line.startswith("| # | 维度 | 通道 id"):
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith("|"):
            break
        if set(line) <= set("|-: "):
            continue
        match = _ROW2.match(line)
        assert match, f"表 2 行无法解析：{line[:60]}"
        rows.append((match.group(1), match.group(2), match.group(3), match.group(4)))
    return rows


def test_appendix_tables_exist():
    assert _table1(), "智能体介绍.md §6.2 的 Supplementary Table 1 不见了"
    assert _table2(), "智能体介绍.md §6.3 的 Supplementary Table 2 不见了"


def test_table1_covers_every_primitive_in_code_order():
    """31 行、集合一致、家族一致、家族内顺序一致 —— 四项一起守。"""
    rows = _table1()
    code_order = [p.id for family in FAMILIES for p in PRIMITIVES if p.family == family]
    assert [pid for _fam, pid, _zh, _rest in rows] == code_order

    by_id = {p.id: p for p in PRIMITIVES}
    for family_zh, pid, name_zh, _rest in rows:
        assert FAMILIES[by_id[pid].family] == family_zh, f"{pid} 的家族与代码不一致"
        assert by_id[pid].name_zh == name_zh, f"{pid} 的中文名与代码不一致"


def test_table1_rows_all_carry_literature():
    """每一行都要有理论依据与 PMID（文献稿不能只剩标题）。"""
    for _fam, pid, _zh, rest in _table1():
        cells = [cell.strip() for cell in rest.split("|")]
        theory, pmid_cell = cells[-2], cells[-1]
        assert theory, f"{pid} 缺理论依据"
        pmids = [chunk.strip() for chunk in pmid_cell.split(",") if chunk.strip()]
        assert pmids, f"{pid} 缺 PMID"
        assert all(pmid.isdigit() and len(pmid) >= 7 for pmid in pmids), (
            f"{pid} 的 PMID 不像 PMID：{pmids}"
        )


def test_table2_covers_the_scored_channels_and_flags_the_control():
    rows = _table2()
    dims = [dim for _n, _label, dim, _rest in rows if dim != CONTROL_DIM]
    assert set(dims) == set(PHENOTYPE_DIMS), "表 2 的维度集合与词表不一致"
    assert len(dims) == len(PHENOTYPE_DIMS), "表 2 有重复维度行"

    control = [(_n, label) for _n, label, dim, _rest in rows if dim == CONTROL_DIM]
    assert len(control) == 1, "对照通道必须且只能有一行"
    number, label = control[0]
    assert "不在" in label and "Supplementary Table 2" in label, (
        "对照通道行必须显式标注它不在 Supplementary Table 2 内（它是项目新增的）"
    )
    assert CONTROL_DIM in SCORED_CHANNELS


def test_table2_rows_all_carry_anchor_and_literature():
    for _n, _label, dim, rest in _table2():
        assert DIM_ANCHORS.get(dim, {}).get("watch"), f"{dim} 在代码里没有观察锚点"
        if dim == CONTROL_DIM:
            continue
        cells = [cell.strip() for cell in rest.split("|")]
        # 末尾的「文献依据 · PMID」列：六个表型维度都必须有 PMID
        assert re.search(r"\b\d{7,8}\b", cells[-1]), f"{dim} 那行缺 PMID"
