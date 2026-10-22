"""表型景观（FR-12.1）：`perturbation × dose × 表型读数` 的全景观。"""

from __future__ import annotations

from dataclasses import dataclass

from ..phenotype import PHENOTYPE_DIMS
from ..phenotype import PhenotypeProfile

# 单元格符号（按表型值分档，值越高越「完整」）
BANDS: tuple[tuple[float, str], ...] = (
    (0.85, "##"),
    (0.70, "++"),
    (0.55, "--"),
    (0.40, ".."),
    (0.00, "  "),
)


def cell_symbol(value: float | None) -> str:
    if value is None:
        return "??"
    for threshold, symbol in BANDS:
        if value >= threshold:
            return symbol
    return "  "


@dataclass
class LandscapeCell:
    primitive_id: str
    dose: float
    theta_id: str
    model_id: str
    dims: dict[str, float | None]

    def to_dict(self) -> dict:
        return {
            "primitive_id": self.primitive_id,
            "dose": self.dose,
            "theta_id": self.theta_id,
            "model_id": self.model_id,
            "dims": self.dims,
        }


def landscape_cells(
    profiles: list[PhenotypeProfile], *, model_id: str | None = None
) -> list[LandscapeCell]:
    """把单因素扰动剖面整理成景观单元格（组合与通用退化另表处理）。"""
    cells: list[LandscapeCell] = []
    for p in profiles:
        if model_id and p.model_id != model_id:
            continue
        if p.generic_degradation > 0:
            continue
        active = {k: v for k, v in p.theta.items() if v < 0.999}
        if len(active) != 1:
            continue
        pid, dose = next(iter(active.items()))
        cells.append(
            LandscapeCell(
                primitive_id=pid,
                dose=round(float(dose), 4),
                theta_id=p.theta_id,
                model_id=p.model_id,
                dims=dict(p.dims),
            )
        )
    cells.sort(key=lambda c: (c.primitive_id, c.dose))
    return cells


def grid(
    cells: list[LandscapeCell],
) -> dict[str, dict[float, dict[str, float | None]]]:
    """整理为 `{primitive: {dose: dims}}`。"""
    out: dict[str, dict[float, dict[str, float | None]]] = {}
    for c in cells:
        out.setdefault(c.primitive_id, {})[c.dose] = c.dims
    return out


def render_heatmap(
    cells: list[LandscapeCell],
    dim: str,
    *,
    label: str | None = None,
) -> str:
    """渲染某一维度的 ASCII 热图：行 = primitive，列 = dose。"""
    g = grid(cells)
    if not g:
        return f"（{dim}：无数据）"
    doses = sorted({d for cells_at in g.values() for d in cells_at})
    header = f"{'primitive':<34}" + "".join(f"{d:>6.1f}" for d in doses)
    lines = [f"表型景观 · {label or dim}", "=" * len(header)]
    lines.append(header)
    lines.append("-" * len(header))
    for pid in sorted(g):
        row = f"{pid:<34}"
        for d in doses:
            value = g[pid].get(d, {}).get(dim)
            row += f"{cell_symbol(value):>6}"
        lines.append(row)
    lines.append("-" * len(header))
    lines.append("图例： ## ≥.85  ++ ≥.70  -- ≥.55  .. ≥.40  (空) <.40   ?? 无有效作答")
    return "\n".join(lines)


def render_all_dimensions(cells: list[LandscapeCell]) -> str:
    """各表型维度的热图汇总。"""
    blocks: list[str] = []
    for dim, name in PHENOTYPE_DIMS.items():
        blocks.append(render_heatmap(cells, dim, label=name))
    return "\n\n".join(blocks)


def profile_table(cells: list[LandscapeCell]) -> str:
    """逐单元格罗列各表型维度数值（供报告与人工核对）。"""
    dims = list(PHENOTYPE_DIMS)
    header = f"{'primitive':<34}{'dose':>6}" + "".join(f"{d[:12]:>14}" for d in dims)
    lines = ["单因素扰动表型明细", "=" * len(header), header, "-" * len(header)]
    for c in cells:
        row = f"{c.primitive_id:<34}{c.dose:>6.1f}"
        for d in dims:
            value = c.dims.get(d)
            row += f"{('—' if value is None else f'{value:.2f}'):>14}"
        lines.append(row)
    return "\n".join(lines)
