"""Web API 测试：用真实 HTTP 请求驱动界面所依赖的每个端点。

不启动阻塞式 serve_forever，而是在后台线程里跑 ThreadingHTTPServer，
确保测试与真实运行走同一条代码路径。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pacs.api.server import PacsHandler
from pacs.phenotype import SCORED_CHANNELS
from pacs.config import Settings
from pacs.store.db import Store


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    base = tmp_path_factory.mktemp("pacs_ui")
    settings = Settings.load(base_dir=base)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    PacsHandler.settings = settings
    PacsHandler.store = store
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), PacsHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{port}", base
    httpd.shutdown()
    httpd.server_close()
    store.close()


def _get(base_url: str, path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(base_url + path, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(base_url: str, path: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _wait_job(base_url: str, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _status, payload = _get(base_url, f"/api/job?job_id={job_id}")
        job = payload.get("job") or {}
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(0.25)
    raise AssertionError("任务超时")


# ---------------------------------------------------------------------------
# 静态与只读端点
# ---------------------------------------------------------------------------


def test_index_shell_served(server):
    """界面外壳：无内联脚本/样式，靠 /static 下的模块与样式表组装。"""
    base_url, _ = server
    with urllib.request.urlopen(base_url + "/", timeout=10) as resp:
        html = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "PACS" in html and "可扰动人工认知系统" in html
    # 模块化前端：入口是 ES 模块，样式分四层
    assert '<script type="module" src="/static/js/app.js">' in html
    for css in ("tokens.css", "base.css", "layout.css", "components.css"):
        assert f"/static/css/{css}" in html
    assert "<style>" not in html and "<script>" not in html
    # 骨架：顶栏 / 侧栏（导航 + 当前大脑 + 大脑库）/ 视图容器 / 门禁弹窗
    for node in ('id="topbar-brain"', 'id="sidebar"', 'id="nav"', 'id="view"',
                 'id="side-current"', 'id="brain-list"', 'id="brain-name"',
                 'id="btn-sidebar-save"', 'id="gate"'):
        assert node in html, f"外壳缺少节点：{node}"
    # 打开时要求配置 LLM API：门禁弹窗带两套配置与显式离线入口
    assert "先配置 LLM API" in html and "离线 mock 模式进入" in html
    assert "模拟大脑" in html and "提问者 / 分析者" in html


def test_static_assets_are_served(server):
    """前端资源必须可从 /static 取到，否则界面只是一张空壳。"""
    base_url, _ = server
    assets = [
        "/static/js/app.js",
        "/static/js/core/dom.js",
        "/static/js/core/api.js",
        "/static/js/core/store.js",
        "/static/js/core/prefs.js",
        "/static/js/core/job.js",
        "/static/js/core/format.js",
        "/static/js/ui/toast.js",
        "/static/js/ui/widgets.js",
        "/static/js/ui/sidebar.js",
        "/static/js/ui/modal.js",
        "/static/js/views/model.js",
        "/static/js/views/perturb.js",
        "/static/js/views/phenotype.js",
        "/static/js/views/bank.js",
    ]
    for asset in assets:
        with urllib.request.urlopen(base_url + asset, timeout=10) as resp:
            assert resp.status == 200, f"{asset} 未提供"
            assert resp.read(), f"{asset} 为空"
    # 已删除的 ④ 视图不应还能取到（删掉的东西不留在路由里）
    try:
        urllib.request.urlopen(base_url + "/static/js/views/mechanism.js", timeout=10)
        raise AssertionError("views/mechanism.js 应已删除")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404, f"应返回 404，实际 {exc.code}"


def test_screens_defined_in_nav_module(server):
    """界面定义在导航模块里（外壳不写死标签），且只有三个。"""
    base_url, _ = server
    with urllib.request.urlopen(base_url + "/static/js/ui/sidebar.js", timeout=10) as resp:
        source = resp.read().decode("utf-8")
    for screen in ("模型设置", "扰动注入", "表型观察"):
        assert screen in source
    for screen_id in ('id: "model"', 'id: "perturb"', 'id: "phenotype"'):
        assert screen_id in source
    assert 'id: "mechanism"' not in source
    with urllib.request.urlopen(base_url + "/static/js/app.js", timeout=10) as resp:
        assert "mechanism" not in resp.read().decode("utf-8")


def test_phenotype_readings_come_from_the_single_brain_report(server):
    """③ 的「当前大脑的表型读数」锚定大脑本身，数据取自 /api/brain-report。"""
    base_url, _ = server
    with urllib.request.urlopen(
        base_url + "/static/js/views/phenotype.js", timeout=10
    ) as resp:
        phenotype_src = resp.read().decode("utf-8")
    assert "/api/brain-report" in phenotype_src
    with urllib.request.urlopen(base_url + "/static/js/ui/readings.js", timeout=10) as resp:
        readings_src = resp.read().decode("utf-8")
    # 扰动与表型在界面上不允许有关联：读数卡不得出现 θ、不得做「读数 vs 扰动」的比对
    assert "blind_audit" not in readings_src
    assert "受损过程所支撑" not in readings_src
    assert "perturbationText" not in readings_src


def test_phenotype_auto_runs_both_llms_behind_a_reveal_button(server):
    """③ 两个 LLM 的行为自动运行、界面收起、按钮唤起；默认只剩读数/历史/访谈。

    这一条守的是界面约定的结构性事实（自动观察的链路存在、过程默认藏在抽屉里、
    读数以雷达图呈现），细节靠浏览器实跑看。
    """
    base_url, _ = server
    with urllib.request.urlopen(
        base_url + "/static/js/views/phenotype.js", timeout=10
    ) as resp:
        phenotype_src = resp.read().decode("utf-8")

    # 自动运行：进入界面先看有没有读数，没有就跑一轮（与手动共用 /api/dialogue/start）
    assert "ensureReading" in phenotype_src and "autoObserve" in phenotype_src
    assert "/api/dialogue/start" in phenotype_src
    # 自动观察没有开关了：不再有「要不要自动跑」这个选择，首选项里也就不该再留着它
    assert 'prefs.get("autoObserve")' not in phenotype_src
    with urllib.request.urlopen(base_url + "/static/js/core/prefs.js", timeout=10) as resp:
        prefs_src = resp.read().decode("utf-8")
    assert "autoObserve" not in prefs_src
    # 自动跑失败过就不再反复重试，只提示手动重试
    assert "autoFailed" in phenotype_src

    # 三个「过程」区块不再显示任何内容：标题与状态行都不得出现
    for gone_title in ("读数是怎么来的", "对话探针 · 提问者", "表型反馈 · 分析者"):
        assert gone_title not in phenotype_src, f"{gone_title} 不应再出现在界面上"

    # 隐藏 + 按钮唤起：整段过程默认收起，开关按钮挂在读数卡上
    assert 'id="probe-drawer" hidden' in phenotype_src
    assert 'prefs.get("probeDrawerOpen")' in phenotype_src
    # 预载读数来源那段对话时不得顺手把抽屉拉开（否则「默认收起」形同虚设）
    assert "reveal: false" in phenotype_src
    assert 'id="dlg-list-card" hidden' in phenotype_src
    # 过程卡与它的一堆控件都不在了
    assert 'id="dlg-chat"' in phenotype_src
    for gone in ('id="dlg-turns"', 'id="dlg-goals"', 'id="dlg-probe-mode"',
                 'id="dlg-fb-card"', 'id="dlg-fb"', 'id="dlg-set"',
                 'id="observe-status"', 'id="dlg-bar"', 'id="dlg-auto-observe"',
                 'id="btn-reobserve"', 'id="btn-toggle-probe"'):
        assert gone not in phenotype_src, f"{gone} 应随「去掉过程区块」一并移除"

    # 三块默认内容仍在：读数（雷达图）、历史对话、访谈
    assert 'id="readings-panel"' in phenotype_src
    assert "历史对话" in phenotype_src
    assert "interview-panel" in phenotype_src

    # 观测通路上不许出现 θ：本界面不渲染扰动、不与读数做比对
    assert "perturbationText" not in phenotype_src
    assert "blind_audit" not in phenotype_src

    with urllib.request.urlopen(base_url + "/static/js/ui/readings.js", timeout=10) as resp:
        readings_src = resp.read().decode("utf-8")
    # 读数以雷达图为主呈现：每维一条轴 + 外圈即「与同龄典型无异」
    assert 'id="rd-radar"' in readings_src
    assert "polygon" in readings_src and "dimsAll" in readings_src
    assert "外圈 = 100" in readings_src          # 分制 0–100（协议阶段 10）
    assert "dim_details" in readings_src         # 逐维量化记录：得分/置信度/偏离/观察/任务级测量
    # 读数卡头部只放三个动作（重新观察 / 查看对话过程 / 查看这段对话）
    for control in ('id="rd-reobserve"', 'id="rd-toggle-probe"', 'id="rd-open"'):
        assert control in readings_src, f"{control} 是过程唯一的入口，不能一起删掉"


def test_perturb_brain_name_follows_the_theta(server):
    """② 的名称框必须跟着 θ 走。

    否则「生成并载入」换了一个 θ、名称框还留着上一个大脑的名字，注入出来的新大脑会以
    旧名字入库（大脑库里出现两条名字几乎一样、θ 却不同的记录）。
    """
    base_url, _ = server
    with urllib.request.urlopen(
        base_url + "/static/js/views/perturb.js", timeout=10
    ) as resp:
        src = resp.read().decode("utf-8")
    assert "syncBrainName" in src
    # 旧写法「只在空的时候填」正是这个 bug 的来源，不能再回来
    assert 'if (!$("brain-name").value.trim())' not in src


def test_static_route_blocks_path_traversal(server):
    """静态路由必须把 `..` 挡在 static 目录之外。"""
    base_url, _ = server
    for path in ("/static/../index.html", "/static/../../pyproject.toml"):
        try:
            with urllib.request.urlopen(base_url + path, timeout=10) as resp:
                raise AssertionError(f"{path} 不应可达（HTTP {resp.status}）")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403, f"{path} 应返回 403，实际 {exc.code}"


def test_index_page_has_no_static_reference_tables(server):
    """静态本体/映射表已移出界面（机制分析只呈现与当前大脑有关的结果）。"""
    base_url, _ = server
    with urllib.request.urlopen(base_url + "/", timeout=10) as resp:
        html = resp.read().decode("utf-8")
    for static_block in ('id="ont-prim"', 'id="ont-sys"', 'id="ont-map"', 'id="ont-items"'):
        assert static_block not in html


def test_frontend_assets_are_not_cached(server):
    """本地工具不缓存前端资源，避免「代码已改、界面没变」。"""
    base_url, _ = server
    for path in ("/", "/static/js/app.js"):
        with urllib.request.urlopen(base_url + path, timeout=10) as resp:
            assert "no-store" in (resp.headers.get("Cache-Control") or ""), path


# ---------------------------------------------------------------------------
# 群体运行：不再有 HTTP 入口（界面只有「观察当前大脑」），由 CLI/脚本直接调引擎
# ---------------------------------------------------------------------------


def _seed_run(base_dir, *, mode="all", limit=None, turns=3, seed=11):
    """直接在库里造一次群体运行（走 `run_design` + store，与 CLI 同一条路径）。

    为什么测试也走引擎而不是 HTTP：群体运行没有界面入口了（`/api/run` 与量表定量管线
    随行为电池一并删除）。要在测试里覆盖「分析 / 导出 / 冻结」这些**只读**端点，
    就得先有一批真实剖面 —— 用引擎造，正是 CLI `pacs run` 的做法。
    """
    from pacs.brain.perturb import DEFAULT_DOSE_LEVELS, generic_degradation_control, sample_design
    from pacs.brain import examiner_bank as eb
    from pacs.engine import new_run_id, run_design
    from pacs.llm.provider import MockProvider
    from pacs.phenotype import SCORED_CHANNELS

    settings = Settings.load(base_dir=base_dir)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    designs = (
        sample_design("single", doses=tuple(DEFAULT_DOSE_LEVELS), seed=seed)
        + sample_design("pair", pair_dose=0.2, seed=seed, max_pairs=20)
        + [generic_degradation_control(0.5, seed=seed + rep) for rep in range(5)]
    )
    if mode == "intact":
        designs = [d for d in designs if d.is_intact]
    if limit:
        # 与 CLI 的 `_apply_limit` 同一取舍：完整心智基线 + 通用退化对照**必须留下**，
        # 否则 Control 1 / Control 2 会变成「缺失」而不是「判不了」。
        keep = [d for d in designs if d.is_intact or d.generic_degradation > 0]
        rest = [d for d in designs if not (d.is_intact or d.generic_degradation > 0)]
        designs = keep + rest[: max(0, limit - len(keep))]
    provider = MockProvider()
    run_id = new_run_id("seeded")
    try:
        for d in designs:
            store.save_perturbation(d)
        store.save_run(run_id, "mock:mock-cognitive-mind", {"mode": mode, "n_designs": len(designs)})
        result = run_design(
            provider, designs, run_id=run_id, model_id="mock:mock-cognitive-mind",
            seed=seed, initialize=False, on_profile=store.save_profile,
            examiner=provider, analyst=provider,
            probes=eb.dialogue_probes(goals=tuple(SCORED_CHANNELS),
                                      n=turns, seed=seed),
            dialogue_turns=turns,
        )
    finally:
        store.close()
    return run_id, result


def test_ontology_endpoint_exposes_primitives_and_assays(server):
    base_url, _ = server
    status, data = _get(base_url, "/api/ontology")
    assert status == 200 and data["ok"]
    assert len(data["primitives"]) == 31
    assert len(data["systems"]) >= 3
    # 电池题目不再存在（强制选择题库已删除）；assay 由观察者条目库派生
    assert "items" not in data
    assert len(data["assays"]) == 31
    # 探查登记不再携带任何维度声称（构造属性与测量维度之间不允许有预设对应）
    assert all(a["primary_probe"] and a["primary_probe_text"] for a in data["assays"])
    assert all("target_dim" not in a for a in data["assays"])
    assert data["checks"]["ontology_problems"] == []
    assert data["checks"]["without_assay"] == []


def test_map_endpoint_is_gone(server):
    """预设的 primitive → 维度 映射端点已整体删除：不允许再有任何映射出口。"""
    base_url, _ = server
    status, data = _get(base_url, "/api/map")
    assert status == 404 and data["ok"] is False


def test_config_never_leaks_api_key(server):
    """安全红线：任何 GET 响应都不得包含 Key 明文。"""
    base_url, _ = server
    _post(base_url, "/api/config", {"provider": "openai", "api_key": "DUMMY-KEY-DO-NOT-USE-12345",
                                    "model": "gpt-4o-mini"})
    status, data = _get(base_url, "/api/config")
    assert status == 200
    assert data["api_key_set"] is True
    assert "DUMMY-KEY-DO-NOT-USE-12345" not in json.dumps(data)
    # 复位成 mock，避免影响后续用例
    _post(base_url, "/api/config", {"provider": "mock"})


def test_prompt_endpoint_returns_architecture(server):
    base_url, _ = server
    status, data = _post(base_url, "/api/prompt",
                         {"kind": "single", "primitive": "social_reward_weighting", "dose": 0.2})
    assert status == 200
    assert "社会奖励加权" in data["prompt"]
    assert "BRAIN_FINGERPRINT" in data["prompt"]
    assert "0.20" in data["prompt"]


def test_group_run_endpoints_are_gone(server):
    """群体运行的 HTTP 入口随行为电池一并删除：界面不再有「定量通道」可驱动它。

    这一条是**刻意**断言「端点不存在」的：删掉入口是个能力变更，不能只体现为
    「某个测试文件变短了」。群体运行仍在 CLI（`pacs run`）与引擎里，界面不再暴露。
    """
    base_url, _ = server
    for path in ("/api/plan", "/api/run", "/api/responses"):
        status, data = _post(base_url, path, {"mode": "intact"})
        assert status == 404, path
        assert data["ok"] is False


def test_analysis_before_data_returns_conflict(server):
    base_url, _ = server
    status, data = _get(base_url, "/api/analysis?endpoint=social_communication_pragmatics")
    assert status == 409
    assert data["ok"] is False


# ---------------------------------------------------------------------------
# 动作端点
# ---------------------------------------------------------------------------


def test_init_brain_job_roundtrip(server):
    base_url, _ = server
    status, data = _post(base_url, "/api/init-brain", {"kind": "intact"})
    assert status == 202
    job = _wait_job(base_url, data["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["ok"] is True
    assert "SELF_STATE: OK" in job["result"]["raw"]


def test_run_pipeline_then_analysis_export_freeze(server):
    """一次群体运行 → 表型 → 分析 → 导出 → 冻结（运行由引擎造，其余走 HTTP）。"""
    base_url, base_dir = server
    run_id, result = _seed_run(base_dir, mode="all", limit=90, turns=4)
    assert len(result.profiles) > 5
    assert all(p.parse_rate == 1.0 for p in result.profiles)

    # 表型
    _status, profiles = _get(base_url, f"/api/profiles?run_id={run_id}")
    assert profiles["n"] == len(result.profiles)
    first = profiles["profiles"][0]
    # 剖面 = 计分通道（6 个表型维度 + 对照通道）；对照通道不进雷达图，但必须在剖面里
    assert set(first["dims"]) == set(SCORED_CHANNELS)

    # 分析
    _status, analysis = _get(
        base_url, f"/api/analysis?run_id={run_id}&endpoint=social_communication_pragmatics"
    )
    atlas = analysis["atlas"]
    # 对照通道由分析者评分 → Control 1 判得了（它在最小判别任务集里仍要被排除）
    assert atlas["generic_degradation"] is not None
    assert atlas["convergence"]["n_candidates"] >= 2
    assert atlas["convergence"]["is_heterogeneous"] is True
    assert atlas["convergence"]["minimal_task_set"]
    # 判别维度必须排除终点与对照通道
    assert "general_cognition" not in atlas["convergence"]["minimal_task_set"]
    assert "social_communication_pragmatics" not in atlas["convergence"]["minimal_task_set"]
    assert "机制异质性" in analysis["markdown"]
    assert "不用于医学诊断" in analysis["markdown"]

    # 导出功能已整体删除（不产出任何维度投影/描述集）
    _status, gone = _post(base_url, "/api/export", {"theta_id": profiles["profiles"][0]["theta_id"]})
    assert gone["ok"] is False

    # 由测量估计的机制—维度剖面：对应关系的唯一来源，且必须是测量产物
    mech = atlas["mechanism_profile"]
    assert mech["rows"], "应能从单扰动剖面估计出机制—维度剖面"
    assert mech["rows"][0]["deltas"], "每行要有逐通道 Δ"
    assert any("不得回写" in n for n in mech["notes"])

    # Atlas 报告
    _status, atlas_out = _post(base_url, "/api/atlas-md",
                               {"run_id": run_id, "endpoint": "social_communication_pragmatics"})
    assert atlas_out["ok"] is True
    with urllib.request.urlopen(
        base_url + "/api/download?path=" + urllib.parse.quote(atlas_out["markdown_path"]),
        timeout=15,
    ) as resp:
        md = resp.read().decode("utf-8")
    assert "认知扰动图谱" in md
    for heading in ("## ② 构造效度", "## ④ 机制异质性", "## ⑧ 实测表型出口与维度口径",
                    "## ⑨ 人类实验设计建议与冻结"):
        assert heading in md, heading
    # ⑧ 章必须写明不内建映射，且不得再出现 Model 1 映射表
    assert "不内建任何「构造属性 → 测量维度」的映射" in md
    assert "Model 1" not in md

    # 冻结
    _status, frozen = _post(base_url, "/api/freeze", {"run_id": run_id})
    assert frozen["ok"] is True
    assert frozen["problems"] == []
    assert len(frozen["digests"]) == 5
    assert "negative_predictions" in frozen["human_predictions"]


def test_exports_endpoint_is_gone(server):
    """导出清单端点随导出功能一并删除。"""
    base_url, _ = server
    status, data = _get(base_url, "/api/exports")
    assert status == 404 and data["ok"] is False


def test_download_blocks_path_traversal(server):
    """路径穿越防护：不得下载项目目录之外的文件。"""
    base_url, _ = server
    outside = "C:/Windows/win.ini" if __import__("os").name == "nt" else "/etc/passwd"
    status, data = _get(base_url, "/api/download?path=" + urllib.parse.quote(outside))
    assert status == 403
    assert data["ok"] is False


def test_run_runs_and_counts_endpoints(server):
    base_url, base_dir = server
    _seed_run(base_dir, mode="intact", limit=3, turns=2, seed=5)
    status, data = _get(base_url, "/api/runs")
    assert status == 200
    assert data["runs"]
    assert data["counts"]["phenotypes"] > 0
    # 逐题作答表已随电池删除：新库不再建它，计数里也不该出现
    assert "responses" not in data["counts"]


def test_observation_endpoint_estimates_and_requires_confirmation(server):
    """观察当前大脑是界面唯一的模型调用入口：真实 provider 下必须报出预计调用数。

    轮数 × 2 + 1 是每次观察的调用数（提问 + 作答 + 分析），因此它天然很小；
    这一条守的是「至少要让用户看见要花多少次调用」，而不是电池时代的 200 次阈值。
    """
    base_url, _ = server
    _s, injected = _post(base_url, "/api/inject", {"kind": "intact"})
    _wait_job(base_url, injected["job_id"], timeout=120)
    status, start = _post(base_url, "/api/dialogue/start", {"n_turns": 5})
    assert status == 202
    assert start["estimated_calls"] == 5 * 2 + 1
    job = _wait_job(base_url, start["job_id"], timeout=180)
    assert job["status"] == "done", job
