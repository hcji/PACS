"""四界面新增端点的 API 测试：启动自动初始化、大脑库、系统生成、注入、对话探针、量表上传。

与 `test_api.py` 同一套路：后台线程跑真实的 `ThreadingHTTPServer`，让测试与真实运行
走同一条代码路径（而不是绕开 HTTP 直接调函数）。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pacs.api.server import BOOT, PacsHandler, _launch_auto_init
from pacs.phenotype import SCORED_CHANNELS
from pacs.config import Settings
from pacs.store.db import Store


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """起一台真实的 HTTP 服务，并**真跑一次启动自动初始化**。

    为什么不像 `test_api.py` 那样只起服务：本文件的头几个用例断言「开机就有一具正常大脑」，
    那是界面开局的第一屏所依赖的状态。故这里先把 `BOOT` 清空、同步等初始化线程跑完，
    再对外提供服务 —— 与 `serve()` 的启动顺序一致（只是这里不等后台线程自己跑）。
    """
    base = tmp_path_factory.mktemp("pacs_api_v2")
    settings = Settings.load(base_dir=base)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    PacsHandler.settings = settings
    PacsHandler.store = store
    BOOT.clear()
    _launch_auto_init(settings, store).join(timeout=60)
    assert BOOT.get("status") == "ready", BOOT.get("message")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), PacsHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{port}", base
    httpd.shutdown()
    httpd.server_close()
    store.close()


def _seed_run(base_dir, *, limit=3, turns=3, seed=11):
    """直接在库里造一次群体运行（走 `run_design` + store，与 CLI 同一条路径）。"""
    from pacs.brain import examiner_bank as eb
    from pacs.brain.perturb import DEFAULT_DOSE_LEVELS, sample_design
    from pacs.engine import new_run_id, run_design
    from pacs.llm.provider import MockProvider

    settings = Settings.load(base_dir=base_dir)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    provider = MockProvider()
    run_id = new_run_id("seeded")
    # 与 CLI `_apply_limit` 同一取舍：完整心智基线 + 通用退化对照必须留下，
    # 否则 Control 1 / Control 2 会变成「缺失」而不是「判不了」。
    keep = list(sample_design("intact", seed=seed))
    rest = sample_design("single", doses=tuple(DEFAULT_DOSE_LEVELS), seed=seed)
    designs = (keep + rest)[: max(limit, len(keep))]
    def on_dialogue(session, feedback) -> None:
        """与 API/CLI 一样把每段对话落盘：群体运行的可回溯证据就是这些会话。"""
        store.save_dialogue_session(session.session_id, session.to_dict())
        for index, turn in enumerate(session.turns):
            store.save_dialogue_turn(
                session.session_id, index, turn.get("role", ""), turn.get("content", ""), turn
            )
        store.save_feedback(session.session_id, session.theta_id,
                            feedback.get("model_id", ""), feedback)

    try:
        for d in designs:
            store.save_perturbation(d)
        store.save_run(run_id, "mock:mock-cognitive-mind",
                       {"mode": "seeded", "n_designs": len(designs)})
        run_design(
            provider, designs, run_id=run_id, model_id="mock:mock-cognitive-mind",
            seed=seed, initialize=False, on_profile=store.save_profile,
            on_dialogue=on_dialogue,
            examiner=provider, analyst=provider,
            probes=eb.dialogue_probes(goals=tuple(SCORED_CHANNELS),
                                      n=turns, seed=seed),
            dialogue_turns=turns,
        )
    finally:
        store.close()
    return run_id


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
# 启动与门禁
# ---------------------------------------------------------------------------


def test_boot_reports_gate_and_auto_initialized_normal_brain(server):
    """门禁状态 + 启动自动初始化的完整心智 —— 界面开局就靠这两块。"""
    base_url, _base = server
    status, data = _get(base_url, "/api/boot")
    assert status == 200 and data["ok"] is True
    # 门禁口径：两套 LLM 都就绪才算就绪（离线 mock 下不算就绪，界面据此弹配置窗）
    assert set(data["configured"]) == {"brain", "examiner", "ready"}
    assert data["gate"]["requires_setup"] is (not data["configured"]["ready"])
    assert isinstance(data["gate"]["reasons"], list)

    boot = data["boot"]
    assert boot["status"] == "ready", boot["message"]
    assert boot["init"]["ok"] is True and boot["init"]["fingerprint"]
    # 启动初始化出来的就是当前大脑，而且必须是**完整心智**（无扰动）
    assert boot["active_brain"]["theta_id"] == boot["init"]["theta_id"]
    assert boot["active_brain"]["impaired"] == {}
    assert data["active_brain"]["source"] == "intact"
    assert data["dialogue"]["default_n_turns"] >= 1


def test_boot_lists_the_startup_brain_in_library(server):
    """启动初始化的正常大脑要同时进大脑库并成为当前大脑（规则 7）。"""
    base_url, _base = server
    status, data = _get(base_url, "/api/brains")
    assert status == 200 and data["ok"] is True
    startup = [b for b in data["brains"] if b["source"] == "intact"]
    assert len(startup) == 1, [b["name"] for b in data["brains"]]
    assert startup[0]["name"] == "完整心智（启动基线）"
    assert data["active"]["brain_id"] == startup[0]["brain_id"]
    # 大脑库要带全部 31 维完整 θ（界面按它画滑杆）与逐过程的中文名
    assert set(data["active"]["values"]) == {p["id"] for p in data["primitives"]}
    assert len(data["primitives"]) == 31
    assert all(p["name_zh"] for p in data["primitives"])


def test_config_has_two_slots_and_never_leaks_keys(server):
    base_url, _base = server
    status, _ = _post(base_url, "/api/config", {
        "provider": "openai", "api_key": "DUMMY-BRAIN-KEY-DO-NOT-USE", "model": "gpt-4o-mini",
        "examiner": {"provider": "openai", "api_key": "DUMMY-EXAMINER-KEY-DO-NOT-USE",
                     "model": "deepseek-chat"},
    })
    assert status == 200
    _status, data = _get(base_url, "/api/config")
    assert data["provider"] == "openai" and data["api_key_set"] is True
    assert data["examiner"]["provider"] == "openai"
    assert data["examiner"]["model"] == "deepseek-chat"
    assert data["examiner"]["api_key_set"] is True
    assert data["configured"]["ready"] is True
    raw = json.dumps(data, ensure_ascii=False)
    assert "DUMMY-BRAIN-KEY-DO-NOT-USE" not in raw
    assert "DUMMY-EXAMINER-KEY-DO-NOT-USE" not in raw

    # 复位成 mock，避免影响后续用例
    _post(base_url, "/api/config", {"provider": "mock", "examiner": {"provider": "mock"}})
    _status, back = _get(base_url, "/api/config")
    assert back["configured"]["ready"] is False


def test_config_rejects_unknown_provider(server):
    base_url, _base = server
    status, data = _post(base_url, "/api/config", {"provider": "gemini"})
    assert status == 400 and "未知 provider" in data["error"]
    status, data = _post(base_url, "/api/config", {"examiner": {"provider": "gemini"}})
    assert status == 400 and "未知 examiner provider" in data["error"]


# ---------------------------------------------------------------------------
# 规则 4：提示词构造过程 + 系统生成
# ---------------------------------------------------------------------------


def test_prompt_endpoint_returns_steps_matching_prompt(server):
    base_url, _base = server
    _status, data = _post(base_url, "/api/prompt",
                          {"kind": "custom", "values": {"attention_shifting": 0.2,
                                                        "inhibition": 0.4}})
    steps = data["steps"]
    # 被扰动的 θ 会多一段「机制注入」（协议阶段 3/4/5）
    assert [s["key"] for s in steps] == [
        "persona", "primitives", "injection", "levels", "consequences", "emergence",
        "conduct", "fingerprint",
    ]
    # 过程可视化与实际发送内容同源
    assert "\n\n".join(s["text"] for s in steps) == data["prompt"]
    assert "社会奖励加权" in data["prompt"] and "BRAIN_FINGERPRINT" in data["prompt"]
    assert len(steps[1]["rows"]) == 31


@pytest.mark.parametrize("mode", ["random_single", "random_pair", "random_lhs", "generic", "intact"])
def test_generate_perturbation_modes(server, mode):
    base_url, _base = server
    _status, data = _post(base_url, "/api/perturb/generate", {"mode": mode, "seed": 7})
    assert data["theta_id"].startswith("theta_")
    assert data["describe"]
    assert set(data["values"])
    if mode == "intact":
        assert data["impaired"] == {}
    if mode == "random_pair":
        assert len(data["impaired"]) == 2


def test_generate_perturbation_is_deterministic_by_seed(server):
    base_url, _base = server
    _s, first = _post(base_url, "/api/perturb/generate", {"mode": "random_lhs", "seed": 11})
    _s, second = _post(base_url, "/api/perturb/generate", {"mode": "random_lhs", "seed": 11})
    _s, other = _post(base_url, "/api/perturb/generate", {"mode": "random_lhs", "seed": 12})
    assert first["theta_id"] == second["theta_id"]
    assert first["theta_id"] != other["theta_id"]


def test_generate_perturbation_rejects_unknown_mode(server):
    base_url, _base = server
    status, data = _post(base_url, "/api/perturb/generate", {"mode": "nope"})
    assert status == 400 and "未知生成模式" in data["error"]


# ---------------------------------------------------------------------------
# 规则 4 + 7：注入 + 大脑库保存/切换/重命名/删除
# ---------------------------------------------------------------------------


def test_inject_saves_brain_and_reinitializes(server):
    base_url, _base = server
    _status, data = _post(base_url, "/api/inject", {
        "kind": "custom", "values": {"attention_shifting": 0.2},
        "name": "注意力转移 0.2 组",
    })
    assert data["brain"]["name"] == "注意力转移 0.2 组"
    assert data["steps"] and data["prompt"]
    job = _wait_job(base_url, data["job_id"])
    assert job["status"] == "done"
    assert job["result"]["ok"] is True
    assert job["result"]["theta_id"] == data["theta_id"]

    _s, brains = _get(base_url, "/api/brains")
    assert brains["active"]["name"] == "注意力转移 0.2 组"
    assert brains["active"]["impaired"] == {"attention_shifting": 0.2}


def test_activate_switch_rename_delete_brain(server):
    base_url, _base = server
    _s, injected = _post(base_url, "/api/inject",
                         {"kind": "custom", "values": {"inhibition": 0.4}, "name": "抑制 0.4 组"})
    injected_id = injected["brain"]["brain_id"]

    _s, brains = _get(base_url, "/api/brains")
    baseline = next(b for b in brains["brains"] if b["brain_id"] != injected_id)

    # 切回基线：会重新初始化并改变「当前大脑」
    _status, switched = _post(base_url, "/api/brains/activate", {"brain_id": baseline["brain_id"]})
    _wait_job(base_url, switched["job_id"])
    _s, after = _get(base_url, "/api/brains")
    assert after["active"]["brain_id"] == baseline["brain_id"]

    # 重命名 + 删除
    assert _post(base_url, "/api/brains/rename",
                 {"brain_id": injected_id, "name": "抑制组（已改名）"})[0] == 200
    _s, after = _get(base_url, "/api/brains")
    assert any(b["name"] == "抑制组（已改名）" for b in after["brains"])

    assert _post(base_url, "/api/brains/delete", {"brain_id": injected_id})[1]["deleted"] is True
    _s, after = _get(base_url, "/api/brains")
    assert all(b["brain_id"] != injected_id for b in after["brains"])


def test_brain_endpoints_validate_input(server):
    base_url, _base = server
    assert _post(base_url, "/api/brains/activate", {})[0] == 400
    assert _post(base_url, "/api/brains/activate", {"brain_id": "brain_nope"})[0] == 400
    assert _post(base_url, "/api/brains/rename", {"brain_id": "x"})[0] == 400
    assert _post(base_url, "/api/brains/save", {"name": ""})[0] == 400
    assert _post(base_url, "/api/brains/delete", {})[0] == 400


# ---------------------------------------------------------------------------
# 规则 5：对话探针 → 表型反馈
# ---------------------------------------------------------------------------


def test_dialogue_start_streams_turns_and_forms_feedback(server):
    base_url, _base = server
    status, start = _post(base_url, "/api/dialogue/start", {"n_turns": 3, "max_turns": 3})
    assert status == 202
    assert start["estimated_calls"] == 7
    assert start["brain"]["brain_id"]

    job = _wait_job(base_url, start["job_id"])
    assert job["status"] == "done"
    # 逐轮记录在任务状态里（界面据此做实时可视化）
    assert len(job["transcript"]) == 6
    assert [t["role"] for t in job["transcript"]] == ["examiner", "brain"] * 3

    feedback = job["result"]["feedback"]
    assert feedback["error"] == ""
    # 计分通道的子集：没有被探查到的通道没有读数（记为空，不按典型计）
    assert set(feedback["dims"]) <= set(SCORED_CHANNELS)
    assert feedback["dims"]
    assert 0.0 <= feedback["mean_reading"] <= 1.0
    # 观测通路的输出里不允许出现 θ：扰动只输入给 LLM，表型只由提问与观察得到
    assert "blind_audit" not in feedback
    assert "switching_cost" not in json.dumps(feedback, ensure_ascii=False)
    # 会话与反馈都要落盘，界面可回看
    _s, sessions = _get(base_url, "/api/dialogue/sessions")
    assert any(s["session_id"] == start["session_id"] and s["has_feedback"] for s in sessions["sessions"])

    _s, detail = _get(base_url, f"/api/dialogue?session_id={start['session_id']}")
    assert len(detail["turns"]) == 6
    assert detail["turns"][0]["turn"] == 1          # 历史轮次也要带轮号
    assert detail["feedback"]["dims"]


def test_dialogue_rejects_bad_turn_counts(server):
    base_url, _base = server
    assert _post(base_url, "/api/dialogue/start", {"n_turns": 0})[0] == 400
    # 硬顶 500：超过一律拒绝（99 在新口径下是合法的基础轮数 —— 上限从 20 抬到了 100）
    assert _post(base_url, "/api/dialogue/start", {"n_turns": 501})[0] == 400
    # 上限必须不小于基础轮数，否则配额还没补完就被截断
    assert _post(base_url, "/api/dialogue/start", {"n_turns": 40, "max_turns": 20})[0] == 400
    # 字数上限也走同一道校验
    assert _post(base_url, "/api/dialogue/start", {"n_turns": 1, "text_limit": 0})[0] == 400


def test_dialogue_analyze_reuses_existing_session(server):
    base_url, _base = server
    _s, start = _post(base_url, "/api/dialogue/start", {"n_turns": 2, "max_turns": 2})
    _wait_job(base_url, start["job_id"])

    status, again = _post(base_url, "/api/dialogue/analyze", {"session_id": start["session_id"]})
    assert status == 202
    job = _wait_job(base_url, again["job_id"])
    assert job["status"] == "done"
    assert job["result"]["feedback"]["dims"]
    assert job["result"]["session"]["session_id"] == start["session_id"]

    assert _post(base_url, "/api/dialogue/analyze", {"session_id": "dlg_nope"})[0] == 400
    assert _post(base_url, "/api/dialogue/analyze", {})[0] == 400


def test_dialogue_detail_validates_session_id(server):
    base_url, _base = server
    assert _get(base_url, "/api/dialogue")[0] == 400
    assert _get(base_url, "/api/dialogue?session_id=dlg_nope")[0] == 400


# ---------------------------------------------------------------------------
# 量表上传
# ---------------------------------------------------------------------------


def test_itemset_upload_is_llm_scored_and_deletable(server):
    """上传量表一律由分析者 LLM 逐条打分；带旧字段时要有可见提示。"""
    base_url, _base = server
    scale = {"name": "API 量表", "items": [
        {"id": "q1", "battery_dim": "social_cognition", "prompt": "他会怎么想？",
         "options": ["A", "B", "C"], "key": 1,
         "primitive_sensitivity": {"belief_updating": 0.3, "attention_shifting": 0.2}},
    ]}
    status, up = _post(base_url, "/api/itemsets", {"name": scale["name"], "payload": scale})
    assert status == 200 and up["valid"] is True
    assert up["item_set"]["scoring_mode"] == "llm"
    assert any("只作出处信息保留" in n for n in up["item_set"]["notices"])

    _s, listed = _get(base_url, "/api/itemsets")
    assert any(s["set_id"] == up["item_set"]["set_id"] for s in listed["item_sets"])
    assert _post(base_url, "/api/itemsets/delete", {"set_id": up["item_set"]["set_id"]})[1]["deleted"]


def test_itemset_custom_dimensions_are_flagged(server):
    base_url, _base = server
    scale = {"name": "API 量表（自编维度）", "items": [
        {"id": "c1", "battery_dim": "自编维度", "prompt": "自由作答"}
    ]}
    _s, up = _post(base_url, "/api/itemsets", {"name": scale["name"], "payload": scale})
    assert up["item_set"]["scoring_mode"] == "llm"
    assert any("自定义维度" in n for n in up["item_set"]["notices"])
    _post(base_url, "/api/itemsets/delete", {"set_id": up["item_set"]["set_id"]})


def test_itemset_upload_reports_problems_without_saving(server):
    base_url, _base = server
    _s, before = _get(base_url, "/api/itemsets")
    status, data = _post(base_url, "/api/itemsets", {"payload": {"items": [{"id": "x"}]}})
    assert status == 200 and data["valid"] is False and data["problems"]
    _s, after = _get(base_url, "/api/itemsets")
    assert len(after["item_sets"]) == len(before["item_sets"])


def test_itemset_upload_accepts_json_text(server):
    base_url, _base = server
    text = json.dumps({"name": "文本上传", "items": [
        {"id": "t1", "battery_dim": "repetitive_restricted_behavior", "prompt": "P", "options": ["A", "B"], "key": 0,
         "primitive_sensitivity": {"switching_cost": 0.5}}]})
    _s, up = _post(base_url, "/api/itemsets", {"name": "文本上传", "text": text})
    assert up["valid"] is True and up["item_set"]["scoring_mode"] == "llm"
    _post(base_url, "/api/itemsets/delete", {"set_id": up["item_set"]["set_id"]})
    status, data = _post(base_url, "/api/itemsets", {"text": "{not json"})
    assert status == 400 and "合法 JSON" in data["error"]


# ---------------------------------------------------------------------------
# 原始作答可视化（补上原有的可观测性缺口）
# ---------------------------------------------------------------------------


def test_group_run_stores_conversations_and_profiles(server):
    """群体运行留下的可回溯证据是**对话**，不是逐题作答（那张表已随电池删除）。

    原始作答的可观测性缺口由此换了载体：会话表 + 逐轮记录 + 分析者的原始输出。
    """
    base_url, _base = server
    run_id = _seed_run(_base, limit=3, turns=3, seed=3)
    _status, profiles = _get(base_url, f"/api/profiles?run_id={run_id}")
    assert profiles["n"] >= 1
    profile = profiles["profiles"][0]
    assert profile["instrument"] == "dialogue"
    assert set(profile["dims"]) <= set(SCORED_CHANNELS)
    # 无证据的通道记 None（不是「表现典型」），并在 uncovered_channels 里显式列出
    assert all(v is None or 0.0 <= v <= 1.0 for v in profile["dims"].values())
    assert isinstance(profile["uncovered_channels"], list)
    assert profile["synthetic"] is True          # 离线 mock：标成合成，不构成涌现证据
    assert profile["provider_kind"] == "mock"

    # 每一段对话都能在「历史对话」里找回来，且逐轮记录与分析者原始输出都在
    _status, sessions = _get(base_url, "/api/dialogue/sessions")
    mine = [s for s in sessions["sessions"] if s["session_id"].startswith(run_id)]
    assert len(mine) == profiles["n"]
    _status, detail = _get(base_url, f"/api/dialogue?session_id={mine[0]['session_id']}")
    assert detail["turns"] and detail["feedback"]["dims"]
    assert detail["feedback"]["raw"], "分析者的原始输出必须留存（可回溯的唯一证据）"

# ---------------------------------------------------------------------------
# 单脑机制报告的接口：只呈现与当前大脑有关的结果（界面暂时移出，接口保留）
# ---------------------------------------------------------------------------


def test_analysis_without_measured_brain_is_conflict(server):
    """当前大脑没被测量过 → 409 并给出可操作的指引，而不是端上一堆与它无关的数据。"""
    base_url, _base = server
    _s, injected = _post(base_url, "/api/inject",
                         {"kind": "custom", "values": {"salience_assignment": 0.3},
                          "name": "未测量的大脑"})
    _wait_job(base_url, injected["job_id"])
    status, data = _get(base_url, "/api/analysis?endpoint=social_communication_pragmatics")
    assert status == 409
    assert data["ok"] is False
    assert "还没有被测量过" in data["error"]
    assert "群体表型运行" in data["error"]


def test_analysis_scope_follows_the_current_brain(server):
    """跑一轮包含该大脑的设计后，分析范围应锁定到那次运行，并给出该大脑的定位。"""
    base_url, _base = server
    _s, injected = _post(base_url, "/api/inject",
                         {"kind": "custom", "values": {"attention_shifting": 0.2},
                          "name": "待测量的注意力组"})
    _wait_job(base_url, injected["job_id"])

    # 直接造一轮群体运行（界面已无此入口，与 CLI 同路径），再用 run_id 断言范围
    run_id = _seed_run(_base, limit=3)

    # 该运行不含 attention_shifting=0.2 的 θ → 当前大脑仍未在定量通道中被测量
    status, _data = _get(base_url, "/api/analysis")
    assert status == 409

    # 换成「基线大脑」（同一 θ 已在上面那轮里被测过）→ 应当给出范围
    _s, brains = _get(base_url, "/api/brains")
    baseline = next(b for b in brains["brains"] if not b["impaired"])
    _s, switched = _post(base_url, "/api/brains/activate", {"brain_id": baseline["brain_id"]})
    _wait_job(base_url, switched["job_id"])

    status, data = _get(base_url, "/api/analysis?endpoint=social_communication_pragmatics")
    assert status == 200
    scope, focus = data["scope"], data["focus"]
    assert scope["mode"] == "current_brain"
    assert scope["run_id"] == run_id
    assert scope["brain"]["theta_id"] == baseline["theta_id"]
    assert scope["n_profiles"] >= 1
    # 该大脑确实在这次受试集合里，并带上它自己的表型与机制层匹配
    assert scope["focus_in_run"] is True
    assert focus["theta_id"] == baseline["theta_id"]
    assert focus["in_run"] is True
    assert set(focus["dims"]) >= {"social_cognition", "social_communication_pragmatics"}
    assert focus["impaired"] == []            # 基线大脑没有被削弱的过程
    assert isinstance(focus["candidate_matches"], list)
    # 该大脑被削弱过才有候选机制匹配，因此基线往往是空集 —— 这本身是一个结果
    assert "subtype" in focus


def test_analysis_explicit_run_id_still_supported(server):
    """显式 run_id 仍然可用（脚本 / 复现），此时范围标记为指定运行。"""
    base_url, _base = server
    run_id = _seed_run(_base, limit=3, seed=21)
    status, data = _get(base_url, f"/api/analysis?run_id={run_id}")
    assert status == 200
    assert data["scope"]["mode"] == "explicit_run"
    assert data["scope"]["run_id"] == run_id
    assert data["atlas"]["n_profiles"] >= 1


# ---------------------------------------------------------------------------
# 观察者内置条目库（来源：AsdKB）
# ---------------------------------------------------------------------------


def test_examiner_bank_endpoint_browses_and_filters(server):
    base_url, _base = server
    status, data = _get(base_url, "/api/examiner/bank?limit=5")
    assert status == 200
    assert data["total"] == 441
    assert len(data["items"]) == 5
    stats = data["stats"]
    assert stats["scales"] == 20
    assert stats["probe_ready"] == 262
    assert stats["by_probe_mode"]["scenario"] == 212
    assert stats["by_probe_mode"]["question"] == 50
    assert stats["observation_only"] == 179
    assert stats["covers_36_months"] == 42
    assert (stats["symptoms"], stats["diagnostic_criteria"], stats["treatments"]) == (65, 43, 28)
    # 映射规则与来源说明必须一起返回，界面要能展示「映射是怎么来的」
    assert {"dim_priority", "dim_keywords", "scale_dims", "primitive_keywords"} <= set(
        data["mapping_rules"])
    assert data["meta"]["source"]["id"] == "ASDKB"
    assert data["dims"]["social_communication_pragmatics"]
    # 每条都带出处、施测方式与维度归属
    for item in data["items"]:
        assert item["source_name"] and item["source_ref"] and item["admin"] and item["dims"]


def test_examiner_bank_filters(server):
    base_url, _base = server
    _s, probes = _get(base_url, "/api/examiner/bank?probe_ready=1&limit=3")
    assert probes["total"] == 262
    assert all(i["probe_zh"] for i in probes["items"])

    _s, scenarios = _get(base_url, "/api/examiner/bank?probe_mode=scenario&limit=3")
    assert scenarios["total"] == 212
    assert all(i["probe_mode"] == "scenario" for i in scenarios["items"])
    assert all("情境：" in i["scenario_zh"] for i in scenarios["items"])

    _s, by_dim = _get(base_url, "/api/examiner/bank?dim=social_communication_pragmatics&limit=3")
    assert all("social_communication_pragmatics" in i["dims"] for i in by_dim["items"])

    _s, by_scale = _get(base_url, "/api/examiner/bank?source_id=tools19&limit=3")
    assert by_scale["total"] == 65
    assert all(i["source_id"] == "tools19" for i in by_scale["items"])

    _s, by_admin = _get(base_url, "/api/examiner/bank?admin=self_report&limit=3")
    assert all(i["admin"] == "self_report" for i in by_admin["items"])

    _s, by_age = _get(base_url, "/api/examiner/bank?age=36&limit=3")
    assert by_age["total"] > 0

    _s, searched = _get(base_url, "/api/examiner/bank?q=" + urllib.parse.quote("假装") + "&limit=3")
    assert searched["total"] > 0


def test_dialogue_uses_builtin_bank_by_default(server):
    base_url, _base = server
    _s, start = _post(base_url, "/api/dialogue/start", {"n_turns": 3, "max_turns": 3})
    job = _wait_job(base_url, start["job_id"])
    assert job["status"] == "done"
    session = job["result"]["session"]
    assert len(session["probe_keys"]) == 3
    assert session["probe_sources"] == ["ASDKB"] or all(
        "tools" in k for k in session["probe_keys"])
    assert session["probe_dims"]
    # 分析者的观察提纲也要记进反馈里（可追溯它拿了哪些锚点）
    feedback = job["result"]["feedback"]
    assert feedback["rubric_keys"]


def test_dialogue_accepts_explicit_bank_keys(server):
    base_url, _base = server
    keys = ["tools1:3", "tools14:4"]
    _s, start = _post(base_url, "/api/dialogue/start", {"n_turns": 2, "max_turns": 2, "bank_keys": keys})
    job = _wait_job(base_url, start["job_id"])
    assert job["status"] == "done"
    assert job["result"]["session"]["probe_keys"] == keys
    asked = [t["content"] for t in job["transcript"] if t["role"] == "examiner"]
    assert len(asked) == 2 and all(asked)


def test_dialogue_rejects_unknown_bank_keys(server):
    base_url, _base = server
    status, data = _post(base_url, "/api/dialogue/start",
                         {"n_turns": 1, "bank_keys": ["tools1:0", "nope:1"]})
    assert status == 400
    assert "不在内置条目库" in data["error"]


def test_dialogue_rejects_entries_without_child_facing_probe(server):
    """不可探查的条目（病史/医学事实类）不能用于提问 —— 硬问只会产出空问题或怪句子。"""
    base_url, _base = server
    _s, listing = _get(base_url, "/api/examiner/bank?probe_mode=none&limit=1")
    blocked_key = listing["items"][0]["key"]
    status, data = _post(base_url, "/api/dialogue/start",
                         {"n_turns": 1, "bank_keys": [blocked_key]})
    assert status == 400
    assert "没有儿童面向探查语句" in data["error"]
    assert "观察要点" in data["error"]


def test_dialogue_can_disable_bank(server):
    base_url, _base = server
    _s, start = _post(base_url, "/api/dialogue/start", {"n_turns": 2, "max_turns": 2, "use_bank": False})
    job = _wait_job(base_url, start["job_id"])
    assert job["status"] == "done"
    assert job["result"]["session"]["probe_keys"] == []


def test_dialogue_in_scenario_mode(server):
    """情境式探查：提问者先给情境，再问「这种情况你会怎么处理」。"""
    base_url, _base = server
    _s, start = _post(base_url, "/api/dialogue/start", {"n_turns": 2, "max_turns": 2, "probe_mode": "scenario"})
    job = _wait_job(base_url, start["job_id"])
    assert job["status"] == "done"
    session = job["result"]["session"]
    assert session["probe_mode"] == "scenario"
    assert len(session["probe_keys"]) == 2

    turns = [t for t in job["transcript"] if t["role"] == "examiner"]
    assert len(turns) == 2
    for turn in turns:
        assert turn["question"]
        # 每轮要么用了一条计划里的条目（`probe_key`），要么是提问者自己出题（自报 `focus`）——
        # 两者都没有才是真的丢了出处（自造题的轮次 `probe_key` 记为 None，不再照抄位置条目）
        assert turn["probe_key"] or turn["focus"]
        # 情境式呈现应包含情境描述或明确的探查语句（mock 会给出「设定情境」）
        assert turn["content"]
    # 该模式选出的条目都必须具备情境形态（要点：不只是首选方式为情境的条目）
    _s, listing = _get(base_url, "/api/examiner/bank?probe_ready=1&limit=400")
    by_key = {i["key"]: i for i in listing["items"]}
    assert by_key
    for key in session["probe_keys"]:
        assert by_key[key].get("scenario_zh"), f"{key} 在情境模式下被选中却没有情境形态"


def test_bank_exposes_probe_mode_notes(server):
    """界面要能说明三种探查方式分别是什么。"""
    base_url, _base = server
    _s, data = _get(base_url, "/api/examiner/bank?limit=1")
    notes = data["probe_mode_notes"]
    assert set(notes) == {"question", "scenario", "none"}
    assert "情境" in notes["scenario"]


# ---------------------------------------------------------------------------
# 用户直接提问（自定义访谈）
# ---------------------------------------------------------------------------


def test_interview_start_records_brain_and_self_report(server):
    """开始访谈：绑定当前大脑，先做提示词初始化并把自我报告回传。"""
    base_url, _base = server
    status, data = _post(base_url, "/api/interview/start", {})
    assert status == 200
    assert data["session_id"].startswith("dlg_")
    assert data["brain"]["theta_id"] == data["init"]["theta_id"]
    assert data["init"]["fingerprint"]
    assert data["init"]["raw"]


def test_interview_ask_keeps_multi_turn_history(server):
    """逐轮提问：大脑要拿到此前的对话历史（多轮，而不是每轮孤立作答）。"""
    base_url, _base = server
    _s, start = _post(base_url, "/api/interview/start", {"include_self_report": False})
    session_id = start["session_id"]

    questions = ["你今天在幼儿园做了什么？", "如果有小朋友抢你的玩具，你会怎么办？"]
    for index, question in enumerate(questions, start=1):
        status, data = _post(base_url, "/api/interview/ask",
                            {"session_id": session_id, "question": question})
        assert status == 200
        assert data["turn"] == index
        assert data["answer"]

    _s, detail = _get(base_url, f"/api/dialogue?session_id={session_id}")
    assert detail["session"]["mode"] == "interview"
    assert "user" in detail["session"]["examiner_model"]
    assert len(detail["turns"]) == 4                      # 2 问 + 2 答
    assert [t["role"] for t in detail["turns"]] == ["user", "brain", "user", "brain"]
    assert detail["turns"][0]["content"] == questions[0]


def test_interview_rejects_bad_input(server):
    base_url, _base = server
    assert _post(base_url, "/api/interview/ask", {"question": "x"})[0] == 400
    _s, start = _post(base_url, "/api/interview/start", {"include_self_report": False})
    status, data = _post(base_url, "/api/interview/ask",
                         {"session_id": start["session_id"], "question": "   "})
    assert status == 400 and "不能为空" in data["error"]
    status, data = _post(base_url, "/api/interview/ask",
                         {"session_id": "dlg_nope", "question": "x"})
    assert status == 400 and "不存在" in data["error"]


def test_interview_sessions_are_listed_separately(server):
    """用户访谈与提问者 LLM 的对话探针要能分开列出。"""
    base_url, _base = server
    _s, interview = _post(base_url, "/api/interview/start", {"include_self_report": False})
    _post(base_url, "/api/interview/ask",
          {"session_id": interview["session_id"], "question": "你喜欢什么玩具？"})

    _s, user_list = _get(base_url, "/api/interview/sessions")
    ids = {s["session_id"] for s in user_list["sessions"]}
    assert interview["session_id"] in ids
    assert all(s["mode"] == "interview" for s in user_list["sessions"])

    # 对话探针那一侧不含用户访谈
    _s, probe_list = _get(base_url, "/api/interview/sessions?mode=probe")
    probe_ids = {s["session_id"] for s in probe_list["sessions"]}
    assert interview["session_id"] not in probe_ids
    assert all(s["mode"] == "probe" for s in probe_list["sessions"])


def test_interview_can_be_scored_by_the_analyst(server):
    """用户自己问的对话同样可以交给分析者 LLM 形成表型反馈。"""
    base_url, _base = server
    _s, start = _post(base_url, "/api/interview/start", {"include_self_report": False})
    for question in ["你想要什么玩具？", "看不到妈妈你会怎么办？"]:
        _post(base_url, "/api/interview/ask", {"session_id": start["session_id"], "question": question})

    _s, job_start = _post(base_url, "/api/dialogue/analyze", {"session_id": start["session_id"]})
    job = _wait_job(base_url, job_start["job_id"])
    assert job["status"] == "done"
    feedback = job["result"]["feedback"]
    assert feedback["error"] == ""
    assert set(feedback["dims"]) == set(SCORED_CHANNELS)
    assert feedback["n_brain_turns"] == 2
    # 分析者拿到的是「提问者 ↔ 孩子」的记录，人类提问也被标为提问者
    _s, detail = _get(base_url, f"/api/dialogue?session_id={start['session_id']}")
    assert detail["feedback"]["mean_reading"] is not None


# ---------------------------------------------------------------------------
# 单脑机制报告：只使用这一个大脑的数据，不含任何群体分析
# ---------------------------------------------------------------------------


def test_brain_report_intact_brain_has_no_mapping(server):
    """完整心智没有被削弱的过程 → 映射为空，但要明确说明原因而不是留白。"""
    base_url, _base = server
    _s, injected = _post(base_url, "/api/inject", {"kind": "intact", "name": "完整基线"})
    _wait_job(base_url, injected["job_id"])

    status, data = _get(base_url, "/api/brain-report")
    assert status == 200
    assert data["intact"] is True
    assert data["primitives"] == []
    assert data["brain"]["theta_id"]
    assert len(data["reference"]) == 31          # 31 个基础过程的参考表
    assert data["notes"]["scope"]                # 说明为什么这里没有映射可展开


def test_brain_report_maps_each_impaired_primitive(server):
    """受损过程要带上「高阶构念 + 题库条目标签 + 探查来源」。

    注意这里**没有**任何 primitive → 测量维度 的映射：那条链（含 Model 1 13 维）已整体
    删除，因为构造属性与测量维度之间不允许有预设对应。
    """
    base_url, _base = server
    _s, injected = _post(base_url, "/api/inject", {
        "kind": "custom",
        "values": {"salience_assignment": 0.3, "switching_cost": 0.4},
        "name": "双过程受损",
    })
    _wait_job(base_url, injected["job_id"])

    status, data = _get(base_url, "/api/brain-report")
    assert status == 200 and data["intact"] is False
    ids = [p["id"] for p in data["primitives"]]
    assert ids == ["salience_assignment", "switching_cost"]
    # 完整度低的排前面
    assert data["primitives"][0]["value"] < data["primitives"][1]["value"]
    for entry in data["primitives"]:
        assert entry["definition"] and entry["computational_role"]
        assert entry["impairment_signature"]
        assert entry["bank_item_dim_labels"]           # 题库条目的覆盖标签（题目设计元数据）
        assert "dims" not in entry                     # 旧键已改名，避免被读成测量期望
        assert "dim_map_targets" not in entry          # 该映射已整体删除
        assert "target_dim" not in (entry.get("assay") or {})
        assert entry["probes_total"] >= 0
        assert "systems" in entry and "symptoms" in entry and "criteria" in entry
    # 报告不得包含任何群体层面的键
    for banned in ("atlas", "convergence", "subtypes", "specificity", "dose_response"):
        assert banned not in data, banned
    assert data["notes"]["scope"]


def test_brain_report_uses_this_brain_own_readings(server):
    """读数必须来自该大脑自己的对话，而不是任何受试集合。"""
    base_url, _base = server
    _s, injected = _post(base_url, "/api/inject", {
        "kind": "custom", "values": {"switching_cost": 0.4}, "name": "待观察大脑",
    })
    _wait_job(base_url, injected["job_id"])
    assert _get(base_url, "/api/brain-report")[1]["readings"] is None

    _s, start = _post(base_url, "/api/dialogue/start", {"n_turns": 5, "max_turns": 5, "seed": 5})
    job = _wait_job(base_url, start["job_id"], timeout=180)
    assert job["status"] == "done", job

    status, data = _get(base_url, "/api/brain-report")
    assert status == 200
    readings = data["readings"]
    assert readings is not None
    assert readings["session_id"] == start["session_id"]
    assert set(readings["dims"]) == set(SCORED_CHANNELS)
    assert readings["mean_reading"] is not None
    # 读数里没有 θ，也没有并排展示的基准真值 —— 扰动与表型之间不存在已知条件的关联
    assert "blind_audit" not in readings
    assert "switching_cost" not in json.dumps(readings, ensure_ascii=False)
    # 该大脑的对话记录也列出来了
    assert [s["session_id"] for s in data["sessions"]] == [start["session_id"]]
    assert data["sessions"][0]["has_feedback"] is True


def test_export_endpoints_are_gone(server):
    """导出功能已整体删除（不产出任何维度投影/描述集）。"""
    base_url, _base = server
    assert _post(base_url, "/api/export", {})[0] == 404
    assert _get(base_url, "/api/exports")[0] == 404
