"""观察通路的两处脆弱点（真实模型下会踩到）的回归测试。

一、**历史对话必须锚定当前大脑**。读数卡说的是「当前大脑最近一次被观察」，历史列表
    若列全库，换大脑/重新初始化后就会出现「读数说这具心智还没被观察过，历史里却列着
    别的几十段对话」。归属规则与排序都在服务端一处定死（`list_brain_sessions`）。

二、**空回答与截断必须报错，不能静默变成空串**。推理型模型会把 `max_tokens` 花在
    `reasoning_content` 上，此时 `content` 是空串：
    - 提问者空回答 → 一轮解析失败（旧行为：两次失败即终止整段对话，35 轮的协议退化成 2 轮）；
    - 分析者空回答 → 读数整块空白（旧行为：`dims={}` 存进库，界面把它显示成
      「另一套维度口径」——把「模型没吐出东西」演成「口径不匹配」）；
    - 心智空回答 → 分析者会把「没有回应」当成行为证据，压低读数。
    这三件事都必须以「这次调用失败」的形式暴露出来。
"""

from __future__ import annotations

import json
import pathlib
import urllib.request

import pytest

from pacs.agents.dialogue import (
    EXAMINER_ATTEMPTS,
    analyze_dialogue,
    run_dialogue,
)
from pacs.agents.virtual_agent import VirtualAgent
from pacs.brain.perturb import intact
from pacs.llm.provider import (
    LLMError,
    MockProvider,
    OpenAICompatProvider,
    _content_of,
)
from pacs.store.db import Store
from pacs.config import Settings
from pacs.api.server import register_brain
from pacs.brain.perturb import single

ALL_DIMS = ("social_cognition", "general_cognition")


# ---------------------------------------------------------------------------
# 一、历史对话 = 当前大脑的观察史
# ---------------------------------------------------------------------------


def _session(session_id: str, *, brain_id: str, theta_id: str) -> dict:
    return {
        "session_id": session_id, "theta_id": theta_id, "brain_id": brain_id,
        "brain_model": "mock:m", "examiner_model": "mock:m", "n_turns": 1,
        "payload": {"session_id": session_id, "theta_id": theta_id, "brain_id": brain_id,
                    "n_turns": 1, "mode": "probe", "turns": []},
    }


def test_sessions_are_scoped_to_one_brain(tmp_path):
    store = Store(tmp_path / "pacs.db")
    store.save_dialogue_session("s_a", _session("s_a", brain_id="brain_A", theta_id="theta_A"))
    store.save_dialogue_session("s_b", _session("s_b", brain_id="brain_B", theta_id="theta_B"))
    # 命令行/早期版本留下的会话：没有 brain_id，但 θ 相同 → 算这具心智的历史
    store.save_dialogue_session("s_cli", _session("s_cli", brain_id="", theta_id="theta_A"))

    assert [s["session_id"] for s in store.list_brain_sessions("brain_A", "theta_A")] == ["s_cli", "s_a"]
    assert [s["session_id"] for s in store.list_brain_sessions("brain_B", "theta_B")] == ["s_b"]
    # 刚初始化出来的大脑（新的 brain_id、新的 θ）→ 空列表（界面上就是「清空」）
    assert store.list_brain_sessions("brain_new", "theta_new") == []
    store.close()


def test_same_second_sessions_put_the_newest_first(tmp_path):
    """`created_at` 只到秒：批量/快速连跑时同秒的会话必须靠 rowid 决胜。

    否则「最近一次读数」可能取到同秒里较早的那一段 —— 界面显示的不是刚跑出来的结果。
    """
    store = Store(tmp_path / "pacs.db")
    for index in range(4):
        store.save_dialogue_session(
            f"s{index}", _session(f"s{index}", brain_id="brain_A", theta_id="theta_A")
        )
    order = [s["session_id"] for s in store.list_brain_sessions("brain_A", "theta_A")]
    assert order == ["s3", "s2", "s1", "s0"]
    store.close()


def test_brain_report_uses_the_same_scoping(tmp_path):
    """读数（brain-report）与历史列表必须指向同一批会话，否则两处会各说一套。"""
    from pacs.analysis.brain_report import _latest_feedback

    store = Store(tmp_path / "pacs.db")
    store.save_dialogue_session("s_a", _session("s_a", brain_id="brain_A", theta_id="theta_A"))
    store.save_dialogue_session("s_b", _session("s_b", brain_id="brain_B", theta_id="theta_B"))
    store.save_feedback("s_b", "theta_B", "mock:m",
                        {"session_id": "s_b", "dims": {"social_cognition": 0.5}})

    latest, briefs, newest = _latest_feedback(store, "theta_A", "brain_A")
    assert latest is None and newest is None            # s_a 没有反馈
    assert [b["session_id"] for b in briefs] == ["s_a"]
    latest_b, _, newest_b = _latest_feedback(store, "theta_B", "brain_B")
    assert latest_b and latest_b["session_id"] == "s_b"
    assert newest_b and newest_b["session_id"] == "s_b"
    store.close()


def test_sessions_endpoint_scoped_by_brain_id(tmp_path):
    """界面传 `brain_id`（即使为空）就只看当前大脑；不传参数仍可取全库（CLI/脚本）。"""
    from pacs.api.server import PacsHandler
    from http.server import ThreadingHTTPServer
    import threading
    import time

    settings = Settings.load(base_dir=tmp_path)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    # 真建一具大脑记录（会话的 brain_id 只是字符串，必须有对应的大脑才算「它的历史」）
    brain_b = register_brain(store, "乙", single("inhibition", 0.3), source="user", activate=True)
    brain_a = register_brain(store, "甲", single("attention_shifting", 0.3), source="user",
                             activate=False)   # 乙保持为当前大脑
    store.save_dialogue_session("s_a", _session("s_a", brain_id=brain_a["brain_id"],
                                                theta_id=brain_a["theta_id"]))
    store.save_dialogue_session("s_b", _session("s_b", brain_id=brain_b["brain_id"],
                                                theta_id=brain_b["theta_id"]))
    PacsHandler.settings = settings
    PacsHandler.store = store

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), PacsHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)

    def get(path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        # 当前大脑 = 乙：带上 brain_id 时只列它的历史，甲的那段不出现
        scoped = get("/api/dialogue/sessions?brain_id=")
        assert [s["session_id"] for s in scoped["sessions"]] == ["s_b"]
        assert scoped["total"] == 1
        # 不带参数：全库（CLI/脚本口径）
        assert {s["session_id"] for s in get("/api/dialogue/sessions")["sessions"]} == {"s_a", "s_b"}
        # 换成甲：列表随之变成甲自己的历史（界面换大脑时会走这里）
        assert get("/api/dialogue/sessions?brain_id=")["sessions"][0]["session_id"] == "s_b"
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


# ---------------------------------------------------------------------------
# 二、空回答 / 截断：报错，不静默
# ---------------------------------------------------------------------------


def test_empty_content_is_an_error_with_an_actionable_message():
    with pytest.raises(LLMError) as empty:
        _content_of({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, "analyst")
    assert "分析者返回空内容" in str(empty.value)
    assert "max_tokens" in str(empty.value)

    with pytest.raises(LLMError) as reasoning:
        _content_of({
            "choices": [{"message": {"content": "", "reasoning_content": "想" * 100},
                         "finish_reason": "length"}],
        }, "brain")
    assert "模拟大脑返回空内容" in str(reasoning.value)
    assert "思维链" in str(reasoning.value)

    with pytest.raises(LLMError):
        _content_of({}, "examiner")          # 响应里连 choices 都没有
    # 非空正文照常返回（不因为 finish_reason=length 就丢掉已有内容）
    assert _content_of(
        {"choices": [{"message": {"content": "半句话"}, "finish_reason": "length"}]}, "brain"
    ) == "半句话"


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _provider(monkeypatch, bodies: list[dict]) -> OpenAICompatProvider:
    """一个真跑 HTTP 分支的 provider：`urlopen` 被换成一串预置响应。"""
    from pacs.config import LLMSettings

    provider = OpenAICompatProvider(LLMSettings(provider="openai", api_key="k", model="m"))
    queue = list(bodies)

    def fake_urlopen(request, timeout=None):
        if not queue:
            raise AssertionError("调用的次数超过了预置响应数")
        return _FakeResponse(queue.pop(0))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_a: None)
    return provider


def test_provider_retries_an_empty_answer_once_then_succeeds(monkeypatch):
    provider = _provider(monkeypatch, [
        {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "好了"}, "finish_reason": "stop"}]},
    ])
    from pacs.llm.provider import ChatMessage

    assert provider.chat([ChatMessage("user", "hi")], role="examiner") == "好了"


def test_provider_raises_after_repeated_empty_answers(monkeypatch):
    empties = [{"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}] * 3
    provider = _provider(monkeypatch, empties)
    from pacs.llm.provider import ChatMessage

    with pytest.raises(LLMError) as exc:
        provider.chat([ChatMessage("user", "hi")], role="brain")
    assert "返回空内容" in str(exc.value)


def test_analyst_failure_becomes_a_displayable_feedback_not_an_exception():
    """分析者调用失败 → 反馈里带着**说得清原因**的错误与处理建议（而不是抛出去让界面空白）。"""
    class DeadAnalyst(MockProvider):
        def chat(self, messages, **kwargs):
            raise LLMError("分析者返回空内容（finish_reason=length）；请调大 max_tokens")

    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m", seed=3)
    session = run_dialogue(brain, MockProvider(), n_turns=4, max_turns=4, seed=3)
    feedback = analyze_dialogue(DeadAnalyst(), session)

    assert feedback["dims"] == {} and feedback["dim_details"] == {}
    assert "返回空内容" in feedback["error"] and "max_tokens" in feedback["error"]
    assert "max_tokens" in feedback["notes"]          # 处理建议进 notes（界面可见）
    assert feedback["conclusion"] == ""               # 不编造结论
    # 读数载荷仍然带得出去（界面据此显示「这次观察没有产出读数」而不是「没有读数」）
    from pacs.analysis.brain_report import _readings_payload

    payload = _readings_payload(feedback)
    assert payload["error"] and payload["dims"] == {}


def test_analyst_call_is_marked_as_the_analyst_role():
    """分析者那一轮必须让 provider 知道「这是分析者」：额度下限是按角色给的。

    额度不再由调用方传死值（那样配置里的 `max_tokens` 会被屏蔽）；调用方只声明角色，
    provider 据此给足额度 —— 见 `test_analyst_token_budget_follows_a_raised_config`。
    """
    seen: dict = {}

    class Recorder(MockProvider):
        def chat(self, messages, **kwargs):
            if kwargs.get("role") == "analyst":
                seen["role"] = kwargs.get("role")
                seen["max_tokens"] = kwargs.get("max_tokens")
            return super().chat(messages, **kwargs)

    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m", seed=5)
    session = run_dialogue(brain, MockProvider(), n_turns=4, max_turns=4, seed=5)
    analyze_dialogue(Recorder(), session)
    assert seen["role"] == "analyst"
    assert seen["max_tokens"] is None      # 不写死额度：由 provider 按角色与配置定


def test_examiner_gets_three_attempts_before_the_dialogue_gives_up():
    """偶发的空回答不该让整段对话停摆（旧行为：两次失败即终止）。"""
    assert EXAMINER_ATTEMPTS == 3

    class FlakyExaminer(MockProvider):
        """前两次返回空串，第三次正常 —— 单轮应能救回来。"""

        def __init__(self) -> None:
            super().__init__()
            self.failures = 0

        def chat(self, messages, **kwargs):
            if kwargs.get("role") == "examiner":
                if self.failures < 2:
                    self.failures += 1
                    return ""
                return "我们一起玩滚球好吗？\nQUESTION: 我们一起玩滚球好吗？\nFOCUS: social_cognition"
            return super().chat(messages, **kwargs)

    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m", seed=7)
    session = run_dialogue(brain, FlakyExaminer(), n_turns=4, max_turns=4, seed=7)
    assert session.n_turns_run == 4
    assert not any("没有产出可解析的问题" in p for p in session.problems)


def test_examiner_failure_note_says_how_many_attempts_were_made():
    class SilentExaminer(MockProvider):
        def chat(self, messages, **kwargs):
            if kwargs.get("role") == "examiner":
                return ""
            return super().chat(messages, **kwargs)

    brain = VirtualAgent(MockProvider(), single("inhibition", 0.3), model_id="mock:m", seed=7)
    session = run_dialogue(brain, SilentExaminer(), n_turns=4, max_turns=4, seed=7)
    assert session.turns == []
    assert any("没有产出可解析的问题" in p for p in session.problems)


def test_truncated_analyst_json_is_repaired_into_real_dimensions():
    """输出写到一半被截断是**最常见**的失败：要把已经写出来的维度救回来，并说清是截断。

    真实病例：`deepseek-flash` 的一次运行输出 967 字就断了（JSON 不完整），
    里面其实已经写完了两个完整维度（各带观察与测量）—— 这些是真实读数，不该丢。
    """
    from pacs.llm.parse import parse_analyst_json

    truncated = (
        '{"dimensions": {'
        '"social_cognition": {"score": 85, "confidence": 0.8, '
        '"observations": ["错误信念题答对", "能区分故意与意外"], "deviation": -15}, '
        '"social_interaction_motivation": {"score": 45, "confidence": 0.75, '
        '"observations": ["被叫名字后只抬眼看了下", "拒绝一起玩小汽车", "对难过无情绪回应'
    )
    payload, error = parse_analyst_json(
        truncated, allowed={"social_cognition", "social_interaction_motivation"}
    )
    assert payload["dims"] == {"social_cognition": 0.85, "social_interaction_motivation": 0.45}
    # 逐维记录（观察/测量）也一并救回，不只剩一个分数
    assert len(payload["dim_details"]["social_cognition"]["observations"]) == 2
    assert len(payload["dim_details"]["social_interaction_motivation"]["observations"]) == 2
    assert "截断" in error
    assert "max_tokens" in payload["notes"]        # 处理办法写在明面上


def test_structural_fields_are_never_read_as_dimensions():
    """结构字段（confidence / deviation …）不是测量通道，正则兜底也不许把它们读成维度。

    旧行为会把截断输出里的 `"confidence": 0.8` 读成「有一个叫 confidence 的维度得了 80 分」：
    界面上于是多了一条叫「confidence」的轴、平均分 80 —— 一条凭空造出来的「读数」，
    比如实报「没有读数」糟得多，因为它看起来像数据。
    """
    from pacs.llm.parse import parse_analyst_json

    payload, error = parse_analyst_json(
        '{"dimensions": {"social_cognition": {"score": 85, "confidence": 0.8}, ',
        allowed={"social_cognition", "social_interaction_motivation"},
    )
    assert list(payload["dims"]) == ["social_cognition"]          # 只救回真维度
    payload, error = parse_analyst_json("模型写到一半断了：confidence: 0.8, deviation: -15")
    assert payload is None and "无法解析" in error                 # 一个真维度都没有 → 如实失败
    assert "confidence" not in error


def test_regex_fallback_only_keeps_channels_the_caller_asked_for():
    """给了通道清单就只认清单内的键：上传量表的条目 id 与观察目标之外的一概不认。"""
    from pacs.llm.parse import parse_analyst_json

    text = 'social_cognition: 0.5，item_q1 = 0.75，made_up_channel = 0.9'
    payload, _error = parse_analyst_json(text, allowed={"social_cognition", "item_q1"})
    assert payload["dims"] == {"social_cognition": 0.5, "item_q1": 0.75}
    assert "made_up_channel" not in payload["dims"]


def test_a_salvaged_partial_reading_is_flagged_in_the_feedback():
    """救回的部分读数必须带上「这次读数不完整」的说明，否则它看起来像完整读数。"""
    from pacs.llm.parse import parse_analyst_json

    class TruncatingAnalyst(MockProvider):
        def chat(self, messages, **kwargs):
            if kwargs.get("role") == "analyst":
                return ('{"dimensions": {"social_cognition": {"score": 85, "confidence": 0.8, '
                        '"observations": ["错误信念题答对"], "deviation": -15}, '
                        '"social_interaction_motivation": {"score": 45, "confidence": 0.7, '
                        '"observations": ["拒绝一起玩小汽车", "对难过无回应')
            return super().chat(messages, **kwargs)

    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m", seed=11)
    session = run_dialogue(brain, MockProvider(), n_turns=4, max_turns=4, seed=11)
    feedback = analyze_dialogue(TruncatingAnalyst(), session)
    assert feedback["dims"] == {
        "social_cognition": 0.85, "social_interaction_motivation": 0.45,
    }
    assert "截断" in feedback["error"]
    assert feedback["notes"].startswith("⚠")           # 界面上以警示样式显示
    assert "不完整" in feedback["notes"] and "max_tokens" in feedback["notes"]
    # 分析者没有给出的那一段结论不编造
    assert feedback["conclusion"] == ""
    assert parse_analyst_json(feedback["raw"])[0]["dims"]["social_cognition"] == 0.85


def test_analyst_token_budget_follows_a_raised_config(monkeypatch):
    """「把 max_tokens 调大」必须对**分析者**也生效。

    它是唯一要吐一整段 JSON 的角色，输出最长；若把调用额度写死成常量，界面上那句
    「调大 max_tokens」对它就完全免疫 —— 用户照做了却依然只拿到被截断的读数。
    """
    from pacs.config import LLMSettings
    from pacs.llm.provider import ANALYST_MIN_TOKENS, ChatMessage

    seen: list[int] = []

    def fake_urlopen(request, timeout=None):
        seen.append(json.loads(request.data.decode("utf-8"))["max_tokens"])
        return _FakeResponse(
            {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    small = OpenAICompatProvider(
        LLMSettings(provider="openai", api_key="k", model="m", max_tokens=512)
    )
    small.chat([ChatMessage("user", "hi")], role="analyst")
    assert seen[-1] == ANALYST_MIN_TOKENS             # 配置比下限小 → 抬起下限
    # 额度要够写完一整段 JSON：实测一次真实会话在 3000 之下只写到 967 字就被截断
    # （推理型模型的思维链吃掉大头），故下限必须在数千量级而不是一次问答的几百
    assert ANALYST_MIN_TOKENS >= 6000
    small.chat([ChatMessage("user", "hi")], role="brain")
    assert seen[-1] == 512                            # 下限只属于分析者，别的角色听配置

    big = OpenAICompatProvider(
        LLMSettings(provider="openai", api_key="k", model="m", max_tokens=8192)
    )
    big.chat([ChatMessage("user", "hi")], role="analyst")
    assert seen[-1] == 8192                           # 配置更大 → 以配置为准


def test_a_rejected_token_budget_falls_back_to_the_configured_value(monkeypatch):
    """端点不认我们要求的额度时（有些模型的输出上限更低）退回配置值重试。

    不这么做的话，「额度要多了」会让整次观察直接失败 —— 那比额度小更难查：
    错误信息说的是 HTTP 400，与读数、模型、提示词都看不出关系。
    """
    import io

    from pacs.config import LLMSettings
    from pacs.llm.provider import ANALYST_MIN_TOKENS, ChatMessage

    class _Rejected(urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__("http://x", 400, "Bad Request", {}, io.BytesIO(
                b'{"error":{"message":"max_tokens is too large: must be <= 4096"}}'
            ))

    seen: list[int] = []

    def fake_urlopen(request, timeout=None):
        seen.append(json.loads(request.data.decode("utf-8"))["max_tokens"])
        if len(seen) == 1:
            raise _Rejected()
        return _FakeResponse(
            {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatProvider(
        LLMSettings(provider="openai", api_key="k", model="m", max_tokens=1024)
    )
    assert provider.chat([ChatMessage("user", "hi")], role="analyst") == "{}"
    assert seen == [ANALYST_MIN_TOKENS, 1024]      # 先要足，被拒后退回配置值


def test_brain_report_prefers_the_latest_usable_reading(tmp_path):
    """最近一次观察失败时，读数卡要拿**此前最近一份可用读数**，并说明最近那次失败了。

    否则一具此前测出过读数的心智，只要再跑一次而这次分析者没吐出东西，读数卡就变成一张
    空卡 —— 看上去像「这具心智测不出东西」，而不是「这一次调用失败了」。
    """
    from pacs.analysis.brain_report import _latest_feedback, _readings_payload

    store = Store(tmp_path / "pacs.db")
    store.save_dialogue_session("s_ok", _session("s_ok", brain_id="brain_A", theta_id="theta_A"))
    store.save_dialogue_session("s_bad", _session("s_bad", brain_id="brain_A", theta_id="theta_A"))
    store.save_feedback("s_ok", "theta_A", "mock:m",
                        {"session_id": "s_ok", "dims": {"social_cognition": 0.5}})
    # 分析者调用失败也会落一条反馈：`dims` 空、`error` 写明原因
    store.save_feedback("s_bad", "theta_A", "openai:m",
                        {"session_id": "s_bad", "dims": {}, "error": "分析者返回空内容"})

    latest, briefs, newest = _latest_feedback(store, "theta_A", "brain_A")
    assert latest["session_id"] == "s_ok"          # 有读数的那一份（界面据此画雷达图）
    assert newest["session_id"] == "s_bad"         # 最近一次尝试（哪怕它失败了）
    assert [b["session_id"] for b in briefs] == ["s_bad", "s_ok"]

    payload = _readings_payload(latest, newest)
    assert payload["dims"] == {"social_cognition": 0.5}
    assert payload["newest_attempt"]["session_id"] == "s_bad"
    assert "返回空内容" in payload["newest_attempt"]["error"]
    # 最近一次就是有读数的那一次 → 不报「过期」
    assert _readings_payload(latest, latest)["newest_attempt"] is None
    # 一具从没测出过读数的心智：退回最近一次**尝试**（哪怕它是失败的），界面据此
    # 说明「为什么没有读数」——而不是显示成「还没被观察过」
    failed, _, failed_newest = _latest_feedback(store, "theta_B", "brain_B")
    assert failed is None and failed_newest is None       # 没有任何会话
    store.close()

    store = Store(tmp_path / "pacs2.db")
    store.save_dialogue_session("s_bad", _session("s_bad", brain_id="brain_C", theta_id="theta_C"))
    store.save_feedback("s_bad", "theta_C", "openai:m",
                        {"session_id": "s_bad", "dims": {}, "error": "分析者返回空内容"})
    only_failed, _, _ = _latest_feedback(store, "theta_C", "brain_C")
    assert only_failed["session_id"] == "s_bad"
    assert _readings_payload(only_failed, only_failed)["newest_attempt"] is None
    store.close()


def test_ui_history_distinguishes_no_reading_from_no_feedback():
    """历史列表的「表型读数」列：有读数 / 无读数 / 未评分 是三件事。

    分析者调用失败也会落一条反馈，若把它显示成「有读数」，用户会以为这一段已经量出来了 ——
    与读数卡上的说明自相矛盾。
    """
    import pacs
    from pacs.analysis.brain_report import feedback_has_reading

    assert feedback_has_reading({"dims": {"social_cognition": 0.5}}) is True
    assert feedback_has_reading({"dims": {}}) is False          # 分析者空回答
    assert feedback_has_reading({"dims": {"social_cognition": None}}) is False
    assert feedback_has_reading({"dims": {"social_cognition": True}}) is False
    # 结构字段不是维度：早期版本把截断输出里的 confidence 存成了维度，那种记录不算有读数
    assert feedback_has_reading({"dims": {"confidence": 0.8}}) is False
    assert feedback_has_reading({"dims": {"confidence": 0.8, "social_cognition": 0.5}}) is True
    assert feedback_has_reading(None) is False

    js = (pathlib.Path(pacs.__file__).parent / "web" / "static" / "js" / "views" / "phenotype.js"
          ).read_text(encoding="utf-8")
    assert "function readingCell(" in js
    assert "has_reading" in js and "无读数" in js and "未评分" in js
    # 单看「有没有反馈记录」已经不足以决定这一列显示什么
    assert "s.has_feedback ?" not in js


def test_ui_draws_the_radar_even_when_some_channels_have_no_reading():
    """分析者只给出部分维度时，已有读数必须照画。

    旧行为是「缺一维就整块不画雷达图」，只留一段解释 —— 用户要的是读数，不是解释。
    缺的轴照画但标 `—（无读数）`：它**不是 0 分**（0 分在这个分制里意味着严重偏离）。
    """
    import pacs

    js = (pathlib.Path(pacs.__file__).parent / "web" / "static" / "js" / "ui" / "readings.js"
          ).read_text(encoding="utf-8")
    assert "虚线轴 = 没有读数" in js
    assert "known.filter" not in js                     # 「缺一维就拒画」的旧条件已去掉
    assert "entries.some(([dim]) => isPhenotypeDim(dim))" in js   # 只有全是外来维度才不画
    # 缺读数的表型维度在得分表里照样占一行（消失掉就没人知道这一维没被读到）
    assert "dimsAll().map((dim) => [dim, valueOf(dims[dim])])" in js
    # 冗余的解释性文字不再常显：图例整块删掉，失败说明只留可操作的那两句
    assert "参考环（25 / 50 / 75 / 100 分）" not in js
    assert '<div class="legend">' not in js


# ---------------------------------------------------------------------------
# 三、界面：把「分析者没产出读数」说清楚，并让历史列表跟着大脑走
# ---------------------------------------------------------------------------


def test_ui_explains_a_failed_analysis_instead_of_blaming_the_vocabulary():
    """一个维度都没有 ≠ 用了别的维度口径：界面上这两句话必须分开。

    旧行为会把空读数显示成「这份读数用的是另一套维度口径，请重新观察」——
    用户看到的是「口径不匹配」，实际原因是分析者没吐出东西，根本无从下手。
    """
    import pacs

    js = (pathlib.Path(pacs.__file__).parent / "web" / "static" / "js" / "ui" / "readings.js"
          ).read_text(encoding="utf-8")
    assert "这次观察没有产出读数" in js
    assert "max_tokens" in js and "思维链" in js
    assert "另一套维度口径" in js          # 那条分支仍然保留（确实是另一件事）
    # 结论块只在真拿到结论时才渲染（空读数时不许出现一张「结论（无）」的假卡）
    assert "if (!text)" in js


def test_ui_history_is_scoped_to_the_current_brain():
    import pacs

    js = (pathlib.Path(pacs.__file__).parent / "web" / "static" / "js" / "views" / "phenotype.js"
          ).read_text(encoding="utf-8")
    assert "brain_id=" in js                      # 请求里显式带上当前大脑
    assert "list_brain_sessions" not in js        # 归属判定只在服务端一处
    # 换大脑时三处一起清：读数 / 对话抽屉 / 历史列表
    on_change = js[js.index("export function onBrainChanged"):][:600]
    assert "readings.reset()" in on_change
    assert 'renderChat([], "")' in on_change
    assert "refreshSessions()" in on_change


# ---------------------------------------------------------------------------
# 四、输出额度要能在界面上改（真实模型下「读数空白」的头号原因是它）
# ---------------------------------------------------------------------------


def test_config_endpoint_round_trips_max_tokens(tmp_path):
    """两个槽位的 max_tokens 都要能改、能读回 —— 否则用户只能去翻 config.json。"""
    import threading
    import time
    from http.server import ThreadingHTTPServer

    from pacs.api.server import PacsHandler

    settings = Settings.load(base_dir=tmp_path)
    settings.ensure_dirs()
    PacsHandler.settings = settings
    PacsHandler.store = Store(settings.db_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), PacsHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)

    def post(body: dict) -> dict:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/config", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        data = post({"max_tokens": 4096, "examiner": {"max_tokens": 8192}})
        assert data["config"]["max_tokens"] == 4096
        assert data["config"]["examiner"]["max_tokens"] == 8192
        assert settings.llm.max_tokens == 4096 and settings.examiner.max_tokens == 8192
        # 不传就保持不变（不要把额度悄悄改回默认）
        again = post({"temperature": 0.5})
        assert again["config"]["max_tokens"] == 4096
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ui_exposes_the_token_budget():
    import pacs

    js = (pathlib.Path(pacs.__file__).parent / "web" / "static" / "js" / "views" / "model.js"
          ).read_text(encoding="utf-8")
    assert "cfg-max" in js and "cfg-emax" in js
    assert "max_tokens" in js
    # 环境卡片上要能一眼看到它（读数空白时第一个该查的数）
    assert "输出额度" in js


# ---------------------------------------------------------------------------
# 五、配好真实 API 之后，「离线 mock」的提示必须消失
# ---------------------------------------------------------------------------


def test_examiner_can_share_the_brains_key_on_the_same_endpoint():
    """只填一套 Key 也要真的能用 —— 否则门禁永远挂着「请配置真实 API」。"""
    from pacs.config import LLMSettings

    same = Settings(
        llm=LLMSettings(provider="openai", model="m", base_url="https://api.deepseek.com/",
                        api_key="DUMMY-SHORT-KEY"),
        # 界面保存配置会写出这一段（api_key 空）——它不该被当成「没配」
        examiner=LLMSettings(provider="openai", model="m", base_url="https://api.deepseek.com",
                             api_key=""),
    )
    assert same.same_endpoint() is True
    assert same.examiner_key() == "DUMMY-SHORT-KEY"
    assert same.examiner_ready() is True and same.is_configured() is True
    assert same.examiner_effective().api_key == "DUMMY-SHORT-KEY"

    other = Settings(
        llm=LLMSettings(provider="openai", base_url="https://a.example.com/v1", api_key="DUMMY-SHORT-KEY"),
        examiner=LLMSettings(provider="openai", base_url="https://b.example.com/v1", api_key=""),
    )
    assert other.same_endpoint() is False
    assert other.examiner_key() == "" and other.examiner_ready() is False

    own = Settings(
        llm=LLMSettings(provider="openai", base_url="https://a/v1", api_key="DUMMY-BRAIN-KEY"),
        examiner=LLMSettings(provider="openai", base_url="https://b/v1", api_key="DUMMY-EXAM-KEY"),
    )
    assert own.examiner_key() == "DUMMY-EXAM-KEY" and own.examiner_ready() is True


def test_examiner_provider_uses_the_fallback_key():
    """判定说「可用」，构造出来的 provider 就必须真带 Key（两处读同一个 Key）。"""
    from pacs.config import LLMSettings
    from pacs.llm.provider import get_examiner_provider

    settings = Settings(
        llm=LLMSettings(provider="openai", model="deepseek-flash",
                        base_url="https://api.deepseek.com", api_key="DUMMY-BOTH-KEY"),
        examiner=LLMSettings(provider="openai", model="deepseek-flash",
                             base_url="https://api.deepseek.com", api_key=""),
    )
    assert settings.examiner_ready() is True
    provider = get_examiner_provider(settings)      # 没 Key 会在这里抛 LLMError
    assert provider.settings.api_key == "DUMMY-BOTH-KEY"


def test_config_endpoint_readiness_follows_the_saved_key(tmp_path):
    """保存真实 Key 之后，`configured.ready` 必须立刻变成 true（横幅据此消失）。"""
    import threading
    import time
    from http.server import ThreadingHTTPServer

    from pacs.api.server import PacsHandler

    settings = Settings.load(base_dir=tmp_path)
    settings.ensure_dirs()
    PacsHandler.settings = settings
    PacsHandler.store = Store(settings.db_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), PacsHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)

    def post(body: dict) -> dict:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/config", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        before = post({"persist": False})
        assert before["config"]["configured"]["ready"] is False       # 默认 mock
        after = post({
            "provider": "openai", "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com", "api_key": "DUMMY-LIVE-KEY",
            "examiner": {"provider": "openai", "model": "deepseek-flash",
                         "base_url": "https://api.deepseek.com"},
            "persist": False,
        })
        # 只填了大脑一把 Key，提问者/分析者同端点 → 也算配好了
        assert after["config"]["configured"]["brain"] is True
        assert after["config"]["configured"]["examiner"] is True
        assert after["config"]["configured"]["ready"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ui_banner_follows_live_config_not_the_sticky_flag():
    """横幅按**当前配置**判定；曾经点过「离线模式进入」不该让它永久挂着。"""
    import pacs

    root = pathlib.Path(pacs.__file__).parent / "web" / "static" / "js"
    app = (root / "app.js").read_text(encoding="utf-8")
    banner_fn = app[app.index("function renderOfflineBanner"):]
    banner_fn = app[app.index("function renderOfflineBanner"):][:2000]
    assert "store.config" in banner_fn                    # 读实时配置
    assert "offlineAcknowledged" not in banner_fn         # 不看历史标记
    assert "configured?.ready" in banner_fn
    # 配置一变就重算（保存真实 API / 关掉门禁弹窗）
    assert "subscribe(() => renderOfflineBanner())" in app
    modal = (root / "ui" / "modal.js").read_text(encoding="utf-8")
    assert 'update({}, "config")' in modal
    # ① 保存配置后，标记与真实状态对齐
    model_view = (root / "views" / "model.js").read_text(encoding="utf-8")
    assert 'prefs.set("offlineAcknowledged", !ready)' in model_view
