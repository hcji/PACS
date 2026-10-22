"""本地存储（NFR-4、NFR-8）：唯一的 sqlite3 访问层。

设计：关键字段落成列便于查询，完整对象以 JSON 存 payload —— 这样表结构在里程碑演进时
不必频繁迁移，同时保留可审计的原始记录（分析者的原始输出也在反馈的 `raw` 里）。

库中不存逐题选择题作答；只存对话、反馈与剖面（反馈的 `raw` 保留分析者原始输出）。
**已经存在的旧库不会删表**（那是历史数据），新库不再创建该表。
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..phenotype import PhenotypeProfile
from ..brain.perturb import Perturbation

SCHEMA = """
CREATE TABLE IF NOT EXISTS perturbations (
    theta_id    TEXT PRIMARY KEY,
    design      TEXT,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    model_id    TEXT,
    created_at  TEXT NOT NULL,
    payload     TEXT
);
CREATE TABLE IF NOT EXISTS phenotypes (
    theta_id    TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    model_id    TEXT,
    replicate   INTEGER NOT NULL DEFAULT 0,
    instrument  TEXT,
    parse_rate  REAL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (theta_id, run_id, model_id, replicate)
);
CREATE TABLE IF NOT EXISTS exports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    path        TEXT,
    payload     TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS freezes (
    freeze_id   TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    event       TEXT NOT NULL,
    detail      TEXT
);
-- 大脑库：一个「模拟大脑」= 具名 θ + 来源。theta_id 唯一，同 θ 不会存成两个大脑。
CREATE TABLE IF NOT EXISTS brains (
    brain_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    source      TEXT,
    theta_id    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_brains_theta ON brains(theta_id);
CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT NOT NULL
);
-- 对话探针会话（表型观察通道 2）：提问者 LLM 提问 → 模拟大脑作答。
CREATE TABLE IF NOT EXISTS dialogues (
    session_id     TEXT PRIMARY KEY,
    brain_id       TEXT,
    theta_id       TEXT,
    brain_model    TEXT,
    examiner_model TEXT,
    n_turns        INTEGER,
    payload        TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dialogue_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    payload     TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON dialogue_turns(session_id);
-- 分析者 LLM 对一段对话形成的表型反馈（通道 2 的表型读数）。
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    theta_id    TEXT,
    model_id    TEXT,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
-- 用户上传的量表/题库（未来可由分析者对话打分）。
CREATE TABLE IF NOT EXISTS item_sets (
    set_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    scoring_mode TEXT,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Store:
    """PACS 的本地库。全部数据留在本地（NFR-8）。"""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：Web 界面用 ThreadingHTTPServer，每个请求在独立工作线程里，
        # 后台运行任务也在另一个线程里写库。sqlite3 默认禁止跨线程复用连接，因此这里放开限制，
        # **并用一把 RLock 串行化所有访问**（sqlite 连接本身不是线程安全的）。
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate_phenotypes()
            self._conn.commit()

    def _migrate_phenotypes(self) -> None:
        """老库的 phenotypes 缺少 replicate / instrument 列 —— 缺哪列补哪列。

        `replicate` 是主键的一部分，sqlite 无法直接改主键，因此走「新建表 → 拷数据 → 换名」。
        `instrument` 只是可空的数据列，直接 ALTER 即可（历史行留空，由 payload 内的字段兜底）。
        """
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(phenotypes)").fetchall()
        }
        if not columns:
            return
        if "replicate" not in columns:
            self._conn.executescript(
                """
                ALTER TABLE phenotypes RENAME TO phenotypes_old;
                CREATE TABLE phenotypes (
                    theta_id    TEXT NOT NULL,
                    run_id      TEXT NOT NULL,
                    model_id    TEXT,
                    replicate   INTEGER NOT NULL DEFAULT 0,
                    instrument  TEXT,
                    parse_rate  REAL,
                    payload     TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    PRIMARY KEY (theta_id, run_id, model_id, replicate)
                );
                INSERT OR IGNORE INTO phenotypes
                    (theta_id, run_id, model_id, replicate, parse_rate, payload, created_at)
                SELECT theta_id, run_id, model_id, 0, parse_rate, payload, created_at
                FROM phenotypes_old;
                DROP TABLE phenotypes_old;
                """
            )
            columns = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(phenotypes)").fetchall()
            }
        if "instrument" not in columns:
            self._conn.execute("ALTER TABLE phenotypes ADD COLUMN instrument TEXT")

    # ---------- 生命周期 ----------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- 写入 ----------

    def audit(self, event: str, detail: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit (ts, event, detail) VALUES (?, ?, ?)",
                (now_iso(), event, detail),
            )
            self._conn.commit()

    def save_perturbation(self, perturbation: Perturbation) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO perturbations (theta_id, design, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    perturbation.theta_id,
                    perturbation.design,
                    json.dumps(perturbation.to_dict(), ensure_ascii=False),
                    now_iso(),
                ),
            )
            self._conn.commit()

    def save_run(self, run_id: str, model_id: str, payload: dict | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, model_id, created_at, payload) "
                "VALUES (?, ?, ?, ?)",
                (run_id, model_id, now_iso(), json.dumps(payload or {}, ensure_ascii=False)),
            )
            self._conn.commit()

    def save_profile(self, profile: PhenotypeProfile) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO phenotypes (theta_id, run_id, model_id, replicate, "
                "instrument, parse_rate, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    profile.theta_id,
                    profile.run_id,
                    profile.model_id,
                    int(profile.replicate),
                    getattr(profile, "instrument", None),
                    profile.parse_rate,
                    json.dumps(profile.to_dict(), ensure_ascii=False),
                    now_iso(),
                ),
            )
            self._conn.commit()

    def save_export(self, kind: str, path: str | Path | None, payload: dict | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO exports (kind, path, payload, created_at) VALUES (?, ?, ?, ?)",
                (
                    kind,
                    str(path) if path else None,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now_iso(),
                ),
            )
            self._conn.commit()

    def save_freeze(self, freeze_id: str, payload: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO freezes (freeze_id, payload, created_at) "
                "VALUES (?, ?, ?)",
                (freeze_id, json.dumps(payload, ensure_ascii=False), now_iso()),
            )
            self._conn.commit()

    # ---------- 大脑库（保存 / 切换）----------

    def save_brain(
        self,
        brain_id: str,
        name: str,
        theta_id: str,
        payload: dict,
        source: str = "user",
    ) -> str:
        """保存一个具名大脑。theta_id 唯一：同一个 θ 重复保存只更新名字，不产生副本。"""
        ts = now_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT brain_id FROM brains WHERE theta_id = ?", (theta_id,)
            ).fetchone()
            if row:
                brain_id = row["brain_id"]
                self._conn.execute(
                    "UPDATE brains SET name = ?, payload = ?, source = ?, updated_at = ? "
                    "WHERE brain_id = ?",
                    (name, json.dumps(payload, ensure_ascii=False), source, ts, brain_id),
                )
            else:
                self._conn.execute(
                    "INSERT INTO brains (brain_id, name, source, theta_id, payload, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (brain_id, name, source, theta_id,
                     json.dumps(payload, ensure_ascii=False), ts, ts),
                )
            self._conn.commit()
        return brain_id

    def list_brains(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT brain_id, name, source, theta_id, payload, created_at, updated_at "
                "FROM brains ORDER BY updated_at DESC"
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item["payload"])
            out.append(item)
        return out

    def load_brain(self, brain_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT brain_id, name, source, theta_id, payload, created_at, updated_at "
                "FROM brains WHERE brain_id = ?",
                (brain_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        return item

    def rename_brain(self, brain_id: str, name: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE brains SET name = ?, updated_at = ? WHERE brain_id = ?",
                (name, now_iso(), brain_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def delete_brain(self, brain_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM brains WHERE brain_id = ?", (brain_id,))
            self._conn.commit()
        return cur.rowcount > 0

    # ---------- 键值元数据（当前大脑等）----------

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now_iso()),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    # ---------- 对话探针 ----------

    def save_dialogue_session(self, session_id: str, payload: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO dialogues (session_id, brain_id, theta_id, "
                "brain_model, examiner_model, n_turns, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    payload.get("brain_id"),
                    payload.get("theta_id"),
                    payload.get("brain_model"),
                    payload.get("examiner_model"),
                    int(payload.get("n_turns") or 0),
                    json.dumps(payload, ensure_ascii=False),
                    now_iso(),
                ),
            )
            self._conn.commit()

    def save_dialogue_turn(
        self, session_id: str, turn_index: int, role: str, content: str, payload: dict | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO dialogue_turns (session_id, turn_index, role, content, payload, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    int(turn_index),
                    role,
                    content,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now_iso(),
                ),
            )
            self._conn.commit()

    def list_dialogue_sessions(self, limit: int = 50) -> list[dict]:
        """最近 N 段对话（新的在前）。

        **同秒决胜用 rowid**：`created_at` 只到秒，批量运行（或 mock 快速连跑）会有一批
        会话落在同一秒里，只按 `created_at` 排序时它们的先后是未定义的 —— 于是「最近一次
        读数」可能取到同秒里较早的那一段，界面显示的就未必是刚跑出来的结果。
        rowid 随插入递增，拿它做第二排序键，同秒内也必然是后来者在前。
        """
        return self._query_sessions("", (), limit)

    def list_brain_sessions(self, brain_id: str, theta_id: str, limit: int = 500) -> list[dict]:
        """某具大脑（或某一具心智）的对话史，新的在前。

        归属规则与界面「历史对话」一致：`brain_id` 相同即同一次大脑记录；`brain_id` 为空
        且 `theta_id` 相同的也算（命令行留下、尚未绑定大脑库的会话）。
        """
        where = "WHERE (brain_id = ? AND brain_id <> '') OR (brain_id = '' AND theta_id = ?)"
        return self._query_sessions(where, (str(brain_id or ""), str(theta_id or "")), limit)

    def _query_sessions(self, where: str, params: tuple, limit: int) -> list[dict]:
        sql = (
            "SELECT session_id, brain_id, theta_id, brain_model, examiner_model, n_turns, "
            "payload, created_at FROM dialogues "
            + where +
            " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (*params, int(limit))).fetchall()
        out: list[dict] = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item["payload"])
            out.append(item)
        return out

    def load_dialogue_turns(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn_index, role, content, payload, created_at FROM dialogue_turns "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item["payload"] or "{}")
            out.append(item)
        return out

    # ---------- 表型反馈（通道 2）----------

    def save_feedback(
        self, session_id: str, theta_id: str, model_id: str, payload: dict
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback (session_id, theta_id, model_id, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, theta_id, model_id,
                 json.dumps(payload, ensure_ascii=False), now_iso()),
            )
            self._conn.commit()

    def load_feedback(self, session_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM feedback WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    # ---------- 用户上传的题库 / 量表 ----------

    def save_item_set(self, set_id: str, name: str, scoring_mode: str, payload: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO item_sets (set_id, name, scoring_mode, payload, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (set_id, name, scoring_mode,
                 json.dumps(payload, ensure_ascii=False), now_iso()),
            )
            self._conn.commit()

    def list_item_sets(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT set_id, name, scoring_mode, payload, created_at FROM item_sets "
                "ORDER BY created_at DESC"
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item["payload"])
            out.append(item)
        return out

    def delete_item_set(self, set_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM item_sets WHERE set_id = ?", (set_id,))
            self._conn.commit()
        return cur.rowcount > 0

    # ---------- 读取 ----------

    def load_profiles(
        self, *, run_id: str | None = None, model_id: str | None = None
    ) -> list[PhenotypeProfile]:
        query = "SELECT payload FROM phenotypes WHERE 1=1"
        params: list[str] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        query += " ORDER BY theta_id"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [PhenotypeProfile.from_dict(json.loads(r["payload"])) for r in rows]

    def runs_for_theta(self, theta_id: str) -> list[dict]:
        """这个 θ 在哪些运行里被测量过（最近的在前）。

        用途：把「机制分析」限定到**当前大脑所属的那次运行** —— 一个 θ 只有在被测量过之后
        才谈得上与它相关的剂量-反应/特异性/亚型分析。

        `instrument` 从 phenotypes 里取（同一运行内一致），界面据此说明读数由什么测出。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.run_id, r.model_id, r.created_at, "
                "(SELECT COUNT(*) FROM phenotypes p WHERE p.run_id = r.run_id) AS n_profiles, "
                "(SELECT p.instrument FROM phenotypes p WHERE p.run_id = r.run_id LIMIT 1) AS instrument "
                "FROM runs r WHERE r.run_id IN "
                "(SELECT DISTINCT run_id FROM phenotypes WHERE theta_id = ?) "
                "ORDER BY r.created_at DESC",
                (theta_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_runs(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.run_id, r.model_id, r.created_at, "
                "(SELECT COUNT(*) FROM phenotypes p WHERE p.run_id = r.run_id) AS n_profiles, "
                "(SELECT p.instrument FROM phenotypes p WHERE p.run_id = r.run_id LIMIT 1) AS instrument "
                "FROM runs r ORDER BY r.created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for table in (
                "perturbations", "runs", "phenotypes", "exports", "freezes",
                "brains", "dialogues", "feedback", "item_sets",
            ):
                row = self._conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
                out[table] = int(row["c"])
        return out

    def load_freeze(self, freeze_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM freezes WHERE freeze_id = ?", (freeze_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def model_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT model_id FROM phenotypes WHERE model_id IS NOT NULL "
                "ORDER BY model_id"
            ).fetchall()
        return [r["model_id"] for r in rows]
