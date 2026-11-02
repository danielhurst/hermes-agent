"""Shared harness for the dispatch-eligibility regression tests.

Every test in this suite drives the REAL dispatcher through ``dispatch_once``
against a throwaway board, so the claim path under test is the production one.

This module holds the pieces both test files need: the ``board`` and
``policy_file`` fixtures, the fresh loader that points at the factory's
eligibility module, and the card/run builders.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import pytest

FACTORY_LIB = Path("/Users/danielhurst/src/hermes-factory/scripts/lib")
# The hermes-agent checkout under test is the one this file lives in. It MUST
# win over the installed package: inserting the live checkout here would
# silently test the OLD dispatcher and every assertion would pass vacuously
# (or fail for the wrong reason).
AGENT_ROOT = Path(__file__).resolve().parents[2]

if str(FACTORY_LIB) not in sys.path:
    sys.path.insert(0, str(FACTORY_LIB))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))
else:
    # Already importable, but possibly behind another copy: move it to the front.
    sys.path.remove(str(AGENT_ROOT))
    sys.path.insert(0, str(AGENT_ROOT))


@pytest.fixture()
def policy_file(tmp_path: Path):
    """A limits.json the tests can mutate."""
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({
        "max_runs_per_card": 4,
        "dispatch_eligibility": {
            "max_in_progress_per_provider": 1,
            "provider_wall_cooldown_seconds": 900,
        },
        "provider_budget": {"max_wall_attempts": 5},
        "ci_budget": {"max_retry_attempts": 3},
    }), encoding="utf-8")
    return path


@pytest.fixture()
def board(tmp_path: Path, policy_file: Path, monkeypatch):
    """A fresh board with the eligibility policy wired in."""
    home = tmp_path / "home"
    os.makedirs(home / "profiles" / "alpha", exist_ok=True)
    (home / "profiles" / "alpha" / "config.yaml").write_text(
        "model:\n  provider: anthropic\n", encoding="utf-8")
    os.makedirs(home / "profiles" / "default", exist_ok=True)
    (home / "profiles" / "default" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    # The eligibility module is resolved by path at runtime; reload it so it
    # picks up the monkeypatched HOME.
    sys.modules.pop("dispatch_eligibility", None)

    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    # Point the eligibility loader at the test's limits file.
    monkeypatch.setattr(
        kbd, "_ELIGIBILITY_SEARCH_PATHS",
        (str(FACTORY_LIB),),
    )
    monkeypatch.setattr(
        kbd, "_load_eligibility_module",
        _fresh_eligibility_loader(),
    )
    # The policy module resolves its limits file at CALL time, so the test
    # points it at the temporary file with an env var rather than by
    # patching a module constant (which would pin the path at import time).
    monkeypatch.setenv("SANWA_LIMITS", str(policy_file))

    with kbc.connect_closing():
        kanban_db.create_board(slug="default", name="Test")
    return {"home": home, "policy": policy_file, "kb": kanban_db, "kbd": kbd, "kbc": kbc}


def _fresh_eligibility_loader():
    """A loader that re-imports the eligibility module per call.

    The production loader caches the module per process; the tests mutate
    HERMES_HOME between cases, so each load must be fresh.
    """
    import importlib.util

    def _loader():
        global _eligibility_module, _eligibility_module_loaded
        candidate = FACTORY_LIB / "dispatch_eligibility.py"
        spec = importlib.util.spec_from_file_location("dispatch_eligibility", candidate)
        module = importlib.util.module_from_spec(spec)
        sys.modules["dispatch_eligibility"] = module
        spec.loader.exec_module(module)
        return module

    return _loader


class _Card:
    """Minimal card handle: the id is what every assertion needs."""

    def __init__(self, card_id: str):
        self.id = card_id


#: A provider override requires a model override (the board rejects a bare
#: provider - a bare ``--provider`` would re-resolve the profile's model
#: against another backend). The model is irrelevant to every gate here; only
#: the provider is.
_PROVIDER_MODELS = {
    "anthropic": "claude-sonnet-4.6",
    "openai-codex": "gpt-5.6-sol",
    "minimax": "MiniMax-M3",
}


def _mk_card(kb, kbc, conn, *, title="card", assignee="alpha", provider=None,
             status="ready", priority=50, max_runtime=None):
    """Create a card in the given state.

    ``create_task`` only accepts ``initial_status`` of ``blocked`` or
    ``running``; ``ready`` is what a card with no incomplete parent gets by
    default, and ``todo`` is what an incomplete parent would produce. The
    requested state is therefore applied after creation with the board's own
    status write, so the test board matches production rows.
    """
    model = _PROVIDER_MODELS.get(provider) if provider else None
    card_id = kb.create_task(
        conn, title=title, body="body", assignee=assignee,
        priority=priority,
        provider_override=provider, model_override=model,
        max_runtime_seconds=max_runtime,
    )
    if status not in ("ready",):
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, "
                "worker_pid = NULL, current_run_id = NULL WHERE id = ?",
                (status, card_id),
            )
    return _Card(card_id)


def _record_run(kb, kbc, conn, task_id, *, profile="alpha", outcome="crashed",
                status="crashed", error=None, metadata=None):
    """Append a CLOSED, REAL run row to the card's claim log.

    A run row is opened by a claim and closed by ``_end_run``. The test needs
    historical runs on a card that is not currently running, so the row is
    written directly here with the board's own columns. Its duration is
    non-zero, so it counts as a worker attempt - the board's synthetic
    transition markers are zero-duration.
    """
    now = int(time.time())
    with kb.write_txn(conn):
        cur = conn.execute(
            "INSERT INTO task_runs "
            "  (task_id, profile, status, outcome, error, metadata, started_at, ended_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, profile, status, outcome, error,
             json.dumps(metadata or {}), now - 120, now),
        )
        return int(cur.lastrowid)



def _spawned_ids(result):
    """The task ids the tick spawned (``spawned`` entries are 3-tuples)."""
    return [entry[0] for entry in result.spawned]

def _ineligible_reasons(result):
    """Map task id -> refusal reason (``ineligible`` entries are 2-tuples)."""
    return dict(result.ineligible)
