"""Policy-contract tests for the dispatch-eligibility gates.

These pin the behaviour of the gate itself, independent of any particular
budget's arithmetic:

* a no-op evaluation does not append a new audit row,
* the pre-existing respawn guard still blocks an active-PR card,
* a board with no policy file is left completely untouched,
* a policy file that exists but is malformed fails CLOSED.

Every test drives the real dispatcher through ``dispatch_once`` against a
throwaway board, so the claim path under test is the production one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.hermes_cli.kanban_eligibility_harness import (
    _ineligible_reasons,
    _mk_card,
    _record_run,
    _spawned_ids,
    board,
    policy_file,
)

def test_no_op_evaluation_does_not_repeat_audit_events(board):
    """A parked card is evaluated every tick; it must not write a row each time."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="quiet", provider="anthropic")
        _record_run(kb, kbc, conn, card.id, outcome="rate_limited", status="rate_limited")
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        first = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'ineligible'",
            (card.id,),
        ).fetchone()[0]
        for _ in range(5):
            kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        second = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'ineligible'",
            (card.id,),
        ).fetchone()[0]
        assert second == first, "repeated no-op evaluations must not append audit rows"


def test_existing_respawn_guard_still_blocks_active_pr(board):
    """The pre-existing active-PR guard survives the new gates."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="has pr")
        kb.add_comment(conn, card.id, "tester", "see https://github.com/o/r/pull/7")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        guarded = dict(result.respawn_guarded)
        assert guarded.get(card.id) == "active_pr"
        assert card.id not in _spawned_ids(result)


# --------------------------------------------------------------- fail closed ---

def test_board_without_a_policy_file_is_untouched(board, tmp_path, monkeypatch):
    """A board with NO policy file has not opted in: it must dispatch normally.

    These budgets are a factory opt-in, not a Hermes-wide requirement. If a
    missing policy froze every board, every non-factory Hermes user would be
    locked out of their own kanban.
    """
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    missing = tmp_path / "does-not-exist.json"
    monkeypatch.setenv("SANWA_LIMITS", str(missing))
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="no policy file")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert card.id in _spawned_ids(result), (
            "a board with no policy file must dispatch normally"
        )


def test_malformed_policy_fails_closed(board, tmp_path, monkeypatch):
    """A policy that EXISTS but cannot be parsed fails closed: refuse, do not guess."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("SANWA_LIMITS", str(bad))
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="bad policy")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert _ineligible_reasons(result).get(card.id) == "policy_unreadable"
        assert result.spawned == []
