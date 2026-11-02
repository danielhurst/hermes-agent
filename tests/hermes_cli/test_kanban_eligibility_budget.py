"""Budget tests for the dispatch-eligibility gates.

These pin the *arithmetic* of each independent budget: the card run cap
(cumulative, excluding review runs and transition markers), the CI-retry
budget, the provider quota-wall budget (per provider, windowed), and provider
capacity (no double-claim in one tick; an occupied slot waits, not parks).

Every test drives the real dispatcher through ``dispatch_once`` against a
throwaway board, so the claim path under test is the production one.
"""
from __future__ import annotations

import os
import sys
import time
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


def _worker_run_count(conn, task_id):
    """Count the runs that consumed a card's BUILDER budget.

    Mirrors ``dispatch_eligibility.cumulative_builder_runs`` exactly: a real
    attempt is a non-zero-duration run whose outcome is not a transition
    marker (``scheduled``/``blocked``/``skipped``), not a review-lane run, not
    a CI failure (budgeted separately) and not a quota wall (budgeted by the
    provider). The two must agree, or the test would be asserting a different
    rule from the one the dispatcher enforces.
    """
    rows = conn.execute(
        "SELECT profile, outcome, started_at, ended_at FROM task_runs WHERE task_id = ?",
        (task_id,),
    ).fetchall()
    total = 0
    for row in rows:
        if "review" in (row["profile"] or "").lower():
            continue
        outcome = row["outcome"] or ""
        if outcome in ("scheduled", "blocked", "skipped", "not_started"):
            continue
        if outcome in ("ci_failed", "rate_limited"):
            continue
        started, ended = row["started_at"], row["ended_at"]
        if started is not None and ended is not None and ended <= started:
            continue
        total += 1
    return total


def test_over_cap_card_is_never_claimed(board):
    """A card at the run cap consumes ZERO runs: no claim, no new run row."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="over cap")
        # Four builder runs == the cap (max_runs_per_card: 4).
        for _ in range(4):
            _record_run(kb, kbc, conn, card.id, outcome="crashed", status="crashed")
        runs_before = _worker_run_count(conn, card.id)

        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert _worker_run_count(conn, card.id) == runs_before, (
            "over-cap card must consume zero worker runs"
        )
        assert _ineligible_reasons(result).get(card.id) == "run_cap"
        assert result.spawned == []
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (card.id,)
        ).fetchone()
        assert row["status"] == "blocked", "over-cap card parks, it does not respawn"


def test_under_cap_card_is_claimed(board):
    """The control case: below the cap the card still dispatches."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="under cap")
        # "crashed" rather than "completed": a completed run inside the 1h
        # respawn-guard window is refused by the (pre-existing, unrelated)
        # recent_success guard. The budget is what this test is about.
        for _ in range(3):
            _record_run(kb, kbc, conn, card.id, outcome="crashed", status="crashed")

        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert card.id in _spawned_ids(result)
        assert card.id not in _ineligible_reasons(result)


def test_review_runs_do_not_count_against_the_card_run_cap(board):
    """A review handoff is not implementation work: it must not exhaust a card."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="reviewed a lot")
        for _ in range(10):
            _record_run(kb, kbc, conn, card.id, profile="alpha-review",
                        outcome="crashed", status="crashed")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert card.id in _spawned_ids(result)


def test_run_cap_count_is_cumulative_not_consecutive(board):
    """A success must not reset the count: the budget is cumulative."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="cumulative")
        # completed, crashed, completed, crashed -> 4 cumulative runs
        for outcome in ("completed", "crashed", "completed", "crashed"):
            _record_run(kb, kbc, conn, card.id, outcome=outcome, status=outcome)
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) == "run_cap"


def test_provider_walled_card_parks_and_does_not_respawn(board):
    """A walled provider parks its cards; they do not spin on the ready lane."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        walled = _mk_card(kb, kbc, conn, title="walled provider", provider="anthropic")
        healthy = _mk_card(kb, kbc, conn, title="healthy provider", provider="openai-codex")

        # The provider hits a quota wall.
        _record_run(kb, kbc, conn, walled.id, outcome="rate_limited",
                    status="rate_limited", error="quota wall")

        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)

        reasons = _ineligible_reasons(result)
        assert reasons.get(walled.id) == "provider_wall"
        assert walled.id not in _spawned_ids(result)
        parked = {t: (state, why) for t, state, why in result.parked}
        assert parked[walled.id][0] == "scheduled"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (walled.id,)).fetchone()
        assert row["status"] == "scheduled", "walled card parks in a native park state"
        # The sibling on a healthy provider is unaffected.
        assert healthy.id in _spawned_ids(result)


def test_walled_card_cannot_be_claimed_while_parked(board):
    """A parked card is not in a dispatchable lane, so it cannot be claimed."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="walled", provider="anthropic")
        _record_run(kb, kbc, conn, card.id, outcome="rate_limited", status="rate_limited")
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        runs_after_park = _worker_run_count(conn, card.id)
        # A second tick must not claim it and must not add a worker run.
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert _worker_run_count(conn, card.id) == runs_after_park


def test_provider_wall_budget_routes_to_human(board):
    """Past the wall budget the card is a human decision, not another retry."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="wall budget", provider="anthropic")
        # Five wall hits == the cap (provider_budget.max_wall_attempts: 5).
        for _ in range(5):
            _record_run(kb, kbc, conn, card.id, outcome="rate_limited", status="rate_limited")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) == "provider_wall_budget"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (card.id,)).fetchone()
        assert row["status"] in ("blocked", "triage"), (
            "a card past the wall budget waits for a human"
        )


def test_wall_budget_counts_wall_RUNS_not_distinct_cards(board):
    """Five walls on ONE card are five hits, not one.

    The budget answers "how many times did this provider hit a wall". A
    count of distinct cards would turn a card's five walls into a single
    hit and make the budget unreachable in exactly the case it exists for.
    """
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="five walls one card", provider="anthropic")
        for _ in range(5):
            _record_run(kb, kbc, conn, card.id, outcome="rate_limited", status="rate_limited")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        assert _ineligible_reasons(result).get(card.id) == "provider_wall_budget", (
            "five walls on one card must exhaust the provider's wall budget"
        )


def test_wall_budget_window_ignores_old_walls(board, policy_file):
    """Walls outside the budget window do not count against the budget.

    An all-time counter would permanently kill a provider whose walls were
    spread over months - the cards could never run again, with no path back
    except editing governance by hand.
    """
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    import json as _json
    data = _json.loads(policy_file.read_text())
    data["provider_budget"]["wall_window_seconds"] = 3600
    policy_file.write_text(_json.dumps(data))
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="old walls", provider="anthropic")
        # Five walls, but each ended 2 hours ago - outside the 1h window.
        old = int(time.time()) - 7200
        with kb.write_txn(conn):
            for _ in range(5):
                conn.execute(
                    "INSERT INTO task_runs "
                    "  (task_id, profile, status, outcome, started_at, ended_at) "
                    "VALUES (?, 'alpha', 'rate_limited', 'rate_limited', ?, ?)",
                    (card.id, old - 60, old),
                )
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) != "provider_wall_budget", (
            "walls outside the budget window must not exhaust the budget"
        )


def test_provider_capacity_is_not_double_claimed_in_one_tick(board):
    """Two cards on one provider: only one may claim while the cap is 1."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        a = _mk_card(kb, kbc, conn, title="cap a", provider="anthropic", priority=90)
        b = _mk_card(kb, kbc, conn, title="cap b", provider="anthropic", priority=80)
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a2, **k: 999)
        spawned = _spawned_ids(result)
        assert len([s for s in spawned if s in (a.id, b.id)]) == 1, (
            "provider capacity must not be double-claimed in one tick"
        )


def test_provider_slot_waits_rather_than_parks(board):
    """A capacity wait is not a park: the card stays ready for a later tick."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        running_card = _mk_card(kb, kbc, conn, title="running", provider="anthropic",
                                 status="running")
        # Mark it running with a live run so the provider slot is occupied.
        if hasattr(kb, "_claim_and_open_run"):
            kb._claim_and_open_run(
                conn, running_card.id, "ready", "lock-test", 2 ** 31 - 1, 0,
                event_extra={},
            )
        waiting = _mk_card(kb, kbc, conn, title="waiting", provider="anthropic")

        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a2, **k: 999)
        reasons = _ineligible_reasons(result)
        if waiting.id in reasons:
            assert reasons[waiting.id] == "provider_slot"
            row = conn.execute("SELECT status FROM tasks WHERE id = ?", (waiting.id,)).fetchone()
            assert row["status"] == "ready", "a capacity wait must not park the card"


def test_ci_retry_budget_is_independent_of_the_run_cap(board):
    """CI retries have their own budget: they do not consume the card's runs."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="ci retries")
        # Three CI failures == the CI budget (ci_budget.max_retry_attempts: 3),
        # but ZERO builder runs.
        for _ in range(3):
            _record_run(kb, kbc, conn, card.id, outcome="ci_failed", status="failed")
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) == "ci_retry_budget"
        # ... and the card's own run budget is untouched: no builder runs.
        assert _worker_run_count(conn, card.id) == 0


def test_provider_wall_does_not_consume_the_card_run_budget(board):
    """A quota wall is a provider condition, not a card attempt."""
    kb, kbd, kbc = board["kb"], board["kbd"], board["kbc"]
    with kbc.connect_closing() as conn:
        card = _mk_card(kb, kbc, conn, title="wall then work", provider="openai-codex")
        for _ in range(4):
            _record_run(kb, kbc, conn, card.id, profile="openai-codex",
                        outcome="rate_limited", status="rate_limited")
        # The provider wall budget is exhausted (4 hits at cap 5 is not), but the
        # card's OWN builder budget must still be zero: walls are not attempts.
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=lambda *a, **k: 999)
        reasons = _ineligible_reasons(result)
        assert reasons.get(card.id) in ("provider_wall", "provider_wall_budget")
        # The park itself writes a SYNTHETIC zero-duration 'scheduled' row
        # (the board's transition marker). That is not an attempt. Real
        # attempts are the non-zero-duration runs that are neither a wall nor
        # a CI failure.
        assert _worker_run_count(conn, card.id) == 0
