# =============================================================================
# app/runtime/fault_state_tracker.py
# M12 Stage 4 (Item 1) — Server-side fault-STATE tracker.
#
# WHY THIS EXISTS
# ---------------
# A pump fault is a PHYSICAL state with memory. The synthetic generators put a
# fault ONSET mid-sequence (gen_acute_fault: onset = n_steps//5..//3), so the
# first windows are genuinely normal (M7 correctly says label 0), the fault then
# matures (M7 names it), then a late window settles and M7 drops back to label 0.
# A memoryless per-window display flickers the fault in and out of existence.
# This tracker is the missing EPISODE-level state: once the validated detector
# (AlertStateMachine) reaches WARN/DANGER and M7 names a fault with enough
# confidence/persistence, the tracker LATCHES that named fault until an operator
# verdict on the Predictions tab clears it.
#
# LOCKED BEHAVIOUR (from M12 Stage 4 session state §2, §6):
#   1. Latch a named WARN/DANGER fault episode. Popup acknowledge/dismiss does
#      NOT clear it; only /api/operator_verdict clears it.
#   2. Escalate UPWARD only (WATCH < WARN < DANGER). NEVER self-de-escalate.
#   3. Physics-ranked conflict resolution: Group B causal chains outrank their
#      component singles; else highest-severity-then-most-confident holds.
#   4. Confidence/persistence floor: a name latches only after it clears
#      latch_min_confidence for latch_min_persistence consecutive windows.
#      Fragile single windows do NOT latch a name.
#   5. WATCH is EXCLUDED from the latch/re-nag/verdict system, BUT the tracker
#      still NAMES the WATCH context (multiple candidate fault types to check),
#      so the dashboard never shows a bare "normal" under a WATCH dial.
#   6. NEVER fabricate a fault name M7 did not predict (§8 lesson #7). When the
#      stack escalated but M7 said label 0, the latched display stays
#      "Unclassified anomaly" — the D2 evidence text — not an invented class.
#
# ARCHITECTURE
#   Lives on app.state.fault_tracker, same async-locked pattern as CUSUMState /
#   RollingState. anomaly.py calls update() AFTER M7 + AFTER the D2 override,
#   passing the already-resolved display fields; the tracker returns the
#   episode-resolved display (which may be the same, or the held latch).
#
#   /api/acknowledge  -> tracker.acknowledge_popup()  (silences popup, NOT latch)
#   /api/operator_verdict -> tracker.clear_latch(verdict)  (the ONLY clear)
#
# Invariant compliance:
#   - Reads alert_state from the validated AlertStateMachine; never invents one.
#   - Never mutates CUSUM/Rolling state (C-25 respected — separate object).
#   - score_C remains advisory-only; not used to drive the latch.
# =============================================================================

import asyncio
from collections import deque
from datetime import datetime
from typing import Optional

# Severity ladder (must mirror alert_state_machine.ESCALATION ordering).
SEVERITY = {"NORMAL": 0, "WATCH": 1, "WARN": 2, "DANGER": 3}
SEVERITY_NAME = {v: k for k, v in SEVERITY.items()}

# Group B compound chains and their physically causal component singles.
# Used so a chain LATCH is never demoted to one of its own components.
GROUP_B_LABELS = {7, 8, 9, 10, 11, 12}
GROUP_B_COMPONENTS = {
    7:  {1, 5},    # bearing + overloading
    8:  {3, 4},    # cavitation + seal
    9:  {2, 1},    # imbalance + bearing
    10: {4, 3},    # seal + cavitation_H
    11: {5, 1},    # overloading + bearing
    12: {2, 3},    # imbalance + cavitation
}

# Labels whose WATCH-stage context is gradual-wear (CUSUM-driven path).
GRADUAL_WEAR_LABELS = {21}


class FaultStateTracker:
    """
    Async-safe episode-level fault-state latch.

    Public async API (all acquire the lock):
        update(...)            -> dict (episode-resolved display + latch meta)
        acknowledge_popup()    -> dict (silence popup; latch UNCHANGED)
        clear_latch(verdict)   -> dict (the ONLY thing that releases a latch)
        get_state()            -> dict (snapshot, no mutation)
        reset()                -> dict (full reset; maintenance ack of L3/L4 only;
                                        does NOT release a WARN/DANGER latch —
                                        see note in acknowledge router wiring)

    The tracker holds at most ONE active latched episode at a time (single-pump
    v14.2). Conflicts within an episode are resolved by physics rank.
    """

    def __init__(
        self,
        latch_min_confidence: float = 0.55,
        latch_min_persistence: int = 2,
        watch_candidate_window: int = 30,
        watch_candidate_max: int = 3,
    ):
        # Latching thresholds. min_confidence is on the CALIBRATED confidence
        # (post temperature+ceiling), expressed as a fraction (0.55 = 55%).
        self.latch_min_confidence = float(latch_min_confidence)
        self.latch_min_persistence = int(latch_min_persistence)

        # WATCH naming: recent label votes drive the "what to watch" candidate set.
        self.watch_candidate_window = int(watch_candidate_window)
        self.watch_candidate_max = int(watch_candidate_max)

        self._lock = asyncio.Lock()

        # ── Latched episode (None when no WARN/DANGER episode is held) ──
        self._latched: Optional[dict] = None
        # _latched schema when active:
        #   {
        #     "label_int": int,          # the held class (0 => Unclassified anomaly)
        #     "label_name": str,         # display label (Field 1)
        #     "severity": str,           # WARN | DANGER (the held level)
        #     "confidence_pct": float,   # confidence at latch/escalation time
        #     "condition": str,          # Field 3 at latch time
        #     "risk": str,               # Field 5
        #     "action": str,             # Field 6
        #     "causal_chain": str|None,
        #     "is_unclassified": bool,   # True when held as D2 "Unclassified anomaly"
        #     "latched_at": iso str,
        #     "escalated_at": iso str,
        #     "popup_acknowledged": bool,# popup silenced but latch still held
        #   }

        # ── Candidate accumulator: count consecutive qualifying windows per name
        #    before a name is allowed to latch. Reset when the candidate changes.
        self._cand_label_int: Optional[int] = None
        self._cand_streak: int = 0

        # ── WATCH context: recent label votes (only when not latched) ──
        self._recent_labels = deque(maxlen=self.watch_candidate_window)

        # ── Bookkeeping ──
        self._n_updates = 0
        self._last_verdict: Optional[dict] = None
        self._created_at = datetime.utcnow()
        # Per-fault episode identity: each distinct latched fault gets a new id.
        self._episode_counter = 0
        # Post-verdict suppression: after a verdict clears a latch, do not
        # re-latch the SAME label for this many updates (lets the dashboard
        # visibly return to live even if the fault is still physically present).
        self._suppress_label = None
        self._suppress_remaining = 0
        self.verdict_suppress_updates = 8

    # =========================================================================
    # MAIN UPDATE — called by anomaly.py after M7 + after D2 override
    # =========================================================================
    async def update(
        self,
        *,
        alert_state: str,
        label_int: int,
        label_name: str,
        confidence_pct: float,
        condition: str,
        risk: str,
        action: str,
        causal_chain: Optional[str] = None,
        is_unclassified: bool = False,
        label_map: Optional[dict] = None,
        physics_ctx: Optional[dict] = None,
    ) -> dict:
        """
        Ingest one window's resolved verdict and return the EPISODE-resolved
        display. anomaly.py passes the values it already computed (after the D2
        override); the tracker decides whether to (a) hold a prior latch, (b)
        escalate it, (c) start a new latch, or (d) name a WATCH context.

        Returns dict:
          {
            "display_label": str,        # Field 1 to show
            "display_severity": str,     # latched/effective alert level for the dial
            "display_confidence_pct": float,
            "condition": str, "risk": str, "action": str,
            "causal_chain": str|None,
            "is_latched": bool,
            "is_unclassified": bool,
            "watch_candidates": [str,...]|None,
            "popup_should_show": bool,   # frontend latch hint (WARN/DANGER + not popup-acked)
            "episode": dict|None,        # full latch snapshot (or None)
            "tracker_reason": str,
          }
        """
        async with self._lock:
            self._n_updates += 1
            # Tick down post-verdict suppression window.
            if self._suppress_remaining > 0:
                self._suppress_remaining -= 1
                if self._suppress_remaining == 0:
                    self._suppress_label = None
            label_map = label_map or {}
            conf_frac = float(confidence_pct) / 100.0
            sev = SEVERITY.get(alert_state, 0)

            # Track recent labels for WATCH naming (only meaningful when not held).
            self._recent_labels.append(int(label_int))

            # ── CASE A: a WARN/DANGER latch is already active ────────────────
            if self._latched is not None:
                return self._handle_active_latch(
                    alert_state, sev, label_int, label_name, conf_frac,
                    confidence_pct, condition, risk, action, causal_chain,
                    is_unclassified,
                )

            # ── CASE B: no latch active. Should we OPEN one? ─────────────────
            # Latch only on WARN/DANGER (WATCH excluded — locked behaviour #5).
            if alert_state in ("WARN", "DANGER"):
                return self._maybe_open_latch(
                    alert_state, sev, label_int, label_name, conf_frac,
                    confidence_pct, condition, risk, action, causal_chain,
                    is_unclassified,
                )

            # ── CASE C: WATCH — name the context, do NOT latch ───────────────
            if alert_state == "WATCH":
                return self._name_watch(
                    label_int, label_name, confidence_pct, condition, risk,
                    action, causal_chain, is_unclassified, label_map, physics_ctx,
                )

            # ── CASE D: NORMAL, no latch — pass through unchanged ────────────
            # Reset the candidate streak: nothing qualifying is happening.
            self._cand_label_int = None
            self._cand_streak = 0
            return self._passthrough(
                alert_state, label_name, confidence_pct, condition, risk,
                action, causal_chain, is_unclassified,
                reason="normal_no_latch",
            )

    # =========================================================================
    # NEW-EPISODE helper — opens a fresh latched episode (re-arms popup)
    # =========================================================================
    def _open_episode(self, *, label_int, label_name, severity, confidence_pct,
                      condition, risk, action, causal_chain, is_unclassified):
        self._episode_counter += 1
        now = datetime.utcnow().isoformat() + "Z"
        self._latched = {
            "episode_id": self._episode_counter,
            "label_int": int(label_int),
            "label_name": label_name,
            "severity": severity,
            "confidence_pct": float(confidence_pct),
            "condition": condition,
            "risk": risk,
            "action": action,
            "causal_chain": causal_chain,
            "is_unclassified": bool(is_unclassified),
            "latched_at": now,
            "escalated_at": now,
            "popup_acknowledged": False,   # fresh episode -> popup re-arms
        }
        return self._latched

    # =========================================================================
    # CASE A — active latch: hold, escalate upward, upgrade name, or OPEN NEW
    # =========================================================================
    def _handle_active_latch(
        self, alert_state, sev, label_int, label_name, conf_frac,
        confidence_pct, condition, risk, action, causal_chain, is_unclassified,
    ) -> dict:
        held = self._latched
        held_sev = SEVERITY[held["severity"]]
        reason_parts = []

        # 0) NEW DISTINCT FAULT -> open a fresh episode (re-arms popup).
        #    A qualifying named fault that is DIFFERENT from the held one and
        #    is NOT a Group-B physics-rank upgrade of it is a genuinely new
        #    condition. Per industrial alerting: a new fault must raise its
        #    own alarm even while a prior one is latched. We close the old
        #    episode and open a new one (new episode_id -> frontend re-pops).
        if (
            label_int != 0
            and label_int != held["label_int"]
            and conf_frac >= self.latch_min_confidence
            and self._is_qualifying_persistent(label_int)
            and not self._physics_rank_wins(new_label=label_int, held_label=held["label_int"])
            and held["label_int"] not in GROUP_B_COMPONENTS.get(label_int, set())
        ):
            # Only treat as a NEW episode when it is not a chain<->component
            # relationship (those are name-upgrades, handled below).
            new_sev = alert_state if alert_state in ("WARN", "DANGER") else held["severity"]
            self._open_episode(
                label_int=label_int, label_name=label_name, severity=new_sev,
                confidence_pct=confidence_pct, condition=condition, risk=risk,
                action=action, causal_chain=causal_chain, is_unclassified=False,
            )
            self._cand_label_int = None
            self._cand_streak = 0
            return self._emit_latched(
                f"NEW episode: distinct fault '{label_name}' opened "
                f"(prev was '{held['label_name']}')"
            )

        # 1) ESCALATE UPWARD: if the live alert is more severe than the held
        #    level, raise the held level (never the reverse — locked behaviour #2).
        if sev > held_sev and alert_state in ("WARN", "DANGER"):
            held["severity"] = alert_state
            held["escalated_at"] = datetime.utcnow().isoformat() + "Z"
            # Re-arm the popup on escalation (operator must see the worse state).
            held["popup_acknowledged"] = False
            reason_parts.append(f"escalated {SEVERITY_NAME[held_sev]}->{alert_state}")
            held_sev = sev

        # 2) PHYSICS-RANKED NAME UPGRADE: if a *new* qualifying classification
        #    outranks the held name, replace the held name (but keep the latch).
        #    A name only upgrades when it clears the confidence/persistence floor.
        if (
            label_int != 0
            and conf_frac >= self.latch_min_confidence
            and self._is_qualifying_persistent(label_int)
        ):
            if self._physics_rank_wins(new_label=label_int, held_label=held["label_int"]):
                held["label_int"] = label_int
                held["label_name"] = label_name
                held["confidence_pct"] = confidence_pct
                held["condition"] = condition
                held["risk"] = risk
                held["action"] = action
                held["causal_chain"] = causal_chain
                held["is_unclassified"] = False
                reason_parts.append(f"name upgraded to '{label_name}' (physics rank)")

        if not reason_parts:
            reason_parts.append("holding latch (no escalation, no rank upgrade)")

        return self._emit_latched("; ".join(reason_parts))

    # =========================================================================
    # CASE B — open a latch if the floor is cleared
    # =========================================================================
    def _maybe_open_latch(
        self, alert_state, sev, label_int, label_name, conf_frac,
        confidence_pct, condition, risk, action, causal_chain, is_unclassified,
    ) -> dict:
        # Determine whether a NAMED fault clears the latch floor.
        named_qualifies = (
            label_int != 0
            and not is_unclassified
            and conf_frac >= self.latch_min_confidence
            and self._is_qualifying_persistent(label_int)
        )

        # Post-verdict suppression: if this exact label was just adjudicated,
        # do not re-latch it during the grace window (the dashboard must visibly
        # return to live). A DIFFERENT label is unaffected and latches normally.
        if (self._suppress_remaining > 0 and self._suppress_label is not None
                and label_int == self._suppress_label):
            return self._passthrough(
                alert_state, label_name, confidence_pct,
                condition, risk, action, causal_chain, is_unclassified,
                reason=f"post-verdict suppression of '{label_name}' "
                       f"({self._suppress_remaining} updates left)")

        if named_qualifies:
            # Latch the named fault as a fresh episode.
            self._open_episode(
                label_int=label_int, label_name=label_name, severity=alert_state,
                confidence_pct=confidence_pct, condition=condition, risk=risk,
                action=action, causal_chain=causal_chain, is_unclassified=False,
            )
            self._cand_label_int = None
            self._cand_streak = 0
            return self._emit_latched(f"latched named fault '{label_name}' at {alert_state}")

        # WARN/DANGER but M7 said normal (or fragile/low-conf): latch the
        # D2 "Unclassified anomaly" episode. NEVER fabricate a class name (#6).
        # The detector is validated; we hold the anomaly, not a guessed label.
        self._open_episode(
            label_int=0, label_name="Unclassified anomaly", severity=alert_state,
            confidence_pct=confidence_pct, condition=condition, risk=risk,
            action=action, causal_chain=None, is_unclassified=True,
        )
        # Keep accumulating candidate evidence so a name can latch on a later
        # window via the active-latch name-upgrade path.
        return self._emit_latched(f"latched UNCLASSIFIED anomaly at {alert_state} "
                                  f"(no qualifying named fault yet)")

    # =========================================================================
    # CASE C — name the WATCH context (no latch)
    # =========================================================================
    def _name_watch(
        self, label_int, label_name, confidence_pct, condition, risk, action,
        causal_chain, is_unclassified, label_map, physics_ctx,
    ) -> dict:
        # Reset latch-candidate streak (WATCH never latches).
        self._cand_label_int = None
        self._cand_streak = 0

        candidates = self._watch_candidates(label_map)

        # If CUSUM-driven gradual wear is in the candidate mix, surface that
        # context explicitly (label 21 path). Otherwise list plausible types.
        if candidates:
            cand_text = ", ".join(candidates)
            watch_condition = (
                "WATCH — sub-threshold drift accumulating. Not a confirmed fault; "
                "the detection stack is tracking possible early-stage development. "
                f"Plausible types to check: {cand_text}. Perform a routine check "
                "(not a maintenance call) and continue monitoring."
            )
        else:
            watch_condition = (
                "WATCH — sub-threshold drift accumulating (CUSUM). No specific "
                "fault type has emerged yet. Perform a routine check and continue "
                "monitoring; this is not yet a maintenance call."
            )

        display_label = "Monitoring — possible early drift"
        return {
            "display_label": display_label,
            "display_severity": "WATCH",
            "display_confidence_pct": float(confidence_pct),
            "condition": watch_condition,
            "risk": (
                "Early-stage drift, unconfirmed. Risk is low now but a developing "
                "fault may not yet be classifiable. Re-check if WATCH persists or "
                "escalates."
            ),
            "action": (
                "Routine inspection of the candidate channels above. Do NOT "
                "schedule maintenance on a WATCH alone — confirm via continued "
                "monitoring or escalation to WARN."
            ),
            "causal_chain": None,
            "is_latched": False,
            "is_unclassified": False,
            "watch_candidates": candidates or None,
            "popup_should_show": False,           # WATCH never pops up (locked)
            "episode_id": None,
            "episode": None,
            "tracker_reason": "watch_context_named",
        }

    # =========================================================================
    # Popup acknowledge — silences popup, latch UNCHANGED (locked behaviour #1)
    # =========================================================================
    async def acknowledge_popup(self) -> dict:
        async with self._lock:
            if self._latched is not None:
                self._latched["popup_acknowledged"] = True
                return {
                    "popup_acknowledged": True,
                    "latch_held": True,
                    "note": ("Popup silenced. The WARN/DANGER state remains LATCHED "
                             "until a verdict is recorded on the Predictions tab."),
                    "episode": dict(self._latched),
                }
            return {
                "popup_acknowledged": True,
                "latch_held": False,
                "note": "No active latch to hold.",
                "episode": None,
            }

    # =========================================================================
    # Verdict — the ONLY thing that releases a WARN/DANGER latch (behaviour #1)
    # =========================================================================
    async def clear_latch(self, verdict: str, note: str = "") -> dict:
        """
        verdict ∈ {confirmed_fixed, action_pending, investigated_no_fault} (or
        the Predictions-tab CORRECT/INCORRECT/UNSURE mapped equivalents).
        Releases the latch and lets the dashboard return to the live state.
        """
        async with self._lock:
            released = self._latched
            self._latched = None
            self._cand_label_int = None
            self._cand_streak = 0
            # Post-verdict suppression: the operator has just adjudicated THIS
            # fault. If the same fault is still physically present, do not
            # immediately re-latch it (that would make the verdict appear to do
            # nothing). Suppress re-latching of this specific label for a grace
            # window of updates. A DIFFERENT fault still latches normally.
            self._suppress_label = released["label_int"] if released else None
            self._suppress_remaining = self.verdict_suppress_updates
            self._last_verdict = {
                "verdict": verdict,
                "note": note,
                "at": datetime.utcnow().isoformat() + "Z",
                "released_episode": released,
            }
            return {
                "latch_released": released is not None,
                "verdict": verdict,
                "released_episode": released,
                "note": ("Latch released. Dashboard returns to live state; re-nag "
                         "stops. System reset for fresh calculation."),
            }

    # =========================================================================
    # reset — used by maintenance acknowledge of L3/L4. Does NOT release a
    # WARN/DANGER latch (only a verdict does). Clears only candidate bookkeeping
    # and WATCH context, matching the locked acknowledge/verdict separation.
    # =========================================================================
    async def reset(self, release_latch: bool = False) -> dict:
        async with self._lock:
            self._cand_label_int = None
            self._cand_streak = 0
            self._recent_labels.clear()
            released = None
            if release_latch:
                released = self._latched
                self._latched = None
            return {
                "reset": True,
                "latch_released": release_latch and released is not None,
                "released_episode": released,
            }

    async def get_state(self) -> dict:
        async with self._lock:
            return {
                "is_latched": self._latched is not None,
                "episode": dict(self._latched) if self._latched else None,
                "candidate_label_int": self._cand_label_int,
                "candidate_streak": self._cand_streak,
                "recent_labels": list(self._recent_labels),
                "n_updates": self._n_updates,
                "last_verdict": self._last_verdict,
                "latch_min_confidence": self.latch_min_confidence,
                "latch_min_persistence": self.latch_min_persistence,
            }

    # =========================================================================
    # Internal helpers
    # =========================================================================
    def _is_qualifying_persistent(self, label_int: int) -> bool:
        """
        Advance the consecutive-window streak for `label_int`. Returns True once
        the streak reaches latch_min_persistence. Changing candidate resets it.
        (Confidence is checked by the caller; this is the persistence half.)
        """
        if self._cand_label_int == label_int:
            self._cand_streak += 1
        else:
            self._cand_label_int = label_int
            self._cand_streak = 1
        return self._cand_streak >= self.latch_min_persistence

    def _physics_rank_wins(self, new_label: int, held_label: int) -> bool:
        """
        Physics-ranked NAME conflict resolution (governs the NAME within an
        episode only; new-episode detection is handled in _handle_active_latch).
          - A Group B compound chain OUTRANKS a held single ONLY IF the chain
            actually CONTAINS that single as a component (the same fault
            progressed into a chain — e.g. seal -> cavitation+seal).
          - An UNRELATED chain (one that does not contain the held single) is a
            DIFFERENT fault, NOT a name upgrade — it must open a new episode, so
            this returns False here.
          - Held D2 unclassified placeholder -> any qualifying name wins.
          - Otherwise keep the held name (no lateral thrash).
        """
        if held_label == new_label:
            return False
        # Held was the D2 unclassified placeholder -> any qualifying name wins.
        if held_label == 0:
            return True
        # New is a chain that CONTAINS the held single -> legitimate progression.
        if new_label in GROUP_B_LABELS and held_label in GROUP_B_COMPONENTS.get(new_label, set()):
            return True
        # Otherwise keep the held name (unrelated faults open a new episode).
        return False

    def _watch_candidates(self, label_map: dict) -> list:
        """
        Build the 'what to watch' candidate set from recent non-normal label
        votes during the WATCH stage. Gradual-wear (21) is always surfaced if
        present since CUSUM is its only detector.
        """
        from collections import Counter
        votes = Counter(l for l in self._recent_labels if l != 0)
        if not votes:
            # Pure CUSUM WATCH with no M7 lean yet -> gradual wear is the prior.
            return [label_map.get(21, "bearing_wear_gradual")] if label_map else []
        ranked = [lab for lab, _ in votes.most_common(self.watch_candidate_max)]
        # Ensure gradual wear shows if it appeared at all.
        if 21 in votes and 21 not in ranked:
            ranked.append(21)
        return [label_map.get(l, f"label_{l}") if label_map else f"label_{l}" for l in ranked]

    def _emit_latched(self, reason: str) -> dict:
        held = self._latched
        popup_should_show = not held["popup_acknowledged"]
        return {
            "display_label": held["label_name"],
            "display_severity": held["severity"],
            "display_confidence_pct": held["confidence_pct"],
            "condition": held["condition"],
            "risk": held["risk"],
            "action": held["action"],
            "causal_chain": held["causal_chain"],
            "is_latched": True,
            "is_unclassified": held["is_unclassified"],
            "watch_candidates": None,
            "popup_should_show": popup_should_show,
            "episode_id": held.get("episode_id"),
            "episode": dict(held),
            "tracker_reason": reason,
        }

    def _passthrough(
        self, alert_state, label_name, confidence_pct, condition, risk, action,
        causal_chain, is_unclassified, reason,
    ) -> dict:
        return {
            "display_label": label_name,
            "display_severity": alert_state,
            "display_confidence_pct": float(confidence_pct),
            "condition": condition,
            "risk": risk,
            "action": action,
            "causal_chain": causal_chain,
            "is_latched": False,
            "is_unclassified": is_unclassified,
            "watch_candidates": None,
            "popup_should_show": False,
            "episode_id": None,
            "episode": None,
            "tracker_reason": reason,
        }