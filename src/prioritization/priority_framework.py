
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from src.utils.logger import logger




class UserPersona(str, Enum):
    """User personas by trust/impact level."""
    EXECUTIVE         = "executive"
    COMPLIANCE        = "compliance"
    ANALYST           = "analyst"
    DEVELOPER         = "developer"
    GENERAL_USER      = "general_user"


PERSONA_IMPACT_WEIGHTS = {
    UserPersona.EXECUTIVE:    5.0,
    UserPersona.COMPLIANCE:   5.0,
    UserPersona.ANALYST:      3.0,
    UserPersona.DEVELOPER:    2.0,
    UserPersona.GENERAL_USER: 1.0,
}


class TopicSensitivity(str, Enum):
    """Topic sensitivity levels."""
    CRITICAL  = "critical"
    HIGH      = "high"
    MEDIUM    = "medium"
    LOW       = "low"


SENSITIVITY_WEIGHTS = {
    TopicSensitivity.CRITICAL: 5.0,
    TopicSensitivity.HIGH:     3.0,
    TopicSensitivity.MEDIUM:   2.0,
    TopicSensitivity.LOW:      1.0,
}


SENSITIVITY_KEYWORDS = {
    TopicSensitivity.CRITICAL:  [
        "medical", "diagnosis", "drug", "legal", "liability", "contract",
        "financial", "investment", "risk", "safety", "compliance", "regulation",
        "gdpr", "hipaa", "audit", "penalty",
    ],
    TopicSensitivity.HIGH: [
        "strategy", "executive", "budget", "acquisition", "merger", "confidential",
        "proprietary", "sensitive", "board", "policy",
    ],
    TopicSensitivity.MEDIUM: [
        "analysis", "report", "research", "data", "performance", "metrics",
    ],
}


@dataclass
class FailureSignal:

    timestamp:    datetime
    query:        str
    query_hash:   str
    failure_type: str
    persona:      UserPersona
    sensitivity:  TopicSensitivity
    user_id:      str
    priority_score: float = 0.0


@dataclass
class ActionItem:

    rank:           int
    issue:          str
    priority_score: float
    frequency:      int
    affected_users: int
    persona_impact: str
    sensitivity:    str
    suggested_fix:  str
    example_queries: List[str]
    urgency:        str




class PriorityFramework:

    # Fix recommendations per failure type
    FAILURE_FIXES = {
        "hallucination":    "Add groundedness scoring + self-correction loop before answer delivery",
        "citation_missing": "Update system prompt to require citations; validate before returning",
        "partial_answer":   "Increase top_k; improve query expansion; check chunk size",
        "retrieval_gap":    "Add semantic chunking; try HyDE query transformation",
        "off_topic":        "Add query classification; raise similarity threshold",
        "wrong_format":     "Add output schema validation; improve system prompt structure",
        "outdated_info":    "Add document staleness tracking; enable web search fallback",
        "general":          "Review failed queries manually; consider A/B testing configurations",
    }

    def __init__(self, recency_window_days: int = 30) -> None:
        self._signals: List[FailureSignal] = []
        self.recency_window_days = recency_window_days

    def detect_sensitivity(self, text: str) -> TopicSensitivity:
        """Auto-detect topic sensitivity from query text."""
        text_lower = text.lower()
        for level in [TopicSensitivity.CRITICAL, TopicSensitivity.HIGH, TopicSensitivity.MEDIUM]:
            kws = SENSITIVITY_KEYWORDS.get(level, [])
            if any(kw in text_lower for kw in kws):
                return level
        return TopicSensitivity.LOW

    def record_query_failure(
        self,
        query:        str,
        failure_type: str              = "general",
        persona:      UserPersona      = UserPersona.GENERAL_USER,
        sensitivity:  Optional[TopicSensitivity] = None,
        user_id:      str              = "anonymous",
    ) -> float:

        if sensitivity is None:
            sensitivity = self.detect_sensitivity(query)


        persona_w     = PERSONA_IMPACT_WEIGHTS.get(persona, 1.0)
        sensitivity_w = SENSITIVITY_WEIGHTS.get(sensitivity, 1.0)
        priority = round(persona_w * sensitivity_w, 2)

        import hashlib
        query_hash = hashlib.md5(query.lower().strip()[:80].encode()).hexdigest()[:8]

        signal = FailureSignal(
            timestamp     = datetime.utcnow(),
            query         = query[:120],
            query_hash    = query_hash,
            failure_type  = failure_type,
            persona       = persona,
            sensitivity   = sensitivity,
            user_id       = user_id,
            priority_score = priority,
        )
        self._signals.append(signal)

        logger.debug(
            f"[Priority] Recorded failure: type={failure_type} "
            f"persona={persona} sensitivity={sensitivity} score={priority}"
        )
        return priority

    def get_action_plan(
        self,
        top_n:         int = 10,
        lookback_days: Optional[int] = None,
    ) -> List[ActionItem]:

        window = lookback_days or self.recency_window_days
        cutoff = datetime.utcnow() - timedelta(days=window)
        recent_signals = [s for s in self._signals if s.timestamp >= cutoff]

        if not recent_signals:
            logger.info("[Priority] No failures recorded in window")
            return []


        groups: DefaultDict[Tuple[str, str], List[FailureSignal]] = defaultdict(list)
        for s in recent_signals:
            key = (s.failure_type, s.sensitivity.value)
            groups[key].append(s)

        scored_groups = []
        for (failure_type, sensitivity_val), sigs in groups.items():

            total_score = 0.0
            for s in sigs:

                age_days = (datetime.utcnow() - s.timestamp).total_seconds() / 86400
                recency  = max(0.1, 1.0 - age_days / window)
                total_score += s.priority_score * recency

            unique_users = len(set(s.user_id for s in sigs))
            personas     = [s.persona.value for s in sigs]
            dominant_persona = max(set(personas), key=personas.count)
            examples     = list({s.query for s in sigs})[:3]


            sensitivity_enum = TopicSensitivity(sensitivity_val)
            if sensitivity_enum in (TopicSensitivity.CRITICAL, TopicSensitivity.HIGH):
                urgency = "immediate"
            elif len(sigs) >= 10:
                urgency = "this_week"
            else:
                urgency = "this_month"

            scored_groups.append({
                "failure_type":    failure_type,
                "sensitivity":     sensitivity_val,
                "total_score":     round(total_score, 2),
                "frequency":       len(sigs),
                "unique_users":    unique_users,
                "dominant_persona": dominant_persona,
                "examples":        examples,
                "urgency":         urgency,
            })

        scored_groups.sort(key=lambda x: -x["total_score"])


        actions = []
        for i, g in enumerate(scored_groups[:top_n], 1):
            ft     = g["failure_type"]
            issue  = (
                f"{g['frequency']} '{ft}' failures on "
                f"{g['sensitivity']} sensitivity topics, "
                f"affecting {g['unique_users']} user(s)"
            )
            actions.append(ActionItem(
                rank           = i,
                issue          = issue,
                priority_score = g["total_score"],
                frequency      = g["frequency"],
                affected_users = g["unique_users"],
                persona_impact = g["dominant_persona"],
                sensitivity    = g["sensitivity"],
                suggested_fix  = self.FAILURE_FIXES.get(ft, self.FAILURE_FIXES["general"]),
                example_queries = g["examples"],
                urgency        = g["urgency"],
            ))

        logger.info(
            f"[Priority] Action plan: {len(actions)} items "
            f"({len(recent_signals)} failures in {window}d window)"
        )
        return actions

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return summary statistics for dashboard display."""
        if not self._signals:
            return {"total_failures": 0}

        cutoff  = datetime.utcnow() - timedelta(days=self.recency_window_days)
        recent  = [s for s in self._signals if s.timestamp >= cutoff]
        total   = len(recent)

        persona_counts: Dict[str, int] = defaultdict(int)
        sensitivity_counts: Dict[str, int] = defaultdict(int)
        failure_type_counts: Dict[str, int] = defaultdict(int)

        for s in recent:
            persona_counts[s.persona.value]         += 1
            sensitivity_counts[s.sensitivity.value] += 1
            failure_type_counts[s.failure_type]     += 1

        critical_count = sensitivity_counts.get(TopicSensitivity.CRITICAL.value, 0)

        return {
            "total_failures":       total,
            "critical_failures":    critical_count,
            "critical_pct":         round(critical_count / max(total, 1) * 100, 1),
            "by_persona":           dict(persona_counts),
            "by_sensitivity":       dict(sensitivity_counts),
            "by_failure_type":      dict(failure_type_counts),
            "unique_users_affected": len(set(s.user_id for s in recent)),
            "window_days":           self.recency_window_days,
        }

    def print_action_plan(self, top_n: int = 10) -> None:
        """Print a human-readable action plan."""
        actions = self.get_action_plan(top_n)
        print(f"\n{'─'*70}")
        print(f"  PRIORITIZED ACTION PLAN (top {len(actions)})")
        print(f"{'─'*70}")
        for a in actions:
            urgency_emoji = {"immediate": "🔴", "this_week": "🟡", "this_month": "🟢"}.get(a.urgency, "⚪")
            print(f"\n  #{a.rank} [{urgency_emoji} {a.urgency.upper()}] score={a.priority_score:.1f}")
            print(f"     Issue:   {a.issue}")
            print(f"     Persona: {a.persona_impact} | Sensitivity: {a.sensitivity}")
            print(f"     Fix:     {a.suggested_fix}")
            if a.example_queries:
                print(f"     Example: '{a.example_queries[0][:60]}'")
        print(f"{'─'*70}")




_framework: Optional[PriorityFramework] = None

def get_priority_framework() -> PriorityFramework:
    global _framework
    if _framework is None:
        _framework = PriorityFramework()
    return _framework
