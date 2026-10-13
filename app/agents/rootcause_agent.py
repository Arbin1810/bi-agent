import json
from app.agents.base_agent import BaseAgent
from config.prompts import ROOT_CAUSE_PROMPT


class RootCauseAgent(BaseAgent):
    name = "RootCauseAgent"

    def run(self, state: dict) -> dict:
        self.log("Analyzing root causes")

        # Only pass the headline + deltas to the LLM. Everything else (the
        # analysis JSON with hallucinated numbers) is deliberately dropped.
        headline = None
        for r in state.get("sql_results", []):
            if r.get("purpose", "").startswith("HEADLINE"):
                headline = r
                break

        deltas = state.get("dimension_deltas") or {}

        prompt = ROOT_CAUSE_PROMPT.format(
            question=state["question"],
            headline=json.dumps(headline.get("rows", []) if headline else [], default=str)[:1000],
            dimension_deltas=json.dumps(deltas, default=str)[:4000],
            anomalies=json.dumps(state.get("anomalies", {}).get("anomalies", []), default=str)[:1500],
        )
        rc = self.llm.complete_json(prompt)

        # ----------------------------------------------------------
        # OVERRIDE: rebuild primary_causes from the deltas directly.
        # contribution_pct = entity's absolute_change / sum of all
        # negative absolute_changes across all dimensions.
        # ----------------------------------------------------------
        flat = []
        for dim, entities in deltas.items():
            for d in entities:
                flat.append(d)

        total_neg = sum(-d["absolute_change"] for d in flat if d["absolute_change"] < 0)

        if flat and total_neg > 0:
            top = sorted(flat, key=lambda d: d["absolute_change"])[:5]
            primary = []
            for d in top:
                contribution = (
                    round(-d["absolute_change"] / total_neg * 100, 2)
                    if d["absolute_change"] < 0 else 0.0
                )
                primary.append({
                    "cause": f"{d['dimension']}={d['entity']} declined",
                    "dimension": d["dimension"],
                    "value": d["entity"],
                    "baseline_value": d["baseline_value"],
                    "target_value": d["target_value"],
                    "absolute_change": d["absolute_change"],
                    "change_pct": d["change_pct"],
                    "contribution_pct": contribution,
                    "evidence": (
                        f"{d['baseline_period']}: {d['baseline_value']:,.2f} → "
                        f"{d['target_period']}: {d['target_value']:,.2f}"
                    ),
                    "confidence": "high" if abs(d["change_pct"]) > 15 else "medium",
                })
            rc["primary_causes"] = primary
            self.log(f"Root-cause override: {len(primary)} causes from deltas")

        rc["known_limitations"] = rc.get("known_limitations") or [
            "Analysis is based on MoM comparison only; YoY not computed.",
            "Contribution percentages are share of total negative absolute change.",
        ]

        # ----------------------------------------------------------
        # SCHEMA ENFORCEMENT: drop any LLM-invented alternative keys
        # (e.g. singular "primary_cause" with a negative contribution_pct).
        # ----------------------------------------------------------
        ALLOWED = {
            "primary_causes",
            "secondary_causes",
            "hypotheses_to_test",
            "known_limitations",
        }
        rc = {k: v for k, v in rc.items() if k in ALLOWED}

        state["root_causes"] = rc
        return state