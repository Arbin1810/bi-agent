import json

from app.agents.base_agent import BaseAgent
from config.prompts import ROOT_CAUSE_PROMPT


class RootCauseAgent(BaseAgent):
    name = "RootCauseAgent"

    def run(self, state: dict) -> dict:
        self.log("Analyzing root causes")

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
        # OVERRIDE (Step 3.2 fixed): rebuild primary_causes from deltas,
        # but compute contribution_pct WITHIN each dimension only.
        #
        # The previous version summed negative absolute_changes across
        # ALL dimensions. That's invalid because dimensions overlap: a
        # single transaction is in one region AND one product AND one
        # branch AND one segment, so summing across dimensions counts
        # the same revenue drop multiple times. Within a single
        # dimension, entities are disjoint (each transaction has exactly
        # one region), so the share is meaningful.
        # ----------------------------------------------------------
        flat = []
        for dim, entities in deltas.items():
            for d in entities:
                flat.append(d)

        # Per-dimension denominators — entities within one dimension are
        # disjoint, so their negative changes can be summed safely.
        dim_total_neg = {}
        for dim, entities in deltas.items():
            neg = sum(-d["absolute_change"] for d in entities if d["absolute_change"] < 0)
            dim_total_neg[dim] = neg

        if flat:
            top = sorted(flat, key=lambda d: d["absolute_change"])[:5]
            primary = []
            for d in top:
                dim = d["dimension"]
                denom = dim_total_neg.get(dim, 0.0)
                # Contribution is share of the *dimension's* total negative
                # change, not the global total. This avoids the cross-
                # dimension double-counting.
                contribution = (
                    round(-d["absolute_change"] / denom * 100, 2)
                    if d["absolute_change"] < 0 and denom > 0 else 0.0
                )
                primary.append({
                    "cause": f"{dim}={d['entity']} declined",
                    "dimension": dim,
                    "value": d["entity"],
                    "baseline_value": d["baseline_value"],
                    "target_value": d["target_value"],
                    "absolute_change": d["absolute_change"],
                    "change_pct": d["change_pct"],
                    "contribution_pct": contribution,
                    "contribution_basis": f"share of {dim} negative change",
                    "note": (
                        f"Dimension: {dim}. Observed change only; "
                        f"not additive with other dimensions."
                    ),
                    "evidence": (
                        f"{d['baseline_period']}: {d['baseline_value']:,.2f} → "
                        f"{d['target_period']}: {d['target_value']:,.2f}"
                    ),
                    "confidence": "high" if abs(d["change_pct"]) > 15 else "medium",
                })
            rc["primary_causes"] = primary
            self.log(f"Root-cause override: {len(primary)} causes from deltas")

        rc["known_limitations"] = rc.get("known_limitations") or []
        # Add the cross-dimension disclaimer to every run, not just when
        # the LLM happens to omit it.
        cross_dim_note = (
            "contribution_pct is measured within each dimension, not across "
            "dimensions. Dimensions overlap (each transaction belongs to one "
            "region, one product, one branch, one segment), so the same "
            "revenue drop appears once in each dimension's list and cannot "
            "be summed to 100% across dimensions."
        )
        if cross_dim_note not in rc["known_limitations"]:
            rc["known_limitations"].append(cross_dim_note)

        ALLOWED = {
            "primary_causes",
            "secondary_causes",
            "hypotheses_to_test",
            "known_limitations",
        }
        rc = {k: v for k, v in rc.items() if k in ALLOWED}

        state["root_causes"] = rc
        return state