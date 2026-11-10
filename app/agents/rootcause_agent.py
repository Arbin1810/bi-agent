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

        flat = []
        for dim, entities in deltas.items():
            flat.extend(entities)

        dim_stats = {}
        for dim, entities in deltas.items():
            total_neg = sum(-d["absolute_change"] for d in entities if d["absolute_change"] < 0)
            total_pos = sum(d["absolute_change"] for d in entities if d["absolute_change"] > 0)
            dim_stats[dim] = {
                "total_negative": round(total_neg, 2),
                "total_positive": round(total_pos, 2),
                "net": round(total_pos - total_neg, 2),
            }

        gross_negative = sum(s["total_negative"] for s in dim_stats.values())
        positive_offset = sum(s["total_positive"] for s in dim_stats.values())

        if flat:
            top = sorted(flat, key=lambda d: d["absolute_change"])[:5]
            primary = []
            for d in top:
                dim = d["dimension"]
                denom = dim_stats.get(dim, {}).get("total_negative", 0.0)
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
                    "contribution_basis": f"share of {dim} gross negative change",
                    "note": (
                        f"Dimension: {dim}. Observed change only; "
                        f"not additive with other dimensions."
                    ),
                    "evidence": (
                        f"{d['baseline_period']}: {d['baseline_value']:,.2f} → "
                        f"{d['target_period']}: {d['target_value']:,.2f}"
                    ),
                    "confidence": "high" if abs(d["change_pct"] or 0) > 15 else "medium",
                })
            rc["primary_causes"] = primary
            self.log(f"Root-cause override: {len(primary)} causes from deltas")

        rc["dimension_totals"] = dim_stats
        rc["gross_negative"] = round(gross_negative, 2)
        rc["positive_offset"] = round(positive_offset, 2)
        rc["net_revenue_change"] = round(positive_offset - gross_negative, 2)

        rc["known_limitations"] = rc.get("known_limitations") or []
        cross_dim_note = (
            "contribution_pct is measured within each dimension, not across "
            "dimensions. Dimensions overlap (each transaction belongs to one "
            "region, one product, one branch, one segment), so the same "
            "revenue drop appears once in each dimension's list and cannot "
            "be summed to 100% across dimensions."
        )
        if cross_dim_note not in rc["known_limitations"]:
            rc["known_limitations"].append(cross_dim_note)

        revenue_only_note = (
            "The anomaly detector z-scores revenue per entity. A pure margin "
            "collapse (revenue unchanged, cost rising) would not be flagged "
            "unless a separate cost-ratio delta is computed per entity."
        )
        if revenue_only_note not in rc["known_limitations"]:
            rc["known_limitations"].append(revenue_only_note)

        # DELIVERY 4 FIX — plain numbers, no $, no thousands separator, so
        # the report validator's payload extractor can trace every value
        # this string renders into the report. The previous version used
        # ${value:,.2f}, whose "$347,241.06." form failed punctuation and
        # comma normalization on the extractor side.
        gross_net_note = (
            f"Total gross negative change across dimensions: "
            f"{gross_negative:.2f} dollars. "
            f"Positive offsets (entities that grew): "
            f"{positive_offset:.2f} dollars. "
            f"Net revenue change: "
            f"{rc['net_revenue_change']:.2f} dollars. "
            f"Contribution percentages are of gross negative change, not "
            f"of the net decline."
        )
        if gross_net_note not in rc["known_limitations"]:
            rc["known_limitations"].append(gross_net_note)

        ALLOWED = {
            "primary_causes",
            "secondary_causes",
            "hypotheses_to_test",
            "known_limitations",
            "dimension_totals",
            "gross_negative",
            "positive_offset",
            "net_revenue_change",
        }
        rc = {k: v for k, v in rc.items() if k in ALLOWED}

        state["root_causes"] = rc
        return state