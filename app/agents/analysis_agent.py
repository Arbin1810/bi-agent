import json

import pandas as pd

from app.agents.base_agent import BaseAgent
from config.prompts import ANALYSIS_PROMPT


def _pct_change(base, target):
    try:
        b = float(base)
        t = float(target)
        return round((t - b) / b * 100, 2) if b else None
    except (TypeError, ValueError):
        return None


def _decompose_drivers(base_rev, tgt_rev, base_cnt, tgt_cnt, base_aov, tgt_aov):
    """
    Three-term exact decomposition of a revenue change:

        ΔR = Δcount × AOV_base       (volume effect)
           + count_base × ΔAOV       (price effect)
           + Δcount × ΔAOV           (interaction)

    The three terms sum EXACTLY to (tgt_rev - base_rev). Tolerance is
    0.10 because inputs may be pre-rounded by SQL or by upstream
    aggregation (e.g. AOV reported to 2 decimals). At revenue scales
    1e3–1e5, 0.10 accommodates standard rounding while still catching
    real decomposition errors (which would be off by whole dollars or
    more). Production headline uses unrounded SQL aggregates, so the
    residual there is typically < 0.01.
    """
    if base_rev is None or tgt_rev is None:
        return None
    try:
        base_rev = float(base_rev)
        tgt_rev = float(tgt_rev)
        base_cnt = float(base_cnt or 0)
        tgt_cnt = float(tgt_cnt or 0)
        base_aov = float(base_aov or 0)
        tgt_aov = float(tgt_aov or 0)
    except (TypeError, ValueError):
        return None

    delta_rev = tgt_rev - base_rev
    delta_cnt = tgt_cnt - base_cnt
    delta_aov = tgt_aov - base_aov

    volume_effect = delta_cnt * base_aov
    price_effect = base_cnt * delta_aov
    interaction_effect = delta_cnt * delta_aov

    residual = (volume_effect + price_effect + interaction_effect) - delta_rev
    assert abs(residual) < 0.10, (
        f"Driver decomposition does not sum exactly: "
        f"volume={volume_effect:.2f} + price={price_effect:.2f} + "
        f"interaction={interaction_effect:.2f} = "
        f"{volume_effect + price_effect + interaction_effect:.2f}, "
        f"expected {delta_rev:.2f} (residual {residual:.4f})"
    )

    def _pct_of_total(effect):
        if abs(delta_rev) < 0.01:
            return None
        return round(effect / delta_rev * 100, 2)

    effects = [
        ("volume", volume_effect),
        ("price",  price_effect),
        ("interaction", interaction_effect),
    ]
    if abs(delta_rev) < 0.01:
        narrative = "Revenue was essentially flat month over month."
    else:
        sign = 1 if delta_rev > 0 else -1
        aligned = [(name, e) for name, e in effects if (e * sign) > 0]
        offsetting = [(name, e) for name, e in effects if (e * sign) < 0]
        aligned.sort(key=lambda kv: abs(kv[1]), reverse=True)
        offsetting.sort(key=lambda kv: abs(kv[1]), reverse=True)

        parts = []
        if aligned:
            name, e = aligned[0]
            parts.append(f"{name} was the primary driver ({e:+,.2f})")
        if offsetting:
            name, e = offsetting[0]
            parts.append(f"{name} partially offset the change ({e:+,.2f})")
        narrative = "; ".join(parts) + "."

    return {
        "volume_effect": round(volume_effect, 2),
        "volume_effect_pct_of_total": _pct_of_total(volume_effect),
        "price_effect": round(price_effect, 2),
        "price_effect_pct_of_total": _pct_of_total(price_effect),
        "interaction_effect": round(interaction_effect, 2),
        "interaction_effect_pct_of_total": _pct_of_total(interaction_effect),
        "total_change": round(delta_rev, 2),
        "sum_check": round(
            volume_effect + price_effect + interaction_effect, 2
        ),
        "narrative": narrative,
    }


class AnalysisAgent(BaseAgent):
    name = "AnalysisAgent"

    def run(self, state: dict) -> dict:
        self.log("Running analysis")

        sql_results = state.get("sql_results")

        if sql_results:
            compact = []
            for r in sql_results:
                compact.append({
                    "purpose": r["purpose"],
                    "columns": r["columns"],
                    "rows": r["rows"][:20],
                })
            payload = json.dumps(compact, default=str)[:5000]
            prompt = ANALYSIS_PROMPT.format(
                question=state["question"],
                sql=f"{len(sql_results)} queries executed",
                stats=payload,
                sample="(see stats)",
                columns="(see stats)",
            )
            llm_result = self.llm.complete_json(prompt)

            if isinstance(llm_result, dict) and llm_result.get("error") == "unparseable_json":
                self.log("LLM JSON unparseable; building analysis purely from deltas")
                result = {}
                raw_excerpt = llm_result.get("raw_excerpt")
                if raw_excerpt:
                    result["llm_raw_excerpt"] = raw_excerpt[:500]
            else:
                result = llm_result

            headline = next(
                (r for r in sql_results if r.get("purpose", "").startswith("HEADLINE")),
                None,
            )
            if headline and len(headline.get("rows", [])) >= 2:
                rows = sorted(headline["rows"], key=lambda r: r["month"])
                base, target = rows[-2], rows[-1]

                intent_type = (state.get("intent") or {}).get("intent_type") or ""
                # R6: diagnostic → completed (realized) revenue;
                # descriptive → gross revenue; anything else → completed.
                #
                # BUT: only use completed_* if the headline actually
                # contains those fields. If a caller provides a headline
                # without completed_rev (e.g. a fixture in older tests),
                # fall back to gross — otherwise we'd format zeros and
                # produce None percentages.
                wants_completed = intent_type != "descriptive"
                has_completed = (
                    base.get("completed_rev") is not None
                    and target.get("completed_rev") is not None
                )
                use_completed = wants_completed and has_completed

                if use_completed:
                    b = float(base.get("completed_rev") or 0)
                    t = float(target.get("completed_rev") or 0)
                    b_cnt = base.get("completed_txn_count") or 0
                    t_cnt = target.get("completed_txn_count") or 0
                    b_aov = base.get("completed_aov") or 0
                    t_aov = target.get("completed_aov") or 0
                    kpi_label = "completed_revenue"
                else:
                    b = float(base.get("total_rev") or 0)
                    t = float(target.get("total_rev") or 0)
                    b_cnt = base.get("txn_count") or 0
                    t_cnt = target.get("txn_count") or 0
                    b_aov = base.get("aov") or 0
                    t_aov = target.get("aov") or 0
                    kpi_label = "gross_revenue"

                driver = _decompose_drivers(b, t, b_cnt, t_cnt, b_aov, t_aov)

                result["overall_summary"] = {
                    "target_period": target["month"],
                    "baseline_period": base["month"],
                    "revenue_definition": kpi_label,
                    "absolute_change": round(t - b, 2),
                    "total_change_pct": _pct_change(b, t),
                    "baseline_total": round(b, 2),
                    "target_total": round(t, 2),
                    "baseline_txn_count": b_cnt,
                    "target_txn_count": t_cnt,
                    "baseline_aov": b_aov,
                    "target_aov": t_aov,
                    "baseline_cancel_rate": base.get("cancel_rate"),
                    "target_cancel_rate": target.get("cancel_rate"),
                    "gross_baseline_total": round(float(base.get("total_rev") or 0), 2),
                    "gross_target_total": round(float(target.get("total_rev") or 0), 2),
                    "driver_decomposition": driver,
                }
                _pct_display = (
                    f"{result['overall_summary']['total_change_pct']:+.2f}%"
                    if result["overall_summary"]["total_change_pct"] is not None
                    else "n/a"
                )
                self.log(
                    f"Headline override: {target['month']} vs {base['month']} "
                    f"({kpi_label}) = {_pct_display} "
                    f"({result['overall_summary']['absolute_change']:+,.2f})"
                )
                if driver:
                    self.log(f"Driver decomposition: {driver['narrative']}")

            deltas = state.get("dimension_deltas") or {}
            if deltas:
                bd = []
                for dim, entities in deltas.items():
                    if not entities:
                        continue
                    worst = entities[0]
                    bd.append({
                        "dimension": dim,
                        "worst_entity": worst["entity"],
                        "baseline_value": worst["baseline_value"],
                        "target_value": worst["target_value"],
                        "absolute_change": worst["absolute_change"],
                        "change_pct": worst["change_pct"],
                        "evidence": (
                            f"{worst['entity']} {dim} fell from "
                            f"{worst['baseline_value']:,.2f} in {worst['baseline_period']} "
                            f"to {worst['target_value']:,.2f} in {worst['target_period']} "
                            f"({worst['change_pct']:+.2f}%)"
                        ),
                    })
                result["dimension_breakdown"] = bd

            if deltas:
                flat = []
                for dim, entities in deltas.items():
                    flat.extend(entities[:3])
                flat.sort(key=lambda d: d["absolute_change"])

                findings = []
                if result.get("overall_summary"):
                    os_ = result["overall_summary"]
                    pct = os_.get("total_change_pct")
                    pct_str = f"{pct:+.2f}%" if pct is not None else "n/a"
                    findings.append(
                        f"Revenue {os_['target_period']} vs {os_['baseline_period']} "
                        f"({os_['revenue_definition']}): {pct_str} "
                        f"({os_['absolute_change']:+,.2f}), "
                        f"txn_count {os_['baseline_txn_count']} → {os_['target_txn_count']}, "
                        f"AOV {os_['baseline_aov']:.2f} → {os_['target_aov']:.2f}"
                    )
                for d in flat[:4]:
                    findings.append(
                        f"{d['dimension']}={d['entity']}: "
                        f"{d['absolute_change']:+,.2f} "
                        f"({d['change_pct']:+.2f}%) "
                        f"[{d['baseline_period']} {d['baseline_value']:,.2f} → "
                        f"{d['target_period']} {d['target_value']:,.2f}]"
                    )
                result["key_findings"] = findings

            ALLOWED = {
                "overall_summary",
                "key_findings",
                "quantitative_evidence",
                "dimension_breakdown",
                "driver_decomposition",
                "limitations",
                "llm_raw_excerpt",
            }
            result = {k: v for k, v in result.items() if k in ALLOWED}

            state["analysis"] = result

            chart_src = next(
                (r for r in sql_results
                 if not r.get("purpose", "").startswith("HEADLINE")
                 and r.get("rows")),
                None,
            )
            if chart_src is not None:
                state["data"] = pd.DataFrame(chart_src["rows"])
            else:
                state["data"] = state.get("dataframe")

            return state

        # Fallback: single SQL
        df = None
        sql = state.get("sql")
        executor = state.get("_executor")
        if sql and executor:
            try:
                df = executor.execute(sql)
                state["data"] = df
            except Exception as e:   # noqa: BLE001
                self.log(f"SQL failed: {e}")

        if df is None:
            if state.get("dataframe") is not None:
                df = state["dataframe"]
            elif state.get("data") is not None:
                df = state["data"]
        if df is None:
            state["analysis"] = {"key_findings": ["No data"], "quantitative_evidence": []}
            return state

        state["data"] = df
        from app.analysis.statistics import summarize_dataframe
        stats = summarize_dataframe(df)

        prompt = ANALYSIS_PROMPT.format(
            question=state["question"],
            sql=(state.get("sql") or "N/A")[:500],
            stats=str(stats)[:1500],
            sample=str(df.head(10).to_dict(orient="records"))[:1000],
            columns=str(list(df.columns))[:500],
        )
        state["analysis"] = self.llm.complete_json(prompt)
        state["stats"] = stats
        return state