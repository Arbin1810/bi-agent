import json

import pandas as pd

from app.agents.base_agent import BaseAgent
from config.prompts import ANALYSIS_PROMPT


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

            # ----------------------------------------------------------
            # If the LLM produced an unparseable JSON stub, drop it and
            # build a clean dict. Our overrides below fully populate it.
            # ----------------------------------------------------------
            if isinstance(llm_result, dict) and llm_result.get("error") == "unparseable_json":
                self.log("LLM JSON unparseable; building analysis purely from deltas")
                result = {}
                raw_excerpt = llm_result.get("raw_excerpt")
                if raw_excerpt:
                    result["llm_raw_excerpt"] = raw_excerpt[:500]
            else:
                result = llm_result

            # ----------------------------------------------------------
            # OVERRIDE 1: canonical headline numbers from Python.
            # ----------------------------------------------------------
            headline = next(
                (r for r in sql_results if r.get("purpose", "").startswith("HEADLINE")),
                None,
            )
            if headline and len(headline.get("rows", [])) >= 2:
                rows = sorted(headline["rows"], key=lambda r: r["month"])
                base, target = rows[-2], rows[-1]
                b = float(base.get("total_rev") or 0)
                t = float(target.get("total_rev") or 0)
                result["overall_summary"] = {
                    "target_period": target["month"],
                    "baseline_period": base["month"],
                    "absolute_change": round(t - b, 2),
                    "total_change_pct": round((t - b) / b * 100, 2) if b else None,
                    "baseline_total": round(b, 2),
                    "target_total": round(t, 2),
                    "baseline_txn_count": base.get("txn_count"),
                    "target_txn_count": target.get("txn_count"),
                    "baseline_aov": base.get("aov"),
                    "target_aov": target.get("aov"),
                    "baseline_cancel_rate": base.get("cancel_rate"),
                    "target_cancel_rate": target.get("cancel_rate"),
                }
                self.log(
                    f"Headline override: {target['month']} vs {base['month']} "
                    f"= {result['overall_summary']['total_change_pct']}% "
                    f"({result['overall_summary']['absolute_change']:+,.2f})"
                )

            # ----------------------------------------------------------
            # OVERRIDE 2: dimension_breakdown from precomputed deltas.
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # OVERRIDE 3: key_findings from the same deltas.
            # ----------------------------------------------------------
            if deltas:
                flat = []
                for dim, entities in deltas.items():
                    for d in entities[:3]:
                        flat.append(d)
                flat.sort(key=lambda d: d["absolute_change"])

                findings = []
                if result.get("overall_summary"):
                    os_ = result["overall_summary"]
                    findings.append(
                        f"Revenue {os_['target_period']} vs {os_['baseline_period']}: "
                        f"{os_['total_change_pct']:+.2f}% "
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

            # ----------------------------------------------------------
            # SCHEMA ENFORCEMENT: keep only keys we control.
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # FIX (Step 3.1): do NOT set state["data"] = None here.
            # ChartAgent reads state["data"] and bails out if it's None,
            # so multi-query runs produced zero charts. Populate it from
            # the first NON-headline result (headline is only 2 rows —
            # useless for charting), falling back to the original frame.
            # ----------------------------------------------------------
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

        # Fallback: single SQL (unchanged)
        df = None
        sql = state.get("sql")
        executor = state.get("_executor")
        if sql and executor:
            try:
                df = executor.execute(sql)
                state["data"] = df
            except Exception as e:
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