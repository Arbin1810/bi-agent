import re
from typing import ClassVar

from app.agents.base_agent import BaseAgent
from app.data.connectors import DataFrameSQLExecutor, db
from app.utils.sql_safety import is_safe_sql
from config.prompts import SQL_GENERATION_PROMPT


class SQLAgent(BaseAgent):
    name = "SQLAgent"

    METRIC_DIMENSIONS: ClassVar[list[str]] = [
        "region", "product_id", "branch", "customer_segment"
    ]

    def run(self, state: dict) -> dict:
        self.log("Generating SQL investigation plan")

        if state.get("dataframe") is not None:
            executor = DataFrameSQLExecutor(state["dataframe"], table_name="transactions")
            state["_executor"] = executor
            dialect = "sqlite"
        elif db.engine is not None:
            state["_executor"] = db
            dialect = db.dialect
        else:
            self.log("No data source")
            return state

        prompt = SQL_GENERATION_PROMPT.format(
            dialect=dialect,
            schema=state["schema_context"][:4000],
            question=state["question"],
        )
        gen = self.llm.complete_json(prompt)

        queries = gen.get("queries", [])
        if not queries:
            self.log("No multi-query plan; falling back to single query")
            sql = (gen.get("sql") or "").strip()
            if sql and is_safe_sql(sql):
                queries = [{"purpose": "single", "sql": sql}]

        if not queries:
            state["sql"] = None
            state["sql_error"] = "No queries generated"
            return state

        results = []
        for q in queries:
            sql = (q.get("sql") or "").strip()
            if not sql:
                continue
            if not is_safe_sql(sql):
                self.log(f"Skipping unsafe SQL: {sql[:100]}")
                continue
            try:
                df = executor.execute(sql)
                results.append({
                    "purpose": q.get("purpose", "query"),
                    "sql": sql,
                    "rows": df.to_dict(orient="records"),
                    "columns": list(df.columns),
                })
                self.log(f"✓ {q.get('purpose','query')}: {len(df)} rows")
            except Exception as e:  # noqa: BLE001
                self.log(f"✗ Query failed ({q.get('purpose')}): {e}")

        headline = self._build_headline(state, executor)
        if headline is not None:
            results.insert(0, headline)
            self.log("✓ HEADLINE: canonical comparison totals computed")

        metric_deltas = self._build_all_metric_deltas(state, executor)
        if metric_deltas:
            total = sum(
                len(ents)
                for metric_table in metric_deltas.values()
                for ents in metric_table.values()
            )
            self.log(
                f"✓ METRIC DELTAS: {total} entity deltas across "
                f"{len(metric_deltas)} metrics × "
                f"{len(next(iter(metric_deltas.values())))} dimensions"
            )

        state["sql"] = queries[0].get("sql") if queries else None
        state["sql_results"] = results
        state["metric_deltas"] = metric_deltas
        state["dimension_deltas"] = metric_deltas.get("revenue", {})
        state["sql_explanation"] = gen.get("explanation", "")
        self.log(f"Executed {len(results)}/{len(queries)} queries")
        return state

    def _build_headline(self, state: dict, executor) -> dict | None:
        intent = state.get("intent") or {}
        time_ref = (intent.get("time_reference") or "").strip()

        target = self._parse_month(time_ref)
        if target is None:
            return None

        y, m = target
        if m == 1:
            base = (y - 1, 12)
        else:
            base = (y, m - 1)

        t_str = f"{y:04d}-{m:02d}"
        b_str = f"{base[0]:04d}-{base[1]:02d}"

        # Delivery 3 (R6): expose completed-order revenue alongside gross
        # revenue. AnalysisAgent picks which KPI to report based on intent.
        # `total_rev` (gross) is unchanged so existing consumers keep working.
        sql = f"""
        SELECT strftime('%Y-%m', date) AS month,
               COUNT(*) AS txn_count,
               SUM(revenue) AS total_rev,
               AVG(revenue) AS aov,
               SUM(CASE WHEN order_status = 'Completed' THEN revenue ELSE 0 END) AS completed_rev,
               SUM(CASE WHEN order_status = 'Completed' THEN 1 ELSE 0 END) AS completed_txn_count,
               AVG(CASE WHEN order_status = 'Completed' THEN revenue END) AS completed_aov,
               SUM(cost) AS total_cost,
               SUM(cost) * 1.0 / NULLIF(SUM(revenue), 0) AS cost_ratio,
               SUM(CASE WHEN order_status = 'Cancelled' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS cancel_rate,
               SUM(quantity) AS total_quantity,
               AVG(quantity) AS avg_quantity
        FROM transactions
        WHERE strftime('%Y-%m', date) IN ('{b_str}', '{t_str}')
        GROUP BY month
        ORDER BY month;
        """.strip()

        try:
            df = executor.execute(sql)
            if df.empty:
                return None
            return {
                "purpose": f"HEADLINE: {b_str} vs {t_str} canonical totals (authoritative comparison)",
                "sql": sql,
                "rows": df.to_dict(orient="records"),
                "columns": list(df.columns),
            }
        except Exception as e:  # noqa: BLE001
            self.log(f"Headline query failed: {e}")
            return None

    def _build_all_metric_deltas(self, state: dict, executor) -> dict:
        intent = state.get("intent") or {}
        time_ref = (intent.get("time_reference") or "").strip()
        target = self._parse_month(time_ref)
        if target is None:
            return {}

        y, m = target
        base = (y - 1, 12) if m == 1 else (y, m - 1)
        t_str = f"{y:04d}-{m:02d}"
        b_str = f"{base[0]:04d}-{base[1]:02d}"

        metric_deltas = {
            "revenue":     {},
            "cost_ratio":  {},
            "cancel_rate": {},
            "quantity":    {},
        }

        for dim in self.METRIC_DIMENSIONS:
            sql = f"""
            SELECT {dim} AS entity,
                   strftime('%Y-%m', date) AS month,
                   SUM(revenue) AS total_rev,
                   SUM(cost) AS total_cost,
                   SUM(quantity) AS total_qty,
                   COUNT(*) AS txn_count,
                   SUM(CASE WHEN order_status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled
            FROM transactions
            WHERE strftime('%Y-%m', date) IN ('{b_str}', '{t_str}')
            GROUP BY {dim}, month
            ORDER BY {dim}, month;
            """.strip()

            try:
                df = executor.execute(sql)
            except Exception as e:  # noqa: BLE001
                self.log(f"✗ Metric delta query failed for {dim}: {e}")
                continue

            if df.empty:
                continue

            pivot = {}
            for row in df.to_dict(orient="records"):
                ent = row.get("entity")
                mo = row.get("month")
                if ent is None or mo is None:
                    continue
                pivot.setdefault(ent, {})[mo] = row

            for ent, months in pivot.items():
                if b_str not in months or t_str not in months:
                    continue
                base_row = months[b_str]
                tgt_row  = months[t_str]

                def _metric(row, metric):
                    rev = float(row.get("total_rev") or 0)
                    cost = float(row.get("total_cost") or 0)
                    qty = float(row.get("total_qty") or 0)
                    cnt = float(row.get("txn_count") or 0)
                    canc = float(row.get("cancelled") or 0)
                    if metric == "revenue":
                        return rev
                    if metric == "cost_ratio":
                        return (cost / rev) if rev else None
                    if metric == "cancel_rate":
                        return (canc / cnt) if cnt else None
                    if metric == "quantity":
                        return (qty / cnt) if cnt else None
                    return None

                for metric, deltas_for_metric in metric_deltas.items():
                    base_v = _metric(base_row, metric)
                    tgt_v  = _metric(tgt_row, metric)
                    if base_v is None or tgt_v is None:
                        continue

                    abs_change = tgt_v - base_v
                    if metric in ("cost_ratio", "cancel_rate"):
                        abs_change_pp = round(abs_change * 100, 2)
                    else:
                        abs_change_pp = round(abs_change, 2)

                    if base_v == 0:
                        pct = None
                    else:
                        pct = round((tgt_v - base_v) / base_v * 100, 2)

                    deltas_for_metric.setdefault(dim, []).append({
                        "entity": str(ent),
                        "dimension": dim,
                        "metric": metric,
                        "baseline_period": b_str,
                        "target_period": t_str,
                        "baseline_value": round(base_v, 4) if metric in ("cost_ratio", "cancel_rate") else round(base_v, 2),
                        "target_value":   round(tgt_v, 4) if metric in ("cost_ratio", "cancel_rate") else round(tgt_v, 2),
                        "absolute_change": abs_change_pp,
                        "change_pct": pct,
                    })

        for per_dim in metric_deltas.values():
            for dim, table in per_dim.items():
                table.sort(key=lambda d: abs(d["absolute_change"]), reverse=True)
                kept = table[:30]
                kept.sort(key=lambda d: d["absolute_change"])
                per_dim[dim] = kept

        return metric_deltas

    @staticmethod
    def _parse_month(text: str):
        if not text:
            return None

        m = re.search(r"(\d{4})-(\d{1,2})", text)
        if m:
            return int(m.group(1)), int(m.group(2))

        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
            "nov": 11, "dec": 12,
        }
        y = re.search(r"(20\d{2})", text)
        for name, num in months.items():
            if re.search(rf"\b{name}\b", text, re.IGNORECASE):
                if y:
                    return int(y.group(1)), num
                return None
        return None