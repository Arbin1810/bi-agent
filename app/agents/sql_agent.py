import json
import re
from app.agents.base_agent import BaseAgent
from config.prompts import SQL_GENERATION_PROMPT
from app.data.connectors import db, DataFrameSQLExecutor
from app.utils.sql_safety import is_safe_sql


class SQLAgent(BaseAgent):
    name = "SQLAgent"

    def run(self, state: dict) -> dict:
        self.log("Generating SQL investigation plan")

        # Pick executor
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

        # Run each query, collect results
        results = []
        executed = []
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
                executed.append(q.get("purpose", "query"))
                self.log(f"✓ {q.get('purpose','query')}: {len(df)} rows")
            except Exception as e:
                self.log(f"✗ Query failed ({q.get('purpose')}): {e}")

        # ------------------------------------------------------------------
        # HEADLINE ANCHOR: canonical target-vs-baseline totals.
        # ------------------------------------------------------------------
        headline = self._build_headline(state, executor)
        if headline is not None:
            results.insert(0, headline)
            self.log("✓ HEADLINE: canonical comparison totals computed")

        # ------------------------------------------------------------------
        # DIMENSION DELTAS: per-entity MoM deltas computed in Python.
        # ------------------------------------------------------------------
        deltas = self._build_dimension_deltas(results)
        if deltas:
            total_dims = sum(len(v) for v in deltas.values())
            self.log(f"✓ DELTAS: {total_dims} entity deltas across {len(deltas)} dimensions")

        state["sql"] = queries[0].get("sql") if queries else None
        state["sql_results"] = results
        state["dimension_deltas"] = deltas
        state["sql_explanation"] = gen.get("explanation", "")
        self.log(f"Executed {len(results)}/{len(queries)} queries")
        return state

    # ----------------------------------------------------------------------
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

        sql = f"""
        SELECT strftime('%Y-%m', date) AS month,
               COUNT(*) AS txn_count,
               SUM(revenue) AS total_rev,
               AVG(revenue) AS aov,
               SUM(CASE WHEN order_status = 'Cancelled' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS cancel_rate
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
        except Exception as e:
            self.log(f"Headline query failed: {e}")
            return None

    # ----------------------------------------------------------------------
    def _build_dimension_deltas(self, results: list) -> dict:
        """
        For each dimension query (columns like [entity, 'month', 'total']),
        compute MoM deltas per entity in Python. Sort by most-negative
        absolute_change and cap to top 10 per dimension.
        """
        deltas = {}
        for r in results:
            purpose = r.get("purpose", "")
            if purpose.startswith("HEADLINE"):
                continue
            cols = r.get("columns", [])
            rows = r.get("rows", [])
            if len(cols) < 3 or "month" not in cols or "total" not in cols:
                continue
            dim = cols[0]
            if dim in ("month", "total"):
                continue

            pivot = {}
            for row in rows:
                ent = row.get(dim)
                mo = row.get("month")
                tot = row.get("total")
                if ent is None or mo is None or tot is None:
                    continue
                try:
                    pivot.setdefault(ent, {})[mo] = float(tot)
                except (TypeError, ValueError):
                    continue

            entity_deltas = []
            for ent, months in pivot.items():
                if len(months) < 2:
                    continue
                sorted_m = sorted(months.items())
                base_m, base_v = sorted_m[-2]
                tgt_m, tgt_v = sorted_m[-1]
                if base_v == 0:
                    continue
                entity_deltas.append({
                    "entity": str(ent),
                    "dimension": dim,
                    "baseline_period": base_m,
                    "target_period": tgt_m,
                    "baseline_value": round(base_v, 2),
                    "target_value": round(tgt_v, 2),
                    "absolute_change": round(tgt_v - base_v, 2),
                    "change_pct": round((tgt_v - base_v) / base_v * 100, 2),
                })

            entity_deltas.sort(key=lambda d: d["absolute_change"])
            deltas[dim] = entity_deltas[:10]

        return deltas

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