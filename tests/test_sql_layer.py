"""
Phase 2 tests — SQL layer without any LLM.

Run with:
    pytest tests/test_sql_layer.py -v

Exit criterion for Phase 2:
    All tests pass. This proves the query executor, the safety gate, the
    schema introspection, and the headline anchor all work without invoking
    the language model.
"""

from pathlib import Path

import pandas as pd
import pytest

from app.data.connectors import DataFrameSQLExecutor
from app.utils.sql_safety import is_safe_sql

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "transactions.csv"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2026-07-01", "2026-07-15", "2026-08-01", "2026-08-15",
        ]),
        "region": ["Region_A", "Region_B", "Region_A", "Region_B"],
        "product_id": ["P007", "P001", "P007", "P001"],
        "revenue": [100.0, 200.0, 70.0, 180.0],
        "cost": [75.0, 160.0, 64.4, 144.0],
        "quantity": [1, 2, 1, 2],
        "order_status": ["Completed", "Completed", "Cancelled", "Completed"],
    })


# ----------------------------------------------------------------------
# 1. DataFrameSQLExecutor — basic aggregation
# ----------------------------------------------------------------------
def test_monthly_aggregation(sample_df):
    ex = DataFrameSQLExecutor(sample_df)
    result = ex.execute("""
        SELECT strftime('%Y-%m', date) AS month,
               SUM(revenue) AS total
        FROM transactions
        GROUP BY month
        ORDER BY month
    """)
    assert len(result) == 2
    assert result.iloc[0]["total"] == pytest.approx(300.0)
    assert result.iloc[1]["total"] == pytest.approx(250.0)


def test_alias_prefix_stripped(sample_df):
    """DataFrameSQLExecutor strips `transactions.` prefixes."""
    ex = DataFrameSQLExecutor(sample_df)
    result = ex.execute(
        "SELECT transactions.region FROM transactions LIMIT 1"
    )
    assert len(result) == 1


def test_table_aliases_all_work(sample_df):
    """All four aliases registered by DataFrameSQLExecutor must resolve."""
    ex = DataFrameSQLExecutor(sample_df)
    for alias in ("transactions", "sales_data", "df", "data"):
        result = ex.execute(f"SELECT COUNT(*) AS n FROM {alias}")
        assert result.iloc[0]["n"] == 4, f"alias {alias} failed"


# ----------------------------------------------------------------------
# 2. SQL safety — the reviewer's baseline
# ----------------------------------------------------------------------
def test_safety_rejects_drop():
    assert not is_safe_sql("DROP TABLE transactions")
    assert not is_safe_sql("drop table transactions")
    assert not is_safe_sql("DrOp TaBlE transactions")


def test_safety_rejects_delete():
    assert not is_safe_sql("DELETE FROM transactions WHERE 1=1")


def test_safety_rejects_update_insert():
    assert not is_safe_sql("UPDATE transactions SET revenue = 0")
    assert not is_safe_sql("INSERT INTO transactions VALUES (1)")


# ----------------------------------------------------------------------
# 3. SQL safety — bypasses the old regex missed
# ----------------------------------------------------------------------
def test_safety_blocks_comment_splitting():
    """DR/**/OP is not matched by \\bDROP\\b but is still a DROP."""
    assert not is_safe_sql("DR/**/OP TABLE transactions")
    assert not is_safe_sql("DE--x\nLETE FROM transactions")
    assert not is_safe_sql("PRAG/*x*/MA writable_schema=ON")


def test_safety_blocks_multi_statement():
    """SELECT 1; <something> is always rejected, even if the second statement
    is not explicitly in the blocklist."""
    assert not is_safe_sql("SELECT 1; PRAGMA writable_schema=ON")
    assert not is_safe_sql("SELECT 1; ATTACH DATABASE '/tmp/x.db' AS x")
    assert not is_safe_sql("SELECT 1; SELECT 2")


def test_safety_blocks_attach_and_pragma():
    assert not is_safe_sql("ATTACH DATABASE 'x' AS y")
    assert not is_safe_sql("PRAGMA table_info(transactions)")
    assert not is_safe_sql("VACUUM")


def test_safety_blocks_leading_write():
    """A write as the first token was already caught; this pins the behavior."""
    assert not is_safe_sql("WITH x AS (SELECT 1) DELETE FROM transactions")


# ----------------------------------------------------------------------
# 4. SQL safety — must NOT reject legitimate queries
# ----------------------------------------------------------------------
def test_safety_allows_string_literal_with_forbidden_word():
    """A forbidden word inside a string literal is not a write statement."""
    assert is_safe_sql(
        "SELECT * FROM transactions WHERE product_name LIKE '%delete%'"
    )
    assert is_safe_sql(
        "SELECT * FROM transactions WHERE order_status = 'Cancelled'"
    )
    assert is_safe_sql(
        "SELECT 'DROP TABLE' AS example FROM transactions LIMIT 1"
    )


def test_safety_allows_plain_select():
    assert is_safe_sql("SELECT region, SUM(revenue) FROM transactions GROUP BY region")
    assert is_safe_sql("WITH x AS (SELECT 1) SELECT * FROM x")
    assert is_safe_sql("SELECT 1;")


# ----------------------------------------------------------------------
# 5. Headline anchor query works against the executor
# ----------------------------------------------------------------------
def test_headline_query_returns_two_months(sample_df):
    """
    The headline SQL from SQLAgent._build_headline must be executable as-is
    on the DataFrame executor. This is a smoke test for the anchor query
    specifically — if it breaks, every downstream agent loses its baseline.
    """
    from app.agents.sql_agent import SQLAgent

    agent = SQLAgent()
    state = {
        "intent": {"time_reference": "2026-08"},
        "dataframe": sample_df,
        "schema_context": "",
    }
    ex = DataFrameSQLExecutor(sample_df)
    headline = agent._build_headline(state, ex)

    assert headline is not None, "headline query returned None"
    assert len(headline["rows"]) == 2
    months = {r["month"] for r in headline["rows"]}
    assert months == {"2026-07", "2026-08"}


# ----------------------------------------------------------------------
# 6. Headline anchor on the real, verified dataset
# ----------------------------------------------------------------------
@pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason=f"{CSV_PATH} not found; run scripts/generate_dataset.py first",
)
def test_headline_query_on_real_csv_matches_ground_truth():
    """
    Runs the headline query on data/transactions.csv and asserts the values
    that scripts/verify_scenarios.py already verified against ground truth:

      * Region_A August revenue is ~30% below its July value (per-txn basis,
        A1-only rows — the aggregate drop is larger due to A4 overlap, so
        we only assert the headline totals exist and are positive here).
      * Total August revenue is lower than July.
      * Cancel rate is higher in August than July (driven by A3).

    The strong per-scenario assertions live in scripts/verify_scenarios.py;
    this test is a lighter integration guard that the SQL layer still returns
    sensible headline numbers after the schema fix.
    """
    from app.agents.sql_agent import SQLAgent

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    agent = SQLAgent()
    state = {
        "intent": {"time_reference": "2026-08"},
        "dataframe": df,
        "schema_context": "",
    }
    ex = DataFrameSQLExecutor(df)
    headline = agent._build_headline(state, ex)
    assert headline is not None

    rows = {r["month"]: r for r in headline["rows"]}
    assert "2026-07" in rows and "2026-08" in rows

    jul = rows["2026-07"]
    aug = rows["2026-08"]

    # Sanity — both months have non-zero totals.
    assert jul["total_rev"] > 0
    assert aug["total_rev"] > 0

    # August revenue is lower than July, as the anomalies require.
    assert aug["total_rev"] < jul["total_rev"], (
        f"August revenue {aug['total_rev']:.0f} should be lower than "
        f"July {jul['total_rev']:.0f}"
    )

    # Cancel rate is higher in August (A3).
    assert aug["cancel_rate"] > jul["cancel_rate"], (
        f"August cancel rate {aug['cancel_rate']:.3f} should exceed "
        f"July {jul['cancel_rate']:.3f}"
    )


# ----------------------------------------------------------------------
# 7. Schema introspection — the reviewer's original bug
# ----------------------------------------------------------------------
def test_schema_introspector_includes_transactions_table(tmp_path):
    """
    Regression test for the `hasattr(loaded, "columns")` bug: after the fix,
    build_context() must emit the 'Queryable table `transactions`' line so
    the SQL agent sees a real table name. Before the fix, this line was
    silently absent and the LLM invented table names, causing every query
    to fail with 'no such table'.
    """
    from app.data.schema_introspector import introspector

    # Build a minimal CSV so we don't depend on data/transactions.csv.
    csv_path = tmp_path / "toy.csv"
    pd.DataFrame({
        "date": ["2026-08-01"],
        "region": ["Region_A"],
        "revenue": [100.0],
    }).to_csv(csv_path, index=False)

    ctx = introspector.build_context([{"path": str(csv_path)}])

    assert "Queryable table `transactions`" in ctx, (
        f"schema context is missing the table name line:\n{ctx}"
    )
    assert "region" in ctx
    assert "revenue" in ctx