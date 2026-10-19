INTENT_PROMPT = """You are an intent classification agent for a Business Intelligence system.

Classify the user's question into ONE of these categories:
- descriptive: "what happened" (trends, summaries, counts)
- diagnostic: "why did it happen" (root cause, drivers)
- predictive: "what will happen" (forecast)
- prescriptive: "what should we do" (recommendation)
- exploratory: open-ended data exploration

Also extract:
- business_domain (loan, sales, customer, operations, finance, marketing, general)
- time_reference (e.g., "August", "this month", "last quarter", null)
- entities (branches, products, segments mentioned)

User question: {question}

Return strict JSON:
{{
  "intent_type": "...",
  "business_domain": "...",
  "time_reference": "...",
  "entities": [...],
  "requires_multi_step": true/false
}}
"""

PLANNER_PROMPT = """You are a planning agent for a Business Intelligence system.

Given the user question, intent, and available data schema, produce a step-by-step analysis plan.

User question: {question}
Intent: {intent}
Business domain: {domain}
Time reference: {time_reference}

Available data sources (tables, columns, descriptions):
{schema_context}

Produce a plan as JSON:
{{
  "steps": [
    {{"step": 1, "action": "identify_data", "description": "..."}},
    {{"step": 2, "action": "query_data", "description": "...", "tables": [...]}},
    {{"step": 3, "action": "statistical_analysis", "description": "..."}},
    {{"step": 4, "action": "trend_analysis", "description": "..."}},
    {{"step": 5, "action": "anomaly_detection", "description": "..."}},
    {{"step": 6, "action": "root_cause", "description": "..."}}
  ],
  "expected_outputs": ["...", "..."],
  "success_criteria": "..."
}}
"""

SQL_GENERATION_PROMPT = """You are an expert SQL analyst investigating a KPI change.

Dialect: {dialect}
Table name: `transactions`
Schema:
{schema}

User question: {question}

STRICT RULES:
- Output a JSON object with TWO keys: "queries" (a list) and "explanation".
- Each query object has: "purpose" (short string) and "sql" (SELECT statement).
- Generate 3-5 queries that decompose the change across ALL relevant dimensions.
- Always include a comparison period (the prior month or same month last year).
- Use these exact SQLite date functions:
  * Month filter: strftime('%Y-%m', date) = '2026-08'
  * Group by month: strftime('%Y-%m', date) AS month

Decomposition checklist — issue one query for EACH of these dimensions:
1. Overall by month (total revenue, transaction count, AOV, cancellation rate)
2. By region (which regions drove the change)
3. By product_id (which products drove the change)
4. By branch (which branches drove the change)
5. By customer_segment (which segments drove the change)
6. By order_status (cancellation/return rate change)

Example output:
{{
  "queries": [
    {{"purpose": "Overall monthly trend",
      "sql": "SELECT strftime('%Y-%m', date) AS month, COUNT(*) AS txn_count, SUM(revenue) AS total_rev, AVG(revenue) AS aov FROM transactions GROUP BY month ORDER BY month;"}},
    {{"purpose": "Revenue by region for Jul/Aug 2026",
      "sql": "SELECT region, strftime('%Y-%m', date) AS month, SUM(revenue) AS total FROM transactions WHERE strftime('%Y-%m', date) IN ('2026-07', '2026-08') GROUP BY region, month ORDER BY region, month;"}}
  ],
  "explanation": "Multi-dimensional decomposition of August 2026 revenue decline"
}}

Now output your JSON:"""

SQL_VALIDATION_PROMPT = """You are a SQL safety validator.

Validate this SQL query for safety and correctness:
Dialect: {dialect}
Schema: {schema}
Query: {sql}

Check:
1. Is it a read-only query (SELECT only)?
2. Are tables/columns valid per schema?
3. Any SQL injection risks?
4. Any dangerous operations (DROP, DELETE, UPDATE, INSERT, ALTER)?

Return strict JSON:
{{"valid": true/false, "issues": [...], "corrected_sql": "..." or null}}
"""

ANALYSIS_PROMPT = """You are a senior business analyst investigating a KPI change.

User question: {question}
SQL context: {sql}
Results payload (JSON list of {{purpose, columns, rows}}): {stats}
Sample rows: {sample}
Columns: {columns}

CRITICAL INSTRUCTIONS:
0. The FIRST result has a purpose starting with "HEADLINE" and exactly TWO
   rows: baseline month and target month. Those two rows are the authoritative
   comparison. Read them for overall_summary.
1. DO NOT invent numbers. Every number you emit must appear verbatim in the
   payload above. If a value isn't in the payload, use null.
2. Compare the target period to the immediately preceding month (MoM).
3. For each dimension present in the payload (region, product_id, branch,
   customer_segment, order_status), name the worst-performing entity and
   report its absolute and percent change from the raw rows.
4. Decompose the total change into:
   - transaction_count change (volume effect)
   - average_order_value change (price effect)
   - cancellation rate change (quality effect)
5. If the target month's transaction_count is HIGHER than the baseline,
   the decline is NOT volume-driven — say so and point at AOV.

STRICT OUTPUT SCHEMA:
- Return ONLY the keys shown below. Do NOT add alternate keys like
  "Overall Summary", "Key Findings", "DimensionBreakdown", or
  "Driver Decomposition".
- All keys must be exactly the lowercase snake_case names shown.

Return strict JSON:
{{
  "overall_summary": {{"target_period": "...", "baseline_period": "...",
                       "total_change_pct": ..., "absolute_change": ...,
                       "baseline_total": ..., "target_total": ...}},
  "key_findings": ["..."],
  "quantitative_evidence": [
    {{"metric": "...", "target": ..., "baseline": ..., "change_pct": ...}}
  ],
  "dimension_breakdown": [
    {{"dimension": "region", "worst_entity": "Region_A",
      "change_pct": -23.1, "contribution_pct": ..., "evidence": "..."}}
  ],
  "driver_decomposition": {{
    "volume_effect_pct": ...,
    "price_effect_pct": ...,
    "cancellation_effect_pct": ...
  }},
  "limitations": ["..."]
}}

Keep the response under 900 words. Prioritize top 4 findings.
"""

ANOMALY_PROMPT = """You are an anomaly detection expert.

User question: {question}
Data summary: {summary}
Basic statistics: {stats}

Consider these anomaly types:
- Point anomaly (a single high/low value)
- Contextual anomaly (normal in general, abnormal for that period/group)
- Collective anomaly (a group of values jointly abnormal)
- Structural break (a change in trend/level between periods)

Look especially for deviations ≥ 2 standard deviations from a baseline,
and for subgroups whose behavior diverges from the overall trend.

Return strict JSON:
{{
  "anomalies": [
    {{
      "entity": "Region_A / P007 / Branch_D1 / Occasional / overall",
      "period": "2026-08",
      "value": ...,
      "expected": ...,
      "deviation_pct": ...,
      "severity": "high|medium|low",
      "reason": "...",
      "anomaly_type": "point|contextual|collective|structural_break"
    }}
  ],
  "method": "z-score + IQR + month-over-month comparison",
  "baseline_period": "...",
  "notes": ["..."]
}}
"""

ROOT_CAUSE_PROMPT = """You are a root cause analysis expert.

User question: {question}

Headline comparison rows (authoritative totals):
{headline}

Precomputed dimension deltas — every entity's MoM change, computed directly
from SQL. These are the ONLY numbers you may cite:
{dimension_deltas}

Anomalies detected (already computed):
{anomalies}

Rules:
1. Rank causes by absolute_change. The most negative absolute_change is your
   strongest candidate.
2. Every cause MUST reference an entity that appears in the dimension deltas.
   If a name is not in the deltas, you cannot claim it.
3. Cite exact numbers from the deltas in "evidence". Do NOT round differently,
   do NOT invent dollar amounts, do NOT rename entities.
4. contribution_pct = entity absolute_change / total negative absolute_change.
   Sum of contribution_pct across primary_causes should be ≤ 100.
5. Anything you cannot support from the deltas goes in "hypotheses_to_test",
   not "primary_causes".

Return strict JSON:
{{
  "primary_causes": [
    {{"cause": "...", "dimension": "...",
      "value": "...", "baseline_value": ..., "target_value": ...,
      "absolute_change": ..., "change_pct": ...,
      "contribution_pct": ..., "evidence": "...",
      "confidence": "high|medium|low"}}
  ],
  "secondary_causes": [...],
  "hypotheses_to_test": [...],
  "known_limitations": [...]
}}
"""

CRITIC_PROMPT = """You are a critic agent. Review the analysis for quality and completeness.

User question: {question}
Findings: {findings}
Root causes: {root_causes}
Plan: {plan}

Evaluate:
- Did the analysis answer the question?
- Are there gaps?
- Weak evidence?
- Missing dimensions?

Return JSON:
{{
  "is_complete": true/false,
  "gaps": [...],
  "weak_evidence": [...],
  "additional_analysis_needed": [...],
  "confidence": "high/medium/low"
}}
"""

REPORT_PROMPT = """You are a management report writer. Produce a concise executive report.

User question: {question}

CRITICAL RULE 0 — MANDATORY SCENARIO DISCLAIMERS:
Every scenario in the list below MUST appear as a bullet in the
"Hypotheses to Investigate" section, verbatim. Do not paraphrase, do not
omit, do not reorder. If the list says "(none)", do not add any.

MANDATORY DISCLAIMERS:
{mandatory_disclaimers}

VERIFIED PAYLOAD (every number you write MUST appear verbatim in this
JSON — do not invent, round differently, or paraphrase):

{verified_payload}

Charts generated: {charts}
Analyst confidence: {confidence}

CRITICAL RULES:

1. Every numeric claim in the report MUST be quoted verbatim from the
   payload. Do NOT perform arithmetic. Do NOT subtract, divide, or
   compute percentages from payload values. Every percentage you might
   want is ALREADY in the payload as a `*_change_pct` field. Quote it.

2. You MUST use this exact four-section structure, in this order:

   ## Observed Changes
   Entity-level deltas from payload["deltas"]. One bullet per top-3
   negative entity per dimension. For each bullet, quote these fields
   from the entity's OWN entry in payload["deltas"][dimension][i]
   VERBATIM:
     - entity name
     - baseline_period and target_period (month strings)
     - baseline_value and target_value (dollar amounts FOR THAT ENTITY)
     - change_pct
     - absolute_change (dollar amount FOR THAT ENTITY)

   CRITICAL: Do NOT use payload["headline"] values here. The headline
   baseline_total and target_total belong to the WHOLE DATASET, not to
   any single entity. If you write Region_G's revenue as the headline
   total, you are factually wrong. Every dollar amount in an Observed
   Changes bullet MUST come from payload["deltas"][dimension][i], never
   from payload["headline"].

   Do NOT cite "contribution" or "contribution_pct" in this section.
   Those belong to the root-cause primary causes, not to observed deltas.

   Format example: "**Region_G:** from baseline_value (baseline_period)
   to target_value (target_period), change_pct% change".

   ## Measured Drivers
   The FIRST bullet MUST state the overall revenue change. Quote these
   four fields from payload["headline"] VERBATIM in one sentence:
     - total_change_pct
     - absolute_change
     - baseline_total
     - target_total
   Example format: "Total revenue changed by X% (Y, from A to B)".
   This sentence must appear BEFORE any per-driver bullets.

   Then add these three bullets, quoting payload["headline"] VERBATIM:
     - aov_change_pct (as "AOV changed by X%")
     - txn_change_pct (as "Transaction count changed by X%")
     - cancel_rate_change_pct (as "Cancellation rate changed by X%")

   If a field is null, state the raw baseline and target values without
   computing a percent. Finally, state whether volume or price drove the
   change, based on the relative magnitudes of txn_change_pct vs
   aov_change_pct.

   ## Hypotheses to Investigate
   Copy the MANDATORY DISCLAIMERS bullets above verbatim as your first
   entries in this section. Then you may add any additional hypotheses
   you can support from the payload. Do NOT include numbers in this
   section unless they already appear under Observed Changes or
   Measured Drivers.

   ## Known Limitations
   Copy payload["known_limitations"] as bullets.

3. Do NOT cite raw JSON key names in prose. Write in sentences.
4. Do NOT add sections beyond the four named above, plus a title and a
   period line.
5. If payload["anomalies"] is empty, say so explicitly under Observed
   Changes, and do NOT claim "no anomalies" — say "no revenue-scope
   anomalies were flagged" to be honest about the detector's scope.

Format (Markdown):

# Revenue Analysis — {confidence} confidence

## Observed Changes
- ...

## Measured Drivers
- ...

## Hypotheses to Investigate
- ...

## Known Limitations
- ...
"""