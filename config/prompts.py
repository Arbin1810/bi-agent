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

SQL_GENERATION_PROMPT = """You are an expert SQL analyst. Generate a valid SQL query to answer the user's question.

Dialect: {dialect}
Schema:
{schema}

User question: {question}
Plan context: {plan_context}

Rules:
- Only SELECT statements. Never modify data.
- Use proper JOINs and aliases.
- Limit results to 10000 rows unless aggregation.
- Return strict JSON: {{"sql": "...", "explanation": "..."}}
"""

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

ANALYSIS_PROMPT = """You are a senior business analyst. Analyze the query results.

User question: {question}
Query executed: {sql}
Result summary (statistics): {stats}
Sample rows: {sample}
Columns: {columns}

Produce structured analysis in JSON:
{{
  "key_findings": ["...", "..."],
  "quantitative_evidence": [{{"metric": "...", "value": "...", "change": "..."}}],
  "patterns": ["..."],
  "comparisons": ["..."],
  "data_quality_notes": ["..."]
}}
"""

ANOMALY_PROMPT = """You are an anomaly detection expert.

User question: {question}
Time series / grouped data summary: {summary}
Statistics: {stats}

Identify anomalies with statistical reasoning (z-score, IQR, deviation from trend).

Return JSON:
{{
  "anomalies": [
    {{"entity": "...", "period": "...", "value": ..., "expected": ..., "severity": "high/medium/low", "reason": "..."}}
  ],
  "method": "z-score/IQR/trend-deviation"
}}
"""

ROOT_CAUSE_PROMPT = """You are a root cause analysis expert using the "5 Whys" and driver decomposition.

User question: {question}
Findings: {findings}
Anomalies: {anomalies}
Additional data: {additional}

Identify the most likely contributing factors ranked by evidence strength.

Return JSON:
{{
  "primary_causes": [
    {{"cause": "...", "evidence": "...", "confidence": "high/medium/low", "contribution_pct": number}}
  ],
  "secondary_causes": [...],
  "hypotheses_to_test": [...]
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

Question: {question}
Findings: {findings}
Root causes: {root_causes}
Anomalies: {anomalies}
Charts generated: {charts}
Confidence: {confidence}

Format (Markdown):

# [Title]

**Period:** ...

## Overall Finding
...

## Main Contributing Factors
1. ...
2. ...
3. ...

## Evidence
- ...

## Anomalies Detected
- ...

## Recommended Investigation
- ...

## Confidence
High/Medium/Low — reason
"""