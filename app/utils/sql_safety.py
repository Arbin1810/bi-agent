"""
SQL safety gate.

Replaces the previous `FORBIDDEN.search(sql)` approach, which had four real
holes that a Phase 2 test suite would surface:

  1. Comment-splitting bypass: `DR/**/OP TABLE ...` never matched `\\bDROP\\b`.
  2. Multi-statement injection: `SELECT 1; PRAGMA writable_schema=ON` passed
     because the leading-keyword regex only checked the first statement and
     `PRAGMA`/`ATTACH` were not in the blocklist.
  3. False positives on string literals: `WHERE product_name LIKE '%delete%'`
     was rejected even though it's a legitimate SELECT.
  4. Missing escalation primitives: `ATTACH`, `PRAGMA`, `VACUUM`, and direct
     reads of `sqlite_master` / `sqlite_schema` were not blocked.

The new implementation strips comments and string literals first, enforces a
single statement, then scans for forbidden tokens. It's deliberately
conservative: the SQL agent should never need to run two statements in one
call, and never needs DDL/DML.
"""

import re

# Match SQL comments and single/double-quoted string literals so we can
# strip them before the safety scan. This defeats both the comment-splitting
# bypass and the "forbidden word inside a string" false positive.
_COMMENT_OR_STRING = re.compile(
    r"--[^\n]*"           # line comment
    r"|/\*.*?\*/"         # block comment (non-greedy, DOTALL)
    r"|'(?:''|[^'])*'"    # single-quoted string, '' is the SQL escape
    r'|"(?:[^"])*"',      # double-quoted string / identifier
    re.DOTALL,
)

FORBIDDEN = re.compile(
    r"\b("
    r"DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"EXEC|EXECUTE|MERGE|REPLACE|"
    r"ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|"
    r"LOAD_EXTENSION|WRITEFILE|READFILE|"
    r"sqlite_master|sqlite_schema|sqlite_temp_master"
    r")\b",
    re.IGNORECASE,
)


def is_safe_sql(sql: str) -> bool:
    if not sql or not sql.strip():
        return False

    # 1. Strip comments and string literals before any other check.
    #    The resulting string is what we scan for "does this look like a
    #    write" — string contents and comments can't smuggle a keyword past.
    stripped = _COMMENT_OR_STRING.sub("", sql).strip()
    if not stripped:
        return False

    # 2. Reject multi-statement input. A single trailing semicolon is fine.
    #    Anything else ("...; ...") is an injection attempt.
    without_trailing_semi = stripped.rstrip(";").rstrip()
    if ";" in without_trailing_semi:
        return False

    # 3. First non-whitespace keyword must be SELECT or WITH.
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        return False

    # 4. No forbidden tokens anywhere in the stripped form.
    if FORBIDDEN.search(stripped):
        return False

    return True