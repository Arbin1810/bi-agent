import re

FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|EXEC|EXECUTE|MERGE|REPLACE)\b",
    re.IGNORECASE,
)

def is_safe_sql(sql: str) -> bool:
    if not sql or not sql.strip():
        return False
    if FORBIDDEN.search(sql):
        return False
    if not re.match(r"^\s*(WITH|SELECT)\b", sql, re.IGNORECASE):
        return False
    return True