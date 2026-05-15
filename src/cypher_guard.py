"""Safety checks for LLM-generated Cypher."""

from __future__ import annotations

import re


class CypherSafetyError(ValueError):
    """Raised when a Cypher query is not safe for read-only QA execution."""


ALLOWED_START_TOKENS = {"MATCH", "OPTIONAL", "WITH"}
FORBIDDEN_TOKENS = {
    "ALTER",
    "CALL",
    "COMMIT",
    "CREATE",
    "DELETE",
    "DENY",
    "DETACH",
    "DROP",
    "FOREACH",
    "GRANT",
    "LOAD",
    "MERGE",
    "REMOVE",
    "REVOKE",
    "ROLLBACK",
    "SET",
    "START",
    "UNWIND",
    "USE",
}
FORBIDDEN_SUBSTRINGS = ("APOC", "DBMS", "DB.", "GDS.")


def mask_string_literals(cypher: str) -> str:
    """Replace quoted strings with blanks so keyword checks ignore values."""
    return re.sub(r"""('([^'\\]|\\.)*'|"([^"\\]|\\.)*")""", "''", cypher)


def normalize_for_checks(cypher: str) -> str:
    masked = mask_string_literals(cypher)
    masked = re.sub(r"//.*?$", " ", masked, flags=re.M)
    masked = re.sub(r"/\*.*?\*/", " ", masked, flags=re.S)
    return masked.upper()


def validate_read_only_cypher(cypher: str) -> str:
    cleaned = str(cypher or "").strip()
    if not cleaned:
        raise CypherSafetyError("Cypher query is empty")
    if ";" in cleaned.rstrip(";"):
        raise CypherSafetyError("Multiple Cypher statements are not allowed")
    cleaned = cleaned.rstrip(";").strip()
    if re.search(r"(^|\s)//|/\*", mask_string_literals(cleaned)):
        raise CypherSafetyError("Cypher comments are not allowed")
    normalized = normalize_for_checks(cleaned)
    first = re.match(r"\s*([A-Z]+)", normalized)
    if not first or first.group(1) not in ALLOWED_START_TOKENS:
        raise CypherSafetyError("Cypher must start with MATCH, OPTIONAL MATCH, or WITH")
    tokens = set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", normalized))
    forbidden = sorted(tokens & FORBIDDEN_TOKENS)
    if forbidden:
        raise CypherSafetyError(f"Forbidden Cypher keyword: {forbidden[0]}")
    if "RETURN" not in tokens:
        raise CypherSafetyError("Cypher must return records")
    for substring in FORBIDDEN_SUBSTRINGS:
        if substring in normalized:
            raise CypherSafetyError(f"Forbidden Cypher function or namespace: {substring}")
    return cleaned


def ensure_limit(cypher: str, *, limit: int = 50) -> str:
    validated = validate_read_only_cypher(cypher)
    normalized = normalize_for_checks(validated)
    if re.search(r"\bLIMIT\s+\d+\b", normalized):
        return validated
    return f"{validated}\nLIMIT {limit}"
