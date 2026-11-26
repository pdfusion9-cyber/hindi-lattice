"""Guarded micro-agent tool execution for NeuroCore.

Model output is untrusted data. Only explicitly registered tools can run.
The first release exposes a bounded arithmetic calculator only.
"""
from __future__ import annotations
import ast
import operator
from pathlib import Path
from typing import Any
from security_agent import SecurityAgent

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def safe_calculate(expression: str) -> float | int:
    if not isinstance(expression, str) or not expression or len(expression) > 256:
        raise ValueError("expression must be a non-empty string of at most 256 characters")
    try:
        root = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic expression") from exc

    def visit(node: ast.AST, depth: int = 0):
        if depth > 20:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
            if abs(value) > 10**12:
                raise ValueError("numeric literal exceeds bound")
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("exponent exceeds bound")
            if isinstance(node.op, (ast.Mult, ast.Pow)) and max(abs(left), abs(right)) > 10**8:
                raise ValueError("operation exceeds numeric bound")
            result = _BIN[type(node.op)](left, right)
            if not isinstance(result, (int, float)) or abs(result) > 10**15:
                raise ValueError("result exceeds numeric bound")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand, depth + 1))
        raise ValueError("only numeric literals and basic arithmetic operators are allowed")

    try:
        return visit(root)
    except (ZeroDivisionError, OverflowError) as exc:
        raise ValueError("invalid arithmetic operation") from exc

class NeuroCoreMicroAgent:
    """Safety-gated tool dispatcher; no shell, filesystem or network tools."""
    def __init__(self, audit_path: str | None = None):
        self.guard = SecurityAgent(audit_path=Path(audit_path) if audit_path else None)
        self.tools = {"calculator": safe_calculate}

    def handle(self, proposal: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(proposal, dict):
            return {"status": "rejected", "reason": "proposal must be an object"}
        name = proposal.get("tool")
        args = proposal.get("arguments")
        if name not in self.tools:
            return {"status": "rejected", "reason": "tool is not allow-listed"}
        if not isinstance(args, dict) or set(args) != {"expression"}:
            return {"status": "rejected", "reason": "invalid tool arguments"}
        expression = args["expression"]
        decision = self.guard.decide(expression, {"trust": "untrusted", "tool": name})
        if decision.action not in ("allow", "sanitize"):
            return {"status": "blocked", "reason": "security policy refused tool input",
                    "action": decision.action}
        try:
            value = self.tools[name](expression)
        except (ValueError, TypeError) as exc:
            return {"status": "rejected", "reason": str(exc)}
        return {"status": "ok", "tool": name, "result": value,
                "security_action": decision.action}
