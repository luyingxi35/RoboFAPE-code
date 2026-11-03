# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
"""Safe AST evaluator and one-dimensional metric progress matching."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np

from .relations import RELATION_FUNCTIONS


MAX_EXPRESSION_LENGTH = 512
MAX_AST_NODES = 100


class UnsafeExpressionError(ValueError):
    pass


def expression_variable_references(expression: str) -> set[str]:
    """Return variable-like identifiers, preserving forbidden dotted source paths."""

    tree = ast.parse(expression.strip(), mode="eval")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    references: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not isinstance(parents.get(node), ast.Attribute):
            dotted = _dotted_name(node)
            if dotted is not None:
                references.add(dotted)
        elif isinstance(node, ast.Name):
            parent = parents.get(node)
            if isinstance(parent, ast.Attribute) and parent.value is node:
                continue
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            references.add(node.id)
    return references


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class SafeExpression:
    """Compile a small arithmetic/relation expression without using eval()."""

    def __init__(self, expression: str, allowed_variables: set[str], allow_condition: bool = False):
        expression = expression.strip()
        if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
            raise UnsafeExpressionError("Expression is empty or too long.")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise UnsafeExpressionError(f"Invalid expression syntax: {exc.msg}.") from exc
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise UnsafeExpressionError("Expression is too complex.")
        self.expression = expression
        self.allowed_variables = set(allowed_variables)
        self.allow_condition = allow_condition
        self._validate(tree.body)
        self._root = tree.body

    def evaluate(self, context: dict[str, Any]) -> float | bool:
        missing = self.allowed_variables.intersection(self._referenced_names()) - set(context)
        if missing:
            raise KeyError(f"Metric context is missing variables: {sorted(missing)}.")
        value = self._evaluate(self._root, context)
        if self.allow_condition:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"End condition must return bool, got {type(value).__name__}.")
            return bool(value)
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Metric result must be finite.")
        return value

    def _referenced_names(self) -> set[str]:
        return {
            node.id
            for node in ast.walk(self._root)
            if isinstance(node, ast.Name) and node.id not in RELATION_FUNCTIONS
        }

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise UnsafeExpressionError("Only numeric and boolean constants are allowed.")
            return
        if isinstance(node, ast.Name):
            if node.id not in self.allowed_variables:
                raise UnsafeExpressionError(f"Unknown variable {node.id!r}.")
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in RELATION_FUNCTIONS:
                raise UnsafeExpressionError("Only registered relation calls are allowed.")
            if node.keywords:
                raise UnsafeExpressionError("Keyword arguments are not allowed.")
            for arg in node.args:
                self._validate(arg)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self._validate(node.left)
            self._validate(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            if isinstance(node.op, ast.Not) and not self.allow_condition:
                raise UnsafeExpressionError("Boolean operators are only allowed in end conditions.")
            self._validate(node.operand)
            return
        if self.allow_condition and isinstance(node, ast.Compare):
            if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                raise UnsafeExpressionError("End conditions support one simple comparison.")
            self._validate(node.left)
            self._validate(node.comparators[0])
            return
        if self.allow_condition and isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                self._validate(value)
            return
        raise UnsafeExpressionError(f"Unsupported syntax: {type(node).__name__}.")

    def _evaluate(self, node: ast.AST, context: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return context[node.id]
        if isinstance(node, ast.Call):
            function = RELATION_FUNCTIONS[node.func.id]
            return function(*(self._evaluate(arg, context) for arg in node.args))
        if isinstance(node, ast.BinOp):
            left, right = self._evaluate(node.left, context), self._evaluate(node.right, context)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if abs(float(right)) < 1e-12:
                    raise ZeroDivisionError("Metric expression divided by zero.")
                return left / right
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand, context)
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.USub):
                return -value
            return not bool(value)
        if isinstance(node, ast.Compare):
            left = self._evaluate(node.left, context)
            right = self._evaluate(node.comparators[0], context)
            op = node.ops[0]
            if isinstance(op, ast.Lt):
                return left < right
            if isinstance(op, ast.LtE):
                return left <= right
            if isinstance(op, ast.Gt):
                return left > right
            if isinstance(op, ast.GtE):
                return left >= right
            if isinstance(op, ast.Eq):
                return left == right
            return left != right
        if isinstance(node, ast.BoolOp):
            values = [bool(self._evaluate(value, context)) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise AssertionError(f"Validated AST node was not handled: {type(node).__name__}.")
