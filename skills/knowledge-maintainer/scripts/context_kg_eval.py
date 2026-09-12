#!/usr/bin/env python3
"""Evaluate context-kg retrieval quality and reading cost."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from context_kg_search import search_with_metrics


def evaluate(context_kg: Path, cases_path: Path, *, limit: int = 5) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集必须是非空 JSON 列表")
    reciprocal_rank = 0.0
    hits = 0
    zero_hits = 0
    fallback_count = 0
    body_reads = 0
    index_levels = 0
    failures: list[str] = []
    for case in cases:
        results, metrics = search_with_metrics(
            context_kg,
            str(case["query"]),
            scope=case.get("scope"),
            history=bool(case.get("history", False)),
            include_body=bool(case.get("body", False)),
            index_levels_expanded=int(case.get("index_levels_expanded", 0)),
            limit=limit,
        )
        paths = [result.path for result in results]
        expected = set(case["expected"])
        rank = next((i for i, path in enumerate(paths, 1) if path in expected), None)
        if rank is not None:
            hits += 1
            reciprocal_rank += 1 / rank
        else:
            failures.append(f"{case['name']}: expected {sorted(expected)}, got {paths}")
        if not results:
            zero_hits += 1
        if metrics.fallback_used != "none":
            fallback_count += 1
        body_reads += metrics.body_documents_read
        index_levels += metrics.index_levels_expanded
        maximum = case.get("max_body_documents")
        if maximum is not None and metrics.body_documents_read > int(maximum):
            failures.append(
                f"{case['name']}: body reads {metrics.body_documents_read} > {maximum}"
            )
        expected_fallback = case.get("expected_fallback")
        if expected_fallback and metrics.fallback_used != expected_fallback:
            failures.append(
                f"{case['name']}: fallback {metrics.fallback_used} != {expected_fallback}"
            )
    total = len(cases)
    return {
        "cases": total,
        "recall_at_k": round(hits / total, 4),
        "mrr": round(reciprocal_rank / total, 4),
        "zero_hit_rate": round(zero_hits / total, 4),
        "body_fallback_rate": round(fallback_count / total, 4),
        "body_documents_read": body_reads,
        "average_index_levels_expanded": round(index_levels / total, 4),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估 context-kg 检索质量与读取成本。")
    parser.add_argument("context_kg", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.context_kg, args.cases, limit=args.limit)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
