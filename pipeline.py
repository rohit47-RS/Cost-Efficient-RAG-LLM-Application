"""
Main judging pipeline CLI.

Mode 1 — score a suite (pointwise):
    python pipeline.py score --suite test_suite.json --out results/report.json

Mode 2 — compare two configs head-to-head (pairwise, with position-bias check),
declaring a winner. Each config file is a JSON list of {"case_id", "output"} — i.e.
the SAME inputs run through two different prompts/models, with their outputs saved
separately beforehand:
    python pipeline.py compare --suite test_suite.json \\
        --config-a outputs/prompt_v1.json --config-b outputs/prompt_v2.json \\
        --out results/ab_report.json

Every judge prompt + raw response is logged to logs/judge_audit.jsonl regardless of
mode (see judge/judge.py). Judge and generator models are independently configurable
via .env (JUDGE_MODEL / GENERATOR_MODEL).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import anthropic

from config import get_settings
from judge.schemas import TestSuite
from judge.judge import judge_pointwise
from judge.aggregate import aggregate_pointwise_suite, declare_ab_winner
from judge.bias import run_position_bias_check, aggregate_flip_rate


def cmd_score(args):
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    suite = TestSuite.model_validate(json.loads(args.suite.read_text()))

    print(f"Scoring {len(suite.cases)} cases with judge={settings.judge_model} "
          f"(generator that produced these outputs, for reference: {settings.generator_model})")

    verdicts = []
    for case in suite.cases:
        v = judge_pointwise(client, settings.judge_model, case)
        verdicts.append(v)
        status = "PASS" if v.passed else "FAIL"
        err = f" [PARSE_ERROR: {v.parse_error}]" if v.parse_error else ""
        print(f"  {case.id:35s} overall={v.overall_score:.2f} {status}{err}")

    summary = aggregate_pointwise_suite(verdicts, settings.pass_threshold)
    report = {
        "summary": summary,
        "judge_model": settings.judge_model,
        "pass_threshold": settings.pass_threshold,
        "per_case": [v.model_dump() for v in verdicts],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("\n=== SUITE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {args.out}")


def cmd_compare(args):
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    suite = TestSuite.model_validate(json.loads(args.suite.read_text()))
    outputs_a = {o["case_id"]: o["output"] for o in json.loads(args.config_a.read_text())}
    outputs_b = {o["case_id"]: o["output"] for o in json.loads(args.config_b.read_text())}

    win_counts = {"A": 0, "B": 0, "tie": 0}
    position_results = []
    per_case_details = []
    total_input_tokens = total_output_tokens = total_judge_calls = 0

    for case in suite.cases:
        if case.id not in outputs_a or case.id not in outputs_b:
            continue
        resp_a, resp_b = outputs_a[case.id], outputs_b[case.id]

        pb_result = run_position_bias_check(
            client, settings.judge_model, case.id, case.input, resp_a, resp_b,
            case.system_prompt, case.expected_output,
        )
        position_results.append(pb_result)
        win_counts[pb_result.final_winner] = win_counts.get(pb_result.final_winner, 0) + 1
        total_input_tokens += pb_result.total_input_tokens
        total_output_tokens += pb_result.total_output_tokens
        total_judge_calls += 2  # run_position_bias_check always makes exactly 2 judge calls

        per_case_details.append({
            "case_id": case.id,
            "winner_order1": pb_result.winner_order1,
            "winner_order2_remapped": pb_result.winner_order2,
            "flipped": pb_result.flipped,
            "final_winner": pb_result.final_winner,
        })
        print(f"  {case.id:35s} order1={pb_result.winner_order1} order2={pb_result.winner_order2} "
              f"{'FLIPPED' if pb_result.flipped else 'stable'} -> final={pb_result.final_winner}")

    flip_rate = aggregate_flip_rate(position_results)
    ab_result = declare_ab_winner(win_counts, tie_margin=args.tie_margin)

    report = {
        "config_a_file": str(args.config_a),
        "config_b_file": str(args.config_b),
        "judge_model": settings.judge_model,
        "position_bias_flip_rate": round(flip_rate, 4),
        "ab_comparison": ab_result,
        "cost": {
            "total_judge_calls": total_judge_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "note": "2 judge calls per case (both orders), for the position-bias check",
        },
        "per_case": per_case_details,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("\n=== A/B COMPARISON ===")
    print(json.dumps(ab_result, indent=2))
    print(f"Position-bias flip rate: {flip_rate:.2%}")
    print(f"\nWritten to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Pointwise-score a test suite against the rubric")
    p_score.add_argument("--suite", type=Path, default=Path("test_suite.json"))
    p_score.add_argument("--out", type=Path, default=Path("results/report.json"))
    p_score.set_defaults(func=cmd_score)

    p_compare = sub.add_parser("compare", help="Pairwise-compare two configs and declare a winner")
    p_compare.add_argument("--suite", type=Path, default=Path("test_suite.json"))
    p_compare.add_argument("--config-a", type=Path, required=True)
    p_compare.add_argument("--config-b", type=Path, required=True)
    p_compare.add_argument("--tie-margin", type=float, default=0.05)
    p_compare.add_argument("--out", type=Path, default=Path("results/ab_report.json"))
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
