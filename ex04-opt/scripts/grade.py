"""Run the exercise's tests like CI does and report which pass; usage: python scripts/grade.py [--json out.json]"""
import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG = Path(".github/classroom/autograding.json")
OPTOUT_FILE = "AI_OPT_OUT.md"
AI_USAGE_FILE = "AI_USAGE.md"
NOT_POINTS = "Pass/Fail only: the tutors grade your submission and decide your points."


def report_optout():
    """Say whether this repository declares an AI opt-out (same rule as provisioning: AI_OPT_OUT.md in the root).

    Returns:
        list of lines for the summary
    """
    # exact name, case-sensitive like GitHub: Path.exists() would also match ai_opt_out.md on Windows and macOS
    if OPTOUT_FILE not in os.listdir("."):
        return [f"AI opt-out: none declared (no file {OPTOUT_FILE} in the root of this repository; "
                "to object, see the instructions on the first exercise sheet)."]
    return [f"AI opt-out registered for the whole team ({OPTOUT_FILE} found). "
            "It applies from this exercise on; its content is up to you."]


def report_ai_usage():
    """Say whether this repository declares AI use (AI_USAGE.md in the root; needed only if AI was used).

    Returns:
        list of lines for the summary
    """
    if AI_USAGE_FILE not in os.listdir("."):  # exact name, as for the opt-out
        return [f"AI use: none declared. Without {AI_USAGE_FILE}, your team states that it used no AI tools for "
                f"this exercise. If you did use any, add {AI_USAGE_FILE} to the root of this repository (which "
                "tools, for what); undeclared AI use gives 0 points for the exercise."]
    return [f"AI use declared ({AI_USAGE_FILE} found). The rules on AI use on the exercise sheet apply."]


def run(cmd, timeout_min):
    """Run one shell command with a timeout.

    Args:
        cmd: the command
        timeout_min: timeout in minutes

    Returns:
        (passed, output)
    """
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_min * 60)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_min} min"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main():
    """Run every test of autograding.json, print Pass/Fail per test, write the --json report and the CI job summary."""
    tests = json.loads(CONFIG.read_text(encoding="utf8"))["tests"]
    rows, got, total = [], 0, 0
    for test in tests:
        print(f"::group::{test['name']}" if os.environ.get("CI") else f"--- {test['name']}", flush=True)
        passed, output = True, ""
        if test.get("setup"):
            passed, output = run(test["setup"], test.get("timeout", 10))
        if passed:
            passed, output = run(test["run"], test.get("timeout", 10))
        print(output)
        if os.environ.get("CI"):
            print("::endgroup::")
        points = test.get("points", 0)
        total += points
        got += points if passed else 0
        rows.append((test["name"], passed, points if passed else 0, points))

    optout = report_optout() + report_ai_usage()
    n_passed = sum(ok for _, ok, _, _ in rows)
    result = f"{n_passed} of {len(rows)} tests passed"
    width = max(len(r[0]) for r in rows)
    table = [f"{'test':<{width}}  result"]
    table += [f"{name:<{width}}  {'Pass' if ok else 'Fail'}" for name, ok, _, _ in rows]
    print("\n" + "\n".join(table) + f"\n\n{result}\n{NOT_POINTS}\n\n" + "\n".join(optout))
    if os.environ.get("CI"):  # shown on the run page itself, without opening the log or the summary
        for title, line in zip(("AI opt-out", "AI use"), optout):
            print(f"::notice title={title}::{line}")

    # used by staff tooling after the deadline; keeps the points of autograding.json (students see only Pass/Fail)
    if len(sys.argv) == 3 and sys.argv[1] == "--json":
        Path(sys.argv[2]).write_text(json.dumps({"points": got, "max": total, "tests": [
            {"name": n, "passed": ok, "points": p, "max": m} for n, ok, p, m in rows]}, indent=2) + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        md = [f"### {result}", "", NOT_POINTS, ""] + [f"> {line}\n" for line in optout]  # one quote each
        md += ["", "| test | result |", "|---|---|"]
        md += [f"| {n} | {':white_check_mark: Pass' if ok else ':x: Fail'} |" for n, ok, _, _ in rows]
        with open(summary, "a", encoding="utf8") as f:
            f.write("\n".join(md) + "\n")
    sys.exit(0 if n_passed == len(rows) else 1)


if __name__ == "__main__":
    main()
