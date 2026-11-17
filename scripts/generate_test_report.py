"""Run pytest and write the executed results to docs/test_report.md (real counts, nothing hand-typed)."""
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    xml = ROOT / "data" / "processed" / "junit.xml"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=ROOT,
                          capture_output=True, text=True)
    suite = ET.parse(xml).getroot()
    suite = suite.find("testsuite") if suite.tag == "testsuites" else suite
    total, fail, err, skip = (int(suite.get(k)) for k in ("tests", "failures", "errors", "skipped"))
    by_file: dict[str, list[int]] = {}
    for case in suite.iter("testcase"):
        f = case.get("classname").split(".")[1] if "." in case.get("classname") else case.get("classname")
        row = by_file.setdefault(f, [0, 0])
        row[0] += 1
        if case.find("failure") is not None or case.find("error") is not None:
            row[1] += 1
    lines = ["# Test report", "",
             f"Executed {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} with `python -m pytest -q` "
             f"(Python {sys.version.split()[0]}, {sys.platform}).", "",
             f"**{total} tests: {total - fail - err - skip} passed, {fail} failed, {err} errors, {skip} skipped.**", "",
             "| Test file | Tests | Failed |", "|---|---|---|"]
    lines += [f"| `{k}.py` | {v[0]} | {v[1]} |" for k, v in sorted(by_file.items())]
    lines += ["", "Covers: dataset loading/validation/download failure, graph construction and caching, Louvain, Greedy Modularity, "
              "modularity (hand-checked), stability/agreement, interest aggregation, Jaccard and cosine (vs brute force), user and "
              "community metrics, SQLite persistence/constraints/transactions, query routing, Gemini context generation, invalid "
              "community IDs, missing API key, API failures/cool-down, empty datasets, missing features, and the evaluation scorer "
              "itself (it must fail wrong answers).", "",
             "Unit tests use small synthetic graphs only; the application never uses synthetic data for its results. The Gemini "
             "client's error handling is unit-tested with a stubbed SDK object; the live API is exercised by "
             "`scripts/run_evaluation.py` (see `evaluation.md`).", "", "```", proc.stdout.strip(), "```"]
    (ROOT / "docs" / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(proc.stdout.strip().splitlines()[-1])
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
