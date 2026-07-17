#!/usr/bin/env python3
"""
Inspect failing GitHub PR checks: fetch GitHub Actions logs and extract a
concise failure snippet.

Requires:
  - `gh auth login` already set up
  - current directory inside a Git repository (or --repo flag)

Exit codes:
  0  — no failing checks detected  (or all failures are log-pending)
  1  — confirmed failing checks with actionable logs
  2  — internal error (GitHub CLI, repo, or unexpected response)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Any, Iterable, Sequence

FAILURE_CONCLUDED = {
    "failure",
    "cancelled",
    "timed_out",
    "action_required",
}

FAILURE_ONGOING = {
    "failure",
    "error",
    "cancelled",
    "timed_out",
    "action_required",
}

FAILURE_BUCKETS = {"fail"}

FAILURE_MARKERS = (
    "error",
    "fail",
    "failed",
    "traceback",
    "exception",
    "assert",
    "panic",
    "fatal",
    "timeout",
    "segmentation fault",
)

DEFAULT_MAX_LINES = 160
DEFAULT_CONTEXT_LINES = 30

# More specific phrases to avoid false-positive substring matches
PENDING_LOG_PATTERNS = (
    "still in progress ... log will be available when it is complete",
    "the log for this run is not yet available",
    "the log is not available yet",
)


class GhResult:
    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# gh CLI helpers
# ---------------------------------------------------------------------------


def run_gh_command(
    args: Sequence[str],
    cwd: Path,
    text: bool = True,
) -> GhResult:
    """Run a `gh` subcommand and return a GhResult.

    When *text* is False the stdout field will be the raw bytes decoded
    with ``errors="replace"``; callers needing binary output should pass
    ``text=False`` and then access ``.stdout_bytes`` (not yet present —
    in that case the stdout is stored as a string via replace-decoding).
    """
    process = subprocess.run(
        ["gh", *args],
        cwd=cwd,
        text=text,
        capture_output=True,
    )
    stderr = process.stderr
    if not text and isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return GhResult(process.returncode, process.stdout, stderr)


def _gh_json(
    args: Sequence[str],
    repo_root: Path,
    label: str = "command",
) -> Any:
    """Run `gh <args> --json` and parse the JSON output.

    Returns the parsed JSON data, or None on failure (printed to stderr).
    """
    result = run_gh_command([*args], cwd=repo_root)
    if result.returncode != 0:
        message = "\n".join(filter(None, [result.stderr, result.stdout])).strip()
        print(f"Error: gh {label} failed.\n{message}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout or "null")
    except json.JSONDecodeError as e:
        print(
            f"Error: unable to parse {label} JSON output: {e}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Entry point & orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect failing GitHub PR checks, fetch GitHub Actions logs, and extract a "
            "failure snippet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Path inside the target Git repository.",
    )
    parser.add_argument(
        "--pr",
        default=None,
        help="PR number or URL (defaults to current branch PR).",
    )
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT_LINES)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = find_git_root(Path(args.repo))
    if repo_root is None:
        print("Error: not inside a Git repository.", file=sys.stderr)
        return 2

    if not ensure_gh_available(repo_root):
        return 2

    pr_value = resolve_pr(args.pr, repo_root)
    if pr_value is None:
        return 2

    checks = fetch_checks(pr_value, repo_root)
    if checks is None:
        return 2

    failing = [c for c in checks if is_failing(c)]
    if not failing:
        print(f"PR #{pr_value}: no failing checks detected.")
        return 0

    analysis_ok = True
    results: list[dict[str, Any]] = []

    for check in failing:
        r = analyze_check(
            check,
            repo_root=repo_root,
            max_lines=max(1, args.max_lines),
            context=max(1, args.context),
        )
        if r.get("status") in ("error", "external") and not r.get("logSnippet"):
            analysis_ok = False
        results.append(r)

    if args.json:
        print(json.dumps({"pr": pr_value, "results": results}, indent=2))
    else:
        render_results(pr_value, results)

    return 1 if analysis_ok else 2


# ---------------------------------------------------------------------------
# Environment verification
# ---------------------------------------------------------------------------


def find_git_root(start: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def ensure_gh_available(repo_root: Path) -> bool:
    if which("gh") is None:
        print("Error: gh is not installed or not on PATH.", file=sys.stderr)
        return False
    result = run_gh_command(["auth", "status"], cwd=repo_root)
    if result.returncode == 0:
        return True
    message = (result.stderr or result.stdout or "").strip()
    print(message or "Error: gh not authenticated.", file=sys.stderr)
    return False


def resolve_pr(pr_value: str | None, repo_root: Path) -> str | None:
    if pr_value:
        return pr_value
    result = run_gh_command(["pr", "view", "--json", "number"], cwd=repo_root)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        print(
            message or "Error: unable to resolve PR for the current branch.",
            file=sys.stderr,
        )
        return None
    data = _gh_json(["pr", "view", "--json", "number"], repo_root, label="pr view")
    if data is None:
        return None
    number = data.get("number")
    if not number:
        print("Error: no PR number found.", file=sys.stderr)
        return None
    return str(number)


# ---------------------------------------------------------------------------
# Check fetching & filtering
# ---------------------------------------------------------------------------


def fetch_checks(pr_value: str, repo_root: Path) -> list[dict[str, Any]] | None:
    primary_fields = ["name", "state", "conclusion", "detailsUrl", "startedAt", "completedAt"]

    result = run_gh_command(
        ["pr", "checks", pr_value, "--json", ",".join(primary_fields)],
        cwd=repo_root,
    )

    if result.returncode == 0:
        return _parse_checks_json(result.stdout, pr_value)

    # Fallback: retry with an older field set
    message = "\n".join(filter(None, [result.stderr, result.stdout])).strip()
    available_fields = parse_available_fields(message)
    if not available_fields:
        print(message or "Error: gh pr checks failed.", file=sys.stderr)
        return None

    fallback_fields = [
        "name",
        "state",
        "bucket",
        "link",
        "startedAt",
        "completedAt",
        "workflow",
    ]
    selected_fields = [f for f in fallback_fields if f in available_fields]
    if not selected_fields:
        print("Error: no usable fields available for gh pr checks.", file=sys.stderr)
        return None

    result = run_gh_command(
        ["pr", "checks", pr_value, "--json", ",".join(selected_fields)],
        cwd=repo_root,
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        print(msg or "Error: gh pr checks failed (fallback).", file=sys.stderr)
        return None

    return _parse_checks_json(result.stdout, pr_value)


def _parse_checks_json(raw: str, pr_value: str) -> list[dict[str, Any]] | None:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError as e:
        print(f"Error: unable to parse checks JSON for PR #{pr_value}: {e}", file=sys.stderr)
        return None
    if not isinstance(data, list):
        print(f"Error: unexpected checks JSON shape for PR #{pr_value}.", file=sys.stderr)
        return None
    return data


def is_failing(check: dict[str, Any]) -> bool:
    conclusion = normalize_field(check.get("conclusion"))
    if conclusion in FAILURE_CONCLUDED:
        return True
    state = normalize_field(check.get("state") or check.get("status"))
    if state in FAILURE_ONGOING:
        return True
    bucket = normalize_field(check.get("bucket"))
    return bucket in FAILURE_BUCKETS


# ---------------------------------------------------------------------------
# Single check analysis
# ---------------------------------------------------------------------------


def analyze_check(
    check: dict[str, Any],
    repo_root: Path,
    max_lines: int,
    context: int,
) -> dict[str, Any]:
    url = check.get("detailsUrl") or check.get("link") or ""
    run_id = extract_run_id(url)
    job_id = extract_job_id(url)

    base: dict[str, Any] = {
        "name": check.get("name", ""),
        "detailsUrl": url,
        "runId": run_id,
        "jobId": job_id,
    }

    if run_id is None:
        base["status"] = "external"
        base["note"] = "No GitHub Actions run id detected in detailsUrl."
        return base

    # Fetch logs first — if they aren't available we skip the metadata call
    log_text, log_error, log_status = fetch_check_log(
        run_id=run_id,
        job_id=job_id,
        repo_root=repo_root,
    )

    if log_status == "pending":
        base["status"] = "log_pending"
        base["note"] = log_error or "Logs are not available yet."
        return base

    if log_error:
        base["status"] = "log_unavailable"
        base["error"] = log_error
        return base

    # Only fetch run metadata now that we know logs are available
    metadata = fetch_run_metadata(run_id, repo_root)
    snippet = extract_failure_snippet(log_text, max_lines=max_lines, context=context)

    base["status"] = "ok"
    base["run"] = metadata or {}
    base["logSnippet"] = snippet
    base["logTail"] = tail_lines(log_text, max_lines)
    return base


# ---------------------------------------------------------------------------
# Run / job artifact extraction
# ---------------------------------------------------------------------------


def extract_run_id(url: str) -> str | None:
    if not url:
        return None
    for pattern in (r"/actions/runs/(\d+)", r"/runs/(\d+)"):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_job_id(url: str) -> str | None:
    if not url:
        return None
    for pattern in (r"/actions/runs/\d+/job/(\d+)", r"/job/(\d+)"):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_run_metadata(run_id: str, repo_root: Path) -> dict[str, Any] | None:
    fields = [
        "conclusion",
        "status",
        "workflowName",
        "name",
        "event",
        "headBranch",
        "headSha",
        "url",
    ]
    return _gh_json(
        ["run", "view", run_id, "--json", ",".join(fields)],
        repo_root,
        label="run view",
    )


def fetch_check_log(
    run_id: str,
    job_id: str | None,
    repo_root: Path,
) -> tuple[str, str, str]:
    """Fetch log text. Returns (text, error, status).

    Status is one of ``"ok"``, ``"pending"``, or ``"error"``.
    """
    log_text, log_error = fetch_run_log(run_id, repo_root)
    if not log_error:
        return log_text, "", "ok"

    if is_log_pending_message(log_error) and job_id:
        job_log, job_error = fetch_job_log(job_id, repo_root)
        if job_log:
            return job_log, "", "ok"
        if job_error and is_log_pending_message(job_error):
            return "", job_error, "pending"
        return "", (job_error or log_error), "pending" if is_log_pending_message(job_error or log_error) else "error"

    if is_log_pending_message(log_error):
        return "", log_error, "pending"

    return "", log_error, "error"


def fetch_run_log(run_id: str, repo_root: Path) -> tuple[str, str]:
    result = run_gh_command(["run", "view", run_id, "--log"], cwd=repo_root)
    if result.returncode != 0:
        error = (result.stderr or result.stdout or "").strip()
        return "", error or "gh run view failed"
    return result.stdout, ""


def fetch_job_log(job_id: str, repo_root: Path) -> tuple[str, str]:
    repo_slug = fetch_repo_slug(repo_root)
    if not repo_slug:
        return "", "Error: unable to resolve repository name for job logs."

    endpoint = f"/repos/{repo_slug}/actions/jobs/{job_id}/logs"
    # Single fetch in binary mode to check zip magic bytes
    process = subprocess.run(
        ["gh", "api", endpoint],
        cwd=repo_root,
        capture_output=True,
    )
    stdout_bytes = process.stdout
    if process.returncode != 0:
        message = process.stderr.decode(errors="replace").strip() if process.stderr else "gh api job logs failed"
        return "", message

    if is_zip_payload(stdout_bytes):
        return "", "Job logs returned a zip archive; unable to parse."
    return stdout_bytes.decode(errors="replace"), ""


def fetch_repo_slug(repo_root: Path) -> str | None:
    data = _gh_json(["repo", "view", "--json", "nameWithOwner"], repo_root, label="repo view")
    if data is None:
        return None
    name_with_owner = data.get("nameWithOwner")
    return str(name_with_owner) if name_with_owner else None


# ---------------------------------------------------------------------------
# Parsing & matching
# ---------------------------------------------------------------------------


def normalize_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def parse_available_fields(message: str) -> list[str]:
    """Extract field names from a ``gh`` error message.

    Handles the standard ``Available fields:`` line followed by space-separated
    field names, as well as a comma-separated fallback for newer gh versions.
    """
    if "Available fields:" not in message and "Fields:" not in message:
        return []

    lines = message.splitlines()
    for i, line in enumerate(lines):
        if "Available fields:" in line or "Fields:" in line:
            # Inline with the header:  "Available fields: name, state, conclusion"
            value = line.split(":", 1)[1].strip()
            if value and "," in value:
                return [f.strip() for f in value.split(",") if f.strip()]
            # Otherwise it's on subsequent lines
            fields: list[str] = []
            for field_line in lines[i + 1:]:
                stripped = field_line.strip()
                if not stripped:
                    continue
                # Skip indented help text (starts with whitespace + non-field text)
                if stripped == field_line and not field_line.startswith(" "):
                    # Space-separated list of field names
                    fields.extend(stripped.split())
            return fields
    return []


def is_log_pending_message(message: str) -> bool:
    lowered = message.lower()
    return any(pattern in lowered for pattern in PENDING_LOG_PATTERNS)


def is_zip_payload(payload: bytes) -> bool:
    return payload[:2] == b"PK"


def extract_failure_snippet(log_text: str, max_lines: int, context: int) -> str:
    lines = log_text.splitlines()
    if not lines:
        return ""

    marker_index = find_failure_index(lines)
    if marker_index is None:
        return "\n".join(lines[-max_lines:])

    start = max(0, marker_index - context)
    end = min(len(lines), marker_index + context)
    window = lines[start:end]
    if len(window) > max_lines:
        window = window[-max_lines:]
    return "\n".join(window)


def find_failure_index(lines: Sequence[str]) -> int | None:
    for idx in range(len(lines) - 1, -1, -1):
        lowered = lines[idx].lower()
        if any(marker in lowered for marker in FAILURE_MARKERS):
            return idx
    return None


def tail_lines(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def render_results(pr_number: str, results: Iterable[dict[str, Any]]) -> None:
    results_list = list(results)
    total = len(results_list)
    actionable = sum(1 for r in results_list if r.get("status") == "ok")
    pending = sum(1 for r in results_list if r.get("status") == "log_pending")
    unavailable = sum(1 for r in results_list if r.get("status") in ("log_unavailable", "external"))

    print(f"PR #{pr_number}: {total} failing check(s) — {actionable} analyzed, {pending} pending, {unavailable} unavailable.")

    for result in results_list:
        print("-" * 60)
        print(f"Check: {result.get('name', '')}")
        if result.get("detailsUrl"):
            print(f"Details: {result['detailsUrl']}")
        run_id = result.get("runId")
        if run_id:
            print(f"Run ID: {run_id}")
        job_id = result.get("jobId")
        if job_id:
            print(f"Job ID: {job_id}")
        status = result.get("status", "unknown")
        print(f"Status: {status}")

        run_meta = result.get("run", {})
        if run_meta:
            branch = run_meta.get("headBranch", "")
            sha = (run_meta.get("headSha") or "")[:12]
            workflow = run_meta.get("workflowName") or run_meta.get("name") or ""
            conclusion = run_meta.get("conclusion") or run_meta.get("status") or ""
            print(f"Workflow: {workflow} ({conclusion})")
            if branch or sha:
                print(f"Branch/SHA: {branch} {sha}")
            if run_meta.get("url"):
                print(f"Run URL: {run_meta['url']}")

        if result.get("note"):
            print(f"Note: {result['note']}")

        if result.get("error"):
            print(f"Error fetching logs: {result['error']}")
            continue

        snippet = result.get("logSnippet") or ""
        if snippet:
            print("Failure snippet:")
            print(indent_block(snippet, prefix="  "))
        else:
            print("No snippet available.")
    print("-" * 60)


def indent_block(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
