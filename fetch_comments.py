#!/usr/bin/env python3
"""
Fetch all PR conversation comments + reviews + review threads (inline threads)
for the PR associated with the current git branch, by shelling out to:

  gh api graphql

Requires:
  - `gh auth login` already set up
  - current branch has an associated (open) PR

Usage:
  python fetch_comments.py > pr_comments.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

MAX_PAGES = 100

QUERY = """\
query(
  $owner: String!,
  $repo: String!,
  $number: Int!,
  $commentsCursor: String,
  $reviewsCursor: String,
  $threadsCursor: String
) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      number
      url
      title
      state
      comments(first: 100, after: $commentsCursor) {
        pageInfo { hasNextPage endCursor }
        nodes { id body createdAt updatedAt author { login } }
      }
      reviews(first: 100, after: $reviewsCursor) {
        pageInfo { hasNextPage endCursor }
        nodes { id state body submittedAt author { login } }
      }
      reviewThreads(first: 100, after: $threadsCursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved isOutdated path line diffSide
          startLine startDiffSide originalLine originalStartLine
          resolvedBy { login }
          comments(first: 100) {
            nodes { id body createdAt updatedAt author { login } }
          }
        }
      }
    }
  }
}
"""


def _run(cmd: list[str], stdin: str | None = None) -> str:
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {p.returncode}): {' '.join(cmd)}\n"
            f"{p.stderr.strip() if p.stderr else '(no stderr)'}"
        )
    return p.stdout


def _run_json(cmd: list[str], stdin: str | None = None) -> dict[str, Any]:
    out = _run(cmd, stdin=stdin)
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from command output: {e}\n"
            f"Raw output:\n{out[:2000]}"
        ) from e


def _safe_author(node: dict[str, Any] | None) -> dict[str, Any]:
    """Return author dict with a login field, even for ghost/deleted users."""
    if node is None:
        return {"login": "<ghost user>"}
    if "login" not in node:
        node["login"] = "<unknown>"
    return node


def _safe_author_nodes(nodes: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Patch each node.author so it never crashes on null."""
    for node in nodes or []:
        if "author" in node:
            node["author"] = _safe_author(node.get("author"))
        if "resolvedBy" in node:
            node["resolvedBy"] = _safe_author(node.get("resolvedBy"))
    return nodes


def gh_pr_view_json(fields: str) -> dict[str, Any]:
    return _run_json(["gh", "pr", "view", "--json", fields])


def get_current_pr_ref() -> tuple[str, str, int]:
    """Resolve the PR for the current branch.

    Works for cross-repo PRs by reading headRepository fields.
    Raises RuntimeError with a clear message if no PR is associated.
    """
    try:
        pr = gh_pr_view_json("number,headRepositoryOwner,headRepository")
    except RuntimeError as e:
        raise RuntimeError(
            "No PR associated with the current branch. "
            "Make sure you are on a branch that has an open pull request.\n"
            f"Cause: {e}"
        ) from None
    owner: str = pr["headRepositoryOwner"]["login"]
    repo: str = pr["headRepository"]["name"]
    number: int = int(pr["number"])
    return owner, repo, number


def gh_api_graphql(
    owner: str,
    repo: str,
    number: int,
    comments_cursor: str | None = None,
    reviews_cursor: str | None = None,
    threads_cursor: str | None = None,
) -> dict[str, Any]:
    """Call gh api graphql, passing query via stdin with -f query=@-.

    Uses -f (string field) for the GraphQL query so the stdin value is
    wrapped as a JSON string. Uses -F (raw field) for scalar variables.
    """
    cmd: list[str] = [
        "gh",
        "api",
        "graphql",
        "-F",
        "query=@-",
        "-F",
        f"owner={owner}",
        "-F",
        f"repo={repo}",
        "-F",
        f"number={number}",
    ]
    if comments_cursor:
        cmd += ["-F", f"commentsCursor={comments_cursor}"]
    if reviews_cursor:
        cmd += ["-F", f"reviewsCursor={reviews_cursor}"]
    if threads_cursor:
        cmd += ["-F", f"threadsCursor={threads_cursor}"]

    return _run_json(cmd, stdin=QUERY)


def fetch_all(owner: str, repo: str, number: int) -> dict[str, Any]:
    conversation_comments: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    review_threads: list[dict[str, Any]] = []

    comments_cursor: str | None = None
    reviews_cursor: str | None = None
    threads_cursor: str | None = None

    pr_meta: dict[str, Any] | None = None

    for _iteration in range(MAX_PAGES):
        payload = gh_api_graphql(
            owner=owner,
            repo=repo,
            number=number,
            comments_cursor=comments_cursor,
            reviews_cursor=reviews_cursor,
            threads_cursor=threads_cursor,
        )

        errors = payload.get("errors")
        if errors:
            raise RuntimeError(
                f"GitHub GraphQL errors:\n{json.dumps(errors, indent=2)}"
            )

        try:
            pr = payload["data"]["repository"]["pullRequest"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected GraphQL response shape: {e}\n"
                f"Payload keys: {list(payload.get('data', {}).keys())}"
            ) from e

        if pr_meta is None:
            pr_meta = {
                "number": pr["number"],
                "url": pr["url"],
                "title": pr["title"],
                "state": pr["state"],
                "owner": owner,
                "repo": repo,
            }

        c = pr["comments"]
        r = pr["reviews"]
        t = pr["reviewThreads"]

        # Sanitise authors to handle ghost/deleted users
        _safe_author_nodes(c.get("nodes") or [])
        _safe_author_nodes(r.get("nodes") or [])
        for thread in t.get("nodes") or []:
            _safe_author_nodes(thread.get("comments", {}).get("nodes") or [])

        conversation_comments.extend(c.get("nodes") or [])
        reviews.extend(r.get("nodes") or [])
        review_threads.extend(t.get("nodes") or [])

        comments_cursor = (
            c["pageInfo"]["endCursor"] if c["pageInfo"]["hasNextPage"] else None
        )
        reviews_cursor = (
            r["pageInfo"]["endCursor"] if r["pageInfo"]["hasNextPage"] else None
        )
        threads_cursor = (
            t["pageInfo"]["endCursor"] if t["pageInfo"]["hasNextPage"] else None
        )

        if not (comments_cursor or reviews_cursor or threads_cursor):
            break
    else:
        raise RuntimeError(
            f"Pagination exceeded {MAX_PAGES} pages; aborting to prevent infinite loop. "
            "This may indicate a GitHub API issue."
        )

    assert pr_meta is not None
    return {
        "pull_request": pr_meta,
        "conversation_comments": conversation_comments,
        "reviews": reviews,
        "review_threads": review_threads,
    }


def main() -> None:
    try:
        owner, repo, number = get_current_pr_ref()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from None

    result = fetch_all(owner, repo, number)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
