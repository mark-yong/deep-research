#!/usr/bin/env bash
# deep-research upstream sync check
# Run periodically; prints what upstream changed relative to this fork's
# content. This repo shares no git history with upstream (snapshot import),
# so syncs are content diffs, never merges.
set -euo pipefail
cd "$(dirname "$0")"

git fetch upstream --quiet

UPSTREAM_HEAD=$(git rev-parse --short upstream/main)
LAST_SYNC_SHA=$(cat .last-upstream-sync 2>/dev/null || echo "none")
echo "upstream/main: $UPSTREAM_HEAD (last recorded sync: $LAST_SYNC_SHA)"

if [ "$UPSTREAM_HEAD" = "$LAST_SYNC_SHA" ]; then
    echo "no new upstream commits since last check."
    exit 0
fi

echo
echo "== commits upstream since last check =="
if [ "$LAST_SYNC_SHA" = "none" ]; then
    git log --oneline -15 upstream/main
else
    git log --oneline "$LAST_SYNC_SHA..upstream/main"
fi

echo
echo "== content deltas vs this fork (src + pyproject) =="
git diff --stat upstream/main HEAD -- src/open_deep_research/configuration.py \
    src/open_deep_research/deep_researcher.py src/open_deep_research/utils.py \
    src/open_deep_research/prompts.py pyproject.toml || true

echo
echo "== dependency drift (uv.lock and requirements-relevant bumps) =="
git log --oneline "$LAST_SYNC_SHA..upstream/main" -- uv.lock 2>/dev/null | head -5 || \
    git log --oneline -5 upstream/main -- uv.lock

echo
echo "Review the per-file diffs above; port changes by hand if relevant."
echo "After reviewing, record the sync point:  echo $UPSTREAM_HEAD > .last-upstream-sync"
