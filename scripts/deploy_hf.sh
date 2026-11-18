#!/usr/bin/env bash
# Deploy the current commit to a Hugging Face Space (Docker SDK).
#
# Hugging Face rejects binary files (mp3/wav/png/...) that are not stored with Git LFS,
# and this repo's history contains such files. Instead of rewriting the team's history,
# this script pushes a clean single-commit snapshot of HEAD *without* binary files
# (they are benchmarks/report assets, not needed to run the app).
#
# One-time setup:
#   git remote add space https://huggingface.co/spaces/<username>/<space-name>
# Usage (Git Bash):
#   bash scripts/deploy_hf.sh            # deploys HEAD
#   bash scripts/deploy_hf.sh my-branch  # deploys another ref
# When asked for a password, paste a Hugging Face *write* token.
#
# Note: this force-pushes to the Space's main branch. The Space repo is a deploy target only.

set -euo pipefail

REMOTE="${HF_REMOTE:-space}"
REF="${1:-HEAD}"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Remote '$REMOTE' not found. Add it first:"
  echo "  git remote add $REMOTE https://huggingface.co/spaces/<username>/<space-name>"
  exit 1
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Warning: you have uncommitted changes. Only committed code ($REF) is deployed."
fi

SHA="$(git rev-parse --short "$REF")"
TMP_INDEX="$(mktemp)"
trap 'rm -f "$TMP_INDEX"' EXIT

# Build a tree from REF without binary files, using a temporary index (working tree untouched).
export GIT_INDEX_FILE="$TMP_INDEX"
git read-tree "$REF"
BINARIES="$(git ls-files | grep -iE '\.(png|jpe?g|gif|mp3|wav|ogg|webm|pdf|zip)$' || true)"
if [ -n "$BINARIES" ]; then
  echo "$BINARIES" | tr '\n' '\0' | xargs -0 git rm --cached --quiet --
  echo "Excluded $(echo "$BINARIES" | wc -l | tr -d ' ') binary files from the snapshot."
fi
TREE="$(git write-tree)"
unset GIT_INDEX_FILE

COMMIT="$(git commit-tree "$TREE" -m "Deploy $SHA")"
echo "Pushing snapshot of $SHA to $REMOTE ..."
git push --force "$REMOTE" "$COMMIT:refs/heads/main"

URL="$(git remote get-url "$REMOTE")"
echo
echo "Done. Build logs: ${URL%.git}?logs=build"
