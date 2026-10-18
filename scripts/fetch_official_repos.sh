#!/usr/bin/env bash
set -euo pipefail

# Fetch the exact upstream source revisions recorded in third_party/SOURCES.md.
# This script intentionally does not fetch weights, datasets, submodules, or
# Git LFS objects.  A destination must be absent or empty; an existing checkout
# is never overwritten.

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [DESTINATION_ROOT]" >&2
  exit 64
fi

root=${1:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party"}
mkdir -p "$root"

clone_fixed() {
  local name=$1 url=$2 sha=$3
  local dest="${root}/${name}"
  if [[ -e "$dest" ]]; then
    if [[ -n "$(find "$dest" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "fetch_official_repos: refusing non-empty destination: $dest" >&2
      exit 73
    fi
    rmdir "$dest"
  fi

  echo "fetch_official_repos: cloning ${name} at ${sha}"
  git clone --filter=blob:none --no-checkout "$url" "$dest"
  git -C "$dest" fetch --depth 1 origin "$sha"
  git -C "$dest" checkout --detach --quiet "$sha"
  actual=$(git -C "$dest" rev-parse HEAD)
  if [[ "$actual" != "$sha" ]]; then
    echo "fetch_official_repos: revision mismatch for ${name}: ${actual}" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\n' "$name" "$actual" "$url"
}

clone_fixed FlexHook \
  https://github.com/buptLwz/FlexHook \
  bd1acc38634b28525d54dc6e0fcb38335f0029f9
clone_fixed YesTrack \
  https://github.com/ggbondrighthere24/YesTrack \
  ed42b8315b28c94dc595d83b0d49bf57002bb281

echo "fetch_official_repos: source fetch complete; weights and datasets were not fetched"
