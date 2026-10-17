#!/usr/bin/env bash
# Download the large certificate files from the GitHub Release and check the SHA-256 hashes of all
# certificate files against certificates/SHA256SUMS (which is in git). The list of files to download
# is taken from SHA256SUMS itself (every .npz listed there), so the two cannot disagree.
# 大きな検証用データを GitHub Release から取得し、すべての検証用データの SHA-256 を certificates/SHA256SUMS（git に入っている）と照合する。
# 取得するファイルの一覧は SHA256SUMS そのもの（そこにあるすべての .npz）から取るので、両者が食い違うことはない。
# Run from the repository root / リポジトリの根から:  bash code/verify/fetch.sh [release tag]
set -euo pipefail
TAG=${1:-certificates-v1}
BASE="https://github.com/roiban1344/112-dimensional-extremal-even-unimodular-lattice-candidate/releases/download/$TAG"
cd "$(dirname "${BASH_SOURCE[0]}")/../../certificates"
for f in $(awk '{print $2}' SHA256SUMS | sed 's/^\*//' | grep '\.npz$'); do
  [ -f "$f" ] || curl -fL -o "$f" "$BASE/$f"
done
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
