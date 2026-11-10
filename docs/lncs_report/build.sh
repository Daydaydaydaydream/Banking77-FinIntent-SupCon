#!/bin/sh
set -eu
cd "$(dirname "$0")"
if command -v xelatex >/dev/null 2>&1; then
  TEX_ENGINE=$(command -v xelatex)
else
  TEX_ENGINE="$PWD/tools/texlive/bin/universal-darwin/xelatex"
fi
for lang in en zh; do
  "$TEX_ENGINE" -interaction=nonstopmode -halt-on-error "report_${lang}.tex"
  "$TEX_ENGINE" -interaction=nonstopmode -halt-on-error "report_${lang}.tex"
done
