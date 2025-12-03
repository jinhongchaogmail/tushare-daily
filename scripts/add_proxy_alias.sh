#!/usr/bin/env bash
# Helper to add proxyon/proxyoff aliases to user's ~/.bashrc

set -euo pipefail

PROFILE_FILE="$HOME/.bashrc"

ALIASES="alias proxyon='export http_proxy=\"socks5://127.0.0.1:1088\"; export https_proxy=\"socks5://127.0.0.1:1088\"; export all_proxy=\"socks5://127.0.0.1:1088\"; echo \"✅ V2Ray 代理已开启 (1088 端口)\"'\nalias proxyoff='unset http_proxy; unset https_proxy; unset all_proxy; echo \"❌ V2Ray 代理已关闭\"'"

echo "Adding proxy aliases to $PROFILE_FILE..."

if grep -F "alias proxyon='export http_proxy" "$PROFILE_FILE" >/dev/null 2>&1; then
  echo "Aliases already present in $PROFILE_FILE, skipping."
else
  printf "\n# V2Ray proxy aliases (added by script)\n%s\n" "$ALIASES" >> "$PROFILE_FILE"
  echo "Aliases appended to $PROFILE_FILE"
fi

echo "Sourcing $PROFILE_FILE to apply changes in current shell (if possible)..."
# shellcheck disable=SC1090
source "$PROFILE_FILE" || true

echo "Done. You can now use 'proxyon' and 'proxyoff' in new shells." 
