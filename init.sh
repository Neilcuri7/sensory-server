#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# init.sh - verification gate. Fails closed: an unconfigured gate exits 2.
# Usage: ./init.sh [--bootstrap] | ./init.sh --feature <id>
# Exit: 0 ok, 1 a check failed, 2 unconfigured, 3 jq missing, 4 bad feature/list, 5 evidence write, 6 done-check, 64 bad args
INSTALL_CMD=""   # only run with --bootstrap
VERIFY_CMD="nix-shell shell.nix --run 'python -m pytest tests -q'"
START_CMD=""     # only run if RUN_START_COMMAND=1, foreground, after verify passes
F="feature_list.json"

die() { # <code> <what> <why> <fix>
  local code="$1"; shift
  printf 'WHAT: %s\nWHY: %s\nFIX: %s\n' "$1" "$2" "$3" >&2
  exit "$code"
}

is_noop() {
  local c; c="$(printf '%s' "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -z "$c" ] || [ "$c" = "true" ] || [ "$c" = ":" ] || [ "$c" = "exit 0" ]
}

require_verify() {
  is_noop "$VERIFY_CMD" && die 2 "VERIFY_CMD is not configured (empty or a no-op)." \
    "an unconfigured gate would report success without verifying anything." \
    "edit init.sh and set VERIFY_CMD to the project's real test/check command."
  return 0
}

require_jq() {
  command -v jq >/dev/null 2>&1 || die 3 "jq is not installed." \
    "init.sh reads and updates $F with jq." "install jq, then re-run."
  jq -e '.features | type == "array"' "$F" >/dev/null 2>&1 \
    || die 4 "$F is not valid JSON with a features array." \
       "the gate needs the feature list to check and record features." "fix $F (run the scaffolder if absent)."
}

run_verify() {
  require_verify
  echo "==> [Harness] verify: $VERIFY_CMD"
  local rc=0; bash -c "$VERIFY_CMD" || rc=$?
  [ "$rc" -eq 0 ] || { echo "==> [Harness] FAILED: VERIFY_CMD exited $rc" >&2; exit 1; }
}

done_check() { # every done feature must carry exit=0 evidence; skipped without a feature list
  [ -f "$F" ] || return 0
  require_jq
  local bad
  bad="$(jq -r '[.features[] | select(.status=="done" and (((.evidence // "") | contains(" exit=0 ")) | not)) | .id] | join(", ")' "$F")"
  [ -z "$bad" ] || die 6 "done feature(s) without exit=0 evidence: $bad." \
    "done must be backed by a passing ./init.sh --feature run." \
    "run ./init.sh --feature <id> for each; if it fails, set status back to in-progress."
}

run_start() {
  if [ "${RUN_START_COMMAND:-0}" = "1" ] && [ -n "$START_CMD" ]; then
    echo "==> [Harness] start: $START_CMD"
    bash -c "$START_CMD"
  fi
}

first_line() {
  local l; l="$(printf '%s\n' "$1" | grep -m1 -v '^[[:space:]]*$' || true)"
  printf '%s' "${l:-(no output)}"
}

run_feature() {
  local id="$1" ts out="" rc code=0 first_fail="" ran cmd o commit attempts
  [ -f "$F" ] || die 4 "$F is missing." "--feature needs the feature list." "restore $F (run the scaffolder if absent)."
  require_jq
  jq -e --arg id "$id" '.features[] | select(.id==$id)' "$F" >/dev/null 2>&1 \
    || die 4 "feature '$id' not found in $F." "the id must match a features[].id entry." \
       "check the id in $F and re-run."
  require_verify
  ran="$(jq -r --arg id "$id" '[.features[] | select(.id==$id) | (.verification // [])[] | select(startswith("manual:") | not)] | length' "$F")"
  [ "$ran" -gt 0 ] || die 2 "feature '$id' has no non-manual verification command." \
    "without a runnable command, passing would prove nothing." \
    "add a runnable command to its verification array in $F."
  local cmds=()
  while IFS= read -r cmd; do cmds+=("$cmd"); done < <(jq -r --arg id "$id" '.features[] | select(.id==$id) | (.verification // [])[]' "$F")
  for cmd in "${cmds[@]}"; do
    case "$cmd" in
      manual:*) echo "==> [Harness] MANUAL (not run): ${cmd#manual:}"; out+="MANUAL:${cmd#manual:}"$'\n'; continue;;
    esac
    echo "==> [Harness] feature $id: $cmd"
    if o="$(bash -c "$cmd" 2>&1)"; then rc=0; else rc=$?; fi
    printf '%s\n' "$o"; out+="$o"$'\n'
    if [ "$rc" -ne 0 ] && [ -z "$first_fail" ]; then
      code=$rc
      first_fail="$cmd | $(first_line "$o")"
    fi
  done
  echo "==> [Harness] verify: $VERIFY_CMD"
  if o="$(bash -c "$VERIFY_CMD" 2>&1)"; then rc=0; else rc=$?; fi
  printf '%s\n' "$o"; out+="$o"$'\n'
  if [ "$rc" -ne 0 ] && [ -z "$first_fail" ]; then
    code=$rc
    first_fail="$VERIFY_CMD | $(first_line "$o")"
  fi
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  commit="$(git rev-parse --short HEAD 2>/dev/null || true)"; commit="${commit:-none}"
  local tail5; tail5="$(printf '%s' "$out" | { grep -v '^[[:space:]]*$' || true; } | tail -n 5 | paste -sd '|' - | sed 's/|/ \/ /g')"
  local tmp
  tmp="$(mktemp "${F}.XXXXXX")" || die 5 "could not create a temp file next to $F." \
    "evidence cannot be recorded atomically." "check write permission on the repo root."
  if ! jq --arg id "$id" --arg ts "$ts" --arg ev "$ts exit=$code commit=$commit cmds=$ran tail: $tail5" --arg lf "$first_fail" --argjson failed "$([ "$code" -eq 0 ] && echo false || echo true)" \
    '(.features[] | select(.id==$id)) |= (.evidence=$ev | .testedAt=$ts | .attempts=(if $failed then (.attempts // 0)+1 else 0 end) | .last_failure=$lf)' \
    "$F" > "$tmp"; then
    rm -f "$tmp"
    die 5 "could not update $F with evidence." "jq failed, so the result was not recorded." "fix $F (valid JSON, attempts numeric) and re-run."
  fi
  chmod --reference="$F" "$tmp" 2>/dev/null || chmod "$(stat -f %Lp "$F" 2>/dev/null || echo 644)" "$tmp"
  mv "$tmp" "$F"
  if [ "$code" -ne 0 ]; then
    attempts="$(jq -r --arg id "$id" '.features[] | select(.id==$id) | .attempts' "$F")"
    echo "==> [Harness] FAILED: $first_fail (exit $code, attempts $attempts)" >&2
    [ "$attempts" -lt 3 ] || echo "BLOCK: 3 consecutive failures for $id. Set status blocked, record why in progress.md, and escalate." >&2
    exit 1
  fi
  done_check
  run_start
}

case "${1:-}" in
  --feature)
    [ -n "${2:-}" ] || die 64 "--feature needs an id." "the gate must know which feature to verify." "run ./init.sh --feature <id>."
    run_feature "$2";;
  --bootstrap)
    if [ -n "$INSTALL_CMD" ]; then
      echo "==> [Harness] install: $INSTALL_CMD"
      bash -c "$INSTALL_CMD" || { echo "==> [Harness] FAILED: INSTALL_CMD" >&2; exit 1; }
    fi
    run_verify; done_check; run_start;;
  "") run_verify; done_check; run_start;;
  *) die 64 "unknown argument '$1'." "only --bootstrap and --feature <id> are supported." "use ./init.sh [--bootstrap] | ./init.sh --feature <id>.";;
esac
echo "==> [Harness] OK"
