# AGENTS.md

## Project

FastAPI sensory server with STT (sherpa-whisper), TTS (Piper ONNX), vision and OCR routers. Hexagonal layout (adapters/domain/routers).

## Run

`./run_sensory_server.sh`. `init.sh` runs `START_CMD` only with `RUN_START_COMMAND=1`, after verify passes.

## Verify

`./init.sh`

Note: `./download_models.sh` is a manual step and is not part of verify.
Known gap: vision and OCR have no tests (out of scope here).

`./init.sh` runs `VERIFY_CMD` (set it in init.sh); `./init.sh --bootstrap` runs `INSTALL_CMD` first. Both need jq when feature_list.json exists (done-check).
`./init.sh --feature <id>` runs that feature's `verification` commands plus `VERIFY_CMD` and records evidence/attempts in feature_list.json. It prints `manual:` steps but does not run them.
Exit codes: 1 = a check failed (raw code in evidence), 2 = VERIFY_CMD unconfigured/no-op or feature has no runnable command, 3 = jq missing, 4 = unknown feature id / bad feature_list.json, 5 = evidence write failed, 6 = a done feature lacks exit=0 evidence, 64 = bad arguments.

## Hard Constraints

- Keep the hexagonal layout: adapters/ for I/O, domain/ for logic, routers/ for HTTP.
- Never commit model weights (see .gitignore); fetch them with `./download_models.sh`.

## Topic Docs

Read topic docs only when the task needs them.

| Path | Read when... |
|------|--------------|
| `README.md` | setting up models or running the server |
| `docs/plans/` | executing a recorded plan |

## Startup

1. `pwd`; read `progress.md` and `feature_list.json`; `git log --oneline -5`
2. `./init.sh` -- if red, fix the baseline first
3. Pick a feature: resume `in-progress` first, else the lowest `priority` number whose dependencies are done.
4. Run `git status --porcelain`; note pre-existing dirty files in progress.md.

## Working Rules

- `status` ∈ not-started|in-progress|blocked|done. One feature at a time; touch only the files it needs.
- Never weaken or delete tests to get green. Never change a feature's `verification` silently; record it in progress.md.
- Never rewrite feature_list.json to hide unfinished work.
- A narrow supporting fix is allowed if it blocks the feature; note it in progress.md.
- `./init.sh --feature` records evidence only and never changes `status`; you set it.
- Manual steps: perform them and log `manual: <step> -> <result>` in progress.md.
- `init.sh` prints BLOCK at 3 consecutive failures: set status blocked, record why in progress.md, escalate.
- Independent review when a change crosses components or modifies existing tests, init.sh, or any feature's verification: a fresh-context reviewer reruns `--feature`; log `review: pass|fail` and who reviewed in progress.md.
- Discard only via `git restore -- <files>` of this feature, only if those files were clean at the start; never `git reset --hard` or `git checkout .`.
- Same review feedback twice: add a check to `VERIFY_CMD` instead of repeating the comment.

## Definition of Done

- `./init.sh --feature <id>` exited 0 in this session AND every `manual:` step is logged in progress.md AND (if review applies) `review: pass` is logged; then set status to done.
- Plain `./init.sh` fails (exit 6) on a done feature without exit=0 evidence.
- A change across components includes an end-to-end/runtime step in `verification`.
- Propose a commit; commit only if the user or project allows it.

## End of Session

1. Verify is green (`./init.sh`).
2. progress.md is updated.
3. feature_list.json matches reality.
4. No temp or debug artifacts left.
5. The startup path above works from a clean checkout.

Unfinished work is recorded as unfinished, never "done". For multi-session work, fill in `session-handoff.md`.
