# Harness Compliance Implementation Plan (rev 2, after Goldfish review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `services/sensory-server` follow the harness-workflow skill. Independent project: does not touch anything outside its own repo.

**Architecture:** Use `~/.claude/skills/harness-workflow/scripts/scaffold-harness.sh` (never `--force`) to create the missing files. entorno_rylai keeps its own names (`features.json`, `PROGRESS.md`): `init.sh` points at them with `F="features.json"`, and the fields are migrated in place.

**Tech Stack:** bash, jq, nix-shell, pytest.

**Spec:** `~/.claude/skills/harness-workflow/SKILL.md` + researcher audits and Goldfish reviews (2026-11-15).

## Global Constraints

- All commands run from `/home/rylai/Documents/proyectos/pet`. `pet/` is not a git repo; each project has its own repo.
- Never run `scaffold-harness.sh --force`.
- Only `./init.sh --feature <id>` writes ` exit=0 ` into `evidence`. Never write it by hand.
- Commits: no `Co-Authored-By` trailer. Use branch `chore/harness-compliance` in each repo.
- `init.sh` runs `cd "$(dirname "$0")"`, so every `verification` command is relative to the project root.

## Review Focus

- The `features.json` migration must be lossless: 31 ids, compared against `git show HEAD:features.json`.
- No `verification` command may contain `entorno_rylai/` or `../` paths, or a bare `nix-shell` with no shell file.
- `VERIFY_CMD` does not download models or start servers.
- A plain `./init.sh` must exit 0 at the end of each project.
- `review: pass` is logged **before** setting `done` on a feature that needs review.

---
## Project A: services/sensory-server

### Task A1: Scaffold and gate

- [ ] **Step 1: Create the branch and scaffold.**
  `git -C services/sensory-server switch -c chore/harness-compliance && bash ~/.claude/skills/harness-workflow/scripts/scaffold-harness.sh services/sensory-server`
  This creates `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `init.sh` and `session-handoff.md`.
- [ ] **Step 2: Confirm the gate starts red.** `services/sensory-server/init.sh` → exit 2.
- [ ] **Step 3: Set the check command.** In `init.sh`, set `VERIFY_CMD="nix-shell shell.nix --run 'python -m pytest tests -q'"`. `.venv` does not exist yet: the first entry into `shell.nix` creates it and runs `pip install`, which needs network access but downloads no models.
- [ ] **Step 4: Confirm the gate is green.** `./init.sh` → exit 0. If it fails, use systematic-debugging before continuing.

### Task A2: AGENTS.md, features and review

- [ ] **Step 1: Fill in `AGENTS.md`.**
  - **Project:** FastAPI sensory server with STT (sherpa-whisper), TTS (Piper ONNX), vision and OCR routers. Hexagonal layout (adapters/domain/routers).
  - **Run:** `./run_sensory_server.sh`
  - **Verify:** `./init.sh`
  - Note: `./download_models.sh` is a manual step and is not part of verify.
  - Known gap: vision and OCR have no tests (out of scope here).
- [ ] **Step 2: Fill in `feature_list.json`.** Use `P="nix-shell shell.nix --run 'python -m pytest tests/test_sensory_endpoints.py -q -k"`. Every feature starts with `status:"in-progress"`, `attempts:0` and `last_failure:null`.

  | id | name | verification `-k` expression |
  |---|---|---|
  | SS-001 | Health + request-id middleware | `health_check or request_logging` |
  | SS-002 | TTS Piper PCM (+ legacy alias, abort, voices) | `speech or tts_legacy or voices` |
  | SS-003 | STT transcriptions | `transcriptions or stt_json` |
  | SS-004 | v1 OpenAPI contract / deprecation | `openapi_versioning` |

- [ ] **Step 3: Verify each feature.** Run `for id in SS-001 SS-002 SS-003 SS-004; do ./init.sh --feature $id; done` from inside the project. Every run must exit 0.
- [ ] **Step 4: Independent review (Step 3b).** The change touches `init.sh` and the verifications, so a review is required. Give a fresh `engineering-agents:reviewer` only:
  - `git diff main -- init.sh feature_list.json AGENTS.md`;
  - the feature list.

  The reviewer reruns `./init.sh --feature SS-001`. Log `review: pass|fail — reviewer: engineering-agents:reviewer — <file:line findings>` in `progress.md`.
- [ ] **Step 5: Mark done and commit.** Only if the review passed: set the 4 features to `done`, then run `./init.sh` → exit 0. Commit with message `chore(harness): adopt harness-workflow`.

---

## Follow-ups
- Tests for the vision and OCR routers.
