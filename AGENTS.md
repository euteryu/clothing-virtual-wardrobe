# Project instructions

Read `PROJECT_BRIEF.md`, `EXPERIMENT_SPEC.md`, and `LOGBOOK.md` before changes.
Treat the latest logbook entry as the durable handover.

- Git is the source of truth; Kaggle notebooks are thin command runners.
- Keep `/kaggle/input` read-only and write artifacts under `/kaggle/working`.
- Use one T4 initially and verify CUDA with a real tensor computation.
- Preserve the deterministic image-level train/validation manifest. Never tune on test.
- Start with transfer learning and the smallest run that answers the question.
- Scripts must accept paths; never hard-code a Kaggle dataset slug.
- Do not commit data, credentials, predictions, or model weights.
- Run `pytest -q` and `ruff check .` after code changes.
- Record meaningful runs, failures, and decisions in `LOGBOOK.md`.
