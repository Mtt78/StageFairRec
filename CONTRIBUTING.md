# Contributing

Use Python 3.10 or later. Install `pip install -e '.[dev]'` and run `pytest -q`
and `python scripts/smoke.py` before proposing changes. Keep tests focused on
mathematical behavior, split visibility, gradient ownership and ranking protocols.

Describe whether each change follows an explicit manuscript equation, resolves
an unspecified implementation choice, or intentionally changes the method.
Do not commit learner data, semantic caches, credentials, model weights or
unverified paper performance claims. Report failures with configuration, random
seed and synthetic reproduction steps, without personal learner records.
