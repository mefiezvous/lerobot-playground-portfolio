# /new-policy — Scaffold a New Policy Wrapper

When the user runs `/new-policy`, do the following:

1. Ask: what is the policy name? (e.g., `smolvla`, `pi0`)
2. Create `src/playground/policies/<name>_wrapper.py` with:
   - SPDX header (Apache-2.0, Arthur Mouraud)
   - Class `<Name>PolicyWrapper` with methods:
     - `from_pretrained(cls, repo_id: str) -> <Name>PolicyWrapper`
     - `select_action(self, obs: Observation) -> Action`
   - Docstrings (Google style)
   - Type hints (mypy strict)
3. Create `configs/policy/<name>.yaml` with Hydra config stub
4. Add test stub to `tests/test_policies.py`
5. Add row to README.md Results table (TBD values)
6. Commit: `feat(policy): scaffold <name> wrapper`
