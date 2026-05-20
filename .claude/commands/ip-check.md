# /ip-check — IP Guardian Audit

When the user runs `/ip-check`, invoke the `ip-guardian` sub-agent to:

1. Run grep audit on all staged and tracked files:
   - Pattern: `_private/` → must be ZERO matches
   - Pattern: `my-robot-stack/` → must be ZERO matches
   - Pattern: `proprietary_` → must be ZERO matches (filenames + content)
   - Pattern: `LicenseRef-Proprietary` → must be ZERO matches
   - Pattern: `All Rights Reserved` → must be ZERO matches
2. Check SPDX headers in all modified `.py` files:
   - Must have: `# SPDX-FileCopyrightText: 2026 Arthur Mouraud`
   - Must have: `# SPDX-License-Identifier: Apache-2.0`
3. Report:
   - PASS: all checks clean
   - FAIL: list every file and pattern that triggered

Run: `bash .git-hooks/pre-commit-anti-leak.sh`
