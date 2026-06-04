# Secret Rotation Procedure

## GH_PAT (GitHub Personal Access Token)

### Scope
Fine-grained PAT. Permissions: **Contents: Read** on `mefiezvous/ml-core` only.
`robotics-platform-template` is cloned over public HTTPS and does not require a token.

### Rotation cadence
Every **90 days**.

### How to rotate

1. Go to <https://github.com/settings/personal-access-tokens> and generate a new
   fine-grained token named `lerobot-playground-ci` with:
   - Resource owner: `mefiezvous`
   - Repository access: only `mefiezvous/ml-core`
   - Permissions: **Contents → Read-only**
   - Expiration: 90 days from today

2. In the `mefiezvous/lerobot-playground-portfolio` repository go to
   **Settings → Secrets and variables → Actions** and update the `GH_PAT` secret
   with the new token value.

3. Delete the old token from GitHub Personal Access Tokens.

4. Update the "Last rotated" date in this file and open a `chore(sec): rotate GH_PAT`
   commit on `main`.

### Rotation log

| Date | Rotated by | Notes |
|------|-----------|-------|
| 2026-06-03 | Arthur Mouraud | Initial token documented (LRB-001) |
</content>
</invoke>