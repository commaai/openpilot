# Proof of Usage Record

This file records the automated systems, models and agents that have processed this repository
under the conditions in [LICENSE](LICENSE).

The condition has two halves. This file is one of them; the other is the credits of whatever the
usage produced. **The Provenance Hash is what ties them together — the same string appears in the
row below and in the product's credits, and the two must match.**

One row per access. Newest at the bottom. If a system read this repository repeatedly for the same
purpose, a single row covering the period is enough.

| System | Operator | Date and Time (UTC) | What | Purpose | Contact | Provenance Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ExampleModel v2 | AI Corp | 2026-08-12T14:30:00Z | whole repository | training | contact@aicorp.com | `471e3dc7467c3c9f83be8199e5ed76b2635a0aefc86b2e3679ffc836fc9c741c` |

<!-- The example row's hash is the real digest of its own four fields against
     https://github.com/commaai/openpilot, so it does not match this repository's URL - it is an
     illustration. Yours is checked: the workflow recomputes it and says so if it disagrees. -->

## Field reference

| Field | Meaning |
| :--- | :--- |
| **System** | Model or product name and version |
| **Operator** | The organisation or person running it |
| **Date and Time (UTC)** | When the access happened, ISO 8601 — `2026-08-12T14:30:00Z` |
| **What** | Files, paths, or "whole repository" |
| **Purpose** | Training, fine-tuning, retrieval, inference, code generation, evaluation |
| **Contact** | An address a question can be sent to |
| **Provenance Hash** | The handshake — see below |

## The handshake hash

```
SHA-256("SystemName:OperatorName:ISODate:TargetRepositoryURL")
```

Four fields, each trimmed of surrounding whitespace, joined by a single colon, encoded as UTF-8.
The digest is written in lowercase hexadecimal. The contact address is deliberately not part of
the input, so that changing it does not invalidate a record already published.

Compute it however you like:

```bash
npx proof-of-usage hash \
  --system "ExampleModel v2" \
  --operator "AI Corp" \
  --date "2026-08-12T14:30:00Z" \
  --repo "https://github.com/commaai/openpilot"
```

```bash
printf '%s' "ExampleModel v2:AI Corp:2026-08-12T14:30:00Z:https://github.com/commaai/openpilot" | shasum -a 256
```

```python
import hashlib
hashlib.sha256("ExampleModel v2:AI Corp:2026-08-12T14:30:00Z:https://github.com/commaai/openpilot".encode()).hexdigest()
```

## Both halves

1. **Here.** Fork this repository, check out the provenance branch (`master`), append your
   row, open a pull request against that branch.
2. **There.** In the credits your product already publishes — credits screen, about page,
   acknowledgements, model card, dataset card, documentation — name this repository and print the
   same hash. Something like:

   ```
   Includes material from commaai/openpilot — https://github.com/commaai/openpilot
   Proof of usage: 471e3dc7467c3c9f83be8199e5ed76b2635a0aefc86b2e3679ffc836fc9c741c
   ```

A reader who finds one half can check it against the other. That is the whole mechanism: no
tracking, no phoning home, nothing hidden — two public strings that either match or do not.

The workflow on pull requests recomputes the hash from the row's own fields and rejects a row
whose fingerprint does not match. It cannot check whether the row is true; that part is on the
operator, which is the point.
