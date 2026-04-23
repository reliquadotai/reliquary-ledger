# Response protocol

How we process findings from the external reviewer. Designed to be
predictable, fast, and publicly auditable — no private-discussion
backchannel the reviewer can't verify.

## Submission

- **Critical findings** (active attack on threat-model properties
  1-6): contact maintainers out-of-band FIRST via the channel
  named in the protocol paper's acknowledgements. Give us a
  fix-then-disclose window before filing publicly.
- **Major + minor findings**: file a GH issue on the relevant repo
  with label `audit-feedback` and the pinned commit SHA in the
  title. Example title: `audit-feedback: reparam guard misses
  diagonal scale (reliquary-inference@c827d5b)`.
- Multiple related findings: one issue per finding, grouped by a
  tracking issue. Easier to triage than a single mega-issue.

## SLA

| Severity | First acknowledgement | Triage decision | Patch landed (testnet) | Mainnet gate |
|---|---|---|---|---|
| Critical | ≤ 24 h | ≤ 72 h | ≤ 7 days | Required before cutover |
| Major    | ≤ 72 h | ≤ 7 days | ≤ 21 days | Documented triage path required |
| Minor    | ≤ 7 days | ≤ 21 days | Next release window | Not gating |

If we miss any SLA tier, the finding auto-escalates one severity
class. If Critical misses the 7-day testnet patch, the mainnet
cutover slips.

## Triage process

1. **Reproducer** — maintainer on duty reproduces the finding
   locally. If we can't reproduce, we ask the reviewer for an
   assist (attached data, exact command line, expected vs observed
   output).
2. **Classification** — the maintainer team agrees on severity per
   the threat-model taxonomy. Reviewer has final say if they
   disagree: if they write "Critical", it's Critical until a
   consensus mitigation lands.
3. **Mitigation scope** — spec-level fix vs implementation-level
   fix. Spec changes require a paper revision; implementation
   changes don't.
4. **Patch** — one PR, one test, one issue referenced. No hidden
   "related fix" bundles.
5. **Review** — the reporter gets a chance to verify the patch
   before landing; if silent for 72 h, we land anyway and
   document the review gap in the commit trailer.

## Disclosure timeline

- **Critical**: coordinated disclosure 7 days after testnet patch
  lands (the mainnet gate window). Reviewer's report can reference
  the finding immediately; we publish our mitigation write-up at
  the 7-day mark.
- **Major + minor**: public from day-0. We don't do embargoes for
  these; the security budget isn't large enough to justify the
  secrecy overhead.

## Writing up findings in the paper

Every accepted Critical + Major finding lands in the protocol paper's
§13 (Empirical validation) with:

- Reviewer credit (opt-in; anonymous attribution supported).
- Commit SHA pre-fix + post-fix.
- One-paragraph summary of the attack class.
- Pointer to the test that now prevents regression.

We do not redact. If the finding is embarrassing, it's an even
better argument for the paper; a protocol that publishes its own
attack history is trustworthier than one that doesn't.

## What we don't do

- **Private bounty negotiations.** Bounty structure is flat +
  documented in the governance charter. No "if you find something
  big, we'll talk about it" side-deals.
- **Silent patches.** Every Critical + Major fix has a commit
  message that names the finding, its severity, and the reporter
  (with consent).
- **NDAs.** No review work is gated behind an NDA. If the reviewer
  wants to anonymize, we support that via a pseudonymous GH
  account or a co-author trailer on the paper.

## For reviewers who find nothing

A "clean review" is itself a deliverable. The report structure from
[`scope.md`](scope.md) still applies — methodology, attacks
attempted, attacks NOT attempted, sign-off. A clean review credits
the reviewer in the paper's §13 with the same attribution as a
finding would.

## Contact

- **Out-of-band channel** for Critical findings: named in the
  protocol paper's acknowledgements (currently the maintainer
  Keybase / Signal handle pinned in the paper header).
- **Public GH issues**: `audit-feedback` label on the relevant
  repo. Issues on `reliquary-protocol` are cross-linked into the
  other three repos by the weekly triage pass.
- **Maintainer on-call rotation**: documented in the operator
  runbook; reviewer doesn't need to know names, just the
  out-of-band handle.

---

*This protocol is part of the Tier 4 Epic 1 governance charter
commitment. It is versioned with the protocol package; changes
require a protocol version bump.*
