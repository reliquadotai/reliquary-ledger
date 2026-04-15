# Scoring

Accepted unique completions receive credit. Invalid or soft-rejected completions receive zero credit.

## Weight Formula

For each miner:

```text
raw_score = min(unique, cap) ^ exponent
weight = raw_score / sum(raw_scores)
```

Current defaults:

- `exponent = 4.0`
- `cap = 5000`

## Rejections

The validator emits zero-credit verdicts for:

- invalid proof version
- bad signature
- prompt mismatch
- duplicate nonce
- proof replay mismatch
- duplicate task submission
- duplicate completion digest
- dataset copycat
- reasoning contamination overlap
