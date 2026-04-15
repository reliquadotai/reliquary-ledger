# Protocol

## Artifact Envelope

All records are plain JSON dictionaries:

```json
{
  "artifact_id": "sha256...",
  "artifact_type": "task_batch | completion | verdict | scorecard | window_manifest | run_manifest",
  "producer_id": "miner-or-validator-id",
  "producer_role": "miner | validator | task_source | operator",
  "window_id": 0,
  "created_at": "2026-04-15T00:00:00+00:00",
  "parent_ids": [],
  "payload": {}
}
```

## Completion Payload

- `task_id`
- `task_source`
- `task_index`
- `prompt_hash`
- `tokens`
- `prompt_length`
- `completion_text`
- `completion_digest`
- `model_name`
- `nonce`
- `proof_version`
- `layer_index`
- `commitments`
- `signature`
- `signature_scheme`
- `signer_id`
- `randomness`
- `upload_ref`

## Verdict Payload

- `completion_id`
- `accepted`
- `hard_fail_reason`
- `soft_fail_reason`
- `proof_summary`
- `task_binding_summary`
- `copycat_summary`
- `task_source`
