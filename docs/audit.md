# Audit Surface

`reliquary-inference` publishes a lightweight static audit surface from finalized `window_manifest` artifacts.

## Stable Outputs

- `audit/index.json`
- `audit/index.html`

These summarize:

- recent finalized windows
- task source
- accepted and submitted counts
- hard and soft failures
- per-window weights
- manifest references suitable for public browsing

## Local Build

```bash
reliquary-inference build-audit-index --limit 25
```

This writes local exports under:

- `RELIQUARY_INFERENCE_EXPORT_DIR/audit/index.json`
- `RELIQUARY_INFERENCE_EXPORT_DIR/audit/index.html`

## Publish

```bash
bash deploy/publish-audit-index.sh
```

Or directly:

```bash
reliquary-inference build-audit-index --limit 25 --publish
```

## Recommended Bucket Posture

Preferred model:

1. keep the main artifact bucket private
2. publish audit pages to a separate public bucket or domain
3. keep raw artifact URLs hidden unless you intentionally want them public

The repo supports this through:

- `RELIQUARY_INFERENCE_AUDIT_BUCKET`
- `RELIQUARY_INFERENCE_AUDIT_ENDPOINT_URL`
- `RELIQUARY_INFERENCE_AUDIT_ACCESS_KEY_ID`
- `RELIQUARY_INFERENCE_AUDIT_SECRET_ACCESS_KEY`
- `RELIQUARY_INFERENCE_PUBLIC_AUDIT_BASE_URL`
- `RELIQUARY_INFERENCE_EXPOSE_PUBLIC_ARTIFACT_URLS`

You can populate the audit credentials directly with S3-compatible values, or derive them at deploy time with `deploy/apply-audit-profile.sh` when a Cloudflare account token is available.

## Security Default

Raw artifact URLs are suppressed by default even when a public audit base URL is configured. This keeps the public audit surface useful without implying that raw artifacts should be directly browsable.
