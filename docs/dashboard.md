# Dashboard

The metrics HTTP server at `:9108` serves four surfaces from one process:

| Path | Content-Type | Purpose |
|---|---|---|
| `/metrics` | text/plain | Prometheus text format — for Grafana scrapes |
| `/healthz` | application/json | Lightweight liveness + freshness probe |
| `/status` | application/json | Full operator state (superset of `/healthz`) |
| `/` + `/dashboard/` | text/html | Static single-page HTML dashboard |

All endpoints emit `Access-Control-Allow-Origin: *` — a third-party
website can fetch them directly without proxying.

## Built-in dashboard

`dashboard/index.html` ships with the repo. `:9108/` serves it. It polls
`/status` + `/healthz` every 10 s and renders:

- Chain state (current block, window, metagraph visibility, scrape age)
- Rolling mining / validation totals (submitted / accepted / hard+soft fail)
- DAPO zone filter gauges (total / in-zone / out-of-zone, in-zone rate,
  mean σ, mean reward)
- Reasoning (MATH) correctness + format-OK + policy-compliance rates
- Runtime: git SHA, model ref, task source, storage backend
- Per-task-source totals table

The health dot at the top flips green / yellow / red based on
`chain_scrape_age_seconds`.

## Customizing

The page accepts a `?endpoint=<url>` query argument so operators can
point one dashboard at a remote validator:

```
https://dashboard.example.com/?endpoint=https://validator.example.com:9108/status
```

## Prometheus gauges

See `reliquary_inference/metrics.py` for the full list. Key gauges:

| Gauge | Meaning |
|---|---|
| `reliquary_rolling_submitted_total` | submitted completions across the rolling audit window |
| `reliquary_rolling_accepted_total` | accepted (passed every check) |
| `reliquary_rolling_hard_failed_total` / `_soft_failed_total` | fail-bucket totals |
| `reliquary_rolling_acceptance_rate` | accepted / submitted |
| `reliquary_rolling_reasoning_correct_rate` | MATH correctness |
| `reliquary_rolling_reasoning_format_rate` | `\boxed{…}` present |
| `reliquary_rolling_zone_total_groups` | rollout groups scored by zone filter |
| `reliquary_rolling_zone_in_zone_groups` | groups with σ ≥ σ_min |
| `reliquary_rolling_zone_out_of_zone_groups` | groups dropped |
| `reliquary_rolling_zone_in_zone_rate` | **single best "is it working"** |
| `reliquary_rolling_zone_mean_sigma` | mean within-group σ |
| `reliquary_rolling_zone_mean_reward` | miner correctness proxy |
| `reliquary_rolling_zone_windows_observed` | audit freshness |
| `reliquary_latest_window_mined` | highest window with mined completions |
| `reliquary_latest_importable_window` | highest finalized window |
| `reliquary_import_lag_windows` | how many windows validators are behind chain |
| `reliquary_chain_current_block` | current chain block height |
| `reliquary_chain_window_id` | current chain window id |
| `reliquary_chain_scrape_age_seconds` | freshness of chain state |
| `reliquary_metagraph_size` | number of hotkeys on the subnet |
| `reliquary_subnet_visible` | 1 iff metagraph was readable this tick |
| `reliquary_audit_window_count` | audit-index window count |

## Healthy value ranges

| Gauge | Green | Yellow | Red |
|---|---|---|---|
| `chain_scrape_age_seconds` | < 60 | 60-180 | > 180 |
| `import_lag_windows` | ≤ 2 | 3-5 | > 5 |
| `acceptance_rate` | ≥ 0.7 | 0.3-0.7 | < 0.3 |
| `zone_in_zone_rate` | ≥ 0.3 | 0.1-0.3 | < 0.1 |
| `zone_mean_reward` | ≥ 0.3 | 0.1-0.3 | < 0.1 |

## Audit-index dependency

The rolling gauges compute their data from the periodic audit index
rebuild. If `reliquary-audit-index.timer` isn't running, rolling gauges
will stay at 0.

Systemd timer cadence: 10 min. Rebuild takes ~60 s on a warm R2 path
with the key-sorted listing. On a cold start (cursor empty) the first
rebuild can take 10-15 min — be patient.

## JSON sample

```json
{
  "rolling_zone_total_groups": 5.0,
  "rolling_zone_in_zone_groups": 3.0,
  "rolling_zone_in_zone_rate": 0.6,
  "rolling_zone_mean_sigma": 0.229,
  "rolling_zone_mean_reward": 0.875,
  "rolling_zone_windows_observed": 5.0,
  "rolling_accepted_total": 35.0,
  "rolling_submitted_total": 40.0,
  "audit_window_count": 5.0,
  "chain_current_block": 6963121,
  "chain_window_id": 6963090,
  "latest_window_mined": 6963060,
  "import_lag_windows": 1,
  "task_source_totals": {
    "math": {"submitted": 40.0, "accepted": 35.0, "correct_total": 35.0, "format_ok_total": 40.0}
  },
  "runtime": {
    "network": "test",
    "netuid": 462,
    "model_ref": "Qwen/Qwen2.5-3B-Instruct",
    "task_source": "math",
    "storage_backend": "r2_rest",
    "git_sha": "<sha>"
  },
  "generated_at": 1776880000.0
}
```

## Polling guidance

- **Dashboard** polls every 10 s (configurable in-page).
- **Prometheus** should scrape every 15-30 s (exporter cache refresh is 30 s).
- **Don't hammer**: the `/status` cache TTL is
  `RELIQUARY_INFERENCE_METRICS_REFRESH_INTERVAL` (default 30 s). Faster
  polling just serves cached snapshots.
- **R2**: dashboards that hit R2 keys directly should cache responses for
  ≥ 60 s. R2 rate-limits at ~1 req/s per account.
