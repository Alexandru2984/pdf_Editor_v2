# Service Level Objectives

Four SLOs cover the user-facing behaviours that matter: can the API be
reached, does it respond quickly, do async jobs eventually succeed, and
do PDF ops produce correct output. Each SLO has a numeric target, an
SLI defined as a PromQL ratio, a 30-day rolling window, and a pair of
multi-window burn-rate alerts wired into Grafana.

The targets aren't aspirational — they reflect what the current single-
VPS deployment can sustain. If they get tightened later (e.g. moving
from one VPS to an HA setup), the burn-rate alert thresholds re-derive
mechanically from the new error budget.

## Why these four

| SLO                 | What it protects             | Who feels a breach          |
| ------------------- | ---------------------------- | --------------------------- |
| API availability    | Site is reachable at all     | Every user, immediately     |
| API latency         | Site doesn't feel broken     | Interactive users, web + UI |
| Job success rate    | Async work finishes          | Users waiting on results    |
| PDF op success rate | Output isn't garbage         | Anyone downloading a result |

The mix is deliberate. Availability + latency capture the *frontend*
experience (the part synthetic monitoring would notice). Job + op
success rates capture the *backend* experience — a request can return
200 OK and still produce a broken PDF an hour later. Without both, a
green dashboard can hide real damage.

## SLO 1 — API availability

| Field          | Value                                                            |
| -------------- | ---------------------------------------------------------------- |
| Target         | **99.9%** of non-5xx responses, rolling 30 days                  |
| Error budget   | 0.1% × 30d = **43m12s** of effective outage per month            |
| SLI numerator  | `sum(rate(django_http_responses_total_by_status_view_method_total{status!~"5.."}[<win>]))` |
| SLI denominator| `sum(rate(django_http_responses_total_by_status_view_method_total[<win>]))` |
| Bad-event rate | `sum(rate(django_http_responses_total_by_status_view_method_total{status=~"5.."}[<win>])) / sum(rate(django_http_responses_total_by_status_view_method_total[<win>]))` |

5xx counts only — 4xx is the client's fault, not ours, and including
it would let auth-spam tank our SLO. `/metrics` and `/healthz` are
scraped frequently and would dominate the denominator at low traffic;
they're acceptable in the ratio because they should never 5xx.

## SLO 2 — API latency

| Field          | Value                                                            |
| -------------- | ---------------------------------------------------------------- |
| Target         | **95%** of requests served in **< 1.0s**, rolling 30 days        |
| Error budget   | 5% of requests may exceed 1s                                     |
| SLI numerator  | `sum(rate(django_http_requests_latency_seconds_by_view_method_bucket{le="1.0"}[<win>]))` |
| SLI denominator| `sum(rate(django_http_requests_latency_seconds_by_view_method_count[<win>]))` |
| Bad-event rate | `1 - SLI`                                                        |

1s is the threshold above which interactive users notice the app feels
sluggish (≈ Nielsen's classic 1s/10s rule). PDF ops that legitimately
take longer run async via Celery — those are governed by SLO 3, not
this one.

## SLO 3 — Job success rate

| Field          | Value                                                            |
| -------------- | ---------------------------------------------------------------- |
| Target         | **99%** of completed jobs succeed, rolling 30 days               |
| Error budget   | 1% of jobs may fail                                              |
| SLI numerator  | `sum(rate(pdfeditor_job_total{status="done"}[<win>]))`           |
| SLI denominator| `sum(rate(pdfeditor_job_total[<win>]))`                          |
| Bad-event rate | `sum(rate(pdfeditor_job_total{status="failed"}[<win>])) / sum(rate(pdfeditor_job_total[<win>]))` |

Canceled jobs are *not* counted — they're user-initiated, not failures.
The counter is incremented exactly once per job in `tasks.py`, so the
denominator equals the number of jobs that reached a terminal state in
the window.

## SLO 4 — PDF op success rate

| Field          | Value                                                            |
| -------------- | ---------------------------------------------------------------- |
| Target         | **99.5%** of PDF operations succeed, rolling 30 days             |
| Error budget   | 0.5% of ops may fail                                             |
| SLI numerator  | `sum(rate(pdfeditor_op_total{outcome="success"}[<win>]))`        |
| SLI denominator| `sum(rate(pdfeditor_op_total[<win>]))`                           |

Op success is the lowest-level signal that an output PDF was actually
produced. It covers both sync ops (counted in `views/_common.py`) and
async ops (counted in `tasks.py`), so a regression that only breaks one
path still shows up. Stricter than the job SLO because most ops are
called multiple times per job — at 99% job success, op success has to
be higher than 99% just to keep the math working.

## Burn-rate alerts

Each SLO has two paged alerts following Google's SRE Workbook multi-
window/multi-burn-rate pattern. Single-window threshold alerts either
fire on every blip (5-minute spike) or take days to notice a slow burn;
multi-window catches both regimes with one false-alarm rate.

| Severity | Window | Burn rate | Budget consumed | Time-to-fire (worst case) |
| -------- | ------ | --------- | --------------- | ------------------------- |
| critical | 1h     | 14.4×     | 2% in 1h        | ≈ 5 min                   |
| warning  | 6h     | 6×        | 5% in 6h        | ≈ 30 min                  |

Burn rate = (bad-event rate observed) ÷ (error budget). At 14.4× a
99.9% SLO, the bad-event rate is ≥ 1.44%; at 6× it's ≥ 0.6%. The
warning is the early-warning channel — it fires *before* a full outage
has consumed enough budget to need an immediate response, but late
enough that intermittent jitter doesn't trip it.

PromQL threshold for each (substitute the bad-event rate from the SLO
section above for `<bad_rate(<win>)>`):

```promql
# critical (1h window)
<bad_rate(1h)> > 14.4 * (1 - <target>)

# warning (6h window)
<bad_rate(6h)> > 6 * (1 - <target>)
```

For example, for SLO 1 (target 99.9%, so 1-target = 0.001):

- Critical: 1h 5xx rate > 1.44%
- Warning : 6h 5xx rate > 0.6%

Provisioned in `docker/grafana/provisioning/alerting/rules.yml`. The
non-SLO alerts already there (replicas down, queue depth) remain — they
catch operational conditions that don't map cleanly to a user-facing
SLO.

## Reading the dashboard

The "PDF Editor — SLOs" dashboard (`docker/grafana/dashboards/slo.json`)
shows for each SLO:

1. Current 30-day SLI as a gauge against the target.
2. Error budget remaining (%).
3. Burn rate over 1h and 6h windows.

If both burn-rate panels are green and the SLI gauge is above target,
nothing user-visible is wrong. If a burn-rate panel goes red, expect a
page within minutes (1h) or half an hour (6h).

## Reviewing the targets

Targets get reviewed every quarter. The forcing function is the error
budget: if we burn through it before the 30 days are up, either the
target was too aggressive or we shipped something that regressed
reliability. Repeated breaches mean the SLO needs tightening (we got
better and the bar should rise) or loosening (we're paying for a level
of reliability users don't actually need). The dashboard makes both
trends obvious without needing to run ad-hoc queries.
