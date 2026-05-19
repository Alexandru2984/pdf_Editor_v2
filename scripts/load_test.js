// Load test for the PDF Editor stack.
//
// Why we bother with this: SLOs are claims ("p95 < 1s, 99.9% availability").
// Without a load test those claims are guesses. This script ramps real
// HTTP traffic against the running stack, fails the run if the SLO
// targets are missed, and gives us a baseline number to detect regressions
// against on future deploys.
//
// What it does NOT do: drive write-heavy ops (upload, compress). Those
// are job-based and not bound by the API-latency SLO — they're bound by
// the job-success-rate SLO, which we exercise separately via the smoke
// suite. This script focuses on the synchronous request path.
//
// ---- Run it ----------------------------------------------------------------
//
// Against the dev stack:
//   BASE_URL=http://localhost:8001 k6 run scripts/load_test.js
//
// Against prod (be sane — coordinate first, this WILL show up in Grafana):
//   BASE_URL=https://pdf.micutu.com k6 run scripts/load_test.js
//
// With an API key to exercise the authenticated read path:
//   BASE_URL=http://localhost:8001 API_KEY=pdfk_xxx k6 run scripts/load_test.js
//
// Throttling: the API endpoints are rate-limited per-IP at 60/h anon and
// 1000/h per api_key (see DEFAULT_THROTTLE_RATES in settings.py). For a
// run with realistic VU counts (>50), restart the stack with
// RATELIMIT_ENABLE=0 first — otherwise k6 measures the throttle, not the
// app. 429s are treated as expected responses here (the throttle is
// designed behavior, not a failure) but they crowd out signal.
//
// See docs/LOAD_TESTING.md for what to read out of the run and what
// "passing" looks like.

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Default http_req_failed in k6 flags any non-2xx as a failure. Our SLO 1
// measures only 5xx + network errors — 4xx (including 429) means "client
// is wrong / hit a limit", not "service is unavailable". Redefine the
// failure predicate so http_req_failed lines up with the SLO numerator.
http.setResponseCallback(http.expectedStatuses({ min: 200, max: 499 }));

// Custom metric so we can split anon vs authed failures in the summary.
// Built-in http_req_failed lumps everything together; this gives us a
// per-scenario view when diagnosing a regression.
const authFailureRate = new Rate('auth_request_failures');

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || '';

// Django's SecureRedirectMiddleware will 301 plain-HTTP requests to HTTPS
// unless it sees a trusted proxy header. When testing against the local
// docker stack (HTTP only — host nginx terminates TLS in prod, not this
// container), we have to pretend to be that trusted proxy or k6 measures
// redirect latency instead of app latency. Harmless in prod: the header
// is already correct in real traffic.
const COMMON_HEADERS = { 'X-Forwarded-Proto': 'https' };

// SHORT=1 collapses the run to ~45s. Use it for "does it run at all" smoke
// checks; never use it as a baseline — the hold window is too short to
// produce meaningful p95 numbers.
const SHORT = __ENV.SHORT === '1';

const anonStages = SHORT
    ? [{ duration: '5s', target: 5 }, { duration: '20s', target: 10 }, { duration: '5s', target: 0 }]
    : [
        { duration: '30s', target: 10 },  // warm up
        { duration: '1m',  target: 50 },  // ramp
        { duration: '2m',  target: 50 },  // hold — this is the SLO-validating window
        { duration: '30s', target: 0  },  // ramp down
      ];
const authStages = SHORT
    ? [{ duration: '5s', target: 2 }, { duration: '20s', target: 5 }, { duration: '5s', target: 0 }]
    : [
        { duration: '30s', target: 5  },
        { duration: '1m',  target: 20 },
        { duration: '2m',  target: 20 },
        { duration: '30s', target: 0  },
      ];

export const options = {
    // Two scenarios in parallel. anon_browse is the bulk of real traffic
    // (HTML pages); auth_read exercises the DRF stack with a session-y
    // workload. Both ramp on the same clock so we can compare them.
    scenarios: {
        anon_browse: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: anonStages,
            gracefulRampDown: '10s',
            exec: 'anonBrowse',
        },
        auth_read: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: authStages,
            gracefulRampDown: '10s',
            exec: 'authRead',
        },
    },

    // Thresholds that map back to our SLOs (docs/SLOs.md). If a threshold
    // fails, k6 exits non-zero — that's the SLO regression signal.
    //
    // SLO 1 (API availability 99.9%) ↔ http_req_failed < 0.001
    // SLO 2 (API latency p95 < 1.0s) ↔ http_req_duration p(95) < 1000
    //
    // These are tighter than the SLOs themselves (we want headroom). If
    // the test is borderline, the production SLO is borderline too.
    thresholds: {
        'http_req_failed': ['rate<0.001'],
        'http_req_duration': ['p(95)<1000', 'p(99)<2500'],
        'http_req_duration{scenario:anon_browse}': ['p(95)<500'],
        'http_req_duration{scenario:auth_read}':   ['p(95)<1000'],
        'auth_request_failures': ['rate<0.001'],
    },

    // Don't summarize per-VU — we want aggregate numbers, not 50 tables.
    summaryTrendStats: ['avg', 'min', 'med', 'p(95)', 'p(99)', 'max'],
};

// ---- Scenarios -------------------------------------------------------------

export function anonBrowse() {
    const params = { headers: COMMON_HEADERS };

    group('liveness/readiness', () => {
        const r = http.get(`${BASE_URL}/healthz`, params);
        check(r, { 'healthz 200': (res) => res.status === 200 });
    });

    group('public pages', () => {
        // Dashboard for an anonymous user — heaviest public template.
        const r = http.get(`${BASE_URL}/`, params);
        check(r, { 'dashboard 2xx': (res) => res.status >= 200 && res.status < 300 });
    });

    group('api root', () => {
        // The /api/v1/ root is the discovery endpoint — no auth required.
        // 429 is treated as success: the throttle is doing its job. Only
        // 5xx + network failures count as availability breaches (that's
        // what SLO 1 measures).
        const r = http.get(`${BASE_URL}/api/v1/`, params);
        check(r, { 'api root 2xx or 429': (res) => res.status < 300 || res.status === 429 });
    });

    // Pace the iteration so a single VU isn't hammering at thousands of
    // RPS. 1s between iterations + 50 VUs ≈ 50 iter/s ≈ 150 req/s across
    // three endpoints, which is the band we want to validate against.
    sleep(1);
}

export function authRead() {
    if (!API_KEY) {
        // No key provided — skip cleanly. The 1s sleep keeps k6 from
        // busy-spinning the (empty) iteration loop at hundreds of
        // thousands per second, which would tank the host CPU.
        sleep(1);
        return;
    }
    const params = { headers: { ...COMMON_HEADERS, 'X-API-Key': API_KEY } };

    group('list outputs', () => {
        // Paginated list endpoint — exercises the heaviest serializer +
        // the most-touched index in the DB.
        const r = http.get(`${BASE_URL}/api/v1/outputs/?page_size=50`, params);
        const ok = check(r, {
            'outputs 2xx or 429': (res) => res.status < 300 || res.status === 429,
        });
        authFailureRate.add(!ok);
    });

    group('list jobs', () => {
        const r = http.get(`${BASE_URL}/api/v1/jobs/?page_size=50`, params);
        const ok = check(r, {
            'jobs 2xx or 429': (res) => res.status < 300 || res.status === 429,
        });
        authFailureRate.add(!ok);
    });

    sleep(1);
}
