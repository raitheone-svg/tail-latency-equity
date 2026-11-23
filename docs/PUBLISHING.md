# Novelty and getting it accepted

Acceptance rides on rigor, framing, and a clear novelty claim — not on any one
trick. These are the levers that move the odds, in priority order.

## Where the novelty is

1. **Quantile-fairness metrics for the latency tail.** Existing fairness-in-
   serving work (VTC, OSDI 2024; QUARTZ, ACL Findings 2026) optimises resource
   share, token throughput, or aggregate SLO attainment. None *measures parity
   of the tail quantile across groups*. That gap is your primary contribution.
2. **The workload-conditioned decomposition.** Separating a tail gap into a
   composition component (groups send different work) and a scheduling-induced
   component (same work, different treatment) is the sharpest novel idea here.
   Lead with it; it answers the obvious reviewer objection ("they just send
   bigger requests") with a number.
3. **A bias-aware controller with a measured trade-off.** Showing that most of
   the gap is a cheap-to-remove artifact of a biased proxy — not an unavoidable
   price of speed — is a concrete, actionable result.

## Framing that raises acceptance

- **Target LLM inference serving.** Use the Azure LLM Inference trace and
  measure fairness on **TTFT (time-to-first-token) tails**. This is the hottest
  subfield and lets you position directly against QUARTZ and VTC. It is the
  single highest-leverage change.
- **Real workload, honest cohorts.** Drive the queue with a real trace's
  arrivals and sizes; define cohorts from a neutral operational dimension
  (application, tenant, request shape). Say loudly that cohorts are operational,
  not demographic. This is far more defensible than a fully synthetic study and
  pre-empts an ethics-desk rejection.
- **Ship this repo as the artifact.** Reproducibility (code + data loader +
  seed) is increasingly rewarded and cheap insurance against "is this real?"
  reviews. Add an artifact-evaluation appendix.
- **Keep the governance section actionable.** AI-ethics venues reward concrete
  safeguards (per-group SLO reporting, the controller, auditing thresholds) over
  pure measurement.

## A credible venue ladder

- *AI and Ethics* (Springer) — the natural home given the framing.
- Systems-and-society workshops: **HotEthics** (ethical systems/architecture).
- Adjacent: **FAccT**, **AIES** for the fairness-metric contribution; systems
  venues (OSDI/NSDI/SoCC workshops) for the serving-systems angle.

## Honest limits to disclose (reviewers will probe these)

- The workload-adjusted gap removes only the workload differences the bins
  capture; use fine bins and bootstrap confidence intervals.
- Fairness metrics are plural and can conflict; over-correction can harm the
  group it helps. Report a panel, not one number.
- Measuring disaggregated harm can itself need sensitive data — reconcile the
  duty to measure with data minimisation, and keep cohorts operational.
