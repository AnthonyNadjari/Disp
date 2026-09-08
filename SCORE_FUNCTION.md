# The optimizer's score function

One page on how a candidate basket is scored. Code: `functions/dispersion/scoring/`
(`metrics.py`, `normalizers.py`, `aggregators.py`, `score.py`).

## The formula

For a candidate basket, the inner solver first derives its weights, which gives one
net P&L series *p* (long leg minus short leg, per observation date). Then

```
score(basket) = Σ_m  w_m · Φ_m( metric_m(p) )        with  Σ_m w_m = 1
```

* `metric_m(p)` is a raw performance number, for example the mean payoff.
* `Φ_m` maps that raw number to **the share of random feasible baskets it beats**,
  a number between 0 and 1.
* `w_m` are the desk's preference weights, set in the interface and renormalised to
  sum to 1.

The score is therefore in [0, 1] and reads directly: 0.87 means the basket sits, on
a weighted average of the active criteria, at the 87th percentile of what the
constraint set can produce.

## The criteria

| Metric | Definition | Direction |
|---|---|---|
| `last_carry` | mean of the most recent payoffs, the entry carry | higher better |
| `mean_payoff` | mean of the payoff series | higher better |
| `hit_ratio` | share of observation dates with a positive payoff | higher better |
| `min_payoff` | worst single payoff, the floor | higher better |
| `cvar_5` | mean of the worst 5% of payoffs | higher better |
| `max_drawdown` | worst peak-to-trough drop of the cumulative payoff curve | lower better |
| `sharpe_payoff` | annualised mean divided by standard deviation of payoffs | higher better |
| `weighted_strike` | net strike of the basket, as an objective | lower better |
| `axe_book_cleaned`, `axe_package_recycled` | vega recycled against the axe book | higher better |

Default weights are a quarter each on last carry, mean payoff, hit ratio and min
payoff. The other criteria sit at weight zero and switch on by being given a weight,
with no code change.

## Why percentiles rather than raw numbers

The raw criteria live on scales that cannot be added: a hit ratio between 0 and 1, a
minimum payoff in P&L points, a strike in vol points. Adding them directly, or
rescaling them by z-score or min-max, makes the same set of weights mean something
different on every universe, and a single fat-tailed observation shifts the scale.

Instead each criterion is passed through its **empirical distribution over a
calibration sample of random feasible baskets** drawn from the same constraint set.
The blend then trades off ranks, which are scale-free, robust to outliers, and read
in plain language. Ties resolve downward: a basket equal to the reference scores
below it, never above.

## The calibration sample

300 random feasible baskets per run, 800 when a tail criterion such as min payoff or
CVaR is active, since tail percentiles need more resolution. Subsets are drawn
uniformly over the allowed sizes, with forced names always included. Their weights
come from a mix of schemes, equal, diversified, spread and random, chosen to cover
the same region of weight space the search itself visits. A reference built only
from equal-weight baskets would misrank corner solutions.

The sample is fixed for the whole run, so every candidate is measured against the
same yardstick. If a criterion has no spread in the reference, the run stops with an
error rather than silently scoring every candidate at zero.

## What the score does not do

Constraints are not part of the score. Basket size, per-name weight bounds, the net
strike cap, sector and region buckets and forced or excluded names are enforced by
construction, so an infeasible basket is never scored rather than scored badly.
Weights are not free variables either: for a given set of names they are derived by
a deterministic solver, which makes the score a well-defined function of the name
selection alone.

## Reproducibility

Same inputs and same seed give the same basket, the same score and the same weights,
including the calibration sample. Every run can be saved as a bundle and replayed.
