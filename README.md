# Asteroid Doomsday-o-meter

Can you tell whether a near-Earth asteroid is classified "potentially hazardous"
from the shape of its orbit alone?

Catch: "potentially hazardous" (PHA) is *defined* as MOID ≤ 0.05 AU and absolute
magnitude H ≤ 22 (close orbit and big enough). Feed the model MOID and H and the
problem is a tautology. So the model never sees MOID, H, or the size proxies
(diameter, albedo). It predicts PHA from orbit **geometry** only: semi-major
axis, eccentricity, inclination, perihelion, aphelion, period, node, argument of
perihelion, rotation. That captures the "does this orbit come close to Earth"
half of the definition. It cannot see the size half. Honest and non-trivial.

Data: the full near-Earth object catalogue from the
[JPL Small-Body Database](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html)
(free, no key, one request). 42,153 NEOs, 2,545 of them flagged PHA (6%).

Built on [Hopsworks](https://www.hopsworks.ai/) as an FTI (feature, training,
inference) system, forked from the
[readme-vaporware-score](https://github.com/MagicLex/readme-vaporware-score)
base patterns.

## Status

- [x] Collector (`collect/collect.py`): JPL SBDB -> `data/neos.jsonl`
- [ ] Feature pipeline -> offline feature group (Hopsworks job)
- [ ] Orbit-geometry-only feature view + PHA classifier (handles 6% imbalance)
- [ ] Register with eval plots; serve / app

## Reproduce

```bash
python collect/collect.py     # pull the NEO catalogue (free, no key)
# feature pipeline + train run as Hopsworks jobs (see pipelines/)
```

## Leakage rules (the honest part)

Excluded from the model: `moid`, `H` (the literal PHA definition) and
`diameter`, `albedo` (size proxies derived from H). Kept: orbital elements +
rotation period + orbit class. The label is `pha`.
