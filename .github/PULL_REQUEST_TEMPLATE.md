## What changed

<!-- One paragraph. What does this add or fix, and why? -->

## Module

- [ ] Failure Prediction (Stage 1)
- [ ] Turbine Health Monitoring (Stage 2)
- [ ] Anomaly Detection & Decision Support (Stage 3)
- [ ] Shared contracts / core utilities
- [ ] Docs, CI or tooling only

## Shared contracts

- [ ] This PR does **not** touch anything listed in `docs/OZAN_STAGE1_HANDOFF.md` §9
- [ ] It does touch a shared contract — described below, and coordinated with the other
      stages

<!-- If you changed a shared contract, say exactly what and why. -->

## Checks

Paste the real output; do not summarise from memory.

```
$ make lint

$ make test

```

## Definition of done

- [ ] Tests and lint pass
- [ ] New temporal features have a leakage test
- [ ] Model outputs subclass `BasePrediction` and keep `advisory_only: true`
- [ ] Missing artifacts degrade gracefully (message + fix command, never a traceback)
- [ ] Every metric quoted in docs comes from a file in `artifacts/`
- [ ] Roadmap tables updated if a module's status changed
