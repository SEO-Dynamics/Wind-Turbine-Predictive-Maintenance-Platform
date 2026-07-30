# Security Policy

## Supported version

This repository is maintained on its default `main` branch. Older commits,
feature branches, generated model artifacts, and unofficial container images
are not independently supported.

| Version | Supported |
|---|---|
| Latest `main` | Yes |
| Older snapshots or forks | No |

## Reporting a vulnerability

Please do not open a public issue for an undisclosed vulnerability.

Use GitHub's private vulnerability reporting form:

<https://github.com/SEO-Dynamics/Wind-Turbine-Predictive-Maintenance-Platform/security/advisories/new>

Include the affected commit, reproduction steps, impact, and any suggested
mitigation. The maintainers will acknowledge a complete report as soon as
practical, investigate it privately, and coordinate disclosure after a fix is
available.

If the private reporting form is unavailable, contact the repository owners
through the SEO-Dynamics GitHub organization without including exploit details
in a public channel.

## Security boundary

The platform is advisory decision support for synthetic hourly SCADA data. It
must not automatically stop, start, derate, or schedule work on real equipment.
It is not a certified safety system.

The bundled Docker Compose configuration is intended for local evaluation. It
binds to `127.0.0.1` by default and does not provide authentication, TLS, rate
limiting, tenant isolation, or an internet-facing reverse proxy. Exposing it to
a network requires those controls to be supplied and reviewed separately.

Generated datasets and model artifacts are deliberately excluded from source
control. Treat externally supplied joblib/pickle artifacts as executable code:
only load artifacts produced by a trusted pipeline and verify their provenance
before serving them.
