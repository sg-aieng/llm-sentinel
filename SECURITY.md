# Security Policy

## Supported versions

Security fixes are applied to the latest version on the default branch. Users
should keep the package and its model dependencies up to date.

## Reporting a vulnerability

Please do not disclose security vulnerabilities in a public issue or pull
request. Use a private GitHub Security Advisory for this repository when that
feature is available. If it is not available, contact the maintainers through
their GitHub profile and request a private channel.

Include, when possible:

- the affected version or commit;
- a concise description of the vulnerability and its impact;
- reproducible steps or a minimal proof of concept;
- relevant configuration, policy, or deployment assumptions;
- any proposed mitigation.

Please allow maintainers reasonable time to investigate and prepare a fix
before making the issue public. We will acknowledge reports, keep reporters
informed when possible, and credit researchers according to their preference.

## Deployment security notes

- The REST sidecar accepts caller-supplied application and identity context.
  Configure `SENTINEL_API_KEY` or protect it with an authenticated trusted
  network boundary such as mTLS or a service mesh.
- Do not expose the example static identity file or example admin role to
  untrusted callers.
- Treat policy files, model configuration, and rule packs as security-sensitive
  deployment inputs.
- Do not enable audit logging that records raw prompts, model outputs, secrets,
  or PII.
