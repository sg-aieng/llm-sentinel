"""Identity resolution — bring your own IAM.

llm-sentinel does not manage users or roles. It resolves a `user_id`
into roles via the IdentityProvider protocol; you supply an
implementation backed by your IAM (Okta, Keycloak, Cognito, LDAP, ...)
and register it in `app.main`:

    from app.identity import Identity, IdentityProvider

    class OktaIdentityProvider:
        def resolve(self, user_id: str) -> Identity:
            profile = my_okta_client.get_user(user_id)
            return Identity(user_id=user_id, roles=profile.roles)

    # in main.py lifespan:
    app.state.identity_provider = OktaIdentityProvider()

The shipped default is StaticIdentityProvider: it reads an optional
identities.yaml and falls back to the "default" role for unknown users.
It exists so the pipeline works out of the box — replace it for anything
real.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from app.config import settings

logger = logging.getLogger("llm_sentinel.identity")

DEFAULT_ROLE = "default"


@dataclass
class Identity:
    user_id: str
    roles: list[str] = field(default_factory=lambda: [DEFAULT_ROLE])
    attributes: dict = field(default_factory=dict)


@runtime_checkable
class IdentityProvider(Protocol):
    def resolve(self, user_id: str) -> Identity:
        """Return the identity for user_id. Must not raise for unknown
        users — return an Identity with the default role instead."""
        ...


class StaticIdentityProvider:
    """Placeholder provider backed by an optional YAML file.

    identities.yaml format:

        users:
          user-42:
            roles: [admin]
          user-99:
            roles: [guest]
            attributes:
              team: support
    """

    def __init__(self, identities_path: Path | None = None):
        path = identities_path or (
            Path(__file__).resolve().parents[1] / settings.identities_file
        )
        self._users: dict = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
            self._users = raw.get("users", {}) or {}
            logger.info("loaded %d static identities from %s", len(self._users), path)

    def resolve(self, user_id: str) -> Identity:
        entry = self._users.get(user_id)
        if not entry:
            return Identity(user_id=user_id)
        return Identity(
            user_id=user_id,
            roles=entry.get("roles") or [DEFAULT_ROLE],
            attributes=entry.get("attributes") or {},
        )
