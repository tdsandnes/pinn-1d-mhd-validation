from dataclasses import dataclass, field
from typing import Any


@dataclass
class PINNMethod:
    """The config for a PINN method."""

    id: int
    module: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, str] = field(default_factory=dict)

    def format_config(self, name: str, config: dict[str, Any], **fields: Any) -> str:
        """The method's config, with its defaults overridden by a suite's config."""
        # Nothing to add, e.g. a family's "off" method
        if self.module is None:
            return ""

        unknown = config.keys() - self.defaults.keys()
        if unknown:
            raise SystemExit(
                f"Cannot set {', '.join(sorted(unknown))} for method '{name}'. "
                f"Use any of {', '.join(sorted(self.defaults))}."
            )

        return self.module.format(**{**self.defaults, **config, **fields})


@dataclass
class MethodFamily:
    """Alternative methods for one role. A run uses at most one of them."""

    default: str
    methods: dict[str, PINNMethod]
    template_field: str | None = None
