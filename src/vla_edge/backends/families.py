"""Model-family loader registry.

The serving and robot layers know only the whole-policy runtime contract.
Family-specific checkpoint loading and compiled-stage composition live behind
this registry, so adding a VLA does not add another branch to ``Pipeline`` or
the HTTP server.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

FamilyLoader = Callable[..., Any]


class ModelFamilyRegistry:
    """Lazy model-family name -> whole-policy runtime loader."""

    def __init__(self) -> None:
        self._loaders: dict[str, FamilyLoader | str] = {}

    def register(self, name: str, loader: FamilyLoader | str) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("model-family name must be a non-empty string")
        if name in self._loaders:
            raise ValueError(f"model family {name!r} is already registered")
        if not callable(loader) and (
            not isinstance(loader, str) or loader.count(":") != 1
        ):
            raise TypeError("loader must be callable or 'module:attribute'")
        self._loaders[name] = loader

    def _resolve(self, name: str) -> FamilyLoader:
        try:
            loader = self._loaders[name]
        except KeyError:
            raise KeyError(
                f"unknown model family {name!r}; available: "
                f"{', '.join(sorted(self._loaders)) or '(none)'}"
            ) from None
        if isinstance(loader, str):
            module_name, attribute = loader.split(":", 1)
            resolved = getattr(importlib.import_module(module_name), attribute)
            if not callable(resolved):
                raise TypeError(f"model-family loader {loader!r} is not callable")
            self._loaders[name] = resolved
            return resolved
        return loader

    def load(self, name: str, **kwargs: Any) -> Any:
        return self._resolve(name)(**kwargs)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaders))


FAMILIES = ModelFamilyRegistry()
FAMILIES.register("molmoact2", "vla_edge.backends.molmoact2:load_backend")
FAMILIES.register("abcvla", "vla_edge.backends.abcvla:load_backend")
FAMILIES.register("pi05", "vla_edge.backends.pi05:load_backend")


def register_model_family(name: str, loader: FamilyLoader | str) -> None:
    """Public extension point for an additional VLA family."""

    FAMILIES.register(name, loader)


__all__ = ["FAMILIES", "ModelFamilyRegistry", "register_model_family"]
