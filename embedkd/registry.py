"""Central registry for user-extensible components.

New distillation objectives, task losses and dataset adapters are added with
a decorator and never require editing the core package:

    from embedkd import registry

    @registry.distill_objective("my_loss")
    class MyLoss(DistillObjective):
        def forward(self, s_emb, t_emb, **kw): ...
"""

from __future__ import annotations

import difflib
from collections import defaultdict


class RegistryError(KeyError):
    pass


class Registry:
    NAMESPACES = ("distill_objective", "task_loss", "dataset")

    def __init__(self) -> None:
        self._store: dict[str, dict[str, type]] = defaultdict(dict)

    def _register(self, namespace: str, name: str, obj: type) -> type:
        if name in self._store[namespace]:
            raise RegistryError(
                f"{namespace} '{name}' is already registered "
                f"({self._store[namespace][name].__qualname__})"
            )
        self._store[namespace][name] = obj
        return obj

    def _decorator(self, namespace: str, name: str):
        def wrap(obj: type) -> type:
            return self._register(namespace, name, obj)

        return wrap

    def distill_objective(self, name: str):
        return self._decorator("distill_objective", name)

    def task_loss(self, name: str):
        return self._decorator("task_loss", name)

    def dataset(self, name: str):
        return self._decorator("dataset", name)

    def get(self, namespace: str, name: str) -> type:
        try:
            return self._store[namespace][name]
        except KeyError:
            close = difflib.get_close_matches(name, self._store[namespace], n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise RegistryError(
                f"Unknown {namespace} '{name}'. "
                f"Available: {sorted(self._store[namespace])}.{hint}"
            ) from None

    def available(self, namespace: str) -> list[str]:
        return sorted(self._store[namespace])

    def contains(self, namespace: str, name: str) -> bool:
        return name in self._store[namespace]


registry = Registry()
