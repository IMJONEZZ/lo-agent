"""LoRA adapter hot-swapping per skill (PEFT)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


class AdapterManager:
    def __init__(self, backend):
        self.backend = backend
        self._wrapped = False

    def _ensure_peft(self):
        from peft import PeftModel

        if not self._wrapped and not isinstance(self.backend.model, PeftModel):
            raise RuntimeError("no adapter loaded yet; call load_adapter first")

    def load_adapter(self, name: str, path: str | Path) -> None:
        from peft import PeftModel

        if isinstance(self.backend.model, PeftModel):
            self.backend.model.load_adapter(str(path), adapter_name=name)
        else:
            self.backend.model = PeftModel.from_pretrained(
                self.backend.model, str(path), adapter_name=name
            )
            self._wrapped = True

    def set_active(self, name: str) -> None:
        self._ensure_peft()
        self.backend.model.set_adapter(name)

    def active(self) -> str | None:
        from peft import PeftModel

        if isinstance(self.backend.model, PeftModel):
            return self.backend.model.active_adapter
        return None

    @contextmanager
    def with_adapter(self, name: str):
        """Temporarily switch adapters (e.g. per-skill) and restore after."""
        previous = self.active()
        self.set_active(name)
        try:
            yield self.backend
        finally:
            if previous is not None:
                self.backend.model.set_adapter(previous)

    @contextmanager
    def disabled(self):
        """Run with adapters bypassed (base model behavior)."""
        self._ensure_peft()
        with self.backend.model.disable_adapter():
            yield self.backend
