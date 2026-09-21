"""ABC-VLA inference; heavy dependencies are imported only when loading a backend."""

from .backend import load_backend

__all__ = ["load_backend"]
