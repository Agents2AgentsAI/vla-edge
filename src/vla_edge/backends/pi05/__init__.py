"""Pi0.5 TensorRT release runtime."""

def load_backend(**kwargs):
    from .backend import load_backend as load
    return load(**kwargs)
