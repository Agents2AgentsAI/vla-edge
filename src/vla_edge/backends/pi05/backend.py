"""Pi0.5 TensorRT policy runtime."""
import threading

import numpy as np

from .bundle import Pi05Error, load_bundle
from .host import Host
from .runtime import Runtime


class Pi05TensorRTBackend:
    name = "tensorrt"

    def __init__(self, *, policy, embodiment, engine_dir, device, dtype, **kwargs):
        import torch
        if dtype != "bfloat16" or torch.device(device).type != "cuda":
            raise Pi05Error("Pi0.5 TensorRT requires CUDA and bfloat16")
        if not engine_dir:
            raise Pi05Error("pass --engine-dir with a local Pi0.5 release bundle")
        self.policy, self.embodiment, self.device = policy, embodiment, torch.device(device)
        self.action_horizon = policy.action_horizon
        self.torch, self.lock, self.closed = torch, threading.RLock(), False
        self.stream = torch.cuda.Stream(device=self.device)
        self.desc = load_bundle(engine_dir, policy, embodiment)
        with torch.cuda.device(self.device):
            self.host = Host(self.desc, self.device)
            try:
                self.runtime = Runtime(self.desc, self.device)
            except BaseException:
                self.host.close()
                raise

    def generate_actions(self, *, images, instruction, state, num_steps,
                         enable_cuda_graph=False, seed=None, noise=None):
        torch = self.torch
        if num_steps != 10:
            raise Pi05Error("this Pi0.5 bundle requires exactly ten diffusion steps")
        if seed is not None and noise is not None:
            raise Pi05Error("supply either seed or noise")
        with self.lock, torch.inference_mode(), torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if self.closed:
                raise Pi05Error("backend is closed")
            pixels, language, mask = self.host.prepare(images, instruction, state)
            shape = (1, self.action_horizon, 32)
            if noise is None:
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(int(seed))
                noise = torch.randn(shape, device=self.device, dtype=torch.float32, generator=generator)
            else:
                # json_numpy decodes arrays over immutable message bytes.
                # Own the storage before constructing a tensor from that view.
                if isinstance(noise, np.ndarray) and not noise.flags.writeable:
                    noise = noise.copy()
                noise = torch.as_tensor(noise, device=self.device, dtype=torch.float32)
                if tuple(noise.shape) != shape or not bool(torch.isfinite(noise).all()):
                    raise Pi05Error(f"noise must be finite with shape {shape}")
            actions = self.runtime.run(pixels, language, mask, noise)
            return self.host.postprocess(actions)

    def warmup(self):
        for _ in range(2):
            self.generate_actions(images=[np.zeros((224,224,3), dtype=np.uint8)] * self.embodiment.num_cameras,
                                  instruction="warmup", state=np.zeros(self.embodiment.state_dim),
                                  num_steps=10, seed=0)

    def close(self):
        with self.lock:
            if not self.closed:
                self.runtime.close()
                self.host.close()
                self.closed = True

def load_backend(*, backend, **kwargs):
    if backend != "tensorrt":
        raise Pi05Error("Pi0.5 releases support --backend tensorrt")
    return Pi05TensorRTBackend(**kwargs)
