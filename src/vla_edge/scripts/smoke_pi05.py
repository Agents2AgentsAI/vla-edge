"""Run Pi0.5 on saved or synthetic RGB observations without robot or camera access."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from vla_edge.backends.pi05.bundle import read_json
from vla_edge.config import get_policy, get_policy_embodiment
from vla_edge.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--observation", type=Path, help="NPZ containing the policy's camera fields and state")
    parser.add_argument("--prompt", default="pick up the red block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="save the action chunk as a NumPy file")
    args = parser.parse_args()
    policy = get_policy(read_json(args.bundle / "pi05-serving.json")["policy"])
    if policy.model_family != "pi05":
        raise ValueError("not a Pi0.5 bundle")
    emb = get_policy_embodiment(policy)
    rng = np.random.default_rng(args.seed)
    if args.observation:
        with np.load(args.observation, allow_pickle=False) as sample:
            images = {name:sample[name].copy() for name in emb.camera_names}
            state = sample["state"].copy()
    else:
        images = {name:rng.integers(0,256,(224,224,3),dtype=np.uint8) for name in emb.camera_names}
        state = np.zeros(emb.state_dim, np.float32)
    for image in images.values():
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("images must be HWC RGB uint8")
    noise = rng.standard_normal((1,policy.action_horizon,32)).astype(np.float32)
    pipeline = Pipeline.load(policy, backend="tensorrt", engine_dir=args.bundle)
    try:
        pipeline.predict(images,args.prompt,state,noise=noise)
        start = perf_counter()
        actions = pipeline.predict(images,args.prompt,state,noise=noise)
        elapsed = (perf_counter()-start)*1000
        print(json.dumps({"policy":policy.name,"shape":list(actions.shape),
                          "finite":bool(np.isfinite(actions).all()),"host_to_actions_ms":elapsed}))
        if args.output:
            np.save(args.output,actions,allow_pickle=False)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
