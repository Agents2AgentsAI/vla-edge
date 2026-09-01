"""Embodiment configuration.

An embodiment is the fixed shape of one robot's interface to a policy: which
cameras exist and in what order, how wide the state vector is, and which
normalization statistics apply. Everything here is a property of the trained
checkpoint, not a tuning knob. If you change one of these values to make
something fit, you have changed the model's input distribution and the policy
will produce confident, wrong actions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Embodiment:
    """One robot's interface contract.

    Attributes:
        name: short identifier used on the command line.
        repo_id: HuggingFace repo (or local path) for the checkpoint.
        norm_tag: selects the normalization statistics inside the
            checkpoint's ``norm_stats.json``. Wrong tag, wrong action scale.
        state_dim: width of the proprioceptive state vector.
        camera_names: request field names, **in training order**. Order is
            load-bearing and silent when wrong. See ``validate_cameras``.
        default_num_steps: flow-matching integration steps. Changing this
            changes the ODE being integrated, so it is a behavioral
            parameter gated by ``docs/gates.md``, not a speed knob.
    """

    name: str
    repo_id: str
    norm_tag: str
    state_dim: int
    camera_names: tuple[str, ...]
    default_num_steps: int = 10
    description: str = ""

    @property
    def num_cameras(self) -> int:
        return len(self.camera_names)

    def validate_cameras(self, provided: list[str]) -> None:
        """Check that exactly the expected camera names are present.

        This validates membership, not sequence. Ordering is enforced by
        construction: the pipeline builds its image list by indexing
        ``camera_names``, so the order the caller happened to serialize its
        fields in cannot affect the result.

        What no check can catch is the wrong *image* in the right field. A
        swapped left/right cable produces confident actions for a scene the
        policy is not looking at, and nothing downstream will flag it. Verify
        camera assignment visually when a rig is first wired, and after any
        USB re-enumeration.
        """
        got, expected = set(provided), set(self.camera_names)
        if got != expected:
            missing = sorted(expected - got)
            unknown = sorted(got - expected)
            detail = []
            if missing:
                detail.append(f"missing: {missing}")
            if unknown:
                detail.append(f"unexpected: {unknown}")
            raise ValueError(
                f"camera mismatch for embodiment {self.name!r} "
                f"({'; '.join(detail)}). "
                f"Expected exactly {list(self.camera_names)} "
                f"(training order, applied automatically)."
            )


#: Built-in embodiments. Add yours here or construct an Embodiment directly.
EMBODIMENTS: dict[str, Embodiment] = {
    "bimanual-yam": Embodiment(
        name="bimanual-yam",
        repo_id="allenai/MolmoAct2-BimanualYAM",
        norm_tag="yam_dual_molmoact2",
        state_dim=14,
        camera_names=("top_cam", "left_cam", "right_cam"),
        default_num_steps=10,
        description="Bimanual YAM arms, 3 cameras, absolute joint control.",
    ),
    "droid": Embodiment(
        name="droid",
        repo_id="allenai/MolmoAct2-DROID",
        norm_tag="franka_droid",
        state_dim=8,
        camera_names=("external_cam", "wrist_cam"),
        default_num_steps=10,
        description="Franka DROID, external + wrist camera.",
    ),
    "libero": Embodiment(
        name="libero",
        repo_id="allenai/MolmoAct2-LIBERO",
        norm_tag="libero",
        state_dim=8,
        camera_names=("image", "wrist_image"),
        default_num_steps=10,
        description="LIBERO single-arm benchmark, scene + wrist camera.",
    ),
}


def get_embodiment(name: str) -> Embodiment:
    if name not in EMBODIMENTS:
        raise KeyError(
            f"unknown embodiment {name!r}; available: "
            f"{', '.join(sorted(EMBODIMENTS))}"
        )
    return EMBODIMENTS[name]
