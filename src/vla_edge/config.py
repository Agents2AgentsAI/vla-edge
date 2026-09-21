"""Robot I/O and policy/checkpoint configuration.

An :class:`Embodiment` describes physical robot I/O. A :class:`PolicySpec`
describes a trained policy attached to that I/O. Keeping those axes separate
lets one robot run MolmoAct2, ABC-VLA, or another VLA without duplicating camera
and state contracts or teaching the robot client about model internals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .action_space import validate_action_space
from .gripper import (
    CLOSED_ZERO_OPEN_ONE,
    validate_gripper_convention,
    validate_gripper_state,
)


@dataclass(frozen=True)
class Embodiment:
    """One robot's model-facing I/O contract.

    Attributes:
        name: short identifier used on the command line.
        state_dim: width of the proprioceptive state vector.
        action_dim: width of one action row sent to the robot.
        camera_names: request field names, **in training order**. Order is
            load-bearing and silent when wrong. See ``validate_cameras``.
        gripper_indices: normalized aperture channels in state/action vectors.
        gripper_convention: robot-wire endpoint meaning for those channels.
    """

    name: str
    state_dim: int
    action_dim: int
    camera_names: tuple[str, ...]
    gripper_indices: tuple[int, ...] = ()
    gripper_convention: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.gripper_indices:
            if tuple(sorted(set(self.gripper_indices))) != self.gripper_indices:
                raise ValueError("gripper_indices must be unique and strictly increasing")
            if any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= min(self.state_dim, self.action_dim)
                for index in self.gripper_indices
            ):
                raise ValueError("gripper_indices fall outside state/action vectors")
            validate_gripper_convention(
                self.gripper_convention,
                label=f"embodiment {self.name!r} gripper convention",
            )
        elif self.gripper_convention is not None:
            raise ValueError("gripper_convention requires gripper_indices")

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


@dataclass(frozen=True)
class PolicySpec:
    """One checkpoint family bound to one robot embodiment.

    ``model_family`` selects a loader from the model-family registry. The
    remaining values are checkpoint behavior, not serving knobs: changing a
    normalization tag, horizon, or integration step count requires a new
    policy spec and its own validation evidence.
    """

    name: str
    model_family: str
    embodiment: str
    repo_id: str
    norm_tag: str
    action_horizon: int
    default_num_steps: int = 10
    gripper_convention: str | None = None
    description: str = ""
    #: ``absolute`` joint targets (every historical policy) or ``delta``: the
    #: checkpoint predicts masked joints relative to the current state and the
    #: serving host adds the state back (see ``vla_edge.action_space``).
    action_space: str = "absolute"
    #: What the training data recorded in the gripper state channel: the
    #: commanded opening (YAM/MolmoAct2 recordings) or the measured one
    #: (ABC-130k). The robot client feeds the policy accordingly.
    gripper_state: str = "commanded"
    #: Optional checkpoint-family configuration identifier.
    training_config: str | None = None
    #: A checkpoint may expose a different action width on the same benchmark.
    action_dim: int | None = None

    def __post_init__(self) -> None:
        if self.action_dim is not None and (type(self.action_dim) is not int or self.action_dim <= 0):
            raise ValueError("action_dim must be a positive integer")
        if self.gripper_convention is not None:
            validate_gripper_convention(
                self.gripper_convention,
                label=f"policy {self.name!r} gripper convention",
            )
        validate_gripper_state(
            self.gripper_state, label=f"policy {self.name!r} gripper_state"
        )
        validate_action_space(
            self.action_space, label=f"policy {self.name!r} action_space"
        )


#: Built-in embodiments. Add yours here or construct an Embodiment directly.
EMBODIMENTS: dict[str, Embodiment] = {
    "bimanual-yam": Embodiment(
        name="bimanual-yam",
        state_dim=14,
        action_dim=14,
        camera_names=("top_cam", "left_cam", "right_cam"),
        gripper_indices=(6, 13),
        gripper_convention=CLOSED_ZERO_OPEN_ONE,
        description="Bimanual YAM arms, 3 cameras, absolute joint control.",
    ),
    "droid": Embodiment(
        name="droid",
        state_dim=8,
        action_dim=8,
        camera_names=("external_cam", "wrist_cam"),
        description="Franka DROID, external + wrist camera.",
    ),
    "libero": Embodiment(
        name="libero",
        state_dim=8,
        action_dim=8,
        camera_names=("image", "wrist_image"),
        description="LIBERO single-arm benchmark, scene + wrist camera.",
    ),
}


#: Built-in trained policies. Multiple policies may share one embodiment.
POLICIES: dict[str, PolicySpec] = {
    "pi05-bimanual-yam": PolicySpec(
        name="pi05-bimanual-yam", model_family="pi05", embodiment="bimanual-yam",
        repo_id="robocurve/pi0.5-yam", norm_tag="yam-bimanual-merged",
        action_horizon=16, gripper_convention=CLOSED_ZERO_OPEN_ONE,
        gripper_state="commanded", description="Pi0.5 on bimanual YAM, ten diffusion steps.",
    ),
    "pi05-libero": PolicySpec(
        name="pi05-libero", model_family="pi05", embodiment="libero",
        repo_id="gs://openpi-assets/checkpoints/pi05_libero", norm_tag="physical-intelligence/libero",
        action_horizon=10, action_dim=7, action_space="native",
        description="Pi0.5 LIBERO, native seven-value end-effector action, ten diffusion steps.",
    ),
    "abcvla-bimanual-yam": PolicySpec(
        name="abcvla-bimanual-yam",
        model_family="abcvla",
        embodiment="bimanual-yam",
        repo_id="amazon-far/abc:vla_abc130k_v2/200000",
        norm_tag="abc130k-vla-v2",
        action_horizon=30,
        default_num_steps=10,
        gripper_convention=CLOSED_ZERO_OPEN_ONE,
        gripper_state="measured",
        description="Released ABC-VLA Gemma 3/SigLIP/DiT policy; native CUDA graph or packed Thor champions.",
    ),
    "molmoact2-bimanual-yam": PolicySpec(
        name="molmoact2-bimanual-yam",
        model_family="molmoact2",
        embodiment="bimanual-yam",
        repo_id="allenai/MolmoAct2-BimanualYAM",
        norm_tag="yam_dual_molmoact2",
        action_horizon=30,
        default_num_steps=10,
        gripper_convention=CLOSED_ZERO_OPEN_ONE,
        description="MolmoAct2 on the bimanual YAM rig.",
    ),
    "molmoact2-droid": PolicySpec(
        name="molmoact2-droid",
        model_family="molmoact2",
        embodiment="droid",
        repo_id="allenai/MolmoAct2-DROID",
        norm_tag="franka_droid",
        action_horizon=30,
        default_num_steps=10,
        description="MolmoAct2 on DROID.",
    ),
    "molmoact2-libero": PolicySpec(
        name="molmoact2-libero",
        model_family="molmoact2",
        embodiment="libero",
        repo_id="allenai/MolmoAct2-LIBERO",
        norm_tag="libero",
        action_horizon=30,
        default_num_steps=10,
        description="MolmoAct2 on LIBERO.",
    ),
}


DEFAULT_POLICIES: dict[str, str] = {
    "bimanual-yam": "molmoact2-bimanual-yam",
    "droid": "molmoact2-droid",
    "libero": "molmoact2-libero",
}

POLICY_ALIASES: dict[str, str] = {}


def get_embodiment(name: str) -> Embodiment:
    if name not in EMBODIMENTS:
        raise KeyError(
            f"unknown embodiment {name!r}; available: "
            f"{', '.join(sorted(EMBODIMENTS))}"
        )
    return EMBODIMENTS[name]


def get_policy(name: str) -> PolicySpec:
    """Resolve a canonical policy name or a compatibility alias."""

    canonical = POLICY_ALIASES.get(name, name)
    if canonical not in POLICIES:
        raise KeyError(
            f"unknown policy {name!r}; available: "
            f"{', '.join(sorted(POLICIES))}"
        )
    return POLICIES[canonical]


def default_policy(embodiment: str | Embodiment) -> PolicySpec:
    """Return the backward-compatible default policy for an embodiment."""

    name = embodiment.name if isinstance(embodiment, Embodiment) else embodiment
    if name in POLICY_ALIASES:
        return get_policy(POLICY_ALIASES[name])
    try:
        return get_policy(DEFAULT_POLICIES[name])
    except KeyError:
        raise KeyError(f"embodiment {name!r} has no default policy") from None


def resolve_policy(
    *,
    policy: str | None = None,
    embodiment: str | None = None,
) -> PolicySpec:
    """Resolve new policy selection and legacy embodiment-only selection.

    If both selectors are supplied, they must describe the same robot contract.
    """

    if policy is None:
        if embodiment is None:
            return get_policy("molmoact2-bimanual-yam")
        return default_policy(embodiment)

    selected = get_policy(policy)
    if embodiment is None:
        return selected
    if embodiment in POLICY_ALIASES:
        legacy = get_policy(embodiment)
        if legacy != selected:
            raise ValueError(
                f"policy {selected.name!r} conflicts with legacy selector "
                f"{embodiment!r}"
            )
        return selected
    if embodiment != selected.embodiment:
        raise ValueError(
            f"policy {selected.name!r} requires embodiment "
            f"{selected.embodiment!r}, got {embodiment!r}"
        )
    return selected


def get_policy_embodiment(policy: PolicySpec) -> Embodiment:
    """Resolve camera/state layout and the checkpoint's action width together."""
    embodiment = get_embodiment(policy.embodiment)
    if policy.action_dim is not None:
        embodiment = replace(embodiment, action_dim=policy.action_dim)
    return embodiment
