
import numpy as np

from gello_min.robot import Robot


class YAMRobot(Robot):
    """Adapter between the rollout client and an I2RT YAM arm.

    Commands go straight to the i2rt motor chain, one setpoint per call. The
    servo handles interpolation between setpoints, matching the teleoperation
    path used to collect the training data.
    """

    def __init__(self, channel="can0", gripper_limits=None):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        limits = None
        if gripper_limits is not None:
            limits = np.asarray(gripper_limits, dtype=float)
            if limits.shape != (2,) or not np.all(np.isfinite(limits)):
                raise ValueError("gripper_limits must contain two finite values")
            if np.isclose(limits[0], limits[1]):
                raise ValueError("gripper_limits must describe nonzero travel")

        self.robot = get_yam_robot(
            channel=channel,
            gripper_type=GripperType.LINEAR_4310,
            gripper_limits_override=limits,
        )
        self._channel = str(channel)
        self._closed = False

        # YAM has 7 joints (6 arm joints + 1 gripper)
        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self._joint_state = self.get_joint_state()  # robot stays where it was when reboot
        # self._joint_state = np.zeros(7)  # robot goes immediately to reset position (avoid using)
        self._joint_velocities = np.zeros(7)  # 7 joints
        self._gripper_state = 0.0 # didn't use because joint_state includes gripper position

    def num_dofs(self) -> int:
        return 7  # YAM has 7 DOFs

    def get_joint_state(self) -> np.ndarray:
        # Abort loudly if the motor chain has died (a motor error fail-fasts
        # i2rt's control loop, but get_joint_pos keeps serving CACHED state and
        # command_joint_pos keeps accepting commands into a dead bus). This is
        # the choke point every observation flows through.
        chain = getattr(self.robot, "motor_chain", None)
        if chain is not None and not getattr(chain, "running", True):
            raise RuntimeError(
                "YAM motor chain is no longer running (a motor errored and "
                "i2rt fail-fasted); aborting instead of operating on cached "
                "joint state. Check motor errors, then re-home."
            )
        # Get actual joint positions from I2RT robot (7 joints total)
        joint_pos = self.robot.get_joint_pos()
        # Ensure we have exactly 7 joints
        if len(joint_pos) > 7:
            joint_pos = joint_pos[:7]
        elif len(joint_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            joint_pos = np.pad(joint_pos, (0, 7 - len(joint_pos)), "constant")

        self._joint_state = joint_pos
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert (
            len(joint_state) == self.num_dofs()
        ), f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"

        dt = 0.01
        self._joint_velocities = (joint_state - self._joint_state) / dt
        self._joint_state = joint_state

        # Command the I2RT robot with all 7 joints (6 arm + 1 gripper)
        self.command_joint_pos(joint_state)

    def get_observations(self) -> dict[str, np.ndarray]:
        ee_pos_quat = np.zeros(7)  # Placeholder for FK
        return {
            "joint_positions": self.get_joint_state(),
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([self._gripper_state]),
        }

    def get_joint_pos(self):
        # Get 7 joints from I2RT robot (6 arm + 1 gripper)
        joint_pos = self.robot.get_joint_pos()
        # Ensure we return exactly 7 joints
        if len(joint_pos) > 7:
            joint_pos = joint_pos[:7]
        elif len(joint_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            joint_pos = np.pad(joint_pos, (0, 7 - len(joint_pos)), "constant")
        return joint_pos

    def command_joint_pos(self, target_pos):
        # Ensure we send exactly 7 joints to the I2RT robot
        if len(target_pos) > 7:
            target_pos = target_pos[:7]
        elif len(target_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            target_pos = np.pad(target_pos, (0, 7 - len(target_pos)), "constant")
        self.robot.command_joint_pos(np.array(target_pos))

    def close(self) -> None:
        """Release I2RT's threads, then disable and verify every motor."""
        if self._closed:
            return
        from home_arms import close_robot_cleanly, disable_and_probe

        close_robot_cleanly(self.robot)
        self._closed = True
        if not disable_and_probe((self._channel,)):
            raise RuntimeError(f"{self._channel}: one or more motors stayed enabled")


def main():
    robot = YAMRobot()
    print(robot.get_observations())


if __name__ == "__main__":
    main()
