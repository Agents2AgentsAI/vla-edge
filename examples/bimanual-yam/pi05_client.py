"""Pi0.5 client for the shared YAM chunk controller."""
import os

import numpy as np

from vla_edge.protocol.client import ActClient


class Pi05Client:
    def __init__(self, server=None):
        self.client = ActClient(server or "127.0.0.1:8202")
        contract = self.client.bind(policy="pi05-bimanual-yam", model_family="pi05",
                                    embodiment="bimanual-yam", cameras=("top_cam","left_cam","right_cam"),
                                    state_dim=14, action_dim=14)
        if contract.action_horizon != 16 or contract.gripper_state != "commanded":
            raise ValueError("Pi0.5 YAM requires 16 actions and commanded gripper state")
        self.action_horizon = int(os.getenv("YAM_ACTION_HORIZON", "16"))
        if not 1 <= self.action_horizon <= 16:
            raise ValueError("YAM_ACTION_HORIZON must be within 1..16 for Pi0.5")
        if os.getenv("YAM_RTC", "0") != "0":
            raise ValueError("this Pi0.5 release does not support RTC")

    def get_action_horizon(self):
        return self.action_horizon

    def set_episode(self, episode_id):
        pass

    def prepare_input(self, obs, instruction):
        return {"cameras": {"top_cam":obs["front_camera_rgb"], "left_cam":obs["left_camera_rgb"],
                            "right_cam":obs["right_camera_rgb"]},
                "instruction": instruction, "state": np.asarray(obs["joint_positions"],dtype=np.float32)}

    def inference(self, input_dict, rtc=None):
        if rtc is not None:
            raise ValueError("this Pi0.5 release does not support RTC")
        actions, dt_ms = self.client.act(**input_dict, num_steps=10)
        return {"actions": actions, "dt_ms": dt_ms}

    def close(self):
        self.client.close()
