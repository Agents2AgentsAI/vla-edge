# Bimanual YAM

This example connects the
[MolmoAct2-BimanualYAM](https://huggingface.co/allenai/MolmoAct2-BimanualYAM)
policy to two I2RT YAM arms and three Intel RealSense cameras. The robot client
captures an observation, calls a running `vla-edge` server, executes the
returned action chunk, and records the rollout.

The reference setup uses one Jetson AGX Thor for both inference and robot
control. The inference server can also run on another machine.

## Hardware

- Two YAM arms, each on its own 1 Mbit/s CAN interface
- Three RealSense cameras: one scene camera and one wrist camera per arm
- A tested hardware e-stop

Before powering the arms, verify the start poses in both config files and the
gripper open direction in `home_arms.py`. The reference grippers open in the
negative motor direction. Configure the motor watchdog as required by I2RT;
the reference homing path assumes it is disabled.

## Setup

Run these commands once from a fresh clone on Jetson Thor:

```bash
git clone https://github.com/Agents2AgentsAI/vla-edge.git
cd vla-edge

python3 -m venv .venv
source .venv/bin/activate
./examples/bimanual-yam/setup_jetson_thor.sh

cd examples/bimanual-yam
python tests/test_rollout_control.py
```

## Configure the rig

```bash
python configure_rig.py
python home_arms.py
python calibrate_grippers.py
```

The setup command finds the cameras and CAN interfaces, saves a labeled camera
snapshot, asks you to confirm the physical left/right assignments and safe
start poses, updates both config files, and brings the selected CAN links up at
1 Mbit/s. Press Enter to keep any value shown in brackets. For CAN mapping, it
reads joint positions without enabling the motors and asks you to hand-move one
joint on the left arm.

`start_joints` is a commanded target, not a measured pose. The first six
values are arm-joint angles in radians; the seventh is the normalized gripper
position, where `0` is closed and `1` is open.

`home_arms.py` moves both arms to encoder zero together, opens the grippers,
and disables all 14 motors. `calibrate_grippers.py` then moves only the
grippers through their physical travel and saves `[closed, open]` limits in
both config files. Run gripper calibration once during setup and again after a
gripper, motor, or motor zero changes. Saved limits also prevent I2RT from
repeating the hard-stop sweep at every launch. Calibration refuses to start if
a gripper rotor is above 45 C and disables each motor when its sweep ends.

The example assumes the six arm-joint encoder zeros are already calibrated.
There is no general YAM joint-zero command here because that operation writes
motor flash and needs a hardware-specific fixture and procedure.

With no controller running, this checks both buses and leaves every motor
disabled without sending a position command. If a controller is active, the
command stops it first, so the controller may park the arms:

```bash
python home_arms.py --status
```

## Test and run

Open three terminals at the repository root and activate the same environment
in each one.

Terminal 1 owns the cameras:

```bash
source .venv/bin/activate
cd examples/bimanual-yam
bash start_camera_server.sh
```

Leave this process running. Starting it again detects the existing healthy
server without resetting or reopening the cameras. Ctrl-C releases all three
cameras cleanly.

Terminal 2 runs the reference inference server. The checkpoint downloads on
the first launch:

```bash
source .venv/bin/activate
vla-edge-serve --embodiment bimanual-yam --backend torch
```

Terminal 3 verifies the server and camera order, then starts a supervised
low-speed rollout:

```bash
source .venv/bin/activate
cd examples/bimanual-yam

curl -f http://127.0.0.1:8202/act
python camera_client.py --mode sub
# Confirm the scene, left wrist, and right wrist panes, then stop the viewer.

YAM_MAX_JOINT_VEL=0.5 bash run_task.sh \
  "pick up the rubik cube and put it in the black box"
```

The camera client opens a window when a graphical display is available. In a
headless or SSH session, it prints a browser URL instead. Open that URL from a
machine on the same network, confirm the three panes, then press Ctrl-C in the
terminal.

Keep the e-stop within reach. From another terminal, end the rollout cleanly:

```bash
touch /tmp/yam_done
```

`run_task.sh` checks the inference server, camera server, and CAN links before
moving. On exit it homes and de-energizes both arms. Omitting
`YAM_MAX_JOINT_VEL` uses the reference rig's 2.2 rad/s limit; raise the limit
only after validating your own rig at low speed.

To use a server on another machine:

```bash
YAM_SERVER=192.168.1.10:8202 YAM_MAX_JOINT_VEL=0.5 \
  bash run_task.sh "pick up the object and put it in the basket"
```

To serve a compatible TensorRT engine bundle, replace the server command in
Terminal 2 with:

```bash
vla-edge-serve --embodiment bimanual-yam --backend tensorrt \
  --engine-dir /path/to/engine-bundle --fast-vision
```

The released bundle includes its processor, normalization data, embeddings,
and compiled-flow weights. Serving it does not contact Hugging Face or load
the full PyTorch checkpoint. A locally built engine set without that compact
host runtime resolves only the required checkpoint weights and says so in the
startup log.

Rollouts and starting-scene snapshots are saved under `yam_eval_runs/`.
Advanced rollout options are documented at the top of `run_task.sh`.
