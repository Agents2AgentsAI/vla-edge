# Bimanual YAM

Run a vision-language-action policy on two I2RT YAM arms and three RealSense
cameras. Setup, camera serving, task launching and recording are shared across
models; model-specific inference and control options are grouped below.

| Model | Policy name | Support |
|---|---|---|
| [ABC-VLA](#abc-vla) | `abcvla-bimanual-yam` | Thor TensorRT; native reference |
| [MolmoAct2](#molmoact2) | `molmoact2-bimanual-yam` | PyTorch or Thor TensorRT |
| [π0.5](#pi05) | `pi05-bimanual-yam` | Thor TensorRT |

**Working directory:** all commands below run from the repository root.
If you are in `examples/bimanual-yam`, first run `cd ../..`. Activate the
repository's environment in each terminal with `source .venv/bin/activate`.

## 1. Shared setup

The reference machine is a Jetson AGX Thor. Inference can also run on a separate
machine. The rig needs:

- Two YAM arms on separate 1 Mbit/s CAN interfaces
- Three RealSense cameras: scene, left wrist and right wrist
- A tested hardware e-stop

From a fresh checkout, install the shared environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
./examples/bimanual-yam/setup_jetson_thor.sh
python examples/bimanual-yam/tests/run_tests.py
```

Setup installs the dependencies for all three models and the
`vla-edge-serve` command into `.venv/bin`. The combined test command covers
both controllers and homing/shutdown. It runs without GPU, camera or robot
access, using quiet output with warnings and failure details reported.

## 2. Configure the rig

```bash
python examples/bimanual-yam/configure_rig.py
python examples/bimanual-yam/home_arms.py
python examples/bimanual-yam/calibrate_grippers.py
```

`configure_rig.py` identifies cameras and CAN interfaces, saves a labeled
camera snapshot, asks for the physical left/right assignments and safe start
poses, and updates `configs/yam_left.yaml` and `configs/yam_right.yaml` inside
this example. It brings the selected CAN links up at 1 Mbit/s. Press Enter to
keep a value shown in brackets. CAN identification reads positions without
enabling motors and asks you to hand-move one joint on the left arm.

Before homing, verify both start poses and the gripper open direction in
`home_arms.py`. The reference grippers open in the negative motor direction.
Configure the motor watchdog as required by I2RT; the reference homing path
assumes it is disabled.

- `start_joints`: six arm angles in radians followed by a normalized gripper
  position (`0` closed, `1` open). These are commanded targets.
- `home_arms.py`: moves both arms to encoder zero, opens the grippers and
  disables all 14 motors.
- `calibrate_grippers.py`: moves the grippers through their travel and saves
  `[closed, open]` limits. Repeat after changing a gripper, motor or motor zero.
  Saved limits avoid a hard-stop calibration sweep at every launch. Calibration
  refuses to start above a gripper rotor temperature of 45 C and disables each
  motor when its sweep ends.

The arm-joint encoder zeros must already be calibrated. Writing joint zeros
requires a hardware-specific fixture and procedure and is not part of setup.

For a disable-and-status check:

```bash
python examples/bimanual-yam/home_arms.py --status
```

This sends no position commands. It stops a running controller first, which
may perform its normal parking sequence, then leaves the motors disabled.

## 3. Start the cameras — terminal 1

```bash
source .venv/bin/activate
./examples/bimanual-yam/start_camera_server.sh
```

Leave this process running. Repeating the command detects an existing healthy
server without reopening the cameras. Ctrl-C releases the cameras.

To check scene/left/right camera order, open a separate terminal:

```bash
source .venv/bin/activate
python examples/bimanual-yam/camera_client.py --mode sub
```

The viewer subscribes to the camera server. It opens a window when a display
is available, or prints a browser URL for headless/SSH use. Confirm the three
panes, then stop the viewer with Ctrl-C. The camera server can remain running.

## 4. Choose a model — terminal 2

Activate `.venv` and start one of the following servers. All examples use
port **8202**, which is also the task launcher's default.

### ABC-VLA

Download the [ABC-VLA Thor bundle](https://huggingface.co/agents2agents/ABC-VLA-Jetson-Thor).
If you already have it locally, use that directory as `--engine-dir`.

```bash
hf download agents2agents/ABC-VLA-Jetson-Thor \
  --local-dir examples/bimanual-yam/engines/abc-vla
vla-edge-serve --policy abcvla-bimanual-yam --backend tensorrt \
  --engine-dir examples/bimanual-yam/engines/abc-vla
```

The bundle requires Thor, CUDA 13.2 and **TensorRT 10.16.2.10**. The server
checks its files and hardware requirements and completes warmup before
accepting requests. It retains the checkpoint's three RGB cameras, 30×14
absolute action shape and 10-step sampler.

**Task settings (terminal 3)**

| Option for `run_task.sh` | Default | Meaning |
|---|---|---|
| `--rtc-prefix-length` | `5` | Remaining committed commands supplied to the next prediction |
| `--execute-chunk-dim` | `6` | New action rows executed per cycle |

ABC's `--check` preflight validates the policy, calibration, camera frames and
CAN links without enabling motors. A controller fault skips homing and
disables both arms.

The prefix must be 1–7 rows and shorter than the executed chunk; their sum
must fit the 30-row model horizon. For example:

```bash
./examples/bimanual-yam/run_task.sh "fold and stack the t-shirts" \
  --rtc-prefix-length 5 --execute-chunk-dim 6
```

ABC uses measured gripper positions and a 30 Hz action clock. It requests the
next prediction while the committed prefix executes. Arm setpoints are
rate-limited before the chunk is committed, so the RTC prefix exactly matches
the commands that will execute. At 3 rad/s, the nominal limit is 0.1 rad per
arm joint per tick. Gripper commands retain native ABC behavior.

**Optional offline checks**

```bash
python -m vla_edge.scripts.verify_release \
  --bundle examples/bimanual-yam/engines/abc-vla
python -m vla_edge.scripts.verify_abcvla_device \
  --bundle examples/bimanual-yam/engines/abc-vla
python -m vla_edge.scripts.smoke_abcvla \
  --bundle examples/bimanual-yam/engines/abc-vla
```

These checks do not open robot or camera hardware. The device and smoke
checks use the GPU. See the [ABC bundle reference](../../recipes/abcvla-jetson-thor/README.md)
for native-checkpoint comparisons, saved observations and wire protocols.

### MolmoAct2

**PyTorch reference**

The [BimanualYAM checkpoint](https://huggingface.co/allenai/MolmoAct2-BimanualYAM)
downloads on the first launch:

```bash
vla-edge-serve --policy molmoact2-bimanual-yam --backend torch
```

The original `--embodiment bimanual-yam --backend torch` command also works.

**Thor TensorRT bundle**

```bash
hf download agents2agents/MolmoAct2-Jetson-Thor \
  --local-dir examples/bimanual-yam/engines/molmoact2
vla-edge-serve --policy molmoact2-bimanual-yam --backend tensorrt \
  --engine-dir examples/bimanual-yam/engines/molmoact2 --fast-vision
```

For a local bundle, skip the download and set `--engine-dir` to its directory.
The released bundle includes the processor, normalization data, embeddings
and compiled-flow weights, so serving it needs no checkpoint download. A
locally built engine set without the compact host runtime resolves the
required checkpoint weights and reports that in its startup log.

**Task settings (terminal 3)**

MolmoAct retains its smoothing, velocity clamp and optional RTC settings.
Its `--check` preflight validates the inference server contract. For example:

```bash
YAM_MAX_JOINT_VEL=0.5 ./examples/bimanual-yam/run_task.sh \
  "pick up the rubik cube and put it in the black box"
```

Advanced options, including `YAM_RTC`, are listed in
[`run_molmoact_task.sh`](run_molmoact_task.sh). ABC's hard-prefix flags apply
only to ABC-VLA.

### Pi0.5

Download the [Pi0.5 BimanualYAM Thor bundle](https://huggingface.co/agents2agents/Pi0.5-BimanualYAM-Jetson-Thor),
or point `--engine-dir` to your existing local bundle. It includes
`pi05-serving.json` and all required inference assets.

```bash
hf download agents2agents/Pi0.5-BimanualYAM-Jetson-Thor \
  --local-dir examples/bimanual-yam/engines/pi05
vla-edge-serve --policy pi05-bimanual-yam --backend tensorrt \
  --engine-dir examples/bimanual-yam/engines/pi05
```

The bundle uses the public `robocurve/pi0.5-yam` checkpoint, three RGB cameras,
16×14 absolute joint actions and ten diffusion steps. It requires Thor,
CUDA 13.2 and TensorRT 10.16.2.10. Gripper state is the last commanded opening.

**Task settings (terminal 3)**

```bash
./examples/bimanual-yam/run_task.sh "fold the t-shirt" --check
YAM_MAX_JOINT_VEL=0.5 ./examples/bimanual-yam/run_task.sh "fold the t-shirt"
```

The task launcher detects Pi0.5 and uses its dedicated controller: synchronous
30 Hz policy chunks feed an independent 100 Hz Ruckig motion process. Motor
targets remain continuous during inference and camera waits, with bounded
velocity, acceleration and jerk. `touch /tmp/yam_done` or Ctrl-C brakes and
homes through the still-powered controller before disabling the motors.

`--check` validates the server, rig, cameras and controller dependencies without
enabling motors. `YAM_MAX_JOINT_VEL` sets the arm speed limit; acceleration and
jerk are bounded at 6 rad/s² and 60 rad/s³. `YAM_ACTION_HORIZON` selects 1–16
executed rows per chunk (default 16). Keep `YAM_ASYNC_PLAN=0` (or unset) and
`YAM_RTC=0`; Pi0.5 does not use the MolmoAct2 queue merger or ABC's prefix flags.

Recordings go to `yam_eval_runs/data/pi05/<timestamp>/`. `commands.jsonl`
contains policy goals, `motion.f64` and its JSON schema record the actual
100 Hz references and motor feedback, and `episode.h5` contains measured
states. Camera frames follow `storage.save_frames` in the rig configuration.

**Optional offline checks**

```bash
python -m vla_edge.scripts.verify_release \
  --bundle examples/bimanual-yam/engines/pi05
python -m vla_edge.scripts.smoke_pi05 \
  --bundle examples/bimanual-yam/engines/pi05
```

These commands do not access the robot or cameras. The smoke check loads
and executes every GPU stage. See the [Pi0.5 bundle guide](../../recipes/pi05-jetson-thor/README.md)
for saved-observation testing and bundle requirements.

## 5. Run and stop a task — terminal 3

Use the same task command for all three models. It reads the server's
policy identity and selects the matching controller.

```bash
source .venv/bin/activate
curl -f http://127.0.0.1:8202/act
./examples/bimanual-yam/run_task.sh "fold and stack the t-shirts" --check
./examples/bimanual-yam/run_task.sh "fold and stack the t-shirts"
```

`--check` performs preflight without enabling motors. The checks specific
to each model are described in its section above.

| Shared setting | Default | Purpose |
|---|---|---|
| `YAM_SERVER` | `127.0.0.1:8202` | Inference server address; override for another host or port |
| `YAM_MAX_JOINT_VEL` | `2.2` | Arm setpoint-rate limit in rad/s |
| `YAM_STAGE_FILE` | `/tmp/yam_done` | Task completion marker |

Validate your rig at low speed and keep the e-stop within reach. To finish the
current task from another terminal:

```bash
touch /tmp/yam_done
```

On a normal stop or Ctrl-C, the active controllers home the arms and open the
grippers while keeping the motors energized. They disable the motors only
after homing. No second controller is opened to home again. If homing fails
or is interrupted, cleanup disables the motors; its fallback never re-enables
them.

Stopping a task leaves the camera and policy servers available for the next
one. Stop those servers separately with Ctrl-C when finished.

## 6. Recordings

Default output directory: `examples/bimanual-yam/yam_eval_runs/`.

| Model | Saved output |
|---|---|
| ABC-VLA | `data/abc-vla/<timestamp>/`: `episode.h5`, `commands.jsonl`, `requests.jsonl`, run manifest |
| MolmoAct2 | Existing session directories, rollout data and starting-scene snapshots |

Set `storage.save_frames: true` in `configs/yam_left.yaml` inside this example
to save RGB frames too. Frame recording adds work to the control loop. Rig
configuration, local bundles and recorded data belong to your local setup.
