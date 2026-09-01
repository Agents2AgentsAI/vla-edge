import importlib.util
import math
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[1] / "examples" / "bimanual-yam" / "configure_rig.py"
)
SPEC = importlib.util.spec_from_file_location("configure_rig", MODULE_PATH)
assert SPEC and SPEC.loader
configure_rig = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = configure_rig
SPEC.loader.exec_module(configure_rig)


def test_render_configs_preserves_unrelated_content_and_comments():
    left = """\
sensors:
  cameras:
    left_camera:
      device_id: "old-left"  # wrist
    front_camera:
      device_id: "old-front"
    right_camera:
      device_id: "old-right"
robot:
  _target_: example.Robot
  channel: old-can
agent:
  start_joints: [0, 0, 0, 0, 0, 0, 0]
hz: 30
"""
    right = """\
robot:
  _target_: example.Robot
  channel: old-can
agent:
  start_joints: [0, 0, 0, 0, 0, 0, 0]
hz: 30
"""

    rendered_left, rendered_right = configure_rig.render_configs(
        left,
        right,
        {
            "front_camera": "front-serial",
            "left_camera": "left-serial",
            "right_camera": "right-serial",
        },
        "can-left",
        "can-right",
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0],
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0],
    )

    assert 'device_id: "left-serial"  # wrist' in rendered_left
    assert 'device_id: "front-serial"' in rendered_left
    assert 'device_id: "right-serial"' in rendered_left
    assert 'channel: "can-left"' in rendered_left
    assert 'channel: "can-right"' in rendered_right
    assert "hz: 30" in rendered_left
    assert "hz: 30" in rendered_right


def test_can_json_parsing_reports_state_bitrate_and_parent():
    links = configure_rig._can_links_from_json(
        [
            {
                "ifname": "can7",
                "flags": ["UP", "LOWER_UP"],
                "parentbus": "usb",
                "parentdev": "1-2.3:1.0",
                "linkinfo": {"info_data": {"bittiming": {"bitrate": 1_000_000}}},
            }
        ]
    )

    assert links == [
        configure_rig.CanLink(
            name="can7", up=True, bitrate=1_000_000, parent="usb/1-2.3:1.0"
        )
    ]


def test_can_motion_identifies_the_bus_that_moved():
    before = {
        "can_left": {1: 0.0, 4: 0.1},
        "can_right": {1: 0.0, 4: -0.1},
    }
    after = {
        "can_left": {1: 0.0, 4: 0.1 + math.radians(18)},
        "can_right": {1: math.radians(0.2), 4: -0.1},
    }

    motion = configure_rig._can_motion_degrees(before, after)

    assert abs(motion["can_left"] - 18.0) < 1e-9
    assert configure_rig._identify_moved_can(motion) == "can_left"


def test_can_motion_rejects_an_inconclusive_result():
    assert (
        configure_rig._identify_moved_can({"can_left": 8.0, "can_right": 6.0}) is None
    )


def test_can_verification_uses_the_hand_moved_bus(monkeypatch, capsys):
    answers = iter(["", "", ""])
    readings = {
        "can_left": iter([{4: 0.0}, {4: math.radians(16)}]),
        "can_right": iter([{4: 0.0}, {4: math.radians(0.2)}]),
    }
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        configure_rig,
        "_read_can_positions",
        lambda channel: next(readings[channel]),
    )
    links = [
        configure_rig.CanLink("can_left", True, 1_000_000, "usb/1-2.2:1.0"),
        configure_rig.CanLink("can_right", True, 1_000_000, "usb/1-2.3:1.0"),
    ]

    assert configure_rig._verify_left_can(links) == "can_left"
    assert "Detected the LEFT arm on can_left." in capsys.readouterr().out


def test_config_writes_keep_first_backup(tmp_path):
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    left.write_text("left-original\n")
    right.write_text("right-original\n")

    configure_rig._write_configs(left, right, "left-new\n", "right-new\n")
    configure_rig._write_configs(left, right, "left-newer\n", "right-newer\n")

    assert left.read_text() == "left-newer\n"
    assert right.read_text() == "right-newer\n"
    assert (tmp_path / "left.yaml.setup.bak").read_text() == "left-original\n"
    assert (tmp_path / "right.yaml.setup.bak").read_text() == "right-original\n"
