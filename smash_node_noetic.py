#!/usr/bin/env python3
"""
MoveIt1/ROS Noetic execution node for topple planner YAML outputs.

- Supports Move and Smash actions from topple_tp.py generated YAML.
- Uses AprilTag perception adapter to refresh object poses in planning scene.
- Plans+executes each action immediately (sequentially).
- Includes lifelong loop mode that repeatedly generates random paired problems
  (same seed with and without topple) and runs both.
"""

import argparse
import copy
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

try:
    import numpy as np
    import rospy
    import moveit_commander
    import tf
    import tf.transformations as tft
    from geometry_msgs.msg import PoseStamped, Pose
except Exception:
    np = None
    rospy = None
    moveit_commander = None
    tf = None
    tft = None


def face_pose_to_cube_center(tag_id: int, face_pose: Pose, cube_size: float = 0.04) -> Pose:
    face = tag_id % 6
    offset = cube_size / 2.0

    face_normals = {
        0: np.array([0, 0, -1]),
        1: np.array([-1, 0, 0]),
        2: np.array([0, -1, 0]),
        3: np.array([+1, 0, 0]),
        4: np.array([0, +1, 0]),
        5: np.array([0, 0, +1]),
    }

    normal_local = face_normals[face] * offset
    q = [
        face_pose.orientation.x,
        face_pose.orientation.y,
        face_pose.orientation.z,
        face_pose.orientation.w,
    ]
    R = tft.quaternion_matrix(q)[0:3, 0:3]
    normal_world = R.dot(normal_local)

    center = Pose()
    center.position.x = face_pose.position.x + normal_world[0]
    center.position.y = face_pose.position.y + normal_world[1]
    center.position.z = face_pose.position.z + normal_world[2]
    center.orientation = face_pose.orientation
    return center

def update_height(height: float, block_size: float, offset: float = 0.001) -> float:
    z_base = block_size / 2.0
    z_inc = block_size
    level = int(round((height - z_base) / z_inc))
    z_updated = round(z_base + level * z_inc, 3)
    z_updated = z_updated + level * offset
    return z_updated


def parse_block_size(size_spec) -> float:
    if isinstance(size_spec, (int, float)):
        return float(size_spec)
    if isinstance(size_spec, str):
        text = size_spec.strip().lower()
        if text.endswith('cm'):
            return float(text[:-2]) / 100.0
        if text.endswith('m'):
            return float(text[:-1])
        return float(text)
    raise ValueError(f"Unsupported block size spec: {size_spec}")


@dataclass
class BlockInfo:
    tag_id: str
    name: str
    size: float
    tag_pose: Pose
    block_pose: Pose


class AprilTagProvider:
    """Adapter for AprilTag perception using ROS TF."""

    def __init__(self, tag_to_block: Dict[str, Tuple[str, float]]):
        self.tag_to_block = tag_to_block
        self.listener = tf.TransformListener()

    def _lookup_tag_pose(self, tag_frame: str) -> Optional[Pose]:
        rate = rospy.Rate(10)
        attempts = 0

        while attempts < 10 and not rospy.is_shutdown():
            try:
                trans, rot = self.listener.lookupTransform('/world', '/' + tag_frame, rospy.Time(0))
                pose = Pose()
                pose.position.x = trans[0]
                pose.position.y = trans[1]
                pose.position.z = trans[2]
                pose.orientation.x = rot[0]
                pose.orientation.y = rot[1]
                pose.orientation.z = rot[2]
                pose.orientation.w = rot[3]
                return pose
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                attempts += 1
                rate.sleep()
        return None

    def lookup_block(self, block_name: str) -> BlockInfo:
        for tag_id, (name, size) in self.tag_to_block.items():
            if name != block_name:
                continue

            frame = name
            tag_pose = self._lookup_tag_pose(frame)
            if tag_pose is None:
                continue

            block_size = parse_block_size(size)
            cube_pose = face_pose_to_cube_center(int(tag_id), tag_pose, cube_size=block_size)
            cube_pose.position.z = update_height(cube_pose.position.z, block_size=block_size, offset=0.001)
            return BlockInfo(
                tag_id=str(tag_id),
                name=block_name,
                size=block_size,
                tag_pose=tag_pose,
                block_pose=cube_pose,
            )

        raise RuntimeError(f"Block '{block_name}' not detected.")

    def lookup_many(self, block_names: List[str]) -> Dict[str, BlockInfo]:
        result: Dict[str, BlockInfo] = {}
        for name in block_names:
            try:
                result[name] = self.lookup_block(name)
            except RuntimeError:
                pass
        return result


def create_tag_mapping() -> Dict[str, Tuple[str, float]]:
    tag_to_block: Dict[str, Tuple[str, float]] = {}

    # Static block-size map in meters. Edit this map to match your physical setup.
    block_size_map = {
        'blue1': '5cm',
        'red1': '4cm',
        'green1': '5cm',
        'yellow1': '5cm',
        'orange1': '5cm',
        'blue2': '5cm',
    }

    block_names = [
        'blue1',
        'red1',
        'green1',
        'yellow1',
        'orange1',
        'blue2',
    ]

    for n, block in enumerate(block_names):
        block_size = parse_block_size(block_size_map[block])
        for i in range(6):
            tag_id = 6 * n + i
            tag_to_block[str(tag_id)] = (block, block_size)

    return tag_to_block


class ToppleExecutor:
    def __init__(self, group_name: str = "xarm7", ee_link: Optional[str] = None):
        if moveit_commander is None:
            raise RuntimeError("ROS/MoveIt python modules are unavailable in this environment.")

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("topple_executor_noetic", anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.group = moveit_commander.MoveGroupCommander(group_name)
        if ee_link:
            self.group.set_end_effector_link(ee_link)

        self.group.set_planning_time(5.0)
        self.group.set_num_planning_attempts(10)
        self.group.allow_replanning(True)

    def setup_static_scene(self):
        world = "world"
        self.scene.remove_world_object("table")
        self.scene.remove_world_object("wall_left")
        self.scene.remove_world_object("wall_right")
        time.sleep(0.2)

        table = PoseStamped()
        table.header.frame_id = world
        table.pose.position.x = 0.45
        table.pose.position.y = 0.0
        table.pose.position.z = -0.025
        table.pose.orientation.w = 1.0
        self.scene.add_box("table", table, size=(0.8, 0.8, 0.05))

        left = PoseStamped()
        left.header.frame_id = world
        left.pose.position.x = 0.45
        left.pose.position.y = 0.45
        left.pose.position.z = 0.15
        left.pose.orientation.w = 1.0
        self.scene.add_box("wall_left", left, size=(0.8, 0.02, 0.3))

        right = PoseStamped()
        right.header.frame_id = world
        right.pose.position.x = 0.45
        right.pose.position.y = -0.45
        right.pose.position.z = 0.15
        right.pose.orientation.w = 1.0
        self.scene.add_box("wall_right", right, size=(0.8, 0.02, 0.3))

        rospy.sleep(0.5)

    def _pose_stamped(self, pose: Pose, frame: str = "world") -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id = frame
        p.pose = pose
        return p

    def upsert_block_collision(self, block: BlockInfo):
        self.scene.remove_world_object(block.name)
        rospy.sleep(0.05)
        self.scene.add_box(block.name, self._pose_stamped(block.block_pose), (block.size, block.size, block.size))

    def remove_blocks_from_scene(self, block_names: List[str]):
        for name in block_names:
            self.scene.remove_world_object(name)
        rospy.sleep(0.1)

    def refresh_blocks_from_perception(self, provider: AprilTagProvider, block_names: List[str]):
        sensed = provider.lookup_many(block_names)
        for bi in sensed.values():
            self.upsert_block_collision(bi)

    def _set_pose_target_and_exec(self, pose: Pose) -> bool:
        self.group.set_pose_target(pose)
        ok = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        return bool(ok)

    @staticmethod
    def _quat_from_yaw(yaw: float):
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return (0.0, 0.0, qz, qw)

    def _cartesian_push(self, dx: float, dy: float, dz: float = 0.0) -> bool:
        waypoints = []
        cur = self.group.get_current_pose().pose
        tgt = copy.deepcopy(cur)
        tgt.position.x += dx
        tgt.position.y += dy
        tgt.position.z += dz
        waypoints.append(tgt)
        plan, fraction = self.group.compute_cartesian_path(waypoints, 0.005, 0.0)
        if fraction < 0.95:
            return False
        return bool(self.group.execute(plan, wait=True))

    def execute_move(self, block: str, src_pose: Pose, dst_pose: Pose, safe_lift: float = 0.10) -> bool:
        pre_grasp = copy.deepcopy(src_pose)
        pre_grasp.position.z += safe_lift
        grasp = copy.deepcopy(src_pose)
        grasp.position.z += 0.03
        pre_place = copy.deepcopy(dst_pose)
        pre_place.position.z += safe_lift
        place = copy.deepcopy(dst_pose)
        place.position.z += 0.03

        q = self._quat_from_yaw(0.0)
        for p in [pre_grasp, grasp, pre_place, place]:
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q

        # Gripper actions are intentionally placeholders (wire to your gripper controller)
        return (
            self._set_pose_target_and_exec(pre_grasp)
            and self._set_pose_target_and_exec(grasp)
            and self._set_pose_target_and_exec(pre_grasp)
            and self._set_pose_target_and_exec(pre_place)
            and self._set_pose_target_and_exec(place)
            and self._set_pose_target_and_exec(pre_place)
        )

    def execute_smash(
        self,
        block_pose: Pose,
        direction: str,
        approach_offset: float = 0.08,
        push_dist: float = 0.12,
        retreat: float = 0.08,
    ) -> bool:
        direction_map = {
            "+x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, math.pi),
            "+y": (0.0, 1.0, math.pi / 2.0),
            "-y": (0.0, -1.0, -math.pi / 2.0),
        }
        if direction not in direction_map:
            raise ValueError(f"Unsupported smash direction: {direction}")

        ux, uy, yaw = direction_map[direction]
        q = self._quat_from_yaw(yaw)

        pre = copy.deepcopy(block_pose)
        pre.position.x -= ux * approach_offset
        pre.position.y -= uy * approach_offset
        pre.position.z += 0.04
        pre.orientation.x, pre.orientation.y, pre.orientation.z, pre.orientation.w = q

        touch = copy.deepcopy(pre)
        touch.position.x += ux * 0.06
        touch.position.y += uy * 0.06

        if not self._set_pose_target_and_exec(pre):
            return False
        if not self._set_pose_target_and_exec(touch):
            return False
        if not self._cartesian_push(ux * push_dist, uy * push_dist, 0.0):
            return False
        return self._cartesian_push(-ux * retreat, -uy * retreat, 0.02)


def load_problem(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_pose_from_pad(problem: dict, pad: int, height: int) -> Pose:
    coords = problem["pads"][pad]
    z_base = float(problem.get("z_base", 0.025))
    z_inc = float(problem.get("z_increment", 0.05))
    p = Pose()
    p.position.x = float(coords["x"])
    p.position.y = float(coords["y"])
    p.position.z = z_base + height * z_inc
    p.orientation.w = 1.0
    return p


def run_problem_once(yaml_path: str, executor: ToppleExecutor, provider: AprilTagProvider):
    prob = load_problem(yaml_path)
    actions = prob.get("actions", [])

    all_names = []
    for section in ("initial", "goal"):
        for e in prob.get(section, []):
            all_names.append(e["name"])
    all_names = sorted(set(all_names))

    executor.setup_static_scene()
    executor.refresh_blocks_from_perception(provider, all_names)

    for i, action in enumerate(actions):
        typ = action["type"]
        rospy.loginfo(f"[{i+1}/{len(actions)}] Action: {typ}")

        if typ == "Move":
            name = action["block"]
            current = provider.lookup_block(name)
            src = current.block_pose
            to_pad = int(action["to"]["pad"])
            to_h = int(action["to"]["height"])
            dst = get_pose_from_pad(prob, to_pad, to_h)

            ok = executor.execute_move(name, src, dst)
            if not ok:
                raise RuntimeError(f"Move failed for block {name}")
            executor.refresh_blocks_from_perception(provider, [name])

        elif typ == "Smash":
            blocks = list(action["blocks"])
            dirn = action.get("dir", "+x")
            base = blocks[0]  # bottom block according to YAML format
            base_pose = provider.lookup_block(base).block_pose

            executor.remove_blocks_from_scene(blocks)
            ok = executor.execute_smash(base_pose, dirn)
            if not ok:
                raise RuntimeError(f"Smash failed for blocks {blocks}")
            executor.refresh_blocks_from_perception(provider, blocks)

        else:
            raise ValueError(f"Unsupported action type: {typ}")


def generate_problem(
    topple_tp_path: str,
    output_yaml: str,
    topple: int,
    num_objects: int,
    seed: int,
    gurobi_time: int,
    num_locations: int,
    single_goal: int,
):
    cmd = [
        sys.executable,
        topple_tp_path,
        str(topple),
        str(num_objects),
        str(seed),
        str(gurobi_time),
        str(num_locations),
        str(single_goal),
        output_yaml,
    ]
    subprocess.run(cmd, check=True)


def lifelong_demo_loop(args, executor: ToppleExecutor, provider: AprilTagProvider):
    os.makedirs(args.output_dir, exist_ok=True)
    cycle = 0
    while not rospy.is_shutdown():
        cycle += 1
        seed = int(time.time()) % 100000 + cycle

        topple_yaml = os.path.join(args.output_dir, f"problem_topple_seed{seed}.yaml")
        notopple_yaml = os.path.join(args.output_dir, f"problem_notopple_seed{seed}.yaml")

        generate_problem(
            args.topple_tp,
            topple_yaml,
            topple=1,
            num_objects=args.num_objects,
            seed=seed,
            gurobi_time=args.gurobi_time,
            num_locations=args.num_locations,
            single_goal=args.single_goal,
        )
        generate_problem(
            args.topple_tp,
            notopple_yaml,
            topple=0,
            num_objects=args.num_objects,
            seed=seed,
            gurobi_time=args.gurobi_time,
            num_locations=args.num_locations,
            single_goal=args.single_goal,
        )

        rospy.loginfo(f"Cycle {cycle}: execute topple scenario")
        run_problem_once(topple_yaml, executor, provider)
        rospy.loginfo(f"Cycle {cycle}: execute non-topple scenario")
        run_problem_once(notopple_yaml, executor, provider)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topple-tp", default="./topple_tp.py")
    p.add_argument("--group", default="xarm7")
    p.add_argument("--ee-link", default=None)
    p.add_argument("--output-dir", default="./problems/generated")
    p.add_argument("--num-objects", type=int, default=9)
    p.add_argument("--gurobi-time", type=int, default=10)
    p.add_argument("--num-locations", type=int, default=8)
    p.add_argument("--single-goal", type=int, default=0)
    p.add_argument("--single-problem-yaml", default="")
    p.add_argument("--lifelong", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    tag_to_block = create_tag_mapping()
    provider = AprilTagProvider(tag_to_block)
    executor = ToppleExecutor(group_name=args.group, ee_link=args.ee_link)

    if args.lifelong:
        lifelong_demo_loop(args, executor, provider)
    else:
        if not args.single_problem_yaml:
            raise ValueError("Provide --single-problem-yaml when not using --lifelong")
        run_problem_once(args.single_problem_yaml, executor, provider)


if __name__ == "__main__":
    main()
