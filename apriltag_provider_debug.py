#!/usr/bin/env python

import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import numpy as np
import rospy
import tf
import tf.transformations as tft
from geometry_msgs.msg import Pose


def face_pose_to_cube_center(tag_id: int, face_pose: Pose, cube_size: float = 0.04) -> Pose:
    face = tag_id % 6
    offset = cube_size / 2.0

    # face normals in tag frame
    face_normals = {
        0: np.array([0, 0, -1]),    # top
        1: np.array([-1, 0, 0]),    # front
        2: np.array([0, -1, 0]),    # right
        3: np.array([+1, 0, 0]),    # back
        4: np.array([0, +1, 0]),    # left
        5: np.array([0, 0, +1]),    # bottom
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
                trans, rot = self.listener.lookupTransform("/world", "/" + tag_frame, rospy.Time(0))
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
        "blue1": "5cm",
        "red1": "4cm",
        "green1": "5cm",
        "yellow1": "5cm",
        "orange1": "5cm",
        "blue2": "5cm",
    }

    block_names = [
        "blue1",
        "red1",
        "green1",
        "yellow1",
        "orange1",
        "blue2",
    ]

    for n, block in enumerate(block_names):
        block_size = parse_block_size(block_size_map[block])
        for i in range(6):
            tag_id = 6 * n + i
            tag_to_block[str(tag_id)] = (block, block_size)

    return tag_to_block


def main():
    rospy.init_node("apriltag_block_detector")

    tag_mapping = create_tag_mapping()
    provider = AprilTagProvider(tag_mapping)

    block_names = [
        "blue1",
        "red1",
        "green1",
        "yellow1",
        "orange1",
        "blue2",
    ]

    print("\nAprilTag Block Detector Ready")
    print("Commands:")
    print("  block name (blue1/red1/...)")
    print("  all")
    print("  exit\n")

    while not rospy.is_shutdown():
        sys.stdout.write("Enter command: ")
        sys.stdout.flush()
        cmd = sys.stdin.readline().strip()

        if cmd == "exit":
            break

        if cmd == "all":
            results = provider.lookup_many(block_names)
            if not results:
                print("No blocks detected\n")
                continue

            for name, info in results.items():
                tp = info.tag_pose.position
                tq = info.tag_pose.orientation
                bp = info.block_pose.position
                bq = info.block_pose.orientation

                print(f"\n{name}")
                print(f" tag id: {info.tag_id}")
                print(f" tag position: {tp.x:.3f}, {tp.y:.3f}, {tp.z:.3f}")
                print(f" tag orientation: {tq.x:.3f}, {tq.y:.3f}, {tq.z:.3f}, {tq.w:.3f}")
                print(f" block position: {bp.x:.3f}, {bp.y:.3f}, {bp.z:.3f}")
                print(f" block orientation: {bq.x:.3f}, {bq.y:.3f}, {bq.z:.3f}, {bq.w:.3f}")
            print()
            continue

        if cmd not in block_names:
            print("Unknown block\n")
            continue

        try:
            info = provider.lookup_block(cmd)
            tp = info.tag_pose.position
            tq = info.tag_pose.orientation
            bp = info.block_pose.position
            bq = info.block_pose.orientation

            print(f"\n{cmd}")
            print(f" tag id: {info.tag_id}")
            print(f" tag position: {tp.x:.3f}, {tp.y:.3f}, {tp.z:.3f}")
            print(f" tag orientation: {tq.x:.3f}, {tq.y:.3f}, {tq.z:.3f}, {tq.w:.3f}")
            print(f" block position: {bp.x:.3f}, {bp.y:.3f}, {bp.z:.3f}")
            print(f" block orientation: {bq.x:.3f}, {bq.y:.3f}, {bq.z:.3f}, {bq.w:.3f}")
        except RuntimeError:
            print("Block not detected\n")


if __name__ == "__main__":
    main()
