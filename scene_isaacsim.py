#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path
import sys
import yaml
import time
import numpy as np
import random

FRANKA_STAGE_PATH = "/World/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"

SCENE_OFFSET_X = 0
SCENE_OFFSET_Y = 0.0#-0.64
SCENE_OFFSET_Z = 0.01459
#SCENE_OFFSET_Z = 0
HEIGHT_ERROR_OFFSET = 0.001


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="Path to problem.yaml")
    p.add_argument("--headless", action="store_true", help="Run headless")
    p.add_argument("--blocks_root", default="/World/Blocks", help="Container Xform for spawned blocks")
    p.add_argument("--density", type=float, default=500.0, help="kg/m^3 for mass (default: 500)")
    p.add_argument("--friction", type=float, default=1.0, help="static/dynamic friction baseline (default: 1.0)")
    return p.parse_args()

args = parse_args()

# ---------------- Isaac Sim must be created BEFORE any isaac/omni imports ----------------
ISAACSIM_ROOT = Path("/home/luhao/Tools/isaac/isaacsim-5.0.0/_build/linux-x86_64/release")

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "renderer": "RaytracedLighting",
    "headless": args.headless,
    "experience": str(ISAACSIM_ROOT/"apps"/"isaacsim.exp.full.kit"),
})

# Now it is safe to import Isaac/Omni/PXR modules
import carb
import omni.usd
import omni.graph.core as og

from isaacsim.core.api import World
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, prims, rotations, stage, viewports
from isaacsim.storage.native import get_assets_root_path

from isaacsim.core.api.objects import VisualCuboid, DynamicCuboid
from isaacsim.core.prims import RigidPrim, GeometryPrim
from isaacsim.core.api.materials.physics_material import PhysicsMaterial

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators import SingleManipulator


from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics, PhysxSchema
import usdrt.Sdf

# Enable ROS 2 bridge if desired
extensions.enable_extension("isaacsim.ros2.bridge")
extensions.enable_extension("omni.kit.widget.stage")
extensions.enable_extension("omni.kit.widget.layers")
simulation_app.update()

# ROS packages
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from std_srvs.srv import Trigger
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene, ObjectColor
# (Optional) If you prefer the service route, you can also import:
# from moveit_msgs.srv import ApplyPlanningScene

# --------------------------- Helpers ---------------------------
def _quat_mul_xyzw(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    """
    Hamilton product q = q1 ⊗ q2 with XYZW ordering.
    Both inputs and output are [x, y, z, w].
    """
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=float)

def _quat_normalize_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else q

def _mat3_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3) -> quaternion XYZW."""
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return np.array([x, y, z, w], dtype=float)

# Fixed frame transforms between ROS (X forward, Y left, Z up)
# and Isaac/USD (X right, Y forward, Z up):
#   pos:  [x', y', z'] = Rz(+90) * [x, y, z]  with y' then shifted by SCENE_OFFSET_Y
#   quat: q' = qz(+90) ⊗ q
_QZ_PLUS_90_XYZW  = rotations.euler_angles_to_quat(np.array([0.0, 0.0,  np.pi/2]), degrees=False)  # XYZW
_QZ_MINUS_90_XYZW = rotations.euler_angles_to_quat(np.array([0.0, 0.0, -np.pi/2]), degrees=False)  # XYZW

def ros_pose_to_isaac(position_ros, quat_ros_xyzw):
    """
    Convert ROS pose -> Isaac/USD pose.
    Inputs:
      position_ros: (x, y, z) in meters
      quat_ros_xyzw: (x, y, z, w) quaternion (ROS & Isaac both use XYZW)
    Returns:
      position_isaac: (x', y', z')
      quat_isaac_xyzw: (x', y', z', w')
    """
    x_ros, y_ros, z_ros = map(float, position_ros)

    # Position: apply Rz(+90) then add Y offset
    x_isaac = -y_ros + float(SCENE_OFFSET_X)
    y_isaac =  x_ros + float(SCENE_OFFSET_Y)
    z_isaac =  z_ros + float(SCENE_OFFSET_Z)

    # Orientation: q' = qz(+90) ⊗ q_ros   (left-multiply)
    q_ros = np.asarray(quat_ros_xyzw, dtype=float)
    q_isaac = _quat_mul_xyzw(_QZ_PLUS_90_XYZW, q_ros)
    q_isaac = _quat_normalize_xyzw(q_isaac)

    return np.array([x_isaac, y_isaac, z_isaac], dtype=float), q_isaac

def isaac_pose_to_ros(position_isaac, quat_isaac_xyzw):
    """
    Convert Isaac/USD pose -> ROS pose.
    Inputs:
      position_isaac: (x', y', z')
      quat_isaac_xyzw: (x', y', z', w') in XYZW
    Returns:
      position_ros: (x, y, z)
      quat_ros_xyzw: (x, y, z, w)
    """
    x_i, y_i, z_i = map(float, position_isaac)

    x_ros =  y_i - float(SCENE_OFFSET_Y)
    y_ros = -x_i + float(SCENE_OFFSET_X)
    z_ros =  z_i - float(SCENE_OFFSET_Z)

    # Orientation: q_ros = qz(-90) ⊗ q_isaac
    q_i = np.asarray(quat_isaac_xyzw, dtype=float)
    q_ros = _quat_mul_xyzw(_QZ_MINUS_90_XYZW, q_i)
    q_ros = _quat_normalize_xyzw(q_ros)

    return np.array([x_ros, y_ros, z_ros], dtype=float), q_ros

def get_prim_world_pose_xyzw(prim_path: str):
    """Return (position_xyz, quat_xyzw) from composed USD xform."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # 4x4 Gf.Matrix4d
    M_np = np.array(M, dtype=float).reshape(4, 4)
    pos = M_np[:3, 3]
    R = M_np[:3, :3]
    q_xyzw = _mat3_to_quat_xyzw(R)
    print(prim_path, pos, q_xyzw)
    return pos, q_xyzw


def log_and_exit(msg: str, code: int = 1):
    carb.log_error(msg)
    try:
        simulation_app.close()
    except Exception:
        pass
    sys.exit(code)

def load_problem(yaml_path: str):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    for k in ("pads", "z_base", "z_increment", "initial"):
        if k not in data:
            raise ValueError(f"YAML missing required key: {k}")
    return data

_NAME_PALETTE = {
    # RED family (distinct: red, magenta-ish, orange-red)
    "red1":   np.array([0.85, 0.15, 0.15], dtype=float),  # vivid red
    "red2":   np.array([0.75, 0.10, 0.40], dtype=float),  # crimson/magenta
    "red3":   np.array([0.90, 0.35, 0.10], dtype=float),  # orange-red

    # GREEN family (distinct: green, teal, lime)
    "green1": np.array([0.15, 0.75, 0.20], dtype=float),  # classic green
    "green2": np.array([0.05, 0.60, 0.55], dtype=float),  # teal (bluish green)
    "green3": np.array([0.75, 0.75, 0.10], dtype=float),  # lime/yellow-green

    # BLUE family (distinct: blue, sky, violet)
    "blue1":  np.array([0.15, 0.25, 0.85], dtype=float),  # deep blue
    "blue2":  np.array([0.20, 0.65, 0.95], dtype=float),  # sky blue
    "blue3":  np.array([0.55, 0.35, 0.75], dtype=float),  # violet
}

def color_from_name(name: str) -> np.ndarray:
    n = (name or "").strip().lower()

    # Exact per-name mapping for the 9 canonical blocks
    if n in _NAME_PALETTE:
        return _NAME_PALETTE[n].copy()

    # Family fallbacks (if you ever pass e.g., 'red4' or just 'red')
    if n.startswith("red"):
        return np.array([0.85, 0.15, 0.15], dtype=float)
    if n.startswith("green"):
        return np.array([0.15, 0.75, 0.20], dtype=float)
    if n.startswith("blue"):
        return np.array([0.15, 0.25, 0.85], dtype=float)

    # Generic fallback
    return np.array([0.60, 0.60, 0.60], dtype=float)

'''
_GOAL_NAME   = "green1"  # fixed goal name
_GOAL_COLOR  = np.array([0.745098, 0.513725, 0.054902], dtype=float)  # #BE830E (gold)
#_OTHERS_COLOR = np.array([0.70, 0.70, 0.70], dtype=float)             # neutral gray
_OTHERS_COLOR = np.array([0.10, 0.10, 0.10], dtype=float)              # dark gray (near-black)

def color_from_name(name: str) -> np.ndarray:
    n = (name or "").strip().lower()
    return _GOAL_COLOR.copy() if n == _GOAL_NAME else _OTHERS_COLOR.copy()
'''

def sanitize(s: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in (s or ""))
    return s or "block"

def ensure_xform(path: str):
    """Create/replace an Xform container at the prim path."""
    stage_if = omni.usd.get_context().get_stage()
    prim = stage_if.GetPrimAtPath(path)
    if prim.IsValid():
        stage_if.RemovePrim(Sdf.Path(path))
    UsdGeom.Xform.Define(stage_if, Sdf.Path(path))

def list_block_prims(blocks_root: str = "/World/Blocks"):
    stage = omni.usd.get_context().get_stage()
    root  = stage.GetPrimAtPath(blocks_root)
    if not root.IsValid():
        return []
    out = []
    for prim in Usd.PrimRange(root):            # iterate subtree properly
        if prim.GetTypeName() == "Cube":
            name = prim.GetName()
            size_attr = UsdGeom.Cube(prim).GetSizeAttr()
            size = float(size_attr.Get()) if size_attr.HasAuthoredValue() else 0.05
            out.append((name, prim.GetPath().pathString, size))

    print(out)
    return out

BLOCKS = {}  # name -> {"path": str, "rb": RigidPrim, "gp": GeometryPrim, "side": float}
CFGS = {}

def register_block(name: str, prim_path: str, side: float, mass: float = None):
    rb = RigidPrim(prim_path)
    gp = GeometryPrim(prim_path)

    # Optional: set mass/collision once, if not already done
    if mass is not None:
        # set_masses expects array-like
        rb.set_masses(np.array([float(mass)], dtype=float))
    gp.apply_collision_apis()

    BLOCKS[name] = {"path": prim_path, "rb": rb, "gp": gp, "side": float(side)}

    '''
    print("rb:")
    pos, quat = rb.get_world_poses()
    print(isaac_pose_to_ros(pos[0], quat[0]))
    print("gp:")
    pos, quat = gp.get_world_poses()
    print(isaac_pose_to_ros(pos[0], quat[0]))
    '''

def get_block_world_pose(world: World, name: str, usd: bool = False):
    """Return (pos_xyz, quat_xyzw) for a registered block using its RigidPrim handle."""
    h = BLOCKS.get(name)
    if h is None:
        raise KeyError(f"Block not registered: {name}")
    # RigidPrim.get_world_poses returns (N,3),(N,4), even for a single prim
    pos, quat = world.scene.get_object(name).get_world_pose()
    '''
    pos = np.asarray(pos_np, dtype=float).reshape(-1, 3)[0]
    quat = np.asarray(quat_xyzw, dtype=float).reshape(-1, 4)[0]
    '''
    print(pos, quat)

    return pos, quat

def update_height(height, offset):
    level = int(round((height - CFGS['z_base']) / CFGS['z_inc']))
    z_updated = round(CFGS['z_base'] + level * CFGS['z_inc'], 3)
    z_updated = z_updated + level * offset

    return z_updated

def spawn_blocks_from_yaml(
    world: World,
    yaml_path: str,
    blocks_root: str,
):
    """Spawn colored cubes (VisualCuboid + RigidPrim) with physics + material."""
    problem = load_problem(yaml_path)
    pads = problem["pads"]
    z_base = float(problem["z_base"])
    z_inc = float(problem["z_increment"])
    initial = problem["initial"]

    CFGS['z_base'] = z_base
    CFGS['z_inc'] = z_inc

    side = z_inc
    #volume = side * side * side
    #mass = float(density) * volume

    spawned = []
    for item in initial:
        name = str(item["name"])
        pad_id = int(item["pad"])
        h = int(item["height"])
        if pad_id not in pads:
            raise ValueError(f"Pad {pad_id} not found in pads")

        x = float(pads[pad_id]["x"])
        y = float(pads[pad_id]["y"])
        z = float(z_base + h * z_inc)   # z_base already includes half block per your spec

        prim_path = f"{blocks_root}/{sanitize(name)}"
        color = color_from_name(name)

        pos, quat = ros_pose_to_isaac(np.array([x, y, z]), np.array([0, 0, 0, 1]))

        mat_path = "/World/PhysicsMaterials/BlockMaterial"
        material = PhysicsMaterial(
            prim_path=mat_path,
            static_friction=0.20,
            dynamic_friction=0.15,
            restitution=0.0,
        )

        world.scene.add(
            DynamicCuboid(
                name=name,
                position=pos,
                orientation=quat,
                prim_path=prim_path,
                size=float(side),
                color=color,
                physics_material=material,
            )
        )

        '''
        # Enable physics on same prim
        register_block(name=name, prim_path=prim_path, side=side, mass=mass)
        '''
        BLOCKS[name] = {"path": prim_path}

        '''
        # Bind physics material (friction/restitution for contacts)
        try:
            gp.apply_physics_material(material)
        except Exception:
            # If wrapper API changes, you can bind via USD/PhysX schema here.
            pass
        '''

        spawned.append(name)

    carb.log_info(f"[spawn] Created {len(spawned)} blocks at {blocks_root}: {spawned}")


def build_planning_scene_from_isaac(world: World = None,
                                    blocks_root: str = "/World/Blocks",
                                    planning_frame: str = "world") -> PlanningScene:
    ps = PlanningScene()
    ps.is_diff = True  # important: "apply" as a diff

    # Header for individual CollisionObjects
    hdr = Header()
    hdr.frame_id = planning_frame

    object_colors = []
    collision_objects = []

    for name, prim_path, side in list_block_prims(blocks_root):
        # 1) Isaac world pose -> ROS pose in planning_frame
        #pos_i, quat_i = get_block_world_pose(name, usd=True)
        pos_i, quat_i = world.scene.get_object(name).get_world_pose()
        pos_r, quat_r = isaac_pose_to_ros(pos_i, quat_i)

        # 2) CollisionObject as a BOX
        co = CollisionObject()
        co.id = name + "_cube"
        co.header = hdr
        co.operation = CollisionObject.ADD  # add/replace

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(side), float(side), float(side)]  # x, y, z

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [round(v, 3) for v in pos_r.tolist()]#pos_r.tolist()
        #pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat_r.tolist()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [0.0, 0.0, 0.0, 1.0]

        pose.position.z = update_height(pose.position.z, HEIGHT_ERROR_OFFSET)

        co.primitives.append(box)
        co.primitive_poses.append(pose)
        collision_objects.append(co)

        # 3) Color (RViz will display if MotionPlanning scene coloring is enabled)
        rgba = color_from_name(name)
        oc = ObjectColor()
        oc.id = name + "_cube"
        oc.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=1.0)
        object_colors.append(oc)

    ps.world.collision_objects = collision_objects
    ps.object_colors = object_colors
    return ps

def _set(attr, create_fn, value):
    # Use your "get or create then Set" style
    if attr and attr.IsValid():
        attr.Set(value)
    else:
        create_fn().Set(value)

def tune_franka(root="/World/Franka"):
    usd_stage = omni.usd.get_context().get_stage()

    # Map: joint -> (parent link, armature, friction, max_vel)
    J = {
        "panda_joint1": ("panda_link0", 0.1, 0.5, 2.3925),
        "panda_joint2": ("panda_link1", 0.1, 0.5, 2.3925),
        "panda_joint3": ("panda_link2", 0.1, 0.5, 2.3925),
        "panda_joint4": ("panda_link3", 0.1, 0.5, 2.3925),
        "panda_joint5": ("panda_link4", None, 0.2, 2.8710),
        "panda_joint6": ("panda_link5", None, 0.2, 2.8710),
        "panda_joint7": ("panda_link6", None, 0.2, 2.8710),
    }

    for jname, (plink, armature, friction, vmax) in J.items():
        path = f"{root}/{plink}/{jname}"
        prim = usd_stage.GetPrimAtPath(path)
        if not prim:
            print(f"[tune_franka] WARN: joint prim not found: {path}")
            continue

        pj = PhysxSchema.PhysxJointAPI(prim)

        # Armature (only J1–J4)
        if armature is not None:
            _set(pj.GetArmatureAttr(), pj.CreateArmatureAttr, float(armature))

        # Coulomb joint friction
        _set(pj.GetJointFrictionAttr(), pj.CreateJointFrictionAttr, float(friction))

        # Optional: max joint velocity (rad/s)
        #_set(pj.GetMaxJointVelocityAttr(), pj.CreateMaxJointVelocityAttr, float(vmax))

        print(f"[tune_franka] {jname}: armature={armature}, friction={friction}")

    # Persist these overrides into your scene file (uncomment if desired)
    # omni.usd.get_context().save_stage()


class SmashSceneBridge(Node):
    def __init__(self,
                 world: World = None,
                 blocks_root="/World/Blocks",
                 planning_frame="world",
                 gripper: ParallelGripper = None):
        super().__init__("smash_scene_bridge")
        self.blocks_root = blocks_root
        self.planning_frame = planning_frame

        # MoveIt PlanningScene diff publisher (MoveIt listens and republishes to monitored scene)
        self.ps_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

        # Optional: if you prefer service route, you can wire a client to /apply_planning_scene
        # self.apply_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

        # Service that your C++ code calls
        self.update_srv = self.create_service(Trigger, "/update_smash_scene", self._handle_update)
        self.open_srv = self.create_service(Trigger, "/isaac_open_gripper", self._handle_gripper_open)
        self.close_srv = self.create_service(Trigger, "/isaac_close_gripper", self._handle_gripper_close)
        self.close2_srv = self.create_service(Trigger, "/isaac_full_close_gripper", self._handle_gripper_close2)

        self._publish_requested = True     # send an initial snapshot
        self._last_publish_t = 0.0
        self._world = world
        self._gripper = gripper

        self._gripper_requested = False
        self._gripper_open = True
        #self.create_timer(5, self._timer_request_publish)
        #self.create_timer(1, self._timer_gripper)

    def publish_scene_once(self) -> int:
        ps = build_planning_scene_from_isaac(
            world=self._world,
            blocks_root=self.blocks_root,
            planning_frame=self.planning_frame
        )
        self.ps_pub.publish(ps)
        return len(ps.world.collision_objects)

    def _handle_update(self, req, res):
        # just mark; actual publish happens after the sim step (main thread)
        self._publish_requested = True
        res.success = True
        res.message = "Publish requested"
        return res

    def flush_publish_if_requested(self):
        """Call from main thread AFTER simulation_context.step()."""
        if not self._publish_requested:
            return False

        self.publish_scene_once()

        print("=== Block poses (world, XYZW) ===")
        for name in sorted(BLOCKS.keys()):
            #pos, quat = get_block_world_pose(name, usd=False)
            pos, quat = self._world.scene.get_object(name).get_world_pose()
            p, q = isaac_pose_to_ros(pos, quat)
            p[2] = update_height(p[2], HEIGHT_ERROR_OFFSET)
            print(f"{name:>8s}: p=({p[0]: .3f}, {p[1]: .3f}, {p[2]: .3f})  "
                  f"q=({q[0]: .4f}, {q[1]: .4f}, {q[2]: .4f}, {q[3]: .4f})")

        print(f"[scene] Published {len(BLOCKS.keys())} objects to /planning_scene")

        self._publish_requested = False

        return True

    def flush_publish_if_due(self, now_monotonic: float):
        """Call this from the main thread AFTER step()."""
        if not self._publish_requested:
            return
        if now_monotonic - self._last_publish_t < 5:
            return

        # Build & publish scene from live Fabric (safe here, sim is idle)
        self.publish_scene_once()

        # Debug print of every block pose
        print("=== Block poses (world, XYZW) ===")
        for name in sorted(BLOCKS.keys()):
            #pos, quat = get_block_world_pose(name, usd=False)
            pos, quat = self._world.scene.get_object(name).get_world_pose()
            p, q = isaac_pose_to_ros(pos, quat)
            p[2] = update_height(p[2], HEIGHT_ERROR_OFFSET)
            print(f"{name:>8s}: p=({p[0]: .3f}, {p[1]: .3f}, {p[2]: .3f})  "
                  f"q=({q[0]: .4f}, {q[1]: .4f}, {q[2]: .4f}, {q[3]: .4f})")

        print(f"[scene] Published {len(BLOCKS.keys())} objects to /planning_scene")
        self._last_publish_t = now_monotonic
        self._publish_requested = False

    def _timer_request_publish(self):
        self._publish_requested = True

    def _timer_gripper(self):
        '''
        pos = random.uniform(0.02, 0.05)
        print(f"set gripper position={pos}")
        self._gripper.set_joint_positions([pos, pos])
        '''
        is_open = random.randint(0,1)
        if(is_open):
            print("open gripper")
            self._gripper.open()
        else:
            print("close gripper")
            self._gripper.close()

    def gripper_action_if_requested(self):
        """Call from main thread AFTER simulation_context.step()."""
        if not self._gripper_requested:
            return False

        print("request gripper open: ", self._gripper_open)

        if self._gripper_open: 
            #self._gripper.open()
            #self._gripper.set_joint_positions([0.04, 0.04])
            self.open_gripper()
        elif self._gripper_close_full:
            self.close_gripper2()
        else:
            #self._gripper.close()
            #self._gripper.set_joint_positions([0.02, 0.02])
            self.close_gripper()

        self._gripper_requested = False

        return True

    def _handle_gripper_open(self, req, res):
        #self._gripper.set_joint_positions([0.05, 0.05])
        print("request to open gripper")
        #self._gripper.open()
        self._gripper_requested = True
        self._gripper_open = True
        res.success = True
        res.message = "Gripper set to open position"
        return res

    def _handle_gripper_close(self, req, res):
        #self._gripper.set_joint_positions([0.01, 0.01])
        print("request to close gripper")
        #self._gripper.close()
        self._gripper_requested = True
        self._gripper_open = False
        self._gripper_close_full = False
        res.success = True
        res.message = "Gripper set to close position"
        return res

    def _handle_gripper_close2(self, req, res):
        #self._gripper.set_joint_positions([0.01, 0.01])
        print("request to full close gripper")
        #self._gripper.close()
        self._gripper_requested = True
        self._gripper_open = False
        self._gripper_close_full = True
        res.success = True
        res.message = "Gripper set to full close position"
        return res

    def gripper_is_open(self):
        return self._gripper_open

    def open_gripper(self):
        self._gripper.set_joint_positions([0.04, 0.04])

    def close_gripper(self):
        self._gripper.set_joint_positions([0.02, 0.02])

    def close_gripper2(self):
        self._gripper.set_joint_positions([0.00, 0.00])

# --------------------------- Main flow ---------------------------
def main():
    rclpy.init(args=None)

    world = World(stage_units_in_meters=1.0)
    # Create SimulationContext
    #simulation_context = SimulationContext(stage_units_in_meters=1.0)

    # Assets root
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        log_and_exit("Could not find Isaac Sim assets folder")

    # Camera
    viewports.set_camera_view(
        eye=np.array([1.2, 1.2, 0.8], dtype=float),
        target=np.array([0.0, 0.0, 0.5], dtype=float),
    )

    # Loading the simple_room environment
    stage.add_reference_to_stage(
        assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH)

    '''
    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    controller = PickPlaceController(name="pick_place_controller", gripper=franka.gripper, robot_articulation=franka)
    '''

    # Loading the franka robot USD
    robot = prims.create_prim(
        FRANKA_STAGE_PATH,
        "Xform",
        position=np.array([0, SCENE_OFFSET_Y, SCENE_OFFSET_Z]),
        orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
        usd_path=assets_root_path + FRANKA_USD_PATH,
    )

    tune_franka(FRANKA_STAGE_PATH)

    simulation_app.update()

    # Set variant selections for the Franka robot
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

    simulation_app.update()

    gripper = ParallelGripper(
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.04, 0.04]),
        joint_closed_positions=np.array([0.00, 0.00]),
        action_deltas=np.array([0.05, 0.05]),
    )

    franka = world.scene.add(
        SingleManipulator(
            prim_path="/World/Franka",
            name="my_franka",
            end_effector_prim_path="/World/Franka/panda_rightfinger",
            gripper=gripper,
        )
    )
    #world.scene.add_default_ground_plane()
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)

    world.reset()

    #franka.gripper.set_joint_positions(gripper.joint_closed_positions)

    # Creating a action graph with ROS component nodes
    try:
        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    (
                        "SubscribeJointState.outputs:positionCommand",
                        "ArticulationController.inputs:positionCommand",
                    ),
                    (
                        "SubscribeJointState.outputs:velocityCommand",
                        "ArticulationController.inputs:velocityCommand",
                    ),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Setting the /Franka target prim to Articulation Controller node
                    ("ArticulationController.inputs:robotPath", FRANKA_STAGE_PATH),
                    ("PublishJointState.inputs:topicName", "isaac_joint_states"),
                    ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
                    ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(FRANKA_STAGE_PATH)]),
                    #("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(FRANKA_STAGE_PATH)]),
                ],
            },
        )
    except Exception as e:
        print(e)

    simulation_app.update()

    # ---- Spawn blocks from YAML BEFORE physics init ----
    spawn_blocks_from_yaml(
        world=world,
        yaml_path=args.yaml,
        blocks_root=args.blocks_root,
    )
    simulation_app.update()

    bridge_node = SmashSceneBridge(
        world=world,
        blocks_root="/World/Blocks",
        planning_frame="world",       # <- set to your MoveIt planning frame if different
        gripper=franka.gripper
    )
    executor = SingleThreadedExecutor()
    executor.add_node(bridge_node)

    bridge_node.publish_scene_once()

    # Init physics and run
    #simulation_context.initialize_physics()
    #simulation_context.play()

    frame = 0
    while simulation_app.is_running():
        # Render every 10th frame for performance (tweak as you like)
        #simulation_context.step(render=(frame % 10 == 0))
        #frame += 1
        #simulation_context.step()
        #franka.gripper.close()
        world.step(render=True)

        executor.spin_once(timeout_sec=0.0)

        #bridge_node.flush_publish_if_due(time.monotonic())
        bridge_node.flush_publish_if_requested()
        bridge_node.gripper_action_if_requested()
        '''
        if bridge_node.gripper_is_open():
            bridge_node.open_gripper()
        else:
            bridge_node.close_gripper()
        '''

        # If your ActionGraph path exists, tick its OnImpulseEvent each frame
        try:
            attr = og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse")
            if attr.is_valid():
                og.Controller.set(attr, True)
        except Exception:
            pass

    world.reset()
    #simulation_context.stop()
    bridge_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        try:
            simulation_app.close()
        except Exception:
            pass
        sys.exit(1)

