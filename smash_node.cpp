#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
#include <tf2_eigen/tf2_eigen.hpp>
#else
#include <tf2_eigen/tf2_eigen.h>
#endif
#include <std_srvs/srv/trigger.hpp>
#include <blocksworld_scene/srv/update_scene.hpp>

#include <pluginlib/class_loader.hpp>


#include <algorithm>
#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <sstream>
#include <cctype>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <map>
#include <set>
#include <termios.h>
#include <unistd.h>
#include <random>
#include <chrono>

#include "topple_sim_client.hpp"
//#include "helper.hpp"

#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>


static const rclcpp::Logger LOGGER = rclcpp::get_logger("smash_node");
namespace mtc = moveit::task_constructor;

using shape_msgs::msg::SolidPrimitive;
using geometry_msgs::msg::Pose;
using moveit_msgs::msg::PlanningScene;
using moveit_msgs::msg::CollisionObject;
using moveit::planning_interface::PlanningSceneInterface;
using moveit_msgs::msg::ObjectColor;
using std_msgs::msg::ColorRGBA;
using std_srvs::srv::Trigger;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using control_msgs::action::GripperCommand;

#define EXECUTE_ERROR       (110)
#define TASK_INIT_ERROR     (1)
#define PLAN_ERROR          (2)
#define DELAY_PLAN_ERROR    (3)
#define PHYS_SIM_ERROR      (255)

#define HEIGHT_ERROR_OFFSET 0.001
#define GRIPPER_OPEN_POS    0.04
#define GRIPPER_CLOSE_POS   0.00

#define DEBUG_SLEEP_MS      0
#define SINGLE_GOAL_PROBLEM

//#define ROBOT_TYPE_FRANKA_PANDA
#define ROBOT_TYPE_UFACTORY_XARM7

#ifdef ROBOT_TYPE_FRANKA_PANDA
#define ARM_GROUP_NAME      "panda_arm"
#define HAND_GROUP_NAME     "hand"
#define HAND_FRAME          "panda_hand"
#define HOME_NAME           "ready"
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
#define ARM_GROUP_NAME      "xarm7"
#define HAND_GROUP_NAME     "xarm_gripper"
#define HAND_FRAME          "xarm_gripper_base_link"
#define HOME_NAME           "home"
#endif

struct Config {
    std::map<int,std::pair<double,double>> pads;
    double z_base{0.025}, z_inc{0.05};
};

struct Block {
    std::string block;
    int pad;
    int height;
    bool is_toppled; /* also used for is_scooped */
    Pose p;
};

struct Action {
    std::string block;
    std::string type;
    int start_pad;
    int start_height;
    int end_pad;
    int end_height;
    std::string dir;
    std::vector<std::string> blocks; // For Smash: all toppled blocks, bottom→top
};

struct FakeMoveRecord {
  bool is_fake_move = false;  // true if this action was planned as a fake move
  planning_scene::PlanningSceneConstPtr start_scene;  // FixedState seed used in fake
  std::map<std::string, double> end_joints;        // final arm joints at the end of the fake task
};

struct GripperOutcome {
  bool success;           // true if GoalStatus == SUCCEEDED (or stall accepted, see below)
  bool stalled;           // result->stalled
  bool reached_goal;      // result->reached_goal
  double final_position;  // result->position (meters)
  std::string status_text;
};

namespace mtc_debug {

inline std::string join(const std::vector<std::string>& v, const char* sep = ", ")
{
  std::ostringstream oss;
  for (size_t i = 0; i < v.size(); ++i) { if (i) oss << sep; oss << v[i]; }
  return oss.str();
}

inline double duration_to_seconds(const builtin_interfaces::msg::Duration& d)
{
  return static_cast<double>(d.sec) + 1e-9 * static_cast<double>(d.nanosec);
}

inline double totalDurationSec(const moveit_msgs::msg::RobotTrajectory& traj)
{
  double t = 0.0;
  if (!traj.joint_trajectory.points.empty())
    t = std::max(t, duration_to_seconds(traj.joint_trajectory.points.back().time_from_start));
  if (!traj.multi_dof_joint_trajectory.points.empty())
    t = std::max(t, duration_to_seconds(traj.multi_dof_joint_trajectory.points.back().time_from_start));
  return t;
}

inline const moveit::core::JointModelGroup* matchGroupByJoints(
    const moveit::core::RobotModelConstPtr& model,
    const std::vector<std::string>& traj_joints)
{
  if (!model || traj_joints.empty()) return nullptr;  // <-- important: don't guess for empty lists
  const std::set<std::string> traj_set(traj_joints.begin(), traj_joints.end());
  for (const auto* jmg : model->getJointModelGroups()) {
    const auto& active = jmg->getActiveJointModelNames();
    std::set<std::string> active_set(active.begin(), active.end());
    if (std::includes(active_set.begin(), active_set.end(), traj_set.begin(), traj_set.end()))
      return jmg;
  }
  return nullptr;
}

// extract first/last finger positions from a hand trajectory; returns (has_data, q_start, q_end)
inline std::tuple<bool,double,double> fingerStartEnd(const moveit_msgs::msg::RobotTrajectory& traj)
{
  const auto& jt = traj.joint_trajectory;
  if (jt.joint_names.empty() || jt.points.empty()) return {false, 0.0, 0.0};

#ifdef ROBOT_TYPE_FRANKA_PANDA
  auto it = std::find(jt.joint_names.begin(), jt.joint_names.end(), "panda_finger_joint1");
  if (it == jt.joint_names.end())
    it = std::find(jt.joint_names.begin(), jt.joint_names.end(), "panda_finger_joint2");
  if (it == jt.joint_names.end()) return {false, 0.0, 0.0};
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
  auto it = std::find(jt.joint_names.begin(), jt.joint_names.end(), "drive_joint");
  if (it == jt.joint_names.end()) return {false, 0.0, 0.0};
#endif

  const size_t idx = static_cast<size_t>(std::distance(jt.joint_names.begin(), it));
  const auto& p0 = jt.points.front();
  const auto& p1 = jt.points.back();
  if (idx >= p0.positions.size() || idx >= p1.positions.size()) return {false, 0.0, 0.0};

#ifdef ROBOT_TYPE_FRANKA_PANDA
  // clamp to panda limits just for sanity
  auto clamp = [](double v){ return std::min(0.04, std::max(0.0, v)); };
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
  auto clamp = [](double v){ return std::min(0.85, std::max(0.0, v)); };
#endif
  return {true, clamp(p0.positions[idx]), clamp(p1.positions[idx])};
}

inline bool containsFinger(const std::vector<std::string>& names) {
#ifdef ROBOT_TYPE_FRANKA_PANDA
  return std::find(names.begin(), names.end(), "panda_finger_joint1") != names.end() ||
         std::find(names.begin(), names.end(), "panda_finger_joint2") != names.end();
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
  return std::find(names.begin(), names.end(), "drive_joint") != names.end();
#endif
}

// Pretty-print the positions in a JointTrajectory (optionally limited)
inline void printJointTrajectorySamples(const trajectory_msgs::msg::JointTrajectory& jt,
                                        int max_points = -1)
{
  const auto& names = jt.joint_names;
  if (jt.points.empty()) {
    RCLCPP_INFO(LOGGER, "      (no points)");
    return;
  }
  const size_t N = jt.points.size();
  size_t print_N = (max_points < 0) ? N : std::min<size_t>(N, static_cast<size_t>(max_points));
  for (size_t i = 0; i < print_N; ++i) {
    const auto& p = jt.points[i];
    std::ostringstream line;
    line << "      pt[" << std::setw(2) << i << "] t=" << std::fixed << std::setprecision(3)
         << duration_to_seconds(p.time_from_start) << "s  q=[";
    for (size_t j = 0; j < names.size(); ++j) {
      if (j) line << ", ";
      const double q = (j < p.positions.size()) ? p.positions[j] : std::numeric_limits<double>::quiet_NaN();
      line << names[j] << ":" << std::setprecision(5) << q;
    }
    line << "]";
    RCLCPP_INFO(LOGGER, "%s", line.str().c_str());
  }
  if (print_N < N)
    RCLCPP_INFO(LOGGER, "      ... (%zu more points not shown)", N - print_N);
}

inline void stripFingerFromArmHand(moveit_task_constructor_msgs::msg::Solution& sol_msg)
{
  // canonical arm joint set
#ifdef ROBOT_TYPE_FRANKA_PANDA
  static const std::array<const char*, 7> kArm{
      "panda_joint1","panda_joint2","panda_joint3",
      "panda_joint4","panda_joint5","panda_joint6","panda_joint7"
  };
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
  static const std::array<const char*, 7> kArm{
      "joint1","joint2","joint3",
      "joint4","joint5","joint6","joint7"
  };
#endif

  auto is_arm_set = [&](const std::vector<std::string>& names)->bool {
    std::set<std::string> s(names.begin(), names.end());
    for (auto* j : kArm) if (!s.count(j)) return false;
    return true;
  };

  auto find_index = [](const std::vector<std::string>& v, const std::string& key)->int {
    auto it = std::find(v.begin(), v.end(), key);
    return (it == v.end()) ? -1 : static_cast<int>(std::distance(v.begin(), it));
  };

  auto erase_idx = [](auto& vec, size_t idx){
    if (idx < vec.size()) vec.erase(vec.begin() + static_cast<long>(idx));
  };
  auto erase_joint_from_point = [&](trajectory_msgs::msg::JointTrajectoryPoint& pt, size_t idx){
    erase_idx(pt.positions, idx);
    erase_idx(pt.velocities, idx);
    erase_idx(pt.accelerations, idx);
    erase_idx(pt.effort, idx);
  };

  size_t modified = 0;

  for (auto& seg : sol_msg.sub_trajectory) {
    auto& jt    = seg.trajectory.joint_trajectory;
    auto& names = jt.joint_names;
    if (names.empty() || jt.points.empty())
      continue;

    // quick filter: we only touch segments with exactly 8 joints total
    if (names.size() != 8)
      continue;

    // must contain all 7 arm joints
    if (!is_arm_set(names))
      continue;

    // must contain exactly ONE finger joint
#ifdef ROBOT_TYPE_FRANKA_PANDA
    const int idx_f1 = find_index(names, "panda_finger_joint1");
    const int idx_f2 = find_index(names, "panda_finger_joint2");
    int idx_f = -1;
    if (idx_f1 >= 0 && idx_f2 < 0)       idx_f = idx_f1;
    else if (idx_f2 >= 0 && idx_f1 < 0)  idx_f = idx_f2;
    else
      continue; // zero or two fingers -> leave unchanged
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
    const int idx_f1 = find_index(names, "drive_joint");
    int idx_f = -1;
    if (idx_f1 >= 0)       idx_f = idx_f1;
    else
      continue; // zero or two fingers -> leave unchanged
#endif

    // erase the finger joint column
    const size_t eidx = static_cast<size_t>(idx_f);
    erase_idx(names, eidx);
    for (auto& p : jt.points) erase_joint_from_point(p, eidx);

    ++modified;
  }

  // (optional) log somewhere if you like:
  RCLCPP_INFO(LOGGER, "stripFingerFromArmHand(): modified %zu segments", modified);
}

// Main printer with samples
inline void printFirstSolutionMessage(moveit_task_constructor_msgs::msg::Solution& sol_msg,
                                      const moveit::core::RobotModelConstPtr& model,
//inline void printFirstSolutionMessage(moveit::task_constructor::Task& task,
                                      int max_points_per_segment = 3)   // change to -1 to dump all
{
#if 0
  if (task.solutions().empty()) {
    RCLCPP_WARN(LOGGER, "No MTC solutions to print");
    return;
  }

  moveit_task_constructor_msgs::msg::Solution sol_msg;
  task.solutions().front()->toMsg(sol_msg);
  const auto model = task.getRobotModel();
#endif
  //stripFingerFromArmHand(sol_msg);

  RCLCPP_INFO(LOGGER, "=== MTC Solution: %zu sub_trajectory segments ===",
              sol_msg.sub_trajectory.size());

  for (size_t i = 0; i < sol_msg.sub_trajectory.size(); ++i) {
    const auto& seg = sol_msg.sub_trajectory[i];
    const auto& jt  = seg.trajectory.joint_trajectory;
    const auto& md  = seg.trajectory.multi_dof_joint_trajectory;

    const auto* jmg = matchGroupByJoints(model, jt.joint_names);
    const std::string group = jmg ? jmg->getName() : std::string("<none>");

    const auto& ps   = seg.scene_diff;
    const auto& co   = ps.world.collision_objects;
    const auto& aco  = ps.robot_state.attached_collision_objects;

    RCLCPP_INFO(LOGGER,
      "  [%02zu] group=%s  joints=[%s]  points=%zu  mdof_points=%zu  duration=%.3fs  scene_diff: collision_objects=%zu attached=%zu",
      i, group.c_str(),
      join(jt.joint_names).c_str(),
      jt.points.size(), md.points.size(),
      totalDurationSec(seg.trajectory),
      co.size(), aco.size());

    // If there is motion, print a few joint samples
    if (!jt.points.empty())
      printJointTrajectorySamples(jt, max_points_per_segment);
  }
}

inline void printTrajectory(const moveit_msgs::msg::RobotTrajectory& traj,
                            const moveit::core::RobotModelConstPtr& model,
                            int max_points_per_segment = 3)
{
    const auto& jt  = traj.joint_trajectory;
    const auto& md  = traj.multi_dof_joint_trajectory;
    const auto* jmg = matchGroupByJoints(model, jt.joint_names);
    const std::string group = jmg ? jmg->getName() : std::string("<none>");

    RCLCPP_INFO(LOGGER,
      "  group=%s  joints=[%s]  points=%zu  mdof_points=%zu  duration=%.3fs",
      group.c_str(),
      join(jt.joint_names).c_str(),
      jt.points.size(), md.points.size(),
      totalDurationSec(traj));

    if (!jt.points.empty())
      printJointTrajectorySamples(jt, max_points_per_segment);
}


// Compute timestamps for all *non-hand* sub_trajectories in the first solution.
// Outputs a Solution msg with updated (timed) trajectories; hand segments are untouched.
inline bool timeParameterizeArmSegments(
    rclcpp::Logger logger,
    moveit::task_constructor::Task& task,
    moveit_task_constructor_msgs::msg::Solution& sol_msg_out,
    double vel_scaling = 1.0,
    double acc_scaling = 1.0)
{
  if (task.solutions().empty()) {
    RCLCPP_ERROR(logger, "No MTC solutions available");
    return false;
  }

  // Flatten the first solution
  task.solutions().front()->toMsg(sol_msg_out);

  auto model = task.getRobotModel();
  if (!model) {
    RCLCPP_ERROR(logger, "RobotModel is null");
    return false;
  }

  trajectory_processing::IterativeParabolicTimeParameterization iptp;
  size_t timed_segments = 0;

  for (size_t i = 0; i < sol_msg_out.sub_trajectory.size(); ++i) {
    auto& seg = sol_msg_out.sub_trajectory[i];
    auto& jt  = seg.trajectory.joint_trajectory;

    // Skip scene-only or empty
    if (jt.joint_names.empty() || jt.points.empty())
      continue;

    // Skip any segment that involves the gripper fingers
    if (containsFinger(jt.joint_names)) {
      RCLCPP_DEBUG(logger, "[%02zu] hand/combined segment detected; skipping timing", i);
      continue;
    }

    // Find a matching group
    const auto* jmg = matchGroupByJoints(model, jt.joint_names);
    if (!jmg) {
      RCLCPP_WARN(logger, "[%02zu] cannot match MoveIt group for joints; skipping timing", i);
      continue;
    }

    // Build a reference state whose group matches the first waypoint
    moveit::core::RobotState ref(model);
    ref.setToDefaultValues();

    // Map group joint order -> trajectory joint order for the first point
    const auto& gnames = jmg->getActiveJointModelNames();
    std::vector<double> q0(gnames.size(), 0.0);

    // For safety: create a name->index map
    std::unordered_map<std::string, size_t> name_to_idx;
    name_to_idx.reserve(jt.joint_names.size());
    for (size_t k = 0; k < jt.joint_names.size(); ++k) name_to_idx[jt.joint_names[k]] = k;

    const auto& p0 = jt.points.front();
    for (size_t g = 0; g < gnames.size(); ++g) {
      auto it = name_to_idx.find(gnames[g]);
      if (it != name_to_idx.end() && it->second < p0.positions.size())
        q0[g] = p0.positions[it->second];
      // else leave default 0.0 (rare; e.g., fixed joints)
    }
    ref.setJointGroupPositions(jmg, q0);
    ref.update();

    // Wrap into RobotTrajectory, run IPTP, and write back
    robot_trajectory::RobotTrajectory rt(model, jmg->getName());
    rt.setRobotTrajectoryMsg(ref, seg.trajectory);

    if (!iptp.computeTimeStamps(rt, vel_scaling, acc_scaling)) {
      RCLCPP_WARN(logger, "[%02zu] IPTP failed; leaving segment un-timed", i);
      continue;
    }

    rt.getRobotTrajectoryMsg(seg.trajectory);
    ++timed_segments;

    RCLCPP_DEBUG(logger, "[%02zu] timed segment for group='%s'  points=%zu",
                 i, jmg->getName().c_str(), seg.trajectory.joint_trajectory.points.size());
  }

  RCLCPP_INFO(logger, "Time-parameterized %zu non-hand segments", timed_segments);
  return true;
}

#if 0
// Duplicate the last point at t += hold_sec (positions same, zeros for v/a)
inline void appendHoldPoint(moveit_msgs::msg::RobotTrajectory& traj_msg, double hold_sec = 0.3)
{
  auto& jt = traj_msg.joint_trajectory;
  if (jt.points.empty() || jt.joint_names.empty()) return;
  const auto last = jt.points.back();

  trajectory_msgs::msg::JointTrajectoryPoint hold = last;
  // advance time
  hold.time_from_start.sec  += static_cast<int32_t>(hold_sec);
  hold.time_from_start.nanosec += static_cast<uint32_t>((hold_sec - static_cast<int>(hold_sec)) * 1e9);
  // normalize nanosec
  if (hold.time_from_start.nanosec >= 1000000000u) {
    hold.time_from_start.sec += 1;
    hold.time_from_start.nanosec -= 1000000000u;
  }
  // zero v/a/effort (controller should hold position)
  hold.velocities.assign(jt.joint_names.size(), 0.0);
  hold.accelerations.assign(jt.joint_names.size(), 0.0);
  // effort usually ignored; leave as-is or zero it
  jt.points.push_back(hold);
}
#endif

inline void appendHoldPoint(moveit_msgs::msg::RobotTrajectory& traj_msg, double hold_sec = 0.3)
{
  auto& jt = traj_msg.joint_trajectory;
  if (jt.joint_names.empty() || jt.points.empty() || hold_sec <= 0.0) return;

  const auto last = jt.points.back();
  const size_t N = jt.joint_names.size();

  // Split hold duration: 25%, 35%, rest (≈40%)
  const double d1 = 0.25 * hold_sec;
  const double d2 = 0.35 * hold_sec;
  const double d3 = hold_sec - (d1 + d2);  // ensures exact sum == hold_sec

  auto time_to_double = [](const builtin_interfaces::msg::Duration& d) -> double {
    return static_cast<double>(d.sec) + 1e-9 * static_cast<double>(d.nanosec);
  };
  auto set_time = [](builtin_interfaces::msg::Duration& d, double t) {
    if (t < 0.0) t = 0.0;
    const double s = std::floor(t);
    d.sec     = static_cast<int32_t>(s);
    d.nanosec = static_cast<uint32_t>((t - s) * 1e9);
    if (d.nanosec >= 1000000000u) { ++d.sec; d.nanosec -= 1000000000u; }
  };

  const double t_base = time_to_double(last.time_from_start);

  auto make_hold = [&](double dt) {
    trajectory_msgs::msg::JointTrajectoryPoint hp = last;

    // Ensure arrays sized
    if (hp.positions.size()      < N) hp.positions.resize(N, 0.0);
    hp.velocities.assign(N, 0.0);
    hp.accelerations.assign(N, 0.0);
    if (!hp.effort.empty()) hp.effort.assign(N, 0.0);

    // New timestamp = last.time + dt
    set_time(hp.time_from_start, t_base + dt);
    return hp;
  };

  // Append three progressively later hold points
  jt.points.push_back(make_hold(d1));
  jt.points.push_back(make_hold(d1 + d2));
  jt.points.push_back(make_hold(d1 + d2 + d3));
}

// Re-time with IPTP using conservative scales
inline bool retimeTrajectory(const moveit::core::RobotModelConstPtr& model,
                             const moveit::core::JointModelGroup* jmg,
                             const moveit::core::RobotState& start_state,
                             moveit_msgs::msg::RobotTrajectory& traj_msg,
                             double vel_scale = 0.25,
                             double acc_scale = 0.25)
{
  if (!model || !jmg) return false;
  robot_trajectory::RobotTrajectory rt(model, jmg->getName());
  rt.setRobotTrajectoryMsg(start_state, traj_msg);
  trajectory_processing::IterativeParabolicTimeParameterization iptp;
  if (!iptp.computeTimeStamps(rt, vel_scale, acc_scale)) return false;
  rt.getRobotTrajectoryMsg(traj_msg);
  return true;
}

// Snap first waypoint to *current* state if available, then re-time.
// If current state isn't available, re-time using the first waypoint as the start.
// If IPTP fails, fall back to simple uniform timestamps so the message is still valid.
inline bool snapFirstPointToCurrentAndRetime(const moveit::planning_interface::MoveGroupInterface& mgi,
                                             const moveit::core::RobotModelConstPtr& model,
                                             const moveit::core::JointModelGroup* jmg,
                                             moveit_msgs::msg::RobotTrajectory& traj_msg,
                                             double vel_scale = 0.25,
                                             double acc_scale = 0.25)
{
  auto& jt = traj_msg.joint_trajectory;
  if (!model || !jmg || jt.joint_names.empty() || jt.points.empty())
    return false;

  // Try to get the live state
  moveit::core::RobotStatePtr cur = mgi.getCurrentState(1.0 /*sec*/);

  // Build a reference state for time parameterization
  moveit::core::RobotState ref(model);
  ref.setToDefaultValues();

  // Map joint name -> index in trajectory
  std::unordered_map<std::string, size_t> idx;
  idx.reserve(jt.joint_names.size());
  for (size_t k = 0; k < jt.joint_names.size(); ++k) idx[jt.joint_names[k]] = k;

  // Ensure first point arrays are properly sized
  auto& p0 = jt.points.front();
  if (p0.positions.size()     < jt.joint_names.size()) p0.positions.resize(jt.joint_names.size(), 0.0);
  if (p0.velocities.size()    != jt.joint_names.size()) p0.velocities.assign(jt.joint_names.size(), 0.0);
  if (p0.accelerations.size() != jt.joint_names.size()) p0.accelerations.assign(jt.joint_names.size(), 0.0);

  if (cur) {
    // Snap: overwrite first point with live joint values
    for (size_t k = 0; k < jt.joint_names.size(); ++k)
      p0.positions[k] = cur->getVariablePosition(jt.joint_names[k]);
    // Reference state = live state
    ref = *cur;
  } else {
    // No live state — build ref from the first waypoint values for this group
    const auto& gnames = jmg->getActiveJointModelNames();
    std::vector<double> q0(gnames.size(), 0.0);
    for (size_t g = 0; g < gnames.size(); ++g) {
      auto it = idx.find(gnames[g]);
      if (it != idx.end() && it->second < p0.positions.size())
        q0[g] = p0.positions[it->second];
    }
    ref.setJointGroupPositions(jmg, q0);
    ref.update();
  }

  // Re-time with IPTP
  robot_trajectory::RobotTrajectory rt(model, jmg->getName());
  rt.setRobotTrajectoryMsg(ref, traj_msg);
  trajectory_processing::IterativeParabolicTimeParameterization iptp;
  if (iptp.computeTimeStamps(rt, vel_scale, acc_scale)) {
    rt.getRobotTrajectoryMsg(traj_msg);
    return true;  // success (snap+retime or retime-from-first-point)
  }

  // IPTP failed — last-resort timestamps so controller doesn't choke
  //assignMonotonicTimes(jt, /*dt_sec=*/0.25);
  return true;  // timestamps updated, even though not optimal
}

// Wait until current joints are within pos_tol AND |velocity| < vel_tol
// for `settle_count_required` consecutive checks (poll every poll_dt)
inline bool waitUntilSettled(const moveit::planning_interface::MoveGroupInterface& mgi,
                             const std::vector<std::string>& names,
                             const std::vector<double>& goal_positions,
                             double pos_tol = 0.01,           // ~0.6°
                             double vel_tol = 0.05,           // rad/s
                             int settle_count_required = 5,   // e.g., 5 consecutive samples
                             std::chrono::milliseconds poll_dt = std::chrono::milliseconds(50),
                             std::chrono::milliseconds timeout = std::chrono::milliseconds(200))
{
  auto start_tp = std::chrono::steady_clock::now();
  int settled = 0;

  while (std::chrono::steady_clock::now() - start_tp < timeout) {
    auto cur = mgi.getCurrentState(0.5);
    if (!cur) { std::this_thread::sleep_for(poll_dt); continue; }

    bool ok = true;
    for (size_t i = 0; i < names.size(); ++i) {
      double q = cur->getVariablePosition(names[i]);
      double v = cur->getVariableVelocity(names[i]);
      if (std::isnan(v)) v = 0.0;
      if (std::abs(q - goal_positions[i]) > pos_tol || std::abs(v) > vel_tol) { ok = false; break; }
    }
    settled = ok ? settled + 1 : 0;
    if (settled >= settle_count_required) {
      std::this_thread::sleep_for(std::chrono::milliseconds(DEBUG_SLEEP_MS));
      return true;
    }
    std::this_thread::sleep_for(poll_dt);
  }
  return false; // timed out; still return control to avoid deadlock
}

// 1) Remove consecutive duplicate waypoints (all joint positions equal)
inline void removeConsecutiveDuplicates(trajectory_msgs::msg::JointTrajectory& jt,
                                        double pos_eps = 1e-9)
{
  const size_t N = jt.joint_names.size();
  if (jt.points.size() < 2 || N == 0) return;

  auto samepos = [&](const auto& a, const auto& b){
    if (a.positions.size() != N || b.positions.size() != N) return false;
    for (size_t i = 0; i < N; ++i)
      if (std::abs(a.positions[i] - b.positions[i]) > pos_eps) return false;
    return true;
  };

  jt.points.erase(std::unique(jt.points.begin(), jt.points.end(), samepos),
                  jt.points.end());
}

// 2) Ensure strictly increasing time_from_start with a minimum gap
inline void enforceStrictlyIncreasingTime(trajectory_msgs::msg::JointTrajectory& jt,
                                          double min_dt = 1e-3 /* 1 ms */,
                                          double start_at = 0.0)
{
  if (jt.points.empty()) return;

  // If first point time is zeroed/invalid, (re)start from 'start_at'
  auto set_time = [](auto& d, double t){
    if (t < 0.0) t = 0.0;
    d.sec     = static_cast<int32_t>(std::floor(t));
    d.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
  };

  // Initialize t_prev
  double t_prev;
  {
    double t0 = jt.points.front().time_from_start.sec +
                1e-9 * jt.points.front().time_from_start.nanosec;
    if (!(t0 > 0.0)) t0 = start_at;
    set_time(jt.points.front().time_from_start, t0);
    t_prev = t0;
  }

  for (size_t i = 1; i < jt.points.size(); ++i) {
    double t = jt.points[i].time_from_start.sec +
               1e-9 * jt.points[i].time_from_start.nanosec;
    if (!(t > t_prev)) t = t_prev + min_dt;          // not strictly increasing → bump
    else if (t - t_prev < min_dt) t = t_prev + min_dt; // too tight → pad

    set_time(jt.points[i].time_from_start, t);
    t_prev = t;
  }
}

// (optional) One convenience that does both:
inline void sanitizeAndEnforceTiming(trajectory_msgs::msg::JointTrajectory& jt,
                                     double min_dt = 1e-3)
{
  removeConsecutiveDuplicates(jt);
  enforceStrictlyIncreasingTime(jt, min_dt, /*start_at=*/0.0);
}

#ifdef ROBOT_TYPE_FRANKA_PANDA
static const std::array<const char*, 7> kArm{
      "panda_joint1","panda_joint2","panda_joint3",
      "panda_joint4","panda_joint5","panda_joint6","panda_joint7"
};
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
static const std::array<const char*, 7> kArm{
      "joint1","joint2","joint3",
      "joint4","joint5","joint6","joint7"
};
#endif

inline bool hasAllArmJoints(const std::vector<std::string>& names) {
  std::set<std::string> s(names.begin(), names.end());
  for (auto* j : kArm) if (!s.count(j)) return false;
  return true;
}
inline bool hasAnyArmJoint(const std::vector<std::string>& names) {
  for (auto* j : kArm) if (std::find(names.begin(), names.end(), j) != names.end()) return true;
  return false;
}
inline bool hasFinger(const std::vector<std::string>& names) {
#ifdef ROBOT_TYPE_FRANKA_PANDA
  return std::find(names.begin(), names.end(), "panda_finger_joint1") != names.end() ||
         std::find(names.begin(), names.end(), "panda_finger_joint2") != names.end();
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
  return std::find(names.begin(), names.end(), "drive_joint") != names.end();
#endif
}

// Append a column to each point (positions required, others if present)
inline void appendColumn(trajectory_msgs::msg::JointTrajectory& jtraj,
                         const std::string& joint_name,
                         double fill_pos)
{
  jtraj.joint_names.push_back(joint_name);
  for (auto& pt : jtraj.points) {
    // positions (required)
    if (pt.positions.size() < jtraj.joint_names.size()-1) pt.positions.resize(jtraj.joint_names.size()-1, 0.0);
    pt.positions.push_back(fill_pos);
    // velocities/accels/effort (only if vectors already used)
    if (!pt.velocities.empty())    { if (pt.velocities.size() < jtraj.joint_names.size()-1) pt.velocities.resize(jtraj.joint_names.size()-1, 0.0);    pt.velocities.push_back(0.0); }
    if (!pt.accelerations.empty()) { if (pt.accelerations.size() < jtraj.joint_names.size()-1) pt.accelerations.resize(jtraj.joint_names.size()-1, 0.0); pt.accelerations.push_back(0.0); }
    if (!pt.effort.empty())        { if (pt.effort.size() < jtraj.joint_names.size()-1) pt.effort.resize(jtraj.joint_names.size()-1, 0.0);             pt.effort.push_back(0.0); }
  }
}

/// Build combined arm+hand segments by injecting panda_finger_joint1 into arm-only segments.
/// The injected finger value equals the *end* finger position of the *last seen* hand segment.
/// Hand-only segments are left unchanged.
/// If no hand has occurred yet, `default_finger_pos` is used (e.g., 0.04 open).
inline bool addFingerToArmUsingLastHand(
    moveit_task_constructor_msgs::msg::Solution& sol_msg_out,
    double default_finger_pos = 0.04)
{

  double last_finger = default_finger_pos, finger;
  size_t modified = 0;

  for (auto& seg : sol_msg_out.sub_trajectory) {
    auto& jtraj = seg.trajectory.joint_trajectory;
    if (jtraj.points.empty()) continue;

    const bool arm_only   = hasAllArmJoints(jtraj.joint_names) && !hasFinger(jtraj.joint_names);
    const bool hand_only  = hasFinger(jtraj.joint_names) && !hasAnyArmJoint(jtraj.joint_names);

    if (hand_only) {
      // Update last_finger to the *end* of this hand motion
      auto [ok, q0, q1] = fingerStartEnd(seg.trajectory);
      if (ok) last_finger = q1;
      continue; // keep hand segment as-is
    }

    if (arm_only) {
      // Inject constant finger column = last_finger across all points
      if(last_finger > GRIPPER_CLOSE_POS)
        finger = GRIPPER_OPEN_POS;
      else
        finger = GRIPPER_CLOSE_POS;
      //appendColumn(jtraj, "panda_finger_joint1", last_finger);
#ifdef ROBOT_TYPE_FRANKA_PANDA
      appendColumn(jtraj, "panda_finger_joint1", finger);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      appendColumn(jtraj, "drive_joint", finger);
#endif
      ++modified;
      continue;
    }

    // For already-combined segments (arm+hand), do nothing
  }

  // Optional log
  // RCLCPP_INFO(node_->get_logger(), "addFingerToArmUsingLastHand(): modified %zu segments, final finger=%.4f", modified, last_finger);
  return true;
}

inline double toSeconds(const builtin_interfaces::msg::Duration& d) {
  return static_cast<double>(d.sec) + 1e-9 * static_cast<double>(d.nanosec);
}

inline void fromSeconds(builtin_interfaces::msg::Duration& d, double t) {
  if (t < 0.0) t = 0.0;
  const double s = std::floor(t);
  d.sec     = static_cast<int32_t>(s);
  d.nanosec = static_cast<uint32_t>((t - s) * 1e9);
  if (d.nanosec >= 1000000000u) { ++d.sec; d.nanosec -= 1000000000u; }
}

// If all timestamps are 0 (or not strictly increasing), seed a uniform time grid
inline void seedUniformTimesIfNeeded(trajectory_msgs::msg::JointTrajectory& jt, double dt=0.25) {
  if (jt.points.empty()) return;
  bool needs_seed = false;
  double last = toSeconds(jt.points.front().time_from_start);
  if (!(last > 0.0)) needs_seed = true;
  for (size_t i=1;i<jt.points.size();++i) {
    double t = toSeconds(jt.points[i].time_from_start);
    if (!(t > last)) { needs_seed = true; break; }
    last = t;
  }
  if (needs_seed) {
    for (size_t i=0;i<jt.points.size();++i)
      fromSeconds(jt.points[i].time_from_start, dt * static_cast<double>(i));
  }
}

inline void densifyJointTrajectory(moveit_msgs::msg::RobotTrajectory& traj_msg,
                                   double max_joint_step = 0.05,
                                   size_t max_new_pts_cap = 5000)
{
  auto& jt = traj_msg.joint_trajectory;
  const size_t N = jt.joint_names.size();
  if (N == 0 || jt.points.size() < 2) return;

  // Ensure each point has positions sized to N
  for (auto& p : jt.points) if (p.positions.size() < N) p.positions.resize(N, 0.0);

  // Seed/repair timestamps if needed
  seedUniformTimesIfNeeded(jt, /*dt=*/0.25);

  std::vector<trajectory_msgs::msg::JointTrajectoryPoint> new_points;
  new_points.reserve(std::min(max_new_pts_cap, jt.points.size()*5));

  auto lerp = [](double a, double b, double u){ return a + u*(b - a); };

  new_points.push_back(jt.points.front());  // keep first point

  for (size_t i = 0; i + 1 < jt.points.size(); ++i) {
    const auto& a = jt.points[i];
    const auto& b = jt.points[i+1];

    // compute how many subdivisions we need based on the *largest* joint change
    double max_delta = 0.0;
    for (size_t j=0;j<N;++j) {
      max_delta = std::max(max_delta, std::abs(b.positions[j] - a.positions[j]));
    }
    // at least 1 segment; n_div = how many equal sub-segments to split this edge into
    size_t n_div = static_cast<size_t>(std::ceil(std::max(1e-12, max_delta) / std::max(1e-9, max_joint_step)));
    n_div = std::max<size_t>(1, n_div);

    const double t_a = toSeconds(a.time_from_start);
    const double t_b = toSeconds(b.time_from_start);

    // Insert intermediate points (not including the endpoint 'b')
    for (size_t k = 1; k < n_div; ++k) {
      const double u = static_cast<double>(k) / static_cast<double>(n_div);  // (0,1)
      trajectory_msgs::msg::JointTrajectoryPoint p;
      p.positions.resize(N);
      for (size_t j=0;j<N;++j) p.positions[j] = lerp(a.positions[j], b.positions[j], u);
      // timestamps linearly between the endpoints
      fromSeconds(p.time_from_start, lerp(t_a, t_b, u));
      // leave velocities/accelerations empty: JTC can handle pos-only with time
      new_points.push_back(std::move(p));
      if (new_points.size() >= max_new_pts_cap) break;
    }
    if (new_points.size() >= max_new_pts_cap) break;

    // finally add the original endpoint 'b'
    new_points.push_back(b);
  }

  jt.points.swap(new_points);
}

inline bool saveTasksFirstSolutionsYAML(const std::vector<moveit::task_constructor::Task>& tasks,
                                        const std::string& filepath)
{
  std::ofstream out(filepath);
  if (!out.is_open()) return false;

  out << "version: 1\n";
  out << "robot_models:\n";
  for (size_t i = 0; i < tasks.size(); ++i)
    out << "  - " << tasks[i].getRobotModel()->getName() << "\n";

  out << "segments:\n";

  for (size_t ti = 0; ti < tasks.size(); ++ti) {
    if (tasks[ti].solutions().empty()) continue;

    moveit_task_constructor_msgs::msg::Solution sol_msg;
    tasks[ti].solutions().front()->toMsg(sol_msg);
    auto model = tasks[ti].getRobotModel();

    for (size_t si = 0; si < sol_msg.sub_trajectory.size(); ++si) {
      const auto& seg = sol_msg.sub_trajectory[si];
      const auto& jt  = seg.trajectory.joint_trajectory;
      if (jt.joint_names.empty() || jt.points.empty()) continue;  // scene-only, skip

      const auto* jmg = matchGroupByJoints(model, jt.joint_names);
      if (!jmg) continue;

      const std::string group = jmg->getName();
      if (group != ARM_GROUP_NAME && group != HAND_GROUP_NAME) continue;

      // --- write YAML block ---
      out << "  - task_index: " << ti << "\n";
      out << "    segment_index: " << si << "\n";
      out << "    group: " << group << "\n";

      // joint names
      out << "    joint_names: [";
      for (size_t j=0; j<jt.joint_names.size(); ++j) {
        out << jt.joint_names[j];
        if (j + 1 < jt.joint_names.size()) out << ", ";
      }
      out << "]\n";

      // points
      out << "    points:\n";
      for (const auto& p : jt.points) {
        if (p.positions.size() != jt.joint_names.size()) continue;
        out << "      - { t: " << std::fixed << std::setprecision(6) << toSeconds(p.time_from_start) << ", positions: [";
        for (size_t k=0; k<p.positions.size(); ++k) {
          out << std::setprecision(9) << p.positions[k];
          if (k + 1 < p.positions.size()) out << ", ";
        }
        out << "] }\n";
      }
    }
  }

  out.flush();
  return static_cast<bool>(out);
}

} // namespace mtc_debug

class MTCTaskNode
{
public:
    MTCTaskNode(const rclcpp::NodeOptions& options);

    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface();

    void setupPlanningScene();

    bool requestSceneUpdate();
    bool requestSceneClearObjects();
    bool requestSceneAddObjects();
    bool requestSceneUpdateObjects(const std::string&, const std::string&);

    ColorRGBA rgba(float r,float g,float b,float a=1.0);

    CollisionObject makeBox(const std::string& id,
                               double dx, double dy, double dz,
                               double px, double py, double pz);

    std::string getObjId(std::string);
    void waitForEnterKey();
    void waitForAnyKey();
    std::string toLower(const std::string &);
    double sampleAngle();

    Pose getRelPose(Action& a);
    Pose getStartPose(Action& a);
    Pose getTargetPose(Action& a);
    Pose getPoseFromPSI(Action& a);
    Pose getPoseFromPH(int pad, int height);
    double adjustHeight(double height);
    void updateSceneObjectHeights();

    inline std::map<int,std::pair<int,int>> grid_rc(const Config& cfg);
    inline void printBlocks(const Config& cfg,
                         const std::vector<Block>& blocks);
    inline void printPlan(const std::vector<Action>& plan);
    inline void pretty(const Config& cfg,
                   const std::vector<Block>& initials,
                   const std::vector<Block>& goals,
                   const std::vector<Action>& plan);

    // Compose an MTC task from a series of stages.
    int executeReturnHomeTask(planning_scene::PlanningSceneConstPtr start_scene);
    mtc::Task createPreMoveTask(bool open,
                    planning_scene::PlanningSceneConstPtr start_scene);
    mtc::Task createMoveTask(std::string obj_id, Pose src_pose, Pose tgt_pose,
                    planning_scene::PlanningSceneConstPtr start_scene);
    mtc::Task createFakeMoveTask(Pose tgt_pose,
                    planning_scene::PlanningSceneConstPtr start_scene);
    mtc::Task createRealFromFakeMoveTask(std::string obj_id, Pose src_pose, Pose tgt_pose,
                        FakeMoveRecord& rec);
    mtc::Task createSmashTask(std::string& obj_id, Pose tgt_pose, std::string& dir,
                    planning_scene::PlanningSceneConstPtr start_scene, size_t n_blocks);
    mtc::Task createScoopTask(std::string& obj_id, Pose src_pose, Pose tgt_pose,
                    planning_scene::PlanningSceneConstPtr start_scene);
    int doMultiMoveTasks();
    bool areAllMoves();
    std::vector<Pose> getSimTopplePoses(std::vector<std::string>& ids,
                        std::vector<Pose>& poses, std::string tid, std::string dir);

    int executeTaskSequence(std::vector<mtc::Task>& tasks,
                    std::vector<int>& task2action,
                    std::unordered_map<int, FakeMoveRecord>& fake_cache,
                    bool stop_on_failure);

    void printTimings();

    bool executeFirstSolutionWithMGI(moveit::task_constructor::Task& task);
    bool executeSolutionSplitHand(moveit::task_constructor::Task& task);
    bool applyPlanningSceneDiff(moveit_msgs::msg::PlanningScene& diff_msg);

    GripperOutcome sendGripperGoal(double position, double max_effort,
                                 bool accept_stall_as_success);

    bool sendIsaacGripperOpenRequest();
    bool sendIsaacGripperCloseRequest();

private:
    //mtc::Task task_;
    rclcpp::Node::SharedPtr node_;
    struct Config cfg_;
    std::vector<Block> initials_, goals_;
    std::vector<Action> plan_;
    std::vector<std::string> all_blocks_;
    std::vector<std::string> all_objects_;
    PlanningSceneInterface psi_;
    size_t sample_angle_idx_;
    rclcpp::Publisher<PlanningScene>::SharedPtr ps_pub_;
    robot_model_loader::RobotModelLoaderPtr rml_;
    moveit::core::RobotModelConstPtr model_;
    planning_scene_monitor::PlanningSceneMonitorPtr psm_;

    float tamp_time_;
    float gap_planning_time_;
    float execution_time_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr update_client_;

    rclcpp::Node::SharedPtr gripper_client_node_;
    rclcpp_action::Client<GripperCommand>::SharedPtr gripper_client_;

    rclcpp::Node::SharedPtr isaac_open_gripper_client_node_;
    rclcpp::Node::SharedPtr isaac_close_gripper_client_node_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr isaac_open_gripper_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr isaac_close_gripper_client_;
};

/* grid helper: pad_id -> (row,col)  (row 0 = top) ------------------- */
inline std::map<int,std::pair<int,int>> MTCTaskNode::grid_rc(const Config& cfg)
{
    std::set<double, std::greater<>> ys;
    std::set<double> xs;
    for (auto& p: cfg.pads) {
        ys.insert(p.second.second);
        xs.insert(p.second.first);
    }

    std::vector<double> yv(ys.begin(), ys.end());
    std::vector<double> xv(xs.begin(), xs.end());

    std::map<int,std::pair<int,int>> rc;
    for (auto& p: cfg.pads) {
        int row = std::find(yv.begin(), yv.end(), p.second.second) - yv.begin();
        int col = std::find(xv.begin(), xv.end(), p.second.first)  - xv.begin();
        rc[p.first] = {row,col};
    }

    return rc;
}

/* ------------------------------------------------------------------ *
 *  printBlocks – fixed 2×4 grid, correct pad ordering                *
 * ------------------------------------------------------------------ */
inline void MTCTaskNode::printBlocks(const Config& cfg,
                         const std::vector<Block>& blocks)
{
    /* map pad_id -> stack[height]=name */
    std::map<int, std::map<int,std::string>> stack;
    for (const auto& b : blocks) {
        std::string& cell = stack[b.pad][b.height];
        if (cell.empty())
            cell = b.block;
        else
            cell += "/" + b.block;                      // concatenate if overlap
    }

    /* row/col assignment: x asc (top->bottom) , y asc (left->right) */
    std::vector<std::tuple<double,double,int>> tmp; // (x,y,id)
    for (auto& kv : cfg.pads)
        tmp.emplace_back(kv.second.first, kv.second.second, kv.first);
    std::sort(tmp.begin(), tmp.end(),
            [](auto&a,auto&b){
                if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) < std::get<0>(b);   // x ascending
                return std::get<1>(a) < std::get<1>(b);   // y ascending
            });
    std::map<int,std::pair<int,int>> rc;              // id->(row,col)
    for (int i=0;i<static_cast<int>(tmp.size());++i)
        rc[ std::get<2>(tmp[i]) ] = { i/4, i%4 };

    /* geometry */
    const int rows = 2, cols = 4;
    const int stackH = 5;                    // show up to 5 cubes per pad
    const int cellW  = 25;                   // chars
    const int cellH  = 2 + stackH + 1;       // border + id + stack + border

    std::vector<std::string> canvas(rows*cellH, std::string(cols*cellW, ' '));
    auto put = [&](int r,int c,const std::string& s){
        for(size_t i = 0; i < s.size() && c + i < canvas[r].size(); ++i)
            canvas[r][c+i]=s[i];
    };

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            /* pad id assigned to this slot --------------------------------- */
            int id = -1;
            for (auto& kv: rc)
                if(kv.second == std::make_pair(row,col))
                    id = kv.first;

            int offR = row * cellH, offC = col * cellW;

            const char *h="-", *v="|", *tl="+", *tr="+", *bl="+", *br="+";

            /* top border */
            put(offR, offC, std::string(tl) + std::string(cellW-2,*h) + tr);

            /* id line (always show pad id) */
            std::ostringstream lab;
            lab << " Pad" << std::setw(2) << std::left << id;
            std::string id_line = v + lab.str();
            id_line.resize(cellW - 1,' ');
            id_line += v;
            put(offR + 1, offC, id_line);

            for (int hgt = stackH - 1; hgt >= 0; --hgt)
            {
                std::string line = std::string(v) + std::string(cellW-2,' ') + v;
                auto it = stack[id].find(hgt);
                if (it != stack[id].end())
                {
                    /* -- 2. trim long labels neatly --------------------- */
                    std::string lbl = "[" + it->second + "]";
                    if (lbl.size() > cellW-4)
                        lbl.resize(cellW-7), lbl += "...";   // ellipsis

                    line.replace(2, lbl.size(), lbl);
                }
                put(offR + (stackH-hgt) + 1, offC, line);
            }
            /* bottom border */
            put(offR+cellH-1, offC, std::string(bl) + std::string(cellW-2,*h) + br);
        }
    }

    std::cout << "World layout  (row:x asc   col:y asc)\n";
    for(auto& ln : canvas)
        std::cout << ln << "\n";
}

/* ------------------------------------------------------------------ */
inline void MTCTaskNode::printPlan(const std::vector<Action>& plan)
{
    std::cout << "\nAction list:\n";
    for(size_t i = 0; i < plan.size(); ++i) {
        auto&s = plan[i];
        std::cout << " " << std::setw(2) << i+1 << ". "
             << std::setw(6) << s.block << " ";
        if(s.type == "Move") {
             std::cout << std::setw(5) << s.type << "   -->  ";
             std::cout << s.end_pad << "(" << s.end_height << ")\n";
        } else if(s.type == "Smash") {
             std::cout << std::setw(5) << s.type << "   <--  ";
             std::cout << s.dir << "\n";
        }
    }
}

/* master call ------------------------------------------------------ */
inline void MTCTaskNode::pretty(const Config& cfg,
                   const std::vector<Block>& initials,
                   const std::vector<Block>& goals,
                   const std::vector<Action>& plan)
{
    std::cout << "\nStart Blocks:\n";
    printBlocks(cfg, initials);
    std::cout << "\nGoal Blocks:\n";
    printBlocks(cfg, goals);
    printPlan(plan);
}

#if 0
Pose MTCTaskNode::getRelPose(Action& a)
{
    std::string id = getObjId(a.block);
    const auto poses = psi_.getObjectPoses({id});   // a.block_name == id
    auto it = poses.find(id);
    if (it == poses.end())
        throw std::runtime_error("Block " + id + " not found in scene");

    const geometry_msgs::msg::Pose& cur = it->second;

    /* 2. compute target pad/height pose ------------------------------ */
    geometry_msgs::msg::Pose delta;
    const auto& pad_end = cfg_.pads[a.end_pad];
    delta.position.x = pad_end.first  - cur.position.x;
    delta.position.y = pad_end.second - cur.position.y;
    delta.position.z = cfg_.z_base + a.end_height * cfg_.z_inc - cur.position.z + 0.01;

    delta.orientation.w = 1.0;

    RCLCPP_INFO(LOGGER, "======================");
    RCLCPP_INFO(LOGGER, "Object:%s", id.c_str());
    RCLCPP_INFO(LOGGER, "start: %f, %f, %f", cur.position.x, cur.position.y, cur.position.z);
    RCLCPP_INFO(LOGGER, "end: %f, %f, %f", pad_end.first, pad_end.second, cfg_.z_base + a.end_height * cfg_.z_inc);
    RCLCPP_INFO(LOGGER, "delta: %f, %f, %f", delta.position.x, delta.position.y, delta.position.z);
    return delta;
}
#endif

double MTCTaskNode::adjustHeight(double height)
{
    const auto k = static_cast<int>(std::llround((height - cfg_.z_base) / cfg_.z_inc));
    return height + std::max(k, 0) * 0.0005;
}

void MTCTaskNode::updateSceneObjectHeights()
{
    // Grab every object currently registered in the planning scene
    const auto objects = psi_.getObjects();               // map<string, CollisionObject>

    std::vector<moveit_msgs::msg::CollisionObject> diffs;
    diffs.reserve(objects.size());

    for (const auto& [id, obj] : objects)
    {
        std::cout << "updating " << id;
        CollisionObject msg;
        msg.header    = obj.header;                        // keep the frame_id & stamp
        msg.id        = id;                                // same unique identifier
        msg.operation = CollisionObject::MOVE;

        // Only the pose is modified (snap z to your stack grid)
        msg.pose            = obj.pose;
        std::cout << " height before:" << msg.pose.position.z;
        msg.pose.position.z = adjustHeight(msg.pose.position.z);
        std::cout << " height after:" << msg.pose.position.z << std::endl;

        diffs.emplace_back(std::move(msg));
    }

    // Publish the diff in one call; MoveIt updates all objects’ transforms
    psi_.applyCollisionObjects(diffs);
}

Pose MTCTaskNode::getPoseFromPH(int pad, int height)
{
    Pose p;

    p.position.x = cfg_.pads[pad].first;
    p.position.y = cfg_.pads[pad].second;
    p.position.z = cfg_.z_base + height * cfg_.z_inc + HEIGHT_ERROR_OFFSET * height;

    p.orientation.x = 0.0;
    p.orientation.y = 0.0;
    p.orientation.z = 0.0;
    p.orientation.w = 1.0;

    return p;
}

Pose MTCTaskNode::getPoseFromPSI(Action &a)
{
    std::string id;

    id = getObjId(a.block);

    const auto poses = psi_.getObjectPoses({id});
    auto it = poses.find(id);
    if (it != poses.end()) {
#if 0
        return it->second;
#else
        auto p = it->second;
        p.orientation.x = 0.0;
        p.orientation.y = 0.0;
        p.orientation.z = 0.0;
        p.orientation.w = 1.0;
        return p;
#endif
    }

    throw std::runtime_error("Block " + id + " not found in scene");
}

Pose MTCTaskNode::getTargetPose(Action& a)
{
    Pose p;

    p.position.x = cfg_.pads[a.end_pad].first;
    p.position.y = cfg_.pads[a.end_pad].second;
    p.position.z = cfg_.z_base + a.end_height * cfg_.z_inc + HEIGHT_ERROR_OFFSET * a.end_height;
    //p.position.z = adjustHeight(cfg_.z_base + a.end_height * cfg_.z_inc) + 0.01;
    p.orientation.x = 0.0;
    p.orientation.y = 0.0;
    p.orientation.z = 0.0;
    p.orientation.w = 1.0;

    return p;
}

Pose MTCTaskNode::getStartPose(Action& a)
{
    geometry_msgs::msg::Pose p;

    p.position.x = cfg_.pads[a.start_pad].first;
    p.position.y = cfg_.pads[a.start_pad].second;
    p.position.z = cfg_.z_base + a.start_height * cfg_.z_inc + HEIGHT_ERROR_OFFSET * a.start_height;
    p.orientation.x = 0.0;
    p.orientation.y = 0.0;
    p.orientation.z = 0.0;
    p.orientation.w = 1.0;

    return p;
}

std::string MTCTaskNode::toLower(const std::string& input)
{
    std::string lower;

    lower.reserve(input.size());
    for (char c : input)
        lower += static_cast<char>(std::tolower(c));

    return lower;
}

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr MTCTaskNode::getNodeBaseInterface()
{
  return node_->get_node_base_interface();
}

MTCTaskNode::MTCTaskNode(const rclcpp::NodeOptions& options)
  : node_{ std::make_shared<rclcpp::Node>("smash_node", options) }
{
    ps_pub_ = node_->create_publisher<PlanningScene>("/planning_scene", 10);
    rml_ = std::make_shared<robot_model_loader::RobotModelLoader>(node_, "robot_description");
    model_ = rml_->getModel();
    psm_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(node_, rml_);

    psm_->startSceneMonitor();
    psm_->startWorldGeometryMonitor();
    psm_->startStateMonitor();

    psm_->getStateMonitorNonConst()->waitForCompleteState(2.0);   // seconds

    const auto clock_type = node_->get_clock()->get_clock_type();
    psm_->waitForCurrentRobotState(rclcpp::Time(0, 0, clock_type), 2.0);

    tamp_time_ = gap_planning_time_ = execution_time_ = 0.0;

    update_client_ = node_->create_client<Trigger>("/update_smash_scene");
    gripper_client_node_ = std::make_shared<rclcpp::Node>("panda_gripper_client");
    gripper_client_ = rclcpp_action::create_client<GripperCommand>(gripper_client_node_, "/panda_hand_controller/gripper_cmd");

    isaac_open_gripper_client_node_ = std::make_shared<rclcpp::Node>("panda_gripper_client");
    isaac_open_gripper_client_ = isaac_open_gripper_client_node_->create_client<Trigger>("/isaac_open_gripper");
    isaac_close_gripper_client_node_ = std::make_shared<rclcpp::Node>("panda_gripper_client");
    isaac_close_gripper_client_ = isaac_close_gripper_client_node_->create_client<Trigger>("/isaac_close_gripper");


    //std::string problem_file = node_->declare_parameter<std::string>("problem");
    std::string problem_file;

    node_->get_parameter("problem", problem_file);
    RCLCPP_INFO(LOGGER, "load from problem file:%s", problem_file.c_str());

    YAML::Node doc = YAML::LoadFile(problem_file);

    for(const auto & kv : doc["pads"])
        cfg_.pads[kv.first.as<int>()] = {kv.second["x"].as<double>(), kv.second["y"].as<double>()};

    cfg_.z_base = doc["z_base"].as<double>();
    cfg_.z_inc  = doc["z_increment"].as<double>();

    for(const auto & a : doc["initial"]) {
        int pad = a["pad"].as<int>();
        int height = a["height"].as<int>();
        initials_.push_back({ a["name"].as<std::string>(),
                         pad,
                         height,
                         false,
                         getPoseFromPH(pad, height)});
    }

    for (const auto& a : doc["goal"]) {
        const std::string name = a["name"].as<std::string>();

        /* pull pad(s) ------------------------------------------------------ */
        std::vector<int> pads;
        if (a["pad"].IsSequence())
            for (const auto& p : a["pad"]) pads.push_back(p.as<int>());
        else
            pads.push_back( a["pad"].as<int>() );

        /* pull height(s) --------------------------------------------------- */
        std::vector<int> heights;
        if (a["height"].IsSequence())
            for (const auto& h : a["height"]) heights.push_back(h.as<int>());
        else
            heights.push_back( a["height"].as<int>() );

        /* broadcast if only one dimension is a list ----------------------- */
        if (pads.size() == 1 && heights.size() > 1)
            pads.assign(heights.size(), pads.front());
        if (heights.size() == 1 && pads.size() > 1)
            heights.assign(pads.size(), heights.front());

        if (pads.size() != heights.size())
            throw std::runtime_error("Mismatch pads vs heights for " + name);

        /* append one GoalEntry per concrete location ---------------------- */
        for (std::size_t i = 0; i < pads.size(); ++i)
            goals_.push_back({ name, pads[i], heights[i], false,
                         getPoseFromPH(pads[i], heights[i])});
    }

    for(const auto & a : doc["actions"]) {
        std::string type = a["type"].as<std::string>();
        if(type == "Move") {
            plan_.push_back({ a["block"].as<std::string>(),
                              type,
                              a["from"]["pad"].as<int>(),
                              a["from"]["height"].as<int>(),
                              a["to"]["pad"].as<int>(),
                              a["to"]["height"].as<int>(),
                              "",
                              {}});
        } else if(type == "Smash") {
            // Gather full topple stack if available (expected order: bottom→top)
            std::vector<std::string> blocks;
            for (const auto& b : a["blocks"])
                blocks.push_back(b.as<std::string>());
            plan_.push_back({ blocks.front(), //bottom
                              type,
                              a["from"]["pad"].as<int>(),
                              a["from"]["height"].as<int>(),
                              0,
                              0,
                              a["dir"].as<std::string>(),
                              std::move(blocks)});
        } else if(type == "ScoopMove") {
            // Gather full topple stack if available (expected order: bottom→top)
            std::vector<std::string> blocks;
            for (const auto& b : a["blocks"])
                blocks.push_back(b.as<std::string>());
            plan_.push_back({ blocks.front(), //bottom
                              type,
                              a["from"]["pad"].as<int>(),
                              0,
                              a["to"]["pad"].as<int>(),
                              0,
                              "",
                              std::move(blocks)});
        }
    }

    pretty(cfg_, initials_, goals_, plan_);

    if(requestSceneUpdate())
        RCLCPP_INFO(LOGGER, "request scene update success");

    all_objects_ = psi_.getKnownObjectNames();
    int attempt = 0;
    while((all_objects_.size() == 0) && (attempt++ < 20)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        all_objects_ = psi_.getKnownObjectNames();
    }
    std::cout << "\nObjects in the scene:" << std::endl;
    for (const auto& id : all_objects_) {
        std::cout << id << "\n";
        if (toLower(id).find("cube") != std::string::npos) {
            all_blocks_.push_back(id);
        }
    }
       
    //waitForAnyKey();
}

bool MTCTaskNode::requestSceneUpdateObjects(const std::string& name, const std::string& action)
{
    auto client = node_->create_client<blocksworld_scene::srv::UpdateScene>("/update_smash_scene_objects");
    if (!client->wait_for_service(2s))
    {
        RCLCPP_ERROR(node_->get_logger(), "update_planning_scene_objects service not available");
        return false;
    }

    auto req = std::make_shared<blocksworld_scene::srv::UpdateScene::Request>();
    req->name   = name;
    req->action = action;

    auto future = client->async_send_request(req);

    /* wait up to 5 s while *your existing executor thread* spins callbacks */
    if (future.wait_for(5s) == std::future_status::ready)
    {
        auto resp = future.get();
        RCLCPP_INFO(node_->get_logger(), "Scene update objects says: %s", resp->message.c_str());
        return resp->success;
    }
    RCLCPP_ERROR(node_->get_logger(), "Scene update objects timed out");
    return false;
}

bool MTCTaskNode::requestSceneUpdate()
{
    if (!update_client_->wait_for_service(2s))
    {
        RCLCPP_ERROR(node_->get_logger(), "update_planning_scene service not available");
        return false;
    }
    auto req = std::make_shared<Trigger::Request>();
    auto future = update_client_->async_send_request(req);

#if 0
    if (rclcpp::spin_until_future_complete(node_, future, 10s) 
                                == rclcpp::FutureReturnCode::SUCCESS)
    {
        auto resp = future.get();
        RCLCPP_INFO(node_->get_logger(), "Scene update says: %s", resp->message.c_str());
        return resp->success;
    }
    RCLCPP_ERROR(node_->get_logger(), "Scene update timed out");
    return false;
#endif

#if 0
    /* wait up to 5 s while *your existing executor thread* spins callbacks */
    if (future.wait_for(5s) == std::future_status::ready)
    {
        auto resp = future.get();
        RCLCPP_INFO(node_->get_logger(), "Scene update says: %s", resp->message.c_str());
        return resp->success;
    }
    RCLCPP_ERROR(node_->get_logger(), "Scene update timed out");
    return false;
#endif

    return true;
}

bool MTCTaskNode::sendIsaacGripperOpenRequest()
{
    constexpr auto kConnectTimeout = std::chrono::seconds(2);
    constexpr auto kResultTimeout  = std::chrono::seconds(5);

    if (!isaac_open_gripper_client_) {
        std::cerr << "gripper_client_ is null" << std::endl;
        return false;
    }
    if (!isaac_open_gripper_client_->wait_for_service(kConnectTimeout)) {
        std::cerr << "ISAAAC action server not available" << std::endl;
        return false;
    }

    // Use a tiny, private executor that only spins the client node
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(isaac_open_gripper_client_node_);

    auto req = std::make_shared<Trigger::Request>();
    auto future = isaac_open_gripper_client_->async_send_request(req);
    RCLCPP_INFO(node_->get_logger(), "Gripper open request sent.");

    if (exec.spin_until_future_complete(future, kResultTimeout) != rclcpp::FutureReturnCode::SUCCESS) {
        std::cerr << "send isaac gripper request timed out" << std::endl;
        rclcpp::sleep_for(std::chrono::milliseconds(DEBUG_SLEEP_MS));
        return false;
    }

    auto resp = future.get();
    if (!resp) {
        std::cerr << "isaac gripper request rejected by server" << std::endl;
        rclcpp::sleep_for(std::chrono::milliseconds(DEBUG_SLEEP_MS));
        return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Scene update says: %s", resp->message.c_str());
    rclcpp::sleep_for(std::chrono::milliseconds(DEBUG_SLEEP_MS));
    waitForAnyKey();

    return resp->success;
}

bool MTCTaskNode::sendIsaacGripperCloseRequest()
{
    constexpr auto kConnectTimeout = std::chrono::seconds(2);
    constexpr auto kResultTimeout  = std::chrono::seconds(5);

    if (!isaac_close_gripper_client_) {
        std::cerr << "gripper_client_ is null" << std::endl;
        return false;
    }
    if (!isaac_close_gripper_client_->wait_for_service(kConnectTimeout)) {
        std::cerr << "ISAAAC action server not available" << std::endl;
        return false;
    }

    // Use a tiny, private executor that only spins the client node
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(isaac_close_gripper_client_node_);

    auto req = std::make_shared<Trigger::Request>();
    auto future = isaac_close_gripper_client_->async_send_request(req);
    RCLCPP_INFO(node_->get_logger(), "Gripper close request sent.");

    if (exec.spin_until_future_complete(future, kResultTimeout) != rclcpp::FutureReturnCode::SUCCESS) {
        std::cerr << "send isaac gripper request timed out" << std::endl;
        rclcpp::sleep_for(std::chrono::milliseconds(DEBUG_SLEEP_MS));
        return false;
    }

    auto resp = future.get();
    if (!resp) {
        std::cerr << "isaac gripper request rejected by server" << std::endl;
        rclcpp::sleep_for(std::chrono::milliseconds(DEBUG_SLEEP_MS));
        return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Scene update says: %s", resp->message.c_str());

    std::this_thread::sleep_for(std::chrono::milliseconds(DEBUG_SLEEP_MS));
    waitForAnyKey();

    return resp->success;
}

bool MTCTaskNode::requestSceneAddObjects()
{
    auto client = node_->create_client<Trigger>("/add_smash_scene_objects");

    if (!client->wait_for_service(2s))
    {
        RCLCPP_ERROR(node_->get_logger(), "add_smash_scene_objects service not available");
        return false;
    }
    auto req = std::make_shared<Trigger::Request>();
    auto future = client->async_send_request(req);

    /* wait up to 5 s while *your existing executor thread* spins callbacks */
    if (future.wait_for(5s) == std::future_status::ready)
    {
        auto resp = future.get();
        RCLCPP_INFO(node_->get_logger(), "Scene objects add says: %s", resp->message.c_str());
        return resp->success;
    }
    RCLCPP_ERROR(node_->get_logger(), "Scene objects add timed out");
    return false;

}

bool MTCTaskNode::requestSceneClearObjects()
{
    auto client = node_->create_client<Trigger>("/remove_smash_scene_objects");

    if (!client->wait_for_service(2s))
    {
        RCLCPP_ERROR(node_->get_logger(), "remove_smash_scene_objects service not available");
        return false;
    }
    auto req = std::make_shared<Trigger::Request>();
    auto future = client->async_send_request(req);

    /* wait up to 5 s while *your existing executor thread* spins callbacks */
    if (future.wait_for(5s) == std::future_status::ready)
    {
        auto resp = future.get();
        RCLCPP_INFO(node_->get_logger(), "Scene objects remove says: %s", resp->message.c_str());
        return resp->success;
    }
    RCLCPP_ERROR(node_->get_logger(), "Scene objects remove timed out");
    return false;
}

/* helper to build a primitive collision object --------------------------------*/
CollisionObject MTCTaskNode::makeBox(const std::string& id,
                               double dx, double dy, double dz,
                               double px, double py, double pz)
{
    CollisionObject obj;
    obj.id = id;
    obj.header.frame_id = "world";
    obj.operation = CollisionObject::ADD;

    SolidPrimitive prim;
    prim.type = SolidPrimitive::BOX;
    prim.dimensions = {dx, dy, dz};

    Pose pose;
    pose.position.x = px;
    pose.position.y = py;
    pose.position.z = pz;
    pose.orientation.w = 1.0;

    obj.primitives.push_back(prim);
    obj.primitive_poses.push_back(pose);
    return obj;
}

/* helper to build an RGBA colour ---------------------------------------------*/
ColorRGBA MTCTaskNode::rgba(float r, float g, float b, float a)
{
    std_msgs::msg::ColorRGBA c;

    c.r=r;
    c.g=g;
    c.b=b;
    c.a=a;

    return c;
}

/* main call ------------------------------------------------------------------*/
void MTCTaskNode::setupPlanningScene()
{
    CollisionObject object;
    object.id = "object";
    object.header.frame_id = "world";
    object.primitives.resize(1);
    object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    object.primitives[0].dimensions = { 0.2, 0.02 };

    geometry_msgs::msg::Pose pose;
    pose.position.x = 0.5;
    pose.position.y = -0.25;
    pose.position.z = 0.10;
    pose.orientation.w = 1.0;
    object.pose = pose;

    /* 1. describe geometry ---------------------------------------------------- */
    CollisionObject table = makeBox("table",
                    /* size  */ 2.0, 2.0, 0.3,
                    /* pose  */ 0.1, 0.0, -0.151);            // top surface @ z-0.35

    CollisionObject wall = makeBox("wall",
                    /* size  */ 0.15, 2.0, 2.0,
                    /* pose  */ -0.8, 0.0, 1.0);           // thin wall 60 cm ahead

#ifdef ROBOT_TYPE_FRANKA_PANDA
    std::vector<CollisionObject> objs = {table, wall/*, object*/};
#else
    std::vector<CollisionObject> objs = {wall/*, object*/};
#endif

    /* 2. colours -------------------------------------------------------------- */
    ObjectColor col_table, col_wall, col_obj;
    col_table.id    = "table";
    col_table.color = rgba(0.55f, 0.27f, 0.07f);     // brown

    col_wall.id     = "wall";
    col_wall.color  = rgba(0.75f, 0.75f, 0.75f);     // light-grey

    col_obj.id    = "object";
    col_obj.color = rgba(0.00f, 0.00f, 1.00f);     // blue

#ifdef ROBOT_TYPE_FRANKA_PANDA
    std::vector<ObjectColor> cols = {col_table, col_wall/*, col_obj*/};
#else
    std::vector<ObjectColor> cols = {col_wall/*, col_obj*/};
#endif

    /* 3. push both vectors in one call --------------------------------------- */
    psi_.applyCollisionObjects(objs, cols);

    RCLCPP_INFO(node_->get_logger(), "Table, wall, and object added to planning scene");
}

std::string MTCTaskNode::getObjId(std::string color)
{
    std::string id = "";

    for (const auto& name : all_blocks_) {
        //RCLCPP_INFO(LOGGER, "Block name:%s", name.c_str());
        auto poses = psi_.getObjectPoses({name});
        auto it = poses.find(id);
        if (it != poses.end()) {
            auto p = it->second;
            RCLCPP_INFO(LOGGER, "Block name:%s, Pose: (%f, %f, %f), (%f, %f, %f, %f)", name.c_str(),
                    p.position.x, p.position.y, p.position.z,
                    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
        }

        if (toLower(name).find(toLower(color)) != std::string::npos)
            id = name;
    }

    return id;
}

void MTCTaskNode::waitForAnyKey()
{
#if 0
  std::cout << "\nPress any key to continue..." << std::flush;

  termios old_tio{}, new_tio{};
  tcgetattr(STDIN_FILENO, &old_tio);      // save current settings
  new_tio = old_tio;
  new_tio.c_lflag &= ~(ICANON | ECHO);    // raw mode, no echo
  tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);

  getchar();                              // wait for a single key press

  tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);  // restore settings
  std::cout << '\n';
#endif
}

bool MTCTaskNode::areAllMoves() {
  return std::all_of(plan_.begin(), plan_.end(),
                     [](const Action& s){ return s.type == "Move"; });
}

std::vector<Pose> MTCTaskNode::getSimTopplePoses(std::vector<std::string>& ids,
        std::vector<Pose>& poses, std::string tid, std::string dir)
{
#if 0
    std::vector<std::string> ids = {"Red1","g2","Blue3"};  // mixed styles OK
    std::vector<Pose> poses(ids.size());
    const double edge = 0.05;
    for (size_t i=0;i<ids.size();++i) {
        poses[i].position.x = 0.0;
        poses[i].position.y = 0.0;
        poses[i].position.z = (0.5 + i)*edge;
        poses[i].orientation.w = 1.0;
    }
#endif

    topple::ToppleConfig cfg;
    cfg.edge_m = cfg_.z_inc;
    if(dir == "+x" || dir == "+X" || dir == "x" || dir == "X") {
        cfg.dir_x = 1.0;
        cfg.dir_y = 0.0;
    } else if (dir == "+y" || dir == "+Y" || dir == "y" || dir == "Y") {
        cfg.dir_x = 0.0;
        cfg.dir_y = 1.0;
    }
    cfg.mode = "force";
    cfg.gui = false;                 // set true to see GUI
    cfg.package_name = "blocksworld_executor"; // where the script is installed

    auto new_poses = topple::simulate_topple_with_pybullet(ids, poses, tid, cfg);

    std::cout << std::endl;
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto& P = new_poses[i];
        std::cout << ids[i] << " -> ["
                  << P.position.x << ", "
                  << P.position.y << ", "
                  << P.position.z << "]\n";
    }

    return new_poses;
}

int MTCTaskNode::executeTaskSequence(std::vector<mtc::Task>& tasks,
                                      std::vector<int>& task2action,
                                      std::unordered_map<int, FakeMoveRecord>& fake_cache,
                                      bool stop_on_failure)
{
    bool res;

    for (size_t i = 0; i < tasks.size(); ++i) {

        if(requestSceneUpdate())
            RCLCPP_INFO(LOGGER, "request scene update success");
        rclcpp::sleep_for(std::chrono::milliseconds(1000));

        int action_id = task2action[i];
        auto it  = fake_cache.find(action_id);
        bool is_fake = (it != fake_cache.end() && it->second.is_fake_move);

        moveit_task_constructor_msgs::msg::Solution sol_msg;
        tasks[i].solutions().front()->toMsg(sol_msg);
        const auto model = tasks[i].getRobotModel();
        RCLCPP_INFO(LOGGER, "Original Solution Message for Task %d (action id:%d):", i, action_id);
        mtc_debug::printFirstSolutionMessage(sol_msg, model, /*max_points_per_segment=*/6);
        //executeFirstSolutionWithMGI(tasks[i]);

        if(is_fake) {
            std::cout << "delayed task:" << i << std::endl;
            auto& s = plan_[action_id];

            std::string scene_id = getObjId(s.block);
            Pose sp = getPoseFromPSI(s);
            Pose tp = getTargetPose(s);

            RCLCPP_INFO(LOGGER, "Populate Real Task, Block: %s", s.block.c_str());
            RCLCPP_INFO(LOGGER, "Start Pose: (%f, %f, %f), (%f, %f, %f, %f)",
                    sp.position.x, sp.position.y, sp.position.z,
                    sp.orientation.x, sp.orientation.y, sp.orientation.z, sp.orientation.w);
            RCLCPP_INFO(LOGGER, "Target Pose: (%f, %f, %f), (%f, %f, %f, %f)",
                    tp.position.x, tp.position.y, tp.position.z,
                    tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w);

            auto t2 = Clock::now();
            sample_angle_idx_ = 0;
            mtc::Task task_ = createRealFromFakeMoveTask(scene_id, sp, tp, it->second);

            try {
                task_.init();
            } catch (mtc::InitStageException& e) {
                RCLCPP_ERROR_STREAM(LOGGER, e);
                return TASK_INIT_ERROR;
            }

            auto res = task_.plan(10 /* max_solutions */);
            if(res.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
                int retry = 0;
                while((retry++ < 100) && (res.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)) {
                    task_ = createRealFromFakeMoveTask(scene_id, sp, tp, it->second);
                    task_.init();
                    res = task_.plan(10 /* max_solutions */);
                }
                if(res.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
                    return DELAY_PLAN_ERROR;
            }
            gap_planning_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t2).count();

            task_.introspection().publishSolution(*task_.solutions().front());

            RCLCPP_INFO(LOGGER, "%ld solutions generated, use the first one.", task_.solutions().size());
            auto t3 = Clock::now();

#ifdef ROBOT_TYPE_FRANKA_PANDA
            res = executeSolutionSplitHand(task_);
#else
            auto result = task_.execute(*task_.solutions().front());
            if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
                return EXECUTE_ERROR;
            }
#endif
            execution_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t3).count();

        } else {

            auto& task = tasks[i];
            std::cout << "normal task:" << i << ", " << task.numSolutions() << " solutions." << std::endl;
            if (task.numSolutions() == 0) {
                RCLCPP_ERROR(LOGGER, "[%zu/%zu] No solution to execute.", i+1, tasks.size());
                if (stop_on_failure)
                    return -14;
                else
                    continue;
            }

            auto t3 = Clock::now();
            auto sol = task.solutions().front();
            //task.introspection().publishSolution(*sol);  // optional
#ifdef ROBOT_TYPE_FRANKA_PANDA
            res = executeSolutionSplitHand(task);
#else
            auto result = task.execute(*sol);
            if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
                return EXECUTE_ERROR;
            }
#endif
            RCLCPP_INFO(LOGGER, "[%zu/%zu] Execution OK.", i+1, tasks.size());
            execution_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t3).count();
        }

    }

    if(requestSceneUpdate())
        RCLCPP_INFO(LOGGER, "request scene update success");

    return (res != true);
}

// Helper: get a joint-goal map for a group from a RobotState
static std::map<std::string,double>
jointGoalFromState(const moveit::core::RobotState& rs, const std::string& group) {
    std::vector<double> pos;
    std::map<std::string,double> goal;
    const auto* jmg = rs.getJointModelGroup(group);

    rs.copyJointGroupPositions(jmg, pos);
    const auto& names = jmg->getVariableNames();

    for (size_t i = 0; i < names.size(); ++i)
        goal[names[i]] = pos[i];

    return goal;
}

bool extractEndJoints(const mtc::Task& task,
                      const std::string& arm_group,
                      std::map<std::string, double>& end_joints_out)
{
    end_joints_out.clear();
    if (task.numSolutions() == 0)
        return false;

    auto sol = task.solutions().front();
    auto end_scene = sol->end()->scene();
    if (!end_scene)
        return false;

    if (!end_scene->getRobotModel()->hasJointModelGroup(arm_group))
        return false;

    const auto& rs = end_scene->getCurrentState();
    end_joints_out = jointGoalFromState(rs, arm_group);

    return !end_joints_out.empty();
}

int MTCTaskNode::doMultiMoveTasks()
{
    int retry = 0;
    std::vector<mtc::Task> tasks_;
    std::vector<int> task2action;
    std::unordered_map<std::string, Block> blocks_map;
    std::vector<std::string> ids;
    std::vector<Pose> poses;
    std::unordered_map<int, FakeMoveRecord> fakemove_cache;

    blocks_map.reserve(initials_.size());
    for (const auto& b : initials_)
        blocks_map.emplace(b.block, b);

#if 0
    /* return to 'ready' state */
    mtc::Task task_ = createPreMoveTask(true, nullptr);
    try {
        task_.init();
    } catch (mtc::InitStageException& e) {
        RCLCPP_ERROR_STREAM(LOGGER, e);
        return TASK_INIT_ERROR;
    }

    if (!task_.plan(10 /* max_solutions */)) {
        RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
        return PLAN_ERROR;
    }
    task_.introspection().publishSolution(*task_.solutions().front());

    RCLCPP_INFO(LOGGER, "%ld solutions generated, use the first one.", task_.solutions().size());
    auto result = task_.execute(*task_.solutions().front());

    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
        return EXECUTE_ERROR;
    }
#endif

#ifdef ROBOT_TYPE_FRANKA_PANDA
    executeReturnHomeTask(/*start scene*/nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    executeReturnHomeTask(/*start scene*/nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    sendIsaacGripperOpenRequest();
    sendIsaacGripperOpenRequest();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#endif

    if(requestSceneUpdate())
        RCLCPP_INFO(LOGGER, "request scene update success");

    for (const auto& kv : blocks_map) {
        const auto& id = kv.first;
        const auto& b  = kv.second;
        std::cout << id << ":" << std::endl;
        std::cout << "\tpad:" << b.pad << std::endl;
        std::cout << "\theight:" << b.height << std::endl;
        std::cout << "\ttopple flag:" << b.is_toppled << std::endl;
    }

    //tasks_.reserve(plan_.size());
    tasks_.reserve(plan_.size()*2);
    task2action.reserve(plan_.size()*2);
    planning_scene::PlanningSceneConstPtr next_start_scene;

    for (size_t i = 0; i < plan_.size(); ++i) {
        auto& s = plan_[i];
        std::string scene_id = getObjId(s.block);

        RCLCPP_INFO(LOGGER, "Block:%s, Object:%s, Action:%s",
                s.block.c_str(), scene_id.c_str(), s.type.c_str());

#ifndef SINGLE_GOAL_PROBLEM
        auto t1 = Clock::now();

            tasks_.emplace_back();
            mtc::Task& pre_task_ = tasks_.back();
            task2action.push_back(-1);

            /* return 'ready' state */
        if(s.type == "Smash")
            pre_task_ = createPreMoveTask(false, next_start_scene);
        else
            pre_task_ = createPreMoveTask(true, next_start_scene); /* open hand */

        {
            try {
                pre_task_.init();
            } catch (mtc::InitStageException& e) {
                RCLCPP_ERROR_STREAM(LOGGER, e);
                return TASK_INIT_ERROR;
            }

            if (!pre_task_.plan(10 /* max_solutions */)) {
                RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
                return PLAN_ERROR;
            }

            tamp_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t1).count();

            std::cout << "pre-task solution:" << pre_task_.numSolutions() << std::endl;
            pre_task_.introspection().publishSolution(*pre_task_.solutions().front());
        }

        //auto result = task_.execute(*task_.solutions().front());
        //if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        //    RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
        //    return -3;
        //}
        //next_start_scene = pre_task_.solutions().front()->end()->scene();
#endif

        tasks_.emplace_back();
        task2action.push_back(i);
        mtc::Task& task_ = tasks_.back();

        if(s.type == "Smash") {
            Pose p = getPoseFromPSI(s);
            RCLCPP_INFO(LOGGER, "Pose: (%f, %f, %f), (%f, %f, %f, %f)",
                    p.position.x, p.position.y, p.position.z,
                    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
            RCLCPP_INFO(LOGGER, "Direction: %s", s.dir.c_str());

            //requestSceneClearObjects();
            //task_ = createSmashTask(scene_id, p, s.dir, next_start_scene);

            ids.clear();
            poses.clear();

            const int pad = s.start_pad;
            for (const auto& kv : blocks_map) {
                const auto& id = kv.first;
                const auto& b  = kv.second;
                if (b.pad == pad) {
                    ids.push_back(id);
                    if(b.is_toppled || b.height < 0) {
                        std::cerr << "Mismatch topple flag or incorrect height for " + b.block << std::endl;
                        return -10;
                    }
                }
            }
            std::cout << "The stack that are toppled:";
            for (const auto& id : ids) {
                std::cout << id << ",";
                poses.push_back(blocks_map.at(id).p);
            }
            std::cout << std::endl;

            // Simulate toppling
            std::vector<Pose> sim_out = getSimTopplePoses(ids, poses, s.block, s.dir);

            // Build id->pose from sim output
            std::unordered_map<std::string, Pose> sim_map;
            sim_map.reserve(ids.size());
            for (size_t i = 0; i < ids.size(); ++i)
                sim_map.emplace(ids[i], sim_out[i]);

            // Update toppled blocks
            std::cout << "the blocks that are toppled and updated:";
            for (auto& id : s.blocks) {
                std::cout << id.c_str() << ",";
                Block& b = blocks_map.at(id);
                b.p          = sim_map.at(id);
                b.is_toppled = true;
                b.pad        = -1;
                b.height     = 0;

                moveit_msgs::msg::CollisionObject co;
                co.header.frame_id = "world";
                co.id = getObjId(id);
                co.pose = b.p;
                co.operation = moveit_msgs::msg::CollisionObject::MOVE;
                psi_.applyCollisionObject(co);

                moveit_msgs::msg::AttachedCollisionObject aco;
                aco.object.id = getObjId(id);
                aco.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
                psi_.applyAttachedCollisionObject(aco);
            }
            std::cout << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));

            auto t2 = Clock::now();
#if 1
            task_ = createSmashTask(scene_id, p, s.dir, nullptr, s.blocks.size());
#else
            if(next_start_scene) {
                std::string summary = dumpPlanningScene(next_start_scene);
                RCLCPP_INFO(LOGGER, "============NEXT START SCENE======================");
                RCLCPP_INFO_STREAM(node_->get_logger(), "\n" << summary);
                const auto& end_state = next_start_scene->getCurrentState();
                //planning_scene_monitor::LockedPlanningSceneRO ls(psm_);
                //planning_scene::PlanningScenePtr updated_start_scene = ls->diff();   // world from *live* scene
                planning_scene::PlanningScenePtr updated_start_scene = psm_->getPlanningScene()->diff();   // world from *live* scene
                updated_start_scene->setCurrentState(end_state);  
                summary = dumpPlanningScene(updated_start_scene);
                RCLCPP_INFO(LOGGER, "============UPDATED START SCENE======================");
                RCLCPP_INFO_STREAM(node_->get_logger(), "\n" << summary);
                task_ = createSmashTask(scene_id, p, s.dir, updated_start_scene, s.blocks.size());
            } else {
                task_ = createSmashTask(scene_id, p, s.dir, next_start_scene, s.blocks.size());
            }
#endif

            try {
                task_.init();
            } catch (mtc::InitStageException& e) {
                RCLCPP_ERROR_STREAM(LOGGER, e);
                return TASK_INIT_ERROR;
            }

            auto res = task_.plan(5 /* max_solutions */);
            if(res.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
                return PLAN_ERROR;
            }
            tamp_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t2).count();

            task_.introspection().publishSolution(*task_.solutions().front());
            RCLCPP_INFO(LOGGER, "%ld solutions generated, retry:%d.", task_.numSolutions(), retry);
            //std::this_thread::sleep_for(std::chrono::milliseconds(2500));

        } else if(s.type == "ScoopMove") {

            ids.clear();
            poses.clear();

            // Update the scoop transferred blocks
            std::cout << "the blocks that are scoop transfer and updated:";
            for (auto& id : s.blocks) {
                std::cout << id.c_str() << ",";
                Block& b = blocks_map.at(id);
                //b.p          = sim_map.at(id);  // we don't care about where are they for now
                b.is_toppled = true;  /* we borrow this field for scoop  flag */
                b.pad        = -1;
                b.height     = 0;

                moveit_msgs::msg::CollisionObject co;
                co.header.frame_id = "world";
                co.id = getObjId(id);
                co.pose = b.p;
                co.operation = moveit_msgs::msg::CollisionObject::MOVE;
                psi_.applyCollisionObject(co);

                moveit_msgs::msg::AttachedCollisionObject aco;
                aco.object.id = getObjId(id);
                aco.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
                psi_.applyAttachedCollisionObject(aco);
            }
            std::cout << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));

            auto t2 = Clock::now();
            Pose sp = getPoseFromPH(s.start_pad, 0);
            Pose tp = getPoseFromPH(s.end_pad, 0);
            //sp.position.z += 0.015;
            //tp.position.z += 0.015;
            RCLCPP_INFO(LOGGER, "Start Pose: (%f, %f, %f), (%f, %f, %f, %f)",
                        sp.position.x, sp.position.y, sp.position.z,
                        sp.orientation.x, sp.orientation.y, sp.orientation.z, sp.orientation.w);
            RCLCPP_INFO(LOGGER, "Target Pose: (%f, %f, %f), (%f, %f, %f, %f)",
                        tp.position.x, tp.position.y, tp.position.z,
                        tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w);
            task_ = createScoopTask(scene_id, sp, tp, next_start_scene);

            try {
                task_.init();
            } catch (mtc::InitStageException& e) {
                RCLCPP_ERROR_STREAM(LOGGER, e);
                return TASK_INIT_ERROR;
            }

            auto res = task_.plan(5 /* max_solutions */);
            if(res.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
                return PLAN_ERROR;
            }
            tamp_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t2).count();

            task_.introspection().publishSolution(*task_.solutions().front());
            RCLCPP_INFO(LOGGER, "%ld solutions generated, retry:%d.", task_.numSolutions(), retry);
            //std::this_thread::sleep_for(std::chrono::milliseconds(2500));

        } else if (s.type == "Move") {
            auto t2 = Clock::now();
            Block& b = blocks_map.at(s.block);

            if (!b.is_toppled && ((b.pad != s.start_pad) || (b.height != s.start_height))) {
                std::cerr << "Mismatch pads and heights for " + s.block
                          << "(" << b.pad << "," << s.start_pad << ")"
                          << "(" << b.height << "," << s.start_height << ")" << std::endl;
                return -11;
            }

            /* keep track of the scene block position and height */
            b.pad    = s.end_pad;
            b.height = s.end_height;

            Pose tp = getTargetPose(s);
            Pose sp = getPoseFromPSI(s);

            if(b.is_toppled) {
                RCLCPP_INFO(LOGGER, "Plan for fake move.");
                RCLCPP_INFO(LOGGER, "Target Pose: (%f, %f, %f), (%f, %f, %f, %f)",
                            tp.position.x, tp.position.y, tp.position.z,
                            tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w);

                task_ = createFakeMoveTask(tp, next_start_scene);
            } else {
                RCLCPP_INFO(LOGGER, "Start Pose: (%f, %f, %f), (%f, %f, %f, %f)",
                            sp.position.x, sp.position.y, sp.position.z,
                            sp.orientation.x, sp.orientation.y, sp.orientation.z, sp.orientation.w);
                RCLCPP_INFO(LOGGER, "Target Pose: (%f, %f, %f), (%f, %f, %f, %f)",
                            tp.position.x, tp.position.y, tp.position.z,
                            tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w);

                sample_angle_idx_ = 0;
                task_ = createMoveTask(scene_id, sp, tp, next_start_scene);
            }

            b.p = getPoseFromPH(b.pad, b.height);

            try {
                task_.init();
            } catch (mtc::InitStageException& e) {
                RCLCPP_ERROR_STREAM(LOGGER, e);
                return TASK_INIT_ERROR;
            }

            auto res = task_.plan(10 /* max_solutions */);
            if(res.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
#if 1
                retry = 0;
                while((retry++ < 10) && (res.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)) {
                    task_ = createMoveTask(scene_id, sp, tp, next_start_scene);
                    task_.init();
                    res = task_.plan(10 /* max_solutions */);
                }
#endif
                if(res.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
                    return PLAN_ERROR;
            }

            tamp_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t2).count();

            task_.introspection().publishSolution(*task_.solutions().front());
            RCLCPP_INFO(LOGGER, "%ld solutions generated, retry:%d.", task_.numSolutions(), retry);
            //std::this_thread::sleep_for(std::chrono::milliseconds(2500));

            auto t3 = Clock::now();
            if(b.is_toppled) {
                std::map<std::string, double> end_joints;
                const bool ok = extractEndJoints(task_, ARM_GROUP_NAME, end_joints);
                if (!ok) {
                    RCLCPP_ERROR_STREAM(LOGGER, "Extract end scene IK failed");
                    return -11;
                }

                fakemove_cache.emplace(i, FakeMoveRecord{true, next_start_scene, std::move(end_joints)});
            }
            tamp_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t3).count();

            moveit_msgs::msg::CollisionObject co;
            co.header.frame_id = "world";
            co.id = scene_id;
            co.pose = tp;
            co.operation = moveit_msgs::msg::CollisionObject::MOVE;
            psi_.applyCollisionObject(co);

            moveit_msgs::msg::AttachedCollisionObject aco;
            aco.object.id = scene_id;
            aco.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
            psi_.applyAttachedCollisionObject(aco);

            b.is_toppled = false;
        }

#if 0
        if(s.type == "Smash") {
            requestSceneUpdate();

            std::vector<moveit_msgs::msg::CollisionObject> ops;
            ops.reserve(blocks_map.size());

            for (const auto& kv : blocks_map) {
                const auto& id = kv.first;
                const auto& b  = kv.second;
                moveit_msgs::msg::CollisionObject co;
                co.header.frame_id = "world";
                co.id = getObjId(id);
                co.operation = moveit_msgs::msg::CollisionObject::MOVE;
                co.pose = b.p;
                ops.push_back(std::move(co));
            }

            psi_.applyCollisionObjects(ops);
        }
#endif

        moveit_msgs::msg::PlanningScene ps_msg;
        ps_msg.is_diff = true;

        const std::vector<std::string> world_ids = psi_.getKnownObjectNames(/*with_type=*/false);
        const auto world_map = psi_.getObjects(world_ids);
        ps_msg.world.collision_objects.reserve(world_map.size());
        for (const auto& kv : world_map) {
            ps_msg.world.collision_objects.push_back(kv.second);
        }

        ps_pub_->publish(ps_msg);

#if 1
        next_start_scene = task_.solutions().front()->end()->scene();
#else
        auto solutions = task_.solutions();
        solutions.sort();
        auto best_solution = solutions.front();
        next_start_scene = best_solution->end()->scene();
#endif
        std::cout << "update the next start scene." << std::endl;
    }

    std::cout << "\nTask and Motion Plan Successfully!" << std::endl;


    for (size_t i = 0; i < tasks_.size(); ++i) {

        moveit_task_constructor_msgs::msg::Solution sol_msg;
        tasks_[i].solutions().front()->toMsg(sol_msg);
        const auto model = tasks_[i].getRobotModel();
        RCLCPP_INFO(LOGGER, "Original Solution Message for Task %d:", i);
        mtc_debug::printFirstSolutionMessage(sol_msg, model, /*max_points_per_segment=*/6);
    }

    bool ok = mtc_debug::saveTasksFirstSolutionsYAML(tasks_, "/tmp/mtc_export.yaml");
    if (!ok) {
        RCLCPP_ERROR(LOGGER, "Failed to write YAML");
    } else {
        RCLCPP_INFO(LOGGER, "Wrote /tmp/mtc_export.yaml");
    }

    waitForAnyKey();
    //requestSceneClearObjects();
    //requestSceneUpdate();
#ifdef ROBOT_TYPE_FRANKA_PANDA
    int ret = executeTaskSequence(tasks_, task2action, fakemove_cache, true);

    if(ret == 0) {
        std::cout << "\nTask and Motion Plan Execution Successfully!" << std::endl;
    } else {
        std::cout << "\nTask and Motion Plan Execution Failed!" << std::endl;
    }

    return ret;
#else
    return 0;
#endif
}

void MTCTaskNode::printTimings()
{
    if(tamp_time_ > 0)
        std::cout << "task and motion planning time:" << tamp_time_/1000 << "s,";
    if(gap_planning_time_ > 0)
        std::cout << "gap task planning time:" << gap_planning_time_/1000 << "s,";
    if(execution_time_ > 0)
        std::cout << "execution time:" << execution_time_/1000 << "s";

    std::cout << std::endl;
}

mtc::Task MTCTaskNode::createSmashTask([[maybe_unused]] std::string& obj_id,
                                       Pose tgt_pose,
                                       std::string& direction,
                                       planning_scene::PlanningSceneConstPtr start_scene,
                                       size_t n_blocks /*number of blocks to be toppled*/)
{
    mtc::Task task;
    task.stages()->setName("demo task");
    //task.loadRobotModel(node_);
    task.setRobotModel(model_); /* make sure all tasks use the same robot model */

    const auto& arm_group_name = ARM_GROUP_NAME;
    const auto& hand_group_name = HAND_GROUP_NAME;
    const auto& hand_frame = HAND_FRAME;

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    mtc::Stage* current_state_ptr = nullptr;  // Forward current_state on to grasp pose generator
    if(start_scene == nullptr) {
        auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
        current_state_ptr = stage_state_current.get();
        task.add(std::move(stage_state_current));
    } else {
        auto stage_state_current = std::make_unique<mtc::stages::FixedState>("current", start_scene->diff());
        current_state_ptr = stage_state_current.get();
        task.add(std::move(stage_state_current));
    }

    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    sampling_planner->setPlannerId("RRTConnectkConfigDefault");
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);
       
    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setGoal(HOME_NAME);
      //task.add(std::move(stage));
    }

    auto stage_close_hand =
        std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
    stage_close_hand->setGroup(hand_group_name);
    stage_close_hand->setGoal("close");
    //task.add(std::move(stage_close_hand));

    auto stage_move_to_smash = std::make_unique<mtc::stages::Connect>(
               "move to smash",
                mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner } });
    stage_move_to_smash->setTimeout(5.0);
    stage_move_to_smash->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_smash));

    auto smash = std::make_unique<mtc::SerialContainer>("smash object");
    task.properties().exposeTo(smash->properties(), { "eef", "group", "ik_frame" });
    smash->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });
#if 0
    {
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision");
      stage->allowCollisions(all_blocks_, all_links, false);
      //smash->insert(std::move(stage));
    }
#endif
    {
      float pre_smash_relh = n_blocks * 0.05;
      auto stage =
              std::make_unique<mtc::stages::MoveRelative>("pre approach object", cartesian_planner);
      stage->properties().set("marker_ns", "pre approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setMinMaxDistance(pre_smash_relh-0.01, pre_smash_relh);

        // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      smash->insert(std::move(stage));

    }

    {
      auto stage =
              std::make_unique<mtc::stages::MoveRelative>("approach object", cartesian_planner);
      stage->properties().set("marker_ns", "approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.13, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.09, 0.20);
#endif

        // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.x = 1.0;
      vec.vector.z = -0.1;
      stage->setDirection(vec);
      smash->insert(std::move(stage));

    }

    {
      auto stage = std::make_unique<mtc::stages::GeneratePose>("generate smash pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "smash_pose");

      geometry_msgs::msg::PoseStamped target_pose_msg;
      target_pose_msg.header.frame_id = "world";
      tgt_pose.position.x += 0.04;
      tgt_pose.position.y += 0.035;
#if defined(ROBOT_TYPE_UFACTORY_XARM7)
      tgt_pose.position.z -= 0.025;
#endif
      target_pose_msg.pose = tgt_pose;
      target_pose_msg.pose.orientation.x = 0.0;
      target_pose_msg.pose.orientation.y = 0.0;
      target_pose_msg.pose.orientation.z = 0.0;
      target_pose_msg.pose.orientation.w = 1.0;
      stage->setPose(target_pose_msg);
      stage->setMonitoredStage(current_state_ptr);

      // This is the transform from the object frame to the end-effector frame
      Eigen::Isometry3d smash_frame_transform = Eigen::Isometry3d::Identity();
      smash_frame_transform.rotate(
                      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
      //smash_frame_transform.translation().x() = -0.10;
#ifdef ROBOT_TYPE_FRANKA_PANDA
      smash_frame_transform.translation().z() = 0.12;           //     hover 5 cm above
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      smash_frame_transform.translation().z() = 0.20;           //     hover 20 cm above
#endif

      auto wrapper =
              std::make_unique<mtc::stages::ComputeIK>("smash pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(8);
      wrapper->setMinSolutionDistance(1.0);
      wrapper->setIKFrame(smash_frame_transform, hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      smash->insert(std::move(wrapper));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("retreat", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setMinMaxDistance(0.01, 0.06);
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "retreat");

      // Set retreat direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      vec.vector.z = 1;
      stage->setDirection(vec);
      smash->insert(std::move(stage));
    }
    task.add(std::move(smash));

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setGoal(HOME_NAME);
      //task.add(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      //task.add(std::move(stage));
    }

    return task;
}

mtc::Task MTCTaskNode::createScoopTask(std::string& obj_id,
                                       Pose src_pose,
                                       Pose tgt_pose,
                                       planning_scene::PlanningSceneConstPtr start_scene)
{
    mtc::Task task = mtc::Task();
    task.reset();
    task.stages()->setName("demo task");
    //task.loadRobotModel(node_);
    task.setRobotModel(model_); /* make sure all tasks use the same robot model */

    const auto& arm_group_name = ARM_GROUP_NAME;
    const auto& hand_group_name = HAND_GROUP_NAME;
    const auto& hand_frame = HAND_FRAME;

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    mtc::Stage* current_state_ptr = nullptr;  // Forward current_state on to grasp pose generator

    if(start_scene == nullptr) {
        auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
        current_state_ptr = stage_state_current.get();
        task.add(std::move(stage_state_current));
    } else {
        auto stage_state_current = std::make_unique<mtc::stages::FixedState>("current", start_scene->diff());
        current_state_ptr = stage_state_current.get();
        task.add(std::move(stage_state_current));
    }

    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    sampling_planner->setPlannerId("RRTConnectkConfigDefault");
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);

    std::vector<std::string> arm_links = task.getRobotModel()
                                ->getJointModelGroup(arm_group_name)
                                ->getLinkModelNamesWithCollisionGeometry();
    std::vector<std::string> hand_links = task.getRobotModel()
                                ->getJointModelGroup(hand_group_name)
                                ->getLinkModelNamesWithCollisionGeometry();

    std::vector<std::string> all_links;
    all_links.insert(all_links.end(), arm_links.begin(), arm_links.end());
    all_links.insert(all_links.end(), hand_links.begin(), hand_links.end());

    for(const auto&link : all_links) {
        std::cout << "link: " << link << std::endl;
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setGoal(HOME_NAME);
      //task.add(std::move(stage));
    }

    auto stage_open_hand =
        std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
    stage_open_hand->setGroup(hand_group_name);
    stage_open_hand->setGoal("open");
    task.add(std::move(stage_open_hand));

    auto stage_move_to_pick = std::make_unique<mtc::stages::Connect>(
        "move to pick",
        mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner } });
    stage_move_to_pick->setTimeout(5.0);
    stage_move_to_pick->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_pick));

    mtc::Stage* attach_object_stage =
        nullptr;  // Forward attach_object_stage to place pose generator

    auto grasp = std::make_unique<mtc::SerialContainer>("pick object");
    task.properties().exposeTo(grasp->properties(), { "eef", "group", "ik_frame" });
    grasp->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("approach object", cartesian_planner);
      stage->properties().set("marker_ns", "approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif

      // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }

    {
      auto stage =
        std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      task.add(std::move(stage));
    }

    /****************************************************
     *               Generate Pick Pose                *
     ***************************************************/
    {
      /* here we don't choose the mtc::stages::GenerateGraspPose, as this stage accept
       * the object id as the input, and then mtc will acquire the pose based on the id
       * for the planning. However, as the pose information is synced up with Gazebo,
       * the orientation of the object pose (after smashed) is no longer (0,0,0,1),
       * mtc seems has issue on planning with some of the orientation. Therefore, we use
       * the mtc::stages::GeneratePose to precisely control the pick pose by the setPose
       * method (which you see in the following code).
       */
      auto stage = std::make_unique<mtc::stages::GeneratePose>("generate pick pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "pick_pose");

      geometry_msgs::msg::PoseStamped source_pose_msg;
      source_pose_msg.header.frame_id = "world";
      source_pose_msg.pose = src_pose;
      source_pose_msg.pose.orientation.x = 0.0;
      source_pose_msg.pose.orientation.y = 0.0;
      source_pose_msg.pose.orientation.z = 0.0;
      source_pose_msg.pose.orientation.w = 1.0;
      stage->setPose(source_pose_msg);
      stage->setMonitoredStage(current_state_ptr);

      /* This is the transform from the object frame to the end-effector frame.
       * Here, by default the grasp pose orientation is the same as panda frame.
       * However, this pose orientation doen't always result in success grasping, given
       * the randomness positions after smashing. Therefore, we introduce the function
       * 'sampleAngle', which is to generate the orientation (along hand z-axis)
       * randomly (the first few are fixed angles, which we want try first).
       */
      Eigen::Isometry3d grasp_frame_transform = Eigen::Isometry3d::Identity();
      grasp_frame_transform.rotate(
#if 1
                  Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
#else
                  Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) *
                  Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
#endif
#ifdef ROBOT_TYPE_FRANKA_PANDA
      grasp_frame_transform.translation().z() = 0.10;           //     hover 10 cm above
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      grasp_frame_transform.translation().z() = 0.16;           //     hover 10 cm above
#endif

      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(8);
      wrapper->setMinSolutionDistance(1.0);
      wrapper->setIKFrame(grasp_frame_transform, hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      grasp->insert(std::move(wrapper));
    }

    {
      /* normally, we should enable the collision between the hand and the object we want
       * to grasp only. But, here we enable the collision between the hand and all the 
       * blocks, As we might not be able to expect what the scene looks like after smashing,
       * it might be beneficial sometimes to allow the contacts between hand and other blocks.
       */
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (hand,object)");
      stage->allowCollisions(obj_id, hand_links, true);
      //stage->allowCollisions(all_blocks_, all_blocks_, true);
      //std::vector<std::string> bs = all_blocks_;
      //bs.erase(std::remove(bs.begin(), bs.end(), obj_id), bs.end());
      //stage->allowCollisions(bs, bs, true);
      grasp->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("close");
      grasp->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach object");
      stage->attachObject(obj_id, hand_frame);
      attach_object_stage = stage.get();
      grasp->insert(std::move(stage));
    }

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("lift object", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "lift_object");

      // Set upward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }
    task.add(std::move(grasp));

    auto stage_move_to_place = std::make_unique<mtc::stages::Connect>(
        "move to place",
        mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner },
                                                  { hand_group_name, sampling_planner } });
    // clang-format on
    stage_move_to_place->setTimeout(5.0);
    stage_move_to_place->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_place));

    auto place = std::make_unique<mtc::SerialContainer>("place object");
    task.properties().exposeTo(place->properties(), { "eef", "group", "ik_frame" });
    place->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("pre place pose", cartesian_planner);
      stage->properties().set("marker_ns", "approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif

      // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }

    /****************************************************
     *               Generate Place Pose                *
     ***************************************************/
    {
      // Sample place pose
      auto stage = std::make_unique<mtc::stages::GeneratePlacePose>("generate place pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "place_pose");
      stage->setObject(obj_id);

      geometry_msgs::msg::PoseStamped target_pose_msg;
      target_pose_msg.header.frame_id = "world";
      target_pose_msg.pose = tgt_pose;
      target_pose_msg.pose.orientation.x = 0.0;
      target_pose_msg.pose.orientation.y = 0.0;
      target_pose_msg.pose.orientation.z = 0.0;
      target_pose_msg.pose.orientation.w = 1.0;
      stage->setPose(target_pose_msg);
      stage->setMonitoredStage(attach_object_stage);  // Hook into attach_object_stage

      Eigen::Isometry3d place_frame_transform = Eigen::Isometry3d::Identity();
      place_frame_transform.rotate(
                      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) *
                      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
#ifdef ROBOT_TYPE_FRANKA_PANDA
      place_frame_transform.translation().z() = 0.10;           //     hover 10 cm above
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      place_frame_transform.translation().z() = 0.16;           //     hover 10 cm above
#endif
      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("place pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(2);
      wrapper->setMinSolutionDistance(1.0);
      wrapper->setIKFrame(place_frame_transform, hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      place->insert(std::move(wrapper));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      place->insert(std::move(stage));
    }

    {
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (hand,object)");
      stage->allowCollisions(obj_id, hand_links, false);
      //stage->allowCollisions(all_blocks_, all_blocks_, false);
      //std::vector<std::string> bs = all_blocks_;
      //bs.erase(std::remove(bs.begin(), bs.end(), obj_id), bs.end());
      //stage->allowCollisions(bs, bs, false);
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("detach object");
      stage->detachObject(obj_id, hand_frame);
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("retreat", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "retreat");

      // Set retreat direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      //vec.vector.x = -0.5;
      vec.vector.z = 1;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }
    task.add(std::move(place));

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setGoal(HOME_NAME);
      //task.add(std::move(stage));
    }

    return task;
}


double MTCTaskNode::sampleAngle() {
    static std::vector<double> fixed_samples = {
        0, /*default */
        M_PI / 2,
        M_PI / 12,
        -M_PI /12,
        M_PI / 10,
        -M_PI / 10,
        M_PI / 8,
        -M_PI / 8,
        M_PI / 6,
        -M_PI / 6,
        M_PI / 4,
        -M_PI / 4
        -M_PI / 2,
    };

    // Static RNG and distribution
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-M_PI / 2, M_PI / 2);

    /* we can put our preferred angle here in the vector
     * so that the planner can try them first, but if
     * the planner can easily sample feasible solution,
     * we comment out these lines as well.
    */

    if (sample_angle_idx_ < fixed_samples.size()) {
        return fixed_samples[sample_angle_idx_++];
    } else {
        return dist(rng);
    }

    return dist(rng);
}

bool MTCTaskNode::applyPlanningSceneDiff(moveit_msgs::msg::PlanningScene& diff_msg)
{
  using ApplyPS = moveit_msgs::srv::ApplyPlanningScene;

  // Ensure it's a diff (MTC typically sets this already)
  moveit_msgs::msg::PlanningScene req_scene = diff_msg;
  req_scene.is_diff = true;

  auto client = node_->create_client<ApplyPS>("/apply_planning_scene");
  if (!client->wait_for_service(std::chrono::milliseconds(1000))) {
    // Fallback: publish to /planning_scene
    ps_pub_->publish(req_scene);
    RCLCPP_WARN(LOGGER,
                "ApplyPlanningScene service not available; published diff to /planning_scene");
    return true;  // not strictly guaranteed, but good enough as fallback
  }

  auto request = std::make_shared<ApplyPS::Request>();
  request->scene = std::move(req_scene);

  auto future = client->async_send_request(request);
#if 0
  // Use a temporary executor that only spins the client_node
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node_);
  auto rc = exec.spin_until_future_complete(future, std::chrono::milliseconds(1000));
  if (rc != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(LOGGER,
                 "apply_planning_scene timed out/failure");
    return false;
  }
  const bool ok = future.get()->success;
  if (!ok) {
    RCLCPP_ERROR(LOGGER,
                 "apply_planning_scene returned success=false");
  }
  return ok;
#endif
    return true;
}

bool MTCTaskNode::executeSolutionSplitHand(moveit::task_constructor::Task& task)
{
  if (task.solutions().empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No MTC solutions available");
    return false;
  }

  // Flatten first solution to message
  moveit_task_constructor_msgs::msg::Solution sol_msg;
  task.solutions().front()->toMsg(sol_msg);
  auto model = task.getRobotModel();

#ifdef ROBOT_TYPE_FRANKA_PANDA
  mtc_debug::stripFingerFromArmHand(sol_msg);
#else
//#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
  mtc_debug::addFingerToArmUsingLastHand(sol_msg);
#endif
  //mtc_debug::timeParameterizeArmSegments(node_->get_logger(), task, sol_msg, /*vel=*/0.5, /*acc=*/0.5);
  constexpr double kEps = 0.002;  // 2 mm threshold to decide open/close

  for (size_t i = 0; i < sol_msg.sub_trajectory.size(); ++i) {
    auto& seg = sol_msg.sub_trajectory[i];
    const auto& jt  = seg.trajectory.joint_trajectory;

    // scene-only segment?
    const bool has_motion =
        !jt.points.empty() || !seg.trajectory.multi_dof_joint_trajectory.points.empty();
    if (!has_motion) {
      RCLCPP_INFO(node_->get_logger(), "[%02zu] scene-only segment (no trajectory) — skipped", i);
      continue;
    }

    // find controlling group
    const auto* jmg = mtc_debug::matchGroupByJoints(model, jt.joint_names);
    const std::string group = jmg ? jmg->getName() : std::string();

    // HAND segments: determine open/close from finger motion and call your hooks
    if (group == HAND_GROUP_NAME) {
      auto [ok, q_start, q_end] = mtc_debug::fingerStartEnd(seg.trajectory);
      if (!ok) {
        RCLCPP_WARN(node_->get_logger(), "[%02zu] hand segment but no finger data — skipped", i);
        continue;
      }
      if (q_end > q_start + kEps) {
        RCLCPP_INFO(node_->get_logger(), "[%02zu] HAND: OPEN (%.4f → %.4f) → sendIsaacGripperOpenRequest()", i, q_start, q_end);
        sendIsaacGripperOpenRequest();
      } else if (q_end < q_start - kEps) {
        RCLCPP_INFO(node_->get_logger(), "[%02zu] HAND: CLOSE (%.4f → %.4f) → sendIsaacGripperCloseRequest()", i, q_start, q_end);
        sendIsaacGripperCloseRequest();
      } else {
        RCLCPP_INFO(node_->get_logger(), "[%02zu] HAND: hold width (%.4f → %.4f) — no-op", i, q_start, q_end);
#if 1
        if(q_end >= 0.03)
            sendIsaacGripperOpenRequest();
        else
            sendIsaacGripperCloseRequest();
#endif
      }
      continue;  // do not execute hand trajectories via MGI
    }

    // Non-hand segments: execute exactly as planned with MGI
    if (group.empty()) {
      RCLCPP_WARN(node_->get_logger(), "[%02zu] cannot match a MoveIt group for this segment — skipped", i);
      continue;
    }

    //mtc_debug::densifyJointTrajectory(seg.trajectory, /*max_joint_step*/ 0.05, /*max_new_pts_cap*/5000);

    moveit::planning_interface::MoveGroupInterface mgi(node_, group);

    RCLCPP_WARN(node_->get_logger(), "============================================");
    RCLCPP_WARN(node_->get_logger(), "[%02zu] Trajectory points: %d before retiming", i, jt.points.size());
    //mtc_debug::printTrajectory(seg.trajectory, model, 20);

    // 1) re-time with conservative scales
#if 0
    if (!mtc_debug::snapFirstPointToCurrentAndRetime(mgi, model, jmg, seg.trajectory, 0.5, 1.0)) {
        RCLCPP_WARN(node_->get_logger(), "[%02zu] snap/re-time failed; sending as-is", i);
    }
#endif
    moveit::core::RobotStatePtr cur = mgi.getCurrentState(1.0);
#if 1
    if(cur) {
        RCLCPP_WARN(node_->get_logger(), "[%02zu] get current state, re-time... ", i);
        mtc_debug::retimeTrajectory(model, jmg, *cur, seg.trajectory, 1.0/*vel_scale*/, 1.0/*acc_scale*/);
    }
#endif

    // 2) give the controller dwell time at goal
    RCLCPP_WARN(node_->get_logger(), "============================================");
    RCLCPP_WARN(node_->get_logger(), "[%02zu] Trajectory points: %d after retiming and before padding", i, jt.points.size());
    //mtc_debug::printTrajectory(seg.trajectory, model, 20);
    mtc_debug::appendHoldPoint(seg.trajectory, /*hold_sec=*/0.3);

    //mtc_debug::sanitizeAndEnforceTiming(seg.trajectory.joint_trajectory, /*min_dt=*/1e-3);

    // 3) execute
    RCLCPP_WARN(node_->get_logger(), "============================================");
    RCLCPP_WARN(node_->get_logger(), "[%02zu] Trajectory points: %d after retiming and padding, ready to execute:", i, jt.points.size());
    mtc_debug::printTrajectory(seg.trajectory, model, 20);

    RCLCPP_INFO(node_->get_logger(), "[%02zu] executing group='%s' (points=%zu)", i, group.c_str(), jt.points.size());
    auto rc = mgi.execute(seg.trajectory);
    if (rc != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "[%02zu] MGI execute failed for group '%s' (code %d)", i, group.c_str(), rc.val);
      return false;
    }

    // 4) wait until the sim is physically settled
    rclcpp::sleep_for(std::chrono::milliseconds(DEBUG_SLEEP_MS));
    waitForAnyKey();
    const auto& last_pt = seg.trajectory.joint_trajectory.points.back();
    std::vector<double> goal = last_pt.positions; // same order as jt.joint_names
    if (!mtc_debug::waitUntilSettled(mgi, jt.joint_names, goal, 0.005, 0.05,
                            /*settle_count=*/5, std::chrono::milliseconds(50), std::chrono::milliseconds(1000))) {
            RCLCPP_WARN(node_->get_logger(), "[%02zu] settle wait timed out; continuing", i);
    }
  }

  RCLCPP_INFO(node_->get_logger(), "All segments processed (hand → open/close hooks, others via MGI)");
  return true;
}

#if 0
bool MTCTaskNode::executeFirstSolutionWithMGI(moveit::task_constructor::Task& task)
{
  if (task.solutions().empty()) {
    RCLCPP_ERROR(node->get_logger(), "No MTC solutions available");
    return false;
  }
  moveit_task_constructor_msgs::msg::Solution sol_msg;
  task.solutions().front()->toMsg(sol_msg);

  auto model = task.getRobotModel();
  for (const auto& seg : sol_msg.sub_trajectory) {
    const auto& jt = seg.trajectory.joint_trajectory;

    // choose a MoveGroup by matching joints
    const auto* jmg = mtc_debug::matchGroupByJoints(model, jt.joint_names);
    if (!jmg) {
      RCLCPP_WARN(node->get_logger(), "Skipping segment: cannot match a group for joints [%s]",
                  mtc_debug::join(jt.joint_names).c_str());
      continue;
    }

    moveit::planning_interface::MoveGroupInterface mgi(node, jmg->getName());
    auto rc = mgi.execute(seg.trajectory);
    if (rc != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node->get_logger(), "MGI execute failed for group '%s' (code %d)",
                   jmg->getName().c_str(), rc.val);
      return false;
    }

    // If seg.scene_diff contains an attach/detach needed before next motion,
    // apply it here via /apply_planning_scene or your existing PSI helper.
  }
  RCLCPP_INFO(node->get_logger(), "All sub-trajectories executed via MGI");
  return true;
}
#endif

// Helper: pull all positions of panda_finger_joint1 from a hand trajectory
static std::vector<double> extractFingerProfile(moveit_msgs::msg::RobotTrajectory& traj)
{
  std::vector<double> out;
  auto& jt = traj.joint_trajectory;
  if (jt.joint_names.empty() || jt.points.empty()) return out;

#ifdef ROBOT_TYPE_FRANKA_PANDA
  auto it = std::find(jt.joint_names.begin(), jt.joint_names.end(), "panda_finger_joint1");
  if (it == jt.joint_names.end()) {
    // (optional) fall back to joint2 if your planner happened to emit that one
    it = std::find(jt.joint_names.begin(), jt.joint_names.end(), "panda_finger_joint2");
    if (it == jt.joint_names.end()) return out;
  }
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
  auto it = std::find(jt.joint_names.begin(), jt.joint_names.end(), "drive_joint");
  if (it == jt.joint_names.end()) return out;
#endif
  size_t idx = static_cast<size_t>(std::distance(jt.joint_names.begin(), it));

  out.reserve(jt.points.size());
  for (auto& p : jt.points) {
    if (idx < p.positions.size()) {
      double v = std::clamp(p.positions[idx], 0.0, 0.04);  // safety clamp (meters per finger)
      out.push_back(v);
    }
  }
  return out;
}

bool MTCTaskNode::executeFirstSolutionWithMGI(moveit::task_constructor::Task& task)
{
  if (task.solutions().empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No MTC solutions available");
    return false;
  }

  moveit_task_constructor_msgs::msg::Solution sol_msg;
  task.solutions().front()->toMsg(sol_msg);
  mtc_debug::stripFingerFromArmHand(sol_msg);

  auto model = task.getRobotModel();

  // --- set your grasp constraint here ---
  double object_width   = 0.050;   // meters (your cubes)
  double clearance      = 0.001;   // 1 mm safety
  //double min_half_width = object_width * 0.5 - clearance;   // e.g., 0.026
  double min_half_width = 0.0;

  for (size_t i = 0; i < sol_msg.sub_trajectory.size(); ++i) {
    auto& seg = sol_msg.sub_trajectory[i];
    auto& jt  = seg.trajectory.joint_trajectory;

    bool has_motion =
        !jt.points.empty() || !seg.trajectory.multi_dof_joint_trajectory.points.empty();
    if (!has_motion) {
      RCLCPP_INFO(node_->get_logger(), "[%02zu] scene-only (no joint motion)", i);
      continue;
    }

    auto* jmg   = mtc_debug::matchGroupByJoints(model, jt.joint_names);
    std::string group = jmg ? jmg->getName() : std::string();

    // --- HAND: stream waypoints, floored at min_half_width ---
    bool is_hand = (group == HAND_GROUP_NAME) ||
#ifdef ROBOT_TYPE_FRANKA_PANDA
                         (std::find(jt.joint_names.begin(), jt.joint_names.end(),
                                    "panda_finger_joint1") != jt.joint_names.end());
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
                         (std::find(jt.joint_names.begin(), jt.joint_names.end(),
                                    "drive_joint") != jt.joint_names.end());
#endif

    if (is_hand) {
      auto profile = extractFingerProfile(seg.trajectory);
      if (profile.empty()) {
        RCLCPP_WARN(node_->get_logger(), "[%02zu] hand segment has no finger positions; skipping", i);
        continue;
      }
      RCLCPP_INFO(node_->get_logger(),
                  "[%02zu] hand segment: %zu waypoints (min_half_width=%.4f m)",
                  i, profile.size(), min_half_width);

      for (size_t k = 0; k < profile.size(); ++k) {
        double target = std::max(profile[k], min_half_width);  // <-- don’t command inside the object
        RCLCPP_INFO(node_->get_logger(), "    hand wp[%02zu]: cmd=%.4f m", k, target);

        GripperOutcome out = sendGripperGoal(target, true, true);
        while(out.success == false) {
            out = sendGripperGoal(target, true, true);
            rclcpp::sleep_for(std::chrono::milliseconds(500));
        }
      }
      continue;  // next segment
    }

    // --- ARM: execute via MGI as-is ---
    if (group.empty()) {
      RCLCPP_WARN(node_->get_logger(), "[%02zu] cannot match a group; skipping segment", i);
      continue;
    }
    RCLCPP_INFO(node_->get_logger(), "[%02zu] executing group='%s' (points=%zu)",
                i, group.c_str(), jt.points.size());

    moveit::planning_interface::MoveGroupInterface mgi(node_, group);
    auto rc = mgi.execute(seg.trajectory);
    while (rc != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "[%02zu] MGI execute failed for group '%s' (code %d), retry...",
                   i, group.c_str(), rc.val);
      rc = mgi.execute(seg.trajectory);
      rclcpp::sleep_for(std::chrono::milliseconds(200));
    }
  }

  RCLCPP_INFO(node_->get_logger(), "All segments processed (hand floored at grasp width).");
  return true;
}

#if 0
bool MTCTaskNode::executeFirstSolutionWithMGI(moveit::task_constructor::Task& task)
{
  if (task.solutions().empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No MTC solutions available");
    return false;
  }

  // Flatten first solution
  moveit_task_constructor_msgs::msg::Solution sol_msg;
  task.solutions().front()->toMsg(sol_msg);

  auto model = task.getRobotModel();

  // Tiny helpers
  auto to_sec = [](const builtin_interfaces::msg::Duration& d) {
    return static_cast<double>(d.sec) + 1e-9 * static_cast<double>(d.nanosec);
  };
  auto format_point = [&](const trajectory_msgs::msg::JointTrajectoryPoint& pt,
                          const std::vector<std::string>& names) {
    std::ostringstream oss;
    oss << "t=" << std::fixed << std::setprecision(3) << to_sec(pt.time_from_start) << "s  q=[";
    const size_t N = std::min(names.size(), pt.positions.size());
    for (size_t j = 0; j < N; ++j) {
      if (j) oss << ", ";
      oss << names[j] << ":" << std::setprecision(5) << pt.positions[j];
    }
    oss << "]";
    return oss.str();
  };

  for (size_t i = 0; i < sol_msg.sub_trajectory.size(); ++i) {
    const auto& seg = sol_msg.sub_trajectory[i];
    const auto& jt  = seg.trajectory.joint_trajectory;

    // pick a MoveGroup by joint names
    const auto* jmg = mtc_debug::matchGroupByJoints(model, jt.joint_names);
    if (!jmg) {
      // scene-only segments often have empty joints; just skip
      if (jt.joint_names.empty() || jt.points.empty()) {
        RCLCPP_INFO(node_->get_logger(), "[%02zu] (scene-only) no joint motion", i);
        continue;
      }
      RCLCPP_WARN(node_->get_logger(), "[%02zu] Skipping: cannot match a group for joints", i);
      continue;
    }

    // If there is motion, print a concise summary + endpoints
    if (!jt.points.empty()) {
      RCLCPP_INFO(node_->get_logger(),
                  "[%02zu] Executing group='%s'  joints=[%s]  points=%zu  duration=%.3fs",
                  i, jmg->getName().c_str(),
                  [&]{
                    std::ostringstream os;
                    for (size_t k = 0; k < jt.joint_names.size(); ++k) {
                      if (k) os << ", ";
                      os << jt.joint_names[k];
                    }
                    return os.str();
                  }().c_str(),
                  jt.points.size(),
                  to_sec(jt.points.back().time_from_start));

      // First waypoint
      RCLCPP_INFO(node_->get_logger(), "    start: %s",
                  format_point(jt.points.front(), jt.joint_names).c_str());
      // Last waypoint
      RCLCPP_INFO(node_->get_logger(), "    end  : %s",
                  format_point(jt.points.back(), jt.joint_names).c_str());
    } else {
      RCLCPP_INFO(node_->get_logger(), "[%02zu] (no joint points) group='%s'", i, jmg->getName().c_str());
    }

    // Execute exactly the trajectory MTC produced
    moveit::planning_interface::MoveGroupInterface mgi(node_, jmg->getName());
    auto rc = mgi.execute(seg.trajectory);
    while (rc != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "[%02zu] MGI execute failed for group '%s' (code %d), retry...",
                   i, jmg->getName().c_str(), rc.val);
      rc = mgi.execute(seg.trajectory);
      rclcpp::sleep_for(std::chrono::milliseconds(200));
    }
  }

  RCLCPP_INFO(node_->get_logger(), "All sub-trajectories executed via MGI");
  return true;
}
#endif

GripperOutcome MTCTaskNode::sendGripperGoal(double position_m, double max_effort, bool accept_stall_as_success)
{
    constexpr auto kConnectTimeout = std::chrono::seconds(2);
    constexpr auto kResultTimeout  = std::chrono::seconds(10);

    GripperOutcome out;

    if (!gripper_client_) {
        out.status_text = "gripper_client_ is null";
        return out;
    }
    if (!gripper_client_->wait_for_action_server(kConnectTimeout)) {
        out.status_text = "Action server not available";
        return out;
    }

    GripperCommand::Goal goal;
    goal.command.position   = position_m;  // meters (per finger)
    goal.command.max_effort = max_effort;  // -1 => controller decides

    // Use a tiny, private executor that only spins the client node
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(gripper_client_node_);

    // Send goal
    auto gh_future = gripper_client_->async_send_goal(goal);
    if (exec.spin_until_future_complete(gh_future, kResultTimeout) != rclcpp::FutureReturnCode::SUCCESS) {
        out.status_text = "send_goal timed out";
        return out;
    }
    auto goal_handle = gh_future.get();
    if (!goal_handle) {
        out.status_text = "Goal rejected by server";
        return out;
    }

    // Wait for result
    auto res_future = gripper_client_->async_get_result(goal_handle);
    if (exec.spin_until_future_complete(res_future, kResultTimeout) != rclcpp::FutureReturnCode::SUCCESS) {
        out.status_text = "result timed out";
        return out;
    }

    const auto wrapped = res_future.get();  // ClientGoalHandle<...>::WrappedResult
    if (wrapped.result) {
        out.stalled        = wrapped.result->stalled;
        out.reached_goal   = wrapped.result->reached_goal;
        out.final_position = wrapped.result->position;
    }

    using RC = rclcpp_action::ResultCode;
    switch (wrapped.code) {
        case RC::SUCCEEDED:
            out.success = true;
            out.status_text = "SUCCEEDED";
            break;
        case RC::ABORTED:
            if (accept_stall_as_success && out.stalled) {
                out.success = true;
                out.status_text = "ABORTED (stalled) — accepted as success";
            } else {
                out.status_text = "ABORTED";
            }
            break;
        case RC::CANCELED:
            out.status_text = "CANCELED";
            break;
        default:
            out.status_text = "UNKNOWN";
            break;
    }

    RCLCPP_INFO(node_->get_logger(),
              "Gripper outcome: success=%s stalled=%s reached=%s pos=%.4f status=%s",
              out.success ? "true":"false",
              out.stalled ? "true":"false",
              out.reached_goal ? "true":"false",
              out.final_position,
              out.status_text.c_str());

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    return out;
}

int MTCTaskNode::executeReturnHomeTask(planning_scene::PlanningSceneConstPtr start_scene)
{
    mtc::Task task = mtc::Task();
    task.reset();
    task.stages()->setName("premove task");
    //task.loadRobotModel(node_);
    task.setRobotModel(model_); /* make sure all tasks use the same robot model */

    const auto& arm_group_name = ARM_GROUP_NAME;
    const auto& hand_group_name = HAND_GROUP_NAME;
    const auto& hand_frame = HAND_FRAME;

    static bool open_flag = true;
    static bool close_flag = true;

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    if(start_scene == nullptr) {
        auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
        task.add(std::move(stage_state_current));
    } else {
        auto stage_state_current = std::make_unique<mtc::stages::FixedState>("current", start_scene->diff());
        task.add(std::move(stage_state_current));
    }

    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    sampling_planner->setPlannerId("RRTConnectkConfigDefault");
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setGoal(HOME_NAME);
      task.add(std::move(stage));
    }

#ifdef ROBOT_TYPE_UFACTORY_XARM7
    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      task.add(std::move(stage));
    }
#endif

    try {
        task.init();
    } catch (mtc::InitStageException& e) {
        RCLCPP_ERROR_STREAM(LOGGER, e);
        std::exit(PHYS_SIM_ERROR);
    }

    if (!task.plan(10 /* max_solutions */)) {
        RCLCPP_ERROR_STREAM(LOGGER, "Return home task planning failed");
        std::exit(PHYS_SIM_ERROR);
    }
    task.introspection().publishSolution(*task.solutions().front());

    RCLCPP_INFO(LOGGER, "%ld solutions generated, use the first one.", task.solutions().size());
    auto result = task.execute(*task.solutions().front());

    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
        std::exit(PHYS_SIM_ERROR);
    }

    return 0;
}

mtc::Task MTCTaskNode::createPreMoveTask(bool open,
                planning_scene::PlanningSceneConstPtr start_scene)
{
    mtc::Task task = mtc::Task();
    task.reset();
    task.stages()->setName("premove task");
    //task.loadRobotModel(node_);
    task.setRobotModel(model_); /* make sure all tasks use the same robot model */

    const auto& arm_group_name = ARM_GROUP_NAME;
    const auto& hand_group_name = HAND_GROUP_NAME;
    const auto& hand_frame = HAND_FRAME;

    static bool open_flag = true;
    static bool close_flag = true;

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    if(start_scene == nullptr) {
        auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
        task.add(std::move(stage_state_current));
    } else {
        auto stage_state_current = std::make_unique<mtc::stages::FixedState>("current", start_scene->diff());
        task.add(std::move(stage_state_current));
    }

    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    sampling_planner->setPlannerId("RRTConnectkConfigDefault");
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);

    if(open) {
#if 0
        if(open_flag) {
            auto stage_open_hand =
                std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
            stage_open_hand->setGroup(hand_group_name);
            stage_open_hand->setGoal("open");
            //task.add(std::move(stage_open_hand));

            auto stage = std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
            stage->setGroup(hand_group_name);
            stage->setGoal("close");
            //task.add(std::move(stage));

            {
              auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
              stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
              stage->setGoal(HOME_NAME);
              task.add(std::move(stage));
            }

            open_flag = false;
        }
#endif
        auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
        stage->setGroup(hand_group_name);
        stage->setGoal("open");
        task.add(std::move(stage));
    } else {
        if(close_flag) {
            auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
            stage->setGroup(hand_group_name);
            stage->setGoal("open");
            //task.add(std::move(stage));

            {
              auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
              stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
              stage->setGoal(HOME_NAME);
              //task.add(std::move(stage));
            }

            close_flag = false;
        }
        auto stage = std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
        stage->setGroup(hand_group_name);
        stage->setGoal("close");
        //task.add(std::move(stage));
        {
          auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
          stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
          stage->setGoal(HOME_NAME);
          task.add(std::move(stage));
        }
    }

    return task;
}

mtc::Task MTCTaskNode::createMoveTask(std::string obj_id, Pose src_pose, Pose tgt_pose,
        planning_scene::PlanningSceneConstPtr start_scene)
{
    mtc::Task task = mtc::Task();
    task.reset();
    task.stages()->setName("demo task");
    //task.loadRobotModel(node_);
    task.setRobotModel(model_); /* make sure all tasks use the same robot model */

    const auto& arm_group_name = ARM_GROUP_NAME;
    const auto& hand_group_name = HAND_GROUP_NAME;
    const auto& hand_frame = HAND_FRAME;

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    mtc::Stage* current_state_ptr = nullptr;  // Forward current_state on to grasp pose generator

    if(start_scene == nullptr) {
        auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
        current_state_ptr = stage_state_current.get();
        task.add(std::move(stage_state_current));
    } else {
        auto stage_state_current = std::make_unique<mtc::stages::FixedState>("current", start_scene->diff());
        current_state_ptr = stage_state_current.get();
        task.add(std::move(stage_state_current));
    }

    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    sampling_planner->setPlannerId("RRTConnectkConfigDefault");
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);

    std::vector<std::string> arm_links = task.getRobotModel()
                                ->getJointModelGroup(arm_group_name)
                                ->getLinkModelNamesWithCollisionGeometry();
    std::vector<std::string> hand_links = task.getRobotModel()
                                ->getJointModelGroup(hand_group_name)
                                ->getLinkModelNamesWithCollisionGeometry();

    std::vector<std::string> all_links;
    all_links.insert(all_links.end(), arm_links.begin(), arm_links.end());
    all_links.insert(all_links.end(), hand_links.begin(), hand_links.end());

    for(const auto&link : all_links) {
        std::cout << "link: " << link << std::endl;
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setGoal(HOME_NAME);
      //task.add(std::move(stage));
    }

    auto stage_open_hand =
        std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
    stage_open_hand->setGroup(hand_group_name);
    stage_open_hand->setGoal("open");
    task.add(std::move(stage_open_hand));

    auto stage_move_to_pick = std::make_unique<mtc::stages::Connect>(
        "move to pick",
        mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner } });
    stage_move_to_pick->setTimeout(5.0);
    stage_move_to_pick->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_pick));

    mtc::Stage* attach_object_stage =
        nullptr;  // Forward attach_object_stage to place pose generator

    auto grasp = std::make_unique<mtc::SerialContainer>("pick object");
    task.properties().exposeTo(grasp->properties(), { "eef", "group", "ik_frame" });
    grasp->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("approach object", cartesian_planner);
      stage->properties().set("marker_ns", "approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif

      // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }

    {
      auto stage =
        std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      task.add(std::move(stage));
    }

    /****************************************************
     *               Generate Pick Pose                *
     ***************************************************/
    {
      /* here we don't choose the mtc::stages::GenerateGraspPose, as this stage accept
       * the object id as the input, and then mtc will acquire the pose based on the id
       * for the planning. However, as the pose information is synced up with Gazebo,
       * the orientation of the object pose (after smashed) is no longer (0,0,0,1),
       * mtc seems has issue on planning with some of the orientation. Therefore, we use
       * the mtc::stages::GeneratePose to precisely control the pick pose by the setPose
       * method (which you see in the following code).
       */
      auto stage = std::make_unique<mtc::stages::GeneratePose>("generate pick pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "pick_pose");

      geometry_msgs::msg::PoseStamped source_pose_msg;
      source_pose_msg.header.frame_id = "world";
      source_pose_msg.pose = src_pose;
      source_pose_msg.pose.orientation.x = 0.0;
      source_pose_msg.pose.orientation.y = 0.0;
      source_pose_msg.pose.orientation.z = 0.0;
      source_pose_msg.pose.orientation.w = 1.0;
      stage->setPose(source_pose_msg);
      stage->setMonitoredStage(current_state_ptr);

      /* This is the transform from the object frame to the end-effector frame.
       * Here, by default the grasp pose orientation is the same as panda frame.
       * However, this pose orientation doen't always result in success grasping, given
       * the randomness positions after smashing. Therefore, we introduce the function
       * 'sampleAngle', which is to generate the orientation (along hand z-axis)
       * randomly (the first few are fixed angles, which we want try first).
       */
      Eigen::Isometry3d grasp_frame_transform = Eigen::Isometry3d::Identity();
      double sample = sampleAngle();
      std::cout << "Sampled Z-axis angle is:" << sample << std::endl;
      grasp_frame_transform.rotate(
                  Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) *
                  Eigen::AngleAxisd(sample, Eigen::Vector3d::UnitZ()));
#ifdef ROBOT_TYPE_FRANKA_PANDA
      grasp_frame_transform.translation().z() = 0.10;           //     hover 10 cm above
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      grasp_frame_transform.translation().z() = 0.16;           //     hover 10 cm above
#endif

      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(8);
      wrapper->setMinSolutionDistance(1.0);
      wrapper->setIKFrame(grasp_frame_transform, hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      grasp->insert(std::move(wrapper));
    }

    {
      /* normally, we should enable the collision between the hand and the object we want
       * to grasp only. But, here we enable the collision between the hand and all the 
       * blocks, As we might not be able to expect what the scene looks like after smashing,
       * it might be beneficial sometimes to allow the contacts between hand and other blocks.
       */
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (hand,object)");
      stage->allowCollisions(obj_id, hand_links, true);
      //stage->allowCollisions(all_blocks_, all_blocks_, true);
      //std::vector<std::string> bs = all_blocks_;
      //bs.erase(std::remove(bs.begin(), bs.end(), obj_id), bs.end());
      //stage->allowCollisions(bs, bs, true);
      grasp->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("close");
      grasp->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach object");
      stage->attachObject(obj_id, hand_frame);
      attach_object_stage = stage.get();
      grasp->insert(std::move(stage));
    }

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("lift object", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "lift_object");

      // Set upward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }
    task.add(std::move(grasp));

    auto stage_move_to_place = std::make_unique<mtc::stages::Connect>(
        "move to place",
        mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner },
                                                  { hand_group_name, sampling_planner } });
    // clang-format on
    stage_move_to_place->setTimeout(5.0);
    stage_move_to_place->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_place));

    auto place = std::make_unique<mtc::SerialContainer>("place object");
    task.properties().exposeTo(place->properties(), { "eef", "group", "ik_frame" });
    place->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("pre place pose", cartesian_planner);
      stage->properties().set("marker_ns", "approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif

      // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }

    /****************************************************
     *               Generate Place Pose                *
     ***************************************************/
    {
      // Sample place pose
      auto stage = std::make_unique<mtc::stages::GeneratePlacePose>("generate place pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "place_pose");
      stage->setObject(obj_id);

      geometry_msgs::msg::PoseStamped target_pose_msg;
      target_pose_msg.header.frame_id = "world";
      target_pose_msg.pose = tgt_pose;
      stage->setPose(target_pose_msg);
      stage->setMonitoredStage(attach_object_stage);  // Hook into attach_object_stage

      Eigen::Isometry3d place_frame_transform = Eigen::Isometry3d::Identity();
      place_frame_transform.rotate(
                      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
#ifdef ROBOT_TYPE_FRANKA_PANDA
      place_frame_transform.translation().z() = 0.10;           //     hover 10 cm above
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      place_frame_transform.translation().z() = 0.16;           //     hover 10 cm above
#endif
      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("place pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(2);
      wrapper->setMinSolutionDistance(1.0);
      wrapper->setIKFrame(place_frame_transform, hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      place->insert(std::move(wrapper));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      place->insert(std::move(stage));
    }

    {
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (hand,object)");
      stage->allowCollisions(obj_id, hand_links, false);
      //stage->allowCollisions(all_blocks_, all_blocks_, false);
      //std::vector<std::string> bs = all_blocks_;
      //bs.erase(std::remove(bs.begin(), bs.end(), obj_id), bs.end());
      //stage->allowCollisions(bs, bs, false);
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("detach object");
      stage->detachObject(obj_id, hand_frame);
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("retreat", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "retreat");

      // Set retreat direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      //vec.vector.x = -0.5;
      vec.vector.z = 1;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }
    task.add(std::move(place));

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setGoal(HOME_NAME);
      //task.add(std::move(stage));
    }

    return task;
}

mtc::Task MTCTaskNode::createFakeMoveTask(Pose tgt_pose,
                planning_scene::PlanningSceneConstPtr start_scene)
{
    mtc::Task task = mtc::Task();
    task.reset();
    task.stages()->setName("demo task");
    task.setRobotModel(model_); /* make sure all tasks use the same robot model */

    const auto& arm_group_name = ARM_GROUP_NAME;
    const auto& hand_group_name = HAND_GROUP_NAME;
    const auto& hand_frame = HAND_FRAME;

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    mtc::Stage* current_state_ptr = nullptr;  // Forward current_state on to grasp pose generator

    auto stage_state_current = std::make_unique<mtc::stages::FixedState>("current", start_scene->diff());
    current_state_ptr = stage_state_current.get();
    task.add(std::move(stage_state_current));

    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    sampling_planner->setPlannerId("RRTConnectkConfigDefault");
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);

    auto stage_open_hand =
        std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
    stage_open_hand->setGroup(hand_group_name);
    stage_open_hand->setGoal("open");
    task.add(std::move(stage_open_hand));

    auto stage_move_to_place = std::make_unique<mtc::stages::Connect>(
        "move to place",
        mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner },
                                                  { hand_group_name, sampling_planner } });
    // clang-format on
    stage_move_to_place->setTimeout(5.0);
    stage_move_to_place->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_place));

    auto place = std::make_unique<mtc::SerialContainer>("place object");
    task.properties().exposeTo(place->properties(), { "eef", "group", "ik_frame" });
    place->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

    /****************************************************
     *               Generate Place Pose                *
     ***************************************************/
    {
      auto stage = std::make_unique<mtc::stages::GeneratePose>("generate place pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "place_pose");

      geometry_msgs::msg::PoseStamped target_pose_msg;
      target_pose_msg.header.frame_id = "world";
      target_pose_msg.pose = tgt_pose;
      target_pose_msg.pose.orientation.x = 0.0;
      target_pose_msg.pose.orientation.y = 0.0;
      target_pose_msg.pose.orientation.z = 0.0;
      target_pose_msg.pose.orientation.w = 1.0;
      stage->setPose(target_pose_msg);
      stage->setMonitoredStage(current_state_ptr);

      Eigen::Isometry3d place_frame_transform = Eigen::Isometry3d::Identity();
      /* here to avoid any inconsistence, we use fixed orientation */
      place_frame_transform.rotate(
                      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
      place_frame_transform.translation().z() = 0.10;           //     hover 10 cm above

      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("place pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(2);
      wrapper->setMinSolutionDistance(1.0);
      wrapper->setIKFrame(place_frame_transform, hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      place->insert(std::move(wrapper));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("retreat", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif
      //stage->setMinMaxDistance(0.05, 0.051);
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "retreat");

      // Set retreat direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      //vec.vector.x = -0.5;
      vec.vector.z = 1;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }
    task.add(std::move(place));

    return task;
}

mtc::Task MTCTaskNode::createRealFromFakeMoveTask(std::string obj_id, Pose src_pose, Pose tgt_pose,
                        FakeMoveRecord& rec)
{
    mtc::Task task = mtc::Task();
    task.reset();
    task.stages()->setName("demo task");
    //task.loadRobotModel(node_);
    task.setRobotModel(model_); /* make sure all tasks use the same robot model */

    const auto& arm_group_name = ARM_GROUP_NAME;
    const auto& hand_group_name = HAND_GROUP_NAME;
    const auto& hand_frame = HAND_FRAME;

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    mtc::Stage* current_state_ptr = nullptr;  // Forward current_state on to grasp pose generator

    if(false) {
        auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
        current_state_ptr = stage_state_current.get();
        task.add(std::move(stage_state_current));
    } else {
        auto stage_state_current = std::make_unique<mtc::stages::FixedState>("current", rec.start_scene->diff());
        current_state_ptr = stage_state_current.get();
        task.add(std::move(stage_state_current));
    }

    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    sampling_planner->setPlannerId("RRTConnectkConfigDefault");
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);

    std::vector<std::string> arm_links = task.getRobotModel()
                                ->getJointModelGroup(arm_group_name)
                                ->getLinkModelNamesWithCollisionGeometry();
    std::vector<std::string> hand_links = task.getRobotModel()
                                ->getJointModelGroup(hand_group_name)
                                ->getLinkModelNamesWithCollisionGeometry();

    std::vector<std::string> all_links;
    all_links.insert(all_links.end(), arm_links.begin(), arm_links.end());
    all_links.insert(all_links.end(), hand_links.begin(), hand_links.end());

    for(const auto&link : all_links) {
        std::cout << "link: " << link << std::endl;
    }

    auto stage_open_hand =
        std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
    stage_open_hand->setGroup(hand_group_name);
    stage_open_hand->setGoal("open");
    task.add(std::move(stage_open_hand));

    auto stage_move_to_pick = std::make_unique<mtc::stages::Connect>(
        "move to pick",
        mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner } });
    stage_move_to_pick->setTimeout(5.0);
    stage_move_to_pick->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_pick));

    mtc::Stage* attach_object_stage =
        nullptr;  // Forward attach_object_stage to place pose generator

    auto grasp = std::make_unique<mtc::SerialContainer>("pick object");
    task.properties().exposeTo(grasp->properties(), { "eef", "group", "ik_frame" });
    grasp->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("approach object", cartesian_planner);
      stage->properties().set("marker_ns", "approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif

      // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }

    /****************************************************
     *               Generate Pick Pose                *
     ***************************************************/
    {
      /* here we don't choose the mtc::stages::GenerateGraspPose, as this stage accept
       * the object id as the input, and then mtc will acquire the pose based on the id
       * for the planning. However, as the pose information is synced up with Gazebo,
       * the orientation of the object pose (after smashed) is no longer (0,0,0,1),
       * mtc seems has issue on planning with some of the orientation. Therefore, we use
       * the mtc::stages::GeneratePose to precisely control the pick pose by the setPose
       * method (which you see in the following code).
       */
      auto stage = std::make_unique<mtc::stages::GeneratePose>("generate pick pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "pick_pose");

      geometry_msgs::msg::PoseStamped source_pose_msg;
      source_pose_msg.header.frame_id = "world";
      source_pose_msg.pose = src_pose;
      source_pose_msg.pose.orientation.x = 0.0;
      source_pose_msg.pose.orientation.y = 0.0;
      source_pose_msg.pose.orientation.z = 0.0;
      source_pose_msg.pose.orientation.w = 1.0;
      stage->setPose(source_pose_msg);
      stage->setMonitoredStage(current_state_ptr);

      /* This is the transform from the object frame to the end-effector frame.
       * Here, by default the grasp pose orientation is the same as panda frame.
       * However, this pose orientation doen't always result in success grasping, given
       * the randomness positions after smashing. Therefore, we introduce the function
       * 'sampleAngle', which is to generate the orientation (along hand z-axis)
       * randomly (the first few are fixed angles, which we want try first).
       */
      Eigen::Isometry3d grasp_frame_transform = Eigen::Isometry3d::Identity();
      double sample = sampleAngle();
      std::cout << "Sampled Z-axis angle is:" << sample << std::endl;
      grasp_frame_transform.rotate(
                  Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) *
                  Eigen::AngleAxisd(sample, Eigen::Vector3d::UnitZ()));
      grasp_frame_transform.translation().z() = 0.10;           //     hover 10 cm above

      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(8);
      wrapper->setMinSolutionDistance(1.0);
      wrapper->setIKFrame(grasp_frame_transform, hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      grasp->insert(std::move(wrapper));
    }

    {
      /* normally, we should enable the collision between the hand and the object we want
       * to grasp only. But, here we enable the collision between the hand and all the 
       * blocks, As we might not be able to expect what the scene looks like after smashing,
       * it might be beneficial sometimes to allow the contacts between hand and other blocks.
       */
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (hand,object)");
      stage->allowCollisions(obj_id, hand_links, true);
      //stage->allowCollisions(all_blocks_, all_blocks_, true);
      //std::vector<std::string> bs = all_blocks_;
      //bs.erase(std::remove(bs.begin(), bs.end(), obj_id), bs.end());
      //stage->allowCollisions(bs, bs, true);
      grasp->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("close");
      grasp->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach object");
      stage->attachObject(obj_id, hand_frame);
      attach_object_stage = stage.get();
      grasp->insert(std::move(stage));
    }

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("lift object", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "lift_object");

      // Set upward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }
    task.add(std::move(grasp));

    auto stage_move_to_place = std::make_unique<mtc::stages::Connect>(
        "move to place",
        mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner },
                                                  { hand_group_name, sampling_planner } });
    // clang-format on
    stage_move_to_place->setTimeout(5.0);
    stage_move_to_place->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_place));

    auto place = std::make_unique<mtc::SerialContainer>("place object");
    task.properties().exposeTo(place->properties(), { "eef", "group", "ik_frame" });
    place->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("pre place pose", cartesian_planner);
      stage->properties().set("marker_ns", "approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif

      // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.z = 1.0;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }

    /****************************************************
     *               Generate Place Pose                *
     ***************************************************/
    {
      // Sample place pose
      auto stage = std::make_unique<mtc::stages::GeneratePlacePose>("generate place pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "place_pose");
      stage->setObject(obj_id);

      geometry_msgs::msg::PoseStamped target_pose_msg;
      target_pose_msg.header.frame_id = "world";
      target_pose_msg.pose = tgt_pose;
      stage->setPose(target_pose_msg);
      stage->setMonitoredStage(attach_object_stage);  // Hook into attach_object_stage

      Eigen::Isometry3d place_frame_transform = Eigen::Isometry3d::Identity();
      place_frame_transform.rotate(
                      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
      place_frame_transform.translation().z() = 0.10;           //     hover 10 cm above
      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("place pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(2);
      wrapper->setMinSolutionDistance(1.0);
      wrapper->setIKFrame(place_frame_transform, hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      place->insert(std::move(wrapper));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      place->insert(std::move(stage));
    }

    {
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (hand,object)");
      stage->allowCollisions(obj_id, hand_links, false);
      //stage->allowCollisions(all_blocks_, all_blocks_, false);
      //std::vector<std::string> bs = all_blocks_;
      //bs.erase(std::remove(bs.begin(), bs.end(), obj_id), bs.end());
      //stage->allowCollisions(bs, bs, false);
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("detach object");
      stage->detachObject(obj_id, hand_frame);
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("retreat", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
#ifdef ROBOT_TYPE_FRANKA_PANDA
      stage->setMinMaxDistance(0.19, 0.20);
#elif defined(ROBOT_TYPE_UFACTORY_XARM7)
      stage->setMinMaxDistance(0.13, 0.20);
#endif
      //stage->setMinMaxDistance(0.05, 0.051);
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "retreat");

      // Set retreat direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      //vec.vector.x = -0.5;
      vec.vector.z = 1;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }
    {
      //auto stage = std::make_unique<mtc::stages::MoveTo>("sync to fake end", sampling_planner);
      //auto stage = std::make_unique<mtc::stages::MoveTo>("sync to fake end", cartesian_planner);
      auto stage = std::make_unique<mtc::stages::MoveTo>("sync to fake end", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
      stage->setGroup(arm_group_name);
      stage->setGoal(rec.end_joints);
      place->insert(std::move(stage));
    }
    task.add(std::move(place));

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setGoal(HOME_NAME);
      //task.add(std::move(stage));
    }

    return task;
}

int main(int argc, char** argv)
{
    int ret;
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto mtc_task_node = std::make_shared<MTCTaskNode>(options);
    rclcpp::executors::MultiThreadedExecutor executor;

    auto spin_thread = std::make_unique<std::thread>([&executor, &mtc_task_node]() {
            executor.add_node(mtc_task_node->getNodeBaseInterface());
            executor.spin();
            executor.remove_node(mtc_task_node->getNodeBaseInterface());
            });

    mtc_task_node->setupPlanningScene();
    ret = mtc_task_node->doMultiMoveTasks();

    mtc_task_node->printTimings();

    executor.cancel();
    spin_thread->join();
    rclcpp::shutdown();
    return ret;
}
