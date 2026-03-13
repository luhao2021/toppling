# toppling

Task-level tabletop rearrangement with interleaved **Move** (pick-place) and **Smash/Topple** actions.

## Files
- `topple_tp.py`: Gurobi task planner that generates action-sequence YAML.
- `smash_node.cpp`: Existing ROS2/MTC implementation.
- `smash_node_noetic.py`: New ROS1 Noetic + MoveIt1 Python execution pipeline for xArm7.
- `problems/problem_multi_1x9.yaml`: example generated problem.

## Planner usage (`topple_tp.py`)
Parameters:
1. `topple_flag` (`0/1`)
2. `number_of_objects`
3. `seed`
4. `gurobi_max_time`
5. `number_of_total_locations`
6. `single_goal` (`0` multi-goal, `1` single-goal)
7. `output_yaml` (optional, default `my_problem.yaml`)
8. `output_sdf` (optional, default `my_blocks_world.sdf`)

Example:
```bash
python3 topple_tp.py 1 9 42 10 8 0 problems/generated/problem_topple_9.yaml
```

## Noetic execution (`smash_node_noetic.py`)
The script reads planner YAML actions and executes each action sequentially:
- updates planning scene from AprilTag perception
- performs Move actions (pick/place style waypoint sequence)
- performs Smash actions (directional approach + push + retreat)
- refreshes scene after each action

### Lifelong demo mode
Generates paired problems (same seed): with topple and without topple, then executes both in a loop.

```bash
python3 smash_node_noetic.py \
  --lifelong \
  --topple-tp ./topple_tp.py \
  --num-objects 9 \
  --num-locations 8 \
  --single-goal 0
```

> `AprilTagProvider.lookup_block()` is intentionally a stub to connect your real tag pipeline.
