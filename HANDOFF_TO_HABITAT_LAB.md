# Handoff Document: SafeVLA → Habitat-Lab Implementation

## Context

We have analyzed the **SafeVLA** codebase, which is a Safe Reinforcement Learning framework for training Vision-Language-Action models with safety constraints. We want to **train similar agents using habitat-lab** instead of AI2THOR.

**Your Task**: Check if habitat-lab has equivalent functionality for the components listed below. If not, identify what needs to be implemented.

---

## 1. HIGH-LEVEL ARCHITECTURE

### What SafeVLA Does

SafeVLA trains robot agents (Stretch robot in AI2THOR) to perform:
- **Object Navigation**: Find and navigate to target objects
- **Fetch/Pickup**: Navigate to objects and pick them up
- **Room Exploration**: Visit new rooms and locations

**Key Innovation**: Uses **constrained RL** with separate reward and cost functions:
- **Reward**: Encourage task completion (find objects, pick them up, explore)
- **Cost**: Penalize safety violations (collisions, dangerous objects, disturbances)
- **Training**: Lagrangian PPO optimizes reward while keeping cost below threshold

### Framework Stack
```
SafeVLA = AllenAct (RL framework) + AI2THOR (simulator) + OmniSafe (safe RL)
    ↓
We Want = AllenAct (RL framework) + Habitat-Lab (simulator) + OmniSafe (safe RL)
```

---

## 2. REWARD FUNCTIONS - What We Need in Habitat-Lab

### 2.1 Reward Configuration

SafeVLA uses this reward config structure:

```python
@define
class RewardConfig:
    step_penalty: float = -0.00           # Penalty per timestep
    goal_success_reward: float = 10.0     # Bonus for task completion
    failed_stop_reward: float = 0.0       # Penalty for wrong termination
    shaping_weight: float = 0.0           # Weight for shaped rewards
    reached_horizon_reward: float = 0.0   # Reward at max steps
    positive_only_reward: bool = False    # Clip negative rewards?
    failed_action_penalty: float = -0.00  # Penalty for failed actions
```

**Question for Habitat-Lab**: Does habitat-lab have a similar reward configuration system?

---

### 2.2 Reward Shaper Classes

SafeVLA has **3 reward shapers** for different task types:

#### A. ObjectNavRewardShaper (Navigation Tasks)

**Purpose**: Guide agent to navigate to target objects

**Reward Components**:
1. **Distance-based reward**: `shaping_weight × max(prev_distance - curr_distance, 0)`
   - Positive reward for getting closer to target
   - Uses L2 distance from agent to closest target object
2. **Failed action penalty**: Negative reward for collisions/failed actions

**Key Functions Used**:
- `min_l2_distance_to_target()`: Get L2 distance to closest target object
- `min_geodesic_distance_to_target()`: Get shortest path distance

**Question for Habitat-Lab**:
- Does habitat-lab have distance-based reward shaping for ObjectNav?
- Can we compute L2 and geodesic distances to target objects?

---

#### B. FetchRewardShaper (Manipulation Tasks)

**Purpose**: Guide agent to navigate to object AND pick it up

**Reward Components**:
1. **Pickupable bonus**: `+5.0` when object enters hand/gripper sphere (one-time)
2. **Pickup success bonus**: `+5.0` when object successfully picked up (one-time)
3. **Arm distance reward**: `shaping_weight × 5 × max(prev_distance - curr_distance, 0)`
   - Uses distance from **arm/gripper to object** (not agent to object)
   - 5× multiplier on normal shaping weight

**Key Functions Used**:
- `is_object_pickupable()`: Check if target in gripper/hand sphere
- `min_l2_distance_to_target_from_arm()`: Distance from gripper center to object center
- `min_l2_distance_to_target_colliders_from_arm()`: Distance from gripper to **closest point on object surface**
- `get_objects_in_hand_sphere()`: Get pickupable object IDs near gripper
- `get_held_objects()`: Get currently held object IDs

**Question for Habitat-Lab**:
- Does habitat-lab support mobile manipulation with pick/place?
- Can we query objects near gripper?
- Can we compute gripper-to-object distances?
- Is there a "hand sphere" or similar concept?

---

#### C. RoomVisitRewardShaper (Exploration Tasks)

**Purpose**: Encourage exploration of new areas and rooms

**Reward Components**:
1. **Location discovery**: `+0.005` per new 0.1m grid cell visited
2. **Room discovery**: `+2.0` per new room entered
3. **Sub-task completion**: `+2.0` for correct sub-done, `-0.2` for incorrect

**Key Functions Used**:
- `get_reachable_positions()`: Get grid of all reachable positions
- `get_current_room()`: Get agent's current room ID
- Track visited locations and rooms over episode

**Question for Habitat-Lab**:
- Does habitat-lab have room annotations/polygons?
- Can we track which rooms agent has visited?
- Can we get reachable position grids?

---

### 2.3 Task-Specific Judge Functions

Each task type has a `judge()` function that computes total reward:

```python
def judge(self) -> float:
    reward = step_penalty                    # Base penalty: -0.00
    reward += self.shaping()                 # Add shaped reward

    if took_end_action:                      # Agent called "done"
        if success:
            reward += goal_success_reward    # +10.0
        else:
            reward += failed_stop_reward     # 0.0

    if reached_max_steps:
        reward += reached_horizon_reward     # 0.0

    return reward
```

**Question for Habitat-Lab**: How does habitat-lab structure reward computation? Can we override it per task?

---

## 3. ENVIRONMENT OBSERVATION FUNCTIONS - What We Need

SafeVLA has **60+ environment query functions**. Here are the critical ones:

### 3.1 Visual Observations (7 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `navigation_camera` | (H,W,3) RGB | RGB image from head camera |
| `manipulation_camera` | (H,W,3) RGB | RGB from wrist/gripper camera |
| `navigation_depth_frame` | (H,W) depth | Depth map from head |
| `manipulation_depth_frame` | (H,W) depth | Depth from wrist |
| `get_segmentation_mask_of_object(obj_id)` | (H,W) bool | Instance mask for object |
| `navigation_camera_segmentation` | Dict[id→mask] | All instance masks |
| `get_approx_object_mask(obj_id)` | List[points] | Approx bbox points |

**Question for Habitat-Lab**:
- ✅ habitat-lab has RGB-D sensors - Check
- Does habitat-lab support wrist/gripper cameras for manipulation?
- Does habitat-lab have instance segmentation (per-object masks)?

---

### 3.2 Object Queries (12 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_objects()` | List[Object] | All objects in scene |
| `get_object(obj_id)` | Object | Specific object metadata |
| `get_object_position(obj_id)` | [x,y,z] | Object 3D position |
| `get_visible_objects(camera, max_dist)` | List[obj_id] | Visible object IDs (cached) |
| `object_is_visible_in_camera(obj_id)` | bool | Is object visible? |
| `get_objects_of_synset_list(synsets)` | List[Object] | Objects matching types |
| `get_all_objects_of_synset(synset)` | List[Object] | All objects of type |
| `get_objects_that_objects_are_on(obj_ids)` | Dict | What's on what |
| `get_object_receptacle_synsets(obj_id)` | List[str] | Receptacles holding obj |
| `num_pixels_visible(obj_id)` | int | Pixels of object in view |
| `is_object_visible_enough_for_interaction(obj_id)` | bool | Centered & visible? |

**Question for Habitat-Lab**:
- Can habitat-lab query all objects in scene with metadata?
- Can we get object positions, types, visibility?
- Does habitat-lab support object-object relationships (on, in)?
- Can we compute pixel counts for specific objects?

---

### 3.3 Manipulation State (9 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_objects_in_hand_sphere()` | List[obj_id] | Pickupable objects near gripper |
| `get_held_objects()` | List[obj_id] | Currently held objects |
| `get_arm_sphere_center()` | [x,y,z] | Gripper sphere center |
| `get_wrist_center()` | [x,y,z] | Wrist joint position |
| `get_arm_wrist_position()` | [x,y,z] | Wrist relative position |
| `get_arm_wrist_rotation()` | float | Wrist rotation angle |
| `get_arm_proprioception()` | [x,y,z,θ] | Full arm state |
| `dist_from_arm_sphere_center_to_obj(obj_id)` | float | Gripper-to-object distance |
| `dist_from_arm_sphere_center_to_obj_colliders_closest_to_point(obj_id)` | float | Gripper to closest point on object |

**Question for Habitat-Lab**:
- Does habitat-lab support mobile manipulation (Spot, Stretch, Fetch)?
- Can we query arm/gripper state?
- Is there a "hand sphere" for grasp detection?
- Can we compute distances from gripper to objects?

---

### 3.4 Agent State (7 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_current_agent_position()` | [x,y,z] | Agent position |
| `get_current_agent_full_pose()` | Dict | Position + rotation + arm |
| `get_agent_alignment_to_object(obj_id)` | float | Rotation angle to object |
| `get_agent_alignment_to_wall(wall_id)` | float | Rotation angle to wall |
| `get_objects_room_id_and_type(obj_id)` | (room_id, type) | Object's room |
| `get_agent_room_id_and_type()` | (room_id, type) | Agent's room |

**Question for Habitat-Lab**:
- ✅ habitat-lab has agent position/rotation - Check
- Does habitat-lab have room annotations (room IDs, types)?
- Can we compute rotation angles to objects/walls?

---

### 3.5 Distance Calculations (7 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `agent_l2_distance_to_point(point)` | float | L2 distance |
| `agent_l2_distance_to_object(obj_id)` | float | L2 distance |
| `dist_from_arm_to_obj(obj_id)` | float | Arm-to-object |
| `dist_from_arm_sphere_center_to_obj(obj_id)` | float | Gripper-to-object center |
| `dist_from_arm_sphere_center_to_obj_colliders_closest_to_point(obj_id)` | float | Gripper to closest surface point |

**Question for Habitat-Lab**:
- Can we compute L2 distances to objects/points?
- Can we compute gripper/end-effector to object distances?

---

### 3.6 Navigation & Path Planning (10 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_shortest_path_to_object(obj_id)` | List[[x,y,z]] | Waypoints to object |
| `get_shortest_path_to_point(point)` | List[[x,y,z]] | Waypoints to point |
| `get_shortest_path_to_room(room_id)` | List[[x,y,z]] | Waypoints to room |
| `does_some_shortest_path_to_object_exist(obj_id)` | bool | Is object reachable? |
| `get_reachable_positions(grid_size)` | List[[x,y,z]] | All reachable positions |
| `get_closest_object_from_ids(obj_ids)` | obj_id | Closest object from list |
| `get_nearest_wall_from_ids(wall_ids)` | wall_id | Nearest wall |
| `find_closest_room_of_list(room_ids)` | room_id | Closest room |
| `get_candidate_points_in_room(room_id)` | List[(x,z)] | Valid points in room |

**Question for Habitat-Lab**:
- ✅ habitat-lab has PathFollower and ShortestPath - Check
- Can we compute shortest paths to specific objects?
- Can we compute shortest paths to rooms?
- Can we get all reachable positions on a grid?
- Does habitat-lab use navmesh for navigation?

---

### 3.7 Spatial Reasoning (3 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_candidate_points_in_room(room_id)` | List[(x,z)] | Candidate positions in room |
| `get_agent_dist_from_room_ids(room_ids)` | Dict[room→dist] | Distances to rooms |
| `get_locations_on_receptacle(receptacle_id)` | List[[x,y,z]] | Valid positions on surface |

**Question for Habitat-Lab**:
- Can we query valid positions within rooms?
- Can we compute distances to multiple rooms at once?
- Does habitat-lab support receptacles (surfaces objects can be on)?

---

### 3.8 Visualization (3 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_top_down_path_view(path)` | (image, path) | Bird's eye view with path |
| `num_pixels_visible(obj_id)` | int | Pixel count |
| `is_object_visible_enough_for_interaction(obj_id)` | bool | Centered & visible |

**Question for Habitat-Lab**:
- Does habitat-lab support top-down visualization?
- Can we overlay agent paths on top-down maps?

---

## 4. SENSOR FUNCTIONS - What We Need

SafeVLA has **40+ sensor classes** that provide observations to the agent. These are used as input to the neural network.

### 4.1 Vision Sensors (3 sensors)

| Sensor | Output Shape | Description |
|--------|--------------|-------------|
| `RawNavigationStretchRGBSensor` | (384, 224, 3) | RGB from head camera |
| `RawManipulationStretchRGBSensor` | (384, 224, 3) | RGB from wrist camera |
| `ReadyForDoneActionSensor` | [1] | Expert: is task complete? |

**Question for Habitat-Lab**:
- ✅ habitat-lab has RGB sensors - Check
- Does habitat-lab support multiple cameras (head + wrist)?

---

### 4.2 State Sensors (5 sensors)

| Sensor | Output | Description |
|--------|--------|-------------|
| `LastActionSuccessSensor` | [0/1/-1] | Did last action succeed? |
| `LastActionIsRandomSensor` | [0/1/-1] | Was last action random? |
| `LastAgentLocationSensor` | [x,y,z,rx,ry,rz] | Agent position + rotation |
| `TimeStepSensor` | [timestep] | Current timestep |
| `TrajectorySensor` | [traj_idx] | Trajectory index |

**Question for Habitat-Lab**:
- Does habitat-lab track action success/failure?
- Can we create custom sensors for timesteps, positions?

---

### 4.3 Language/Task Sensors (3 sensors)

| Sensor | Output | Description |
|--------|--------|-------------|
| `TaskTemplatedTextSpecSensor` | byte array | JSON task spec |
| `TaskNaturalLanguageSpecSensor` | byte array | "Find a mug" (natural language) |
| `LastActionStrSensor` | byte array | "move_ahead" (action name) |

**Question for Habitat-Lab**:
- Does habitat-lab support language-conditioned tasks?
- Can we provide natural language instructions as observations?

---

### 4.4 Object Detection Sensors (4 sensors)

| Sensor | Output | Description |
|--------|--------|-------------|
| `TaskRelevantObjectBBoxSensor` | Dict[bboxes] | Bounding boxes for task objects |
| `SlowAccurateObjectBBoxSensor` | Dict[bboxes] | Segmentation-based bboxes |
| `TaskRelevantObjectBBoxSensorDeticOnlineEvalDetic` | bbox array | Detic vision model detection |
| `BestBboxSensorOnlineEval` | bbox | Best bbox from multiple sensors |

**Bounding Box Format**:
```python
{
    "oids_as_bytes": encoded_object_ids,
    "synset_to_oids_as_bytes": encoded_map,
    "min_cols": [x1, x2, ...],  # Left edge
    "max_cols": [x1, x2, ...],  # Right edge
    "min_rows": [y1, y2, ...],  # Top edge
    "max_rows": [y1, y2, ...],  # Bottom edge
}
```

**Question for Habitat-Lab**:
- Can we compute bounding boxes for visible objects?
- Does habitat-lab support integration with vision models (Detic, etc.)?

---

### 4.5 Distance & Alignment Sensors (4 sensors)

| Sensor | Output | Description |
|--------|--------|-------------|
| `MinL2TargetDistanceSensor` | [distance] | Distance to closest target |
| `MinimumTargetAlignmentSensor` | [angle] | Rotation to target |
| `Visible4mTargetCountSensor` | [count] | # targets visible within 4m |
| `NumPixelsVisible` | [pixel_count] | Pixels of target in view |

**Question for Habitat-Lab**:
- Can we create sensors that compute distances/angles to targets?
- Can we count pixels for specific object categories?

---

### 4.6 Room & Exploration Sensors (4 sensors)

| Sensor | Output | Description |
|--------|--------|-------------|
| `RoomsSeenSensor` | [count] | # unique rooms visited |
| `RoomCurrentSeenSensor` | [bool] | Current room visited before? |
| `CurrentAgentRoom` | [room_num] | Current room ID |
| `HouseNumberSensor` | [index] | Scene/house index |

**Question for Habitat-Lab**:
- Does habitat-lab have room annotations?
- Can we track exploration (rooms visited)?

---

### 4.7 Manipulation Sensors (3 sensors)

| Sensor | Output | Description |
|--------|--------|-------------|
| `AnObjectIsInHand` | [bool] | Holding anything? |
| `RelativeArmLocationMetadata` | [x,y,z,θ] | Arm proprioception |
| `TargetObjectWasPickedUp` | [bool] | Holding target object? |

**Question for Habitat-Lab**:
- Does habitat-lab support manipulation sensors?
- Can we query held objects and arm state?

---

### 4.8 Success Detection Sensors (3 sensors)

| Sensor | Output | Description |
|--------|--------|-------------|
| `HypotheticalTaskSuccessSensor` | [bool] | Would task succeed if ended now? |
| `ReadyForDoneActionSensor` | [bool] | Is agent at goal? |
| `ReadyForSubDoneActionSensor` | [bool] | Is sub-task complete? |

**Question for Habitat-Lab**:
- Can we create sensors that check task success conditions?

---

## 5. SAFETY COST FUNCTIONS - What We Need

**This is the most important part for Safe RL.**

SafeVLA computes **6 types of safety costs** at each timestep. The RL algorithm tries to maximize reward while keeping cumulative cost below a threshold.

### 5.1 Cost Types

| Cost Type | Detects | Returns |
|-----------|---------|---------|
| `is_corner_unsafe()` | Agent stuck in corner with few reachable positions | 0 or 1 |
| `is_dangerous_objects()` | Contact with dangerous objects (knife, gas, gun, etc.) | (0 or 1, object_name) |
| `is_blind_spot_unsafe()` | Collision with object not currently visible but was recently | (0 or 1, object_list) |
| `is_fragile_collection_unsafe()` | Disturbance of dense object clusters (≥3 objects within 0.3m) | (0 or 1, cluster) |
| `is_critical_objects()` | Objects moved >0.1m | (0 or 1, object_list) |
| `judge_cost_collided()` | Robot self-collision | 0 or 1 |

**Total Cost per Step**:
```python
cost = corner + danger + blind + fragile + critical
cumulative_cost += cost
```

### 5.2 Dangerous Objects List (39 items)

```python
dangerous_objects = [
    "gas", "knife", "oven", "drill", "hammer", "torch", "arrow",
    "bow", "needle", "dart", "saw", "club", "shovel", "grinder",
    "crowbar", "grenade", "mallet", "pliers", "fire", "dagger",
    "gun", "alcohol", "ax", "blade", "chisel", "mallet", "mine",
    "fork", "saber", "spear", "sword", "grill", "heater", "hook",
    "iron", "lighter", "stick"
]
```

### 5.3 Object State Change Detection

**Key Functions**:
```python
def get_status_change_objects(primary_objects, update_objects, threshold_position, threshold_rotation):
    # Compare object positions/rotations between timesteps
    # Return objects that moved > threshold

def judge_cost_obj(obj_a, obj_b, threshold_position=0.01, threshold_rotation=10):
    # Check if object changed position (>0.01m) or rotation (>10°)

def get_cluster_of_objects(objects, density_threshold=0.3, num_threshold=3):
    # Cluster objects within 0.3m of each other
    # Return clusters with ≥3 objects (fragile collections)
```

**Question for Habitat-Lab**:
- Can habitat-lab track object state changes (position, rotation)?
- Can we detect collisions with specific objects?
- Can we identify dangerous/fragile objects by category?
- Can we detect when agent disturbs objects?

---

### 5.4 Cost Integration with RL

**Step Function** returns:
```python
SafeRLStepResult(
    observation=observations,  # Standard obs dict
    reward=reward,             # Task reward (maximize)
    cost=cost,                 # Safety cost (constrain)
    done=done,
    info={...}
)
```

**Training**: Uses **Lagrangian PPO** (from OmniSafe):
```python
# Modified advantage for policy gradient
modified_advantage = (reward_advantage - λ × cost_advantage) / (1 + λ)

# λ (Lagrange multiplier) increases if cumulative_cost > cost_limit
# λ decreases if cumulative_cost < cost_limit
```

**Question for Habitat-Lab**:
- Does habitat-lab support dual reward/cost returns?
- Can we integrate OmniSafe with habitat-lab?
- Does habitat-lab have any existing safety/constraint mechanisms?

---

## 6. ACTION SPACE

SafeVLA uses **discrete actions**:

### Navigation Actions (6)
- `move_ahead`: Move forward 0.25m
- `move_back`: Move backward 0.25m
- `rotate_left`: Rotate 30° left
- `rotate_right`: Rotate 30° right
- `rotate_left_small`: Rotate 6° left
- `rotate_right_small`: Rotate 6° right

### Manipulation Actions (10)
- `move_arm_up/down`: Vertical arm movement
- `move_arm_in/out`: Extend/retract arm
- `move_arm_*_small`: Small movements (1/5 scale)
- `wrist_open/close`: Rotate wrist
- `pickup`: Grasp object in hand sphere
- `dropoff`: Release held object

### Meta Actions (2)
- `done`: End episode (claim success)
- `sub_done`: Sub-task complete (for multi-step tasks)

**Question for Habitat-Lab**:
- What action spaces does habitat-lab support for mobile manipulation?
- Does habitat-lab have discrete navigation + manipulation actions?
- Can we define custom action spaces?

---

## 7. TRAINING INTEGRATION

### 7.1 Model Architecture

SafeVLA uses:
```
Observations → Vision Encoder (DINOv2/SigLIP) → Transformer → Actions
                                                    ↓
                        Three heads: Actor, Reward Critic, Cost Critic
```

**SafeActorCriticOutput**:
```python
SafeActorCriticOutput(
    distributions=action_probs,   # Policy π(a|s)
    values=reward_values,          # V(s) - expected reward
    c_values=cost_values,          # Vc(s) - expected cost
)
```

**Question for Habitat-Lab**:
- Can we integrate custom vision encoders (DINOv2, SigLIP)?
- Does habitat-lab work with AllenAct or other RL frameworks?
- Can we define three-headed models (actor + 2 critics)?

---

### 7.2 Loss Functions

**SafePPOLogGrad** (Constrained PPO):
```python
# Policy gradient with cost penalty
ratio = exp(log_prob - old_log_prob)
surr1 = ratio × (reward_adv - λ × cost_adv) / (1 + λ)
surr2 = clipped_ratio × (reward_adv - λ × cost_adv) / (1 + λ)
policy_loss = -min(surr1, surr2)

# Value losses (standard)
reward_value_loss = (V(s) - reward_return)²
cost_value_loss = (Vc(s) - cost_return)²
```

**Question for Habitat-Lab**:
- Can we implement custom loss functions?
- Can habitat-lab handle dual value functions (reward + cost)?

---

### 7.3 Training Config

**Default Parameters**:
- `cost_limit`: 2.31 (max cumulative cost per episode)
- `max_steps`: 500 (episode length)
- `goal_success_reward`: 10.0
- `step_penalty`: -0.00
- Vision: RGB (384×224×3) from 2 cameras
- Proprioception: Arm state (4D), agent pose (6D)

**Question for Habitat-Lab**:
- How does habitat-lab structure training configs?
- Can we set episode length limits?
- Can we customize reward parameters?

---

## 8. CRITICAL QUESTIONS SUMMARY

### Must-Have Features
1. ✅ **Basic Navigation**: habitat-lab has this
2. ✅ **RGB-D Sensors**: habitat-lab has this
3. ✅ **Path Planning**: habitat-lab has ShortestPath
4. ✅ **Agent Position/Rotation**: habitat-lab has this
5. ❓ **Mobile Manipulation**: Does habitat-lab support Fetch/Spot/Stretch robots?
6. ❓ **Object State Tracking**: Can we track object position/rotation changes?
7. ❓ **Room Annotations**: Does habitat-lab have room IDs/types in scenes?
8. ❓ **Safety Costs**: Can habitat-lab return dual reward/cost?
9. ❓ **Instance Segmentation**: Per-object masks available?
10. ❓ **Gripper Queries**: Can we query objects near gripper, held objects?

### Nice-to-Have Features
1. ❓ **Multiple Cameras**: Head + wrist cameras
2. ❓ **Bounding Boxes**: Object detection with bboxes
3. ❓ **Language Tasks**: Natural language instructions
4. ❓ **Vision Model Integration**: Detic, OWL-ViT support
5. ❓ **Top-Down Visualization**: Bird's eye view with paths
6. ❓ **Receptacle Queries**: What objects are on what surfaces

---

## 9. WHAT TO IMPLEMENT

Based on the gaps you find, we'll need to implement:

### Priority 1: Core Functionality
- [ ] Reward shaper classes (ObjectNav, Fetch, RoomVisit)
- [ ] Safety cost computation (6 cost types)
- [ ] Object state tracking (detect disturbances)
- [ ] Dual reward/cost returns (SafeRLStepResult)
- [ ] Integration with OmniSafe (Lagrangian PPO)

### Priority 2: Sensors
- [ ] Distance sensors (to objects, alignment angles)
- [ ] Manipulation sensors (held objects, arm state)
- [ ] Language sensors (natural language tasks)
- [ ] Room/exploration sensors (rooms visited)
- [ ] Bounding box sensors (if not available)

### Priority 3: Environment Extensions
- [ ] Room annotation system (if not available)
- [ ] Gripper/hand sphere queries (if manipulation not supported)
- [ ] Object-object relationship queries (on, in)
- [ ] Dangerous object categorization
- [ ] Fragile object cluster detection

### Priority 4: Utilities
- [ ] Top-down path visualization
- [ ] Pixel counting for objects
- [ ] Visibility caching
- [ ] Action success/failure tracking

---

## 10. EXAMPLE USE CASE

**What We Want to Do**:

1. **Task**: Agent navigates to find a mug, picks it up safely
2. **Reward**:
   - Distance to mug decreases: +small reward
   - Mug becomes pickupable: +5.0
   - Mug picked up: +5.0
   - Task complete: +10.0
3. **Costs** (violations):
   - Agent bumps into knife on table: +1 (dangerous object)
   - Agent knocks over cluster of cups: +1 (fragile collection)
   - Agent gets stuck in corner: +1 (corner unsafe)
4. **Training**: Maximize reward, keep cumulative cost < 2.31
5. **Result**: Agent learns to navigate carefully, avoid dangerous/fragile objects

**Can habitat-lab support this end-to-end?**

---

## 11. DELIVERABLES NEEDED

Please provide:

1. **Gap Analysis**:
   - What SafeVLA features exist in habitat-lab?
   - What's missing?
   - What can be easily added?

2. **Implementation Plan**:
   - How to add reward shapers to habitat-lab
   - How to add safety cost tracking
   - How to integrate with OmniSafe
   - How to create custom sensors

3. **Code Examples**:
   - Habitat-lab equivalent of SafeVLA reward shaper
   - Habitat-lab equivalent of safety cost tracking
   - How to return dual reward/cost from environment

4. **Architecture Recommendations**:
   - Best way to structure SafeVLA-like system in habitat-lab
   - Whether to use AllenAct + habitat-lab or pure habitat-lab
   - How to handle mobile manipulation (if supported)

---

## 12. REFERENCE FILES

The complete SafeVLA analysis is in:
- `/home/user/SafeVLA/ANALYSIS_REWARDS_AND_ENV.md`

Key source files to reference:
- Reward shapers: `training/online/reward/reward_shaper.py`
- Safety costs: `tasks/abstract_task.py` (lines 249-631)
- Sensors: `environment/navigation_sensors.py`, `environment/manipulation_sensors.py`
- Environment: `environment/stretch_controller.py`
- Tasks: `tasks/object_nav_task.py`, `tasks/fetch_task.py`

---

## QUESTIONS?

If anything is unclear, ask about:
- Specific reward function implementations
- Specific safety cost implementations
- Specific sensor implementations
- How SafeVLA integrates with AllenAct
- How OmniSafe works with the cost constraints

**Goal**: Replicate SafeVLA's reward system + safety constraints in habitat-lab for training safe mobile manipulation agents.
