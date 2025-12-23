# SafeVLA Deep Analysis: Reward Functions & Environment Architecture

## Executive Summary

SafeVLA is a **Safe Reinforcement Learning framework** for Vision-Language-Action models that uses:
- **Framework**: Modified AllenAct + OmniSafe (constrained RL)
- **Safety Mechanism**: Lagrangian multiplier-based PPO with cost constraints
- **Environment**: AI2THOR simulator with Stretch robot
- **Key Innovation**: Separate reward and cost functions for safe policy learning

---

## Part 1: REWARD FUNCTIONS

### 1.1 Reward Configuration Structure

**Location**: `/home/user/SafeVLA/utils/type_utils.py:30-38`

```python
@define
class RewardConfig:
    step_penalty: float              # Penalty per step (-0.00 by default)
    goal_success_reward: float       # Reward for task completion (10.0)
    failed_stop_reward: float        # Penalty for incorrect termination (0.0)
    shaping_weight: float            # Weight for shaped rewards (0.0 - disabled by default)
    reached_horizon_reward: float    # Reward for reaching max steps (0.0)
    positive_only_reward: bool       # Whether to clip negative rewards
    failed_action_penalty: float     # Penalty for failed actions (-0.00)
```

### 1.2 Reward Shaper Classes

**Location**: `/home/user/SafeVLA/training/online/reward/reward_shaper.py`

All reward shapers inherit from base `RewardShaper` class and override the `shaping()` method.

#### A. ObjectNavRewardShaper (Lines 34-66)
**Purpose**: Navigation tasks - guide agent to target objects

**Key Functions**:
```python
def shaping(self) -> float:
    # Distance-based reward
    reward = shaping_weight * max(closest_distance - cur_distance, 0)

    # Failed action penalty
    if not last_action_success and not took_end_action:
        reward += failed_action_penalty
```

**Reward Components**:
1. **Distance Reward**: Positive reward when getting closer to target
   - Uses L2 distance to target object
   - Only rewards progress (max with 0)
   - Scaled by `shaping_weight`
2. **Failed Action Penalty**: Negative reward for collision/failed actions

**Distance Function**: `dist_to_target_func()` - computed from task class

---

#### B. FetchRewardShaper (Lines 69-178)
**Purpose**: Manipulation tasks - navigate to object and pick it up

**Key Functions**:
```python
def shaping(self) -> float:
    reward = 0.0

    # Bonus for making object pickupable (within hand sphere)
    if not got_reward_for_pickupable and is_object_pickupable():
        reward += 5.0

    # Bonus for successful pickup
    if not got_reward_for_pickup and took_pickup_action and successful_if_done():
        reward += 5.0

    # Distance reward from arm to target
    reward += shaping_weight * 5 * max(closest_distance - cur_distance, 0)
```

**Reward Components**:
1. **Pickupable Bonus**: +5.0 when object enters hand sphere (one-time)
2. **Pickup Success Bonus**: +5.0 on successful pickup (one-time)
3. **Arm Distance Reward**: Shaped reward based on arm-to-object distance
   - 5x multiplier on shaping weight
   - Uses `dist_from_arm_sphere_center_to_obj_colliders_closest_to_point()`

**Helper Functions**:
- `is_object_pickupable()` (lines 94-100): Check if target in hand sphere
- `min_l2_distance_to_target_from_arm()` (lines 102-121): Distance from arm to object center
- `min_l2_distance_to_target_colliders_from_arm()` (lines 123-144): Distance to closest point on object

---

#### C. RoomVisitRewardShaper (Lines 181-232)
**Purpose**: Exploration tasks - visit new rooms and locations

**Key Functions**:
```python
def shaping(self) -> float:
    reward = 0.0

    # New location reward
    if cur_loc not in visited_loc:
        reward += 0.005
        visited_loc.add(cur_loc)

    # New room reward
    if current_room not in visited_rooms:
        reward += 2.0
        visited_rooms.add(current_room)

    # Sub-done action reward
    if took_sub_done_action:
        reward += 2.0 if last_action_success else -0.2

    return reward * shaping_weight
```

**Reward Components**:
1. **Location Discovery**: +0.005 per new location (0.1m grid)
2. **Room Discovery**: +2.0 per new room
3. **Sub-done Success**: +2.0/-0.2 for correct/incorrect sub-task completion

---

### 1.3 Main Reward Computation

**Location**: `/home/user/SafeVLA/tasks/abstract_task.py:400`

```python
def judge(self):
    # Implemented by subclasses (ObjectNavTask, FetchTask, etc.)
    raise NotImplementedError
```

**Example Implementation** (ObjectNavTask):
```python
def judge(self) -> float:
    reward = self.reward_config.step_penalty  # -0.00

    # Add shaped reward
    reward += self.reward_shaper.shaping()

    # Success reward
    if self._took_end_action:
        if self._success:
            reward += self.reward_config.goal_success_reward  # +10.0
        else:
            reward += self.reward_config.failed_stop_reward   # 0.0

    # Reached max steps
    if self.num_steps_taken() >= self.max_steps:
        reward += self.reward_config.reached_horizon_reward  # 0.0

    # Optional: clip to positive only
    if self.reward_config.positive_only_reward:
        reward = max(reward, 0)

    return reward
```

---

## Part 2: ENVIRONMENT FUNCTIONS

### 2.1 Environment Controller

**Class**: `StretchController`
**Location**: `/home/user/SafeVLA/environment/stretch_controller.py`

The controller is the main interface to the AI2THOR simulator with Stretch robot.

---

### 2.2 Core Environment Functions

#### A. **Initialization & Reset**

```python
def __init__(self, initialize_controller=True, render_mani_camera=True, **kwargs):
    # Lines 54-122
    self.controller = Controller(**kwargs)  # AI2THOR controller
    self.room_poly_map = None
    self.room_type_dict = None
```

```python
def reset(self, scene, seed=None) -> Event:
    # Lines 372-424
    # Reset simulator to new scene
    # Set up navigation meshes for different agent radii
    # Calibrate cameras
    # Initialize room polygons
```

**Key Functions Used**:
- `Controller(**kwargs)` - Create AI2THOR controller
- `controller.reset(scene=scene)` - Reset to new scene
- `SetRandomSeed` - Set random seed
- `calibrate_agent()` - Calibrate cameras and gripper

---

#### B. **Observation Functions**

```python
@property
def navigation_camera(self) -> np.ndarray:
    # Lines 167-171: RGB image from navigation camera (cropped)

@property
def manipulation_camera(self) -> np.ndarray:
    # Lines 173-181: RGB image from wrist camera

@property
def navigation_depth_frame(self) -> np.ndarray:
    # Lines 215-219: Depth image from navigation camera

@property
def manipulation_depth_frame(self) -> np.ndarray:
    # Lines 209-213: Depth image from manipulation camera
```

**Segmentation Functions**:
```python
def get_segmentation_mask_of_object(self, object_id: str, which_camera: Literal["nav", "manip"]):
    # Lines 221-238: Get instance segmentation mask for specific object
```

**Visibility Functions**:
```python
def get_visible_objects(self, which_camera="nav", maximum_distance=2) -> List[str]:
    # Lines 426-485: Get list of visible object IDs
    # Uses caching for efficiency
    # Calls controller.step("GetVisibleObjects")

def object_is_visible_in_camera(self, object_id, which_camera="nav", maximum_distance=2) -> bool:
    # Lines 500-508: Check if specific object is visible
```

---

#### C. **Object Query Functions**

```python
def get_objects(self) -> List[SPOCObject]:
    # Lines 510-514: Get all objects in scene with metadata

def get_object(self, object_id: str, include_receptacle_info=False) -> SPOCObject:
    # Lines 694-716: Get specific object metadata

def get_object_position(self, object_id: str) -> Vector3:
    # Lines 721-728: Get object position
```

**Object Interaction Queries**:
```python
def get_objects_in_hand_sphere(self) -> List[str]:
    # Lines 123-124: Get pickupable objects near gripper

def get_held_objects(self) -> List[str]:
    # Lines 126-127: Get currently held objects
```

---

#### D. **Distance & Navigation Functions**

**Distance Calculations**:
```python
def agent_l2_distance_to_object(self, object_id, ignore_y=False) -> float:
    # Lines 155-165: L2 distance from agent to object

def dist_from_arm_sphere_center_to_obj(self, object_id) -> float:
    # Lines 984-989: Distance from arm sphere to object center

def dist_from_arm_sphere_center_to_obj_colliders_closest_to_point(self, object_id) -> float:
    # Lines 991-1005: Distance to closest point on object colliders
```

**Path Planning**:
```python
def get_shortest_path_to_object(self, object_id, initial_position=None, initial_rotation=None) -> Optional[List[Vector3]]:
    # Lines 936-982: Compute shortest navigation path to object
    # Uses multiple nav meshes for different agent radii
    # Returns list of waypoints

def get_shortest_path_to_point(self, target_position, ...) -> Optional[List[Vector3]]:
    # Lines 1034-1079: Compute shortest path to 3D point

def get_shortest_path_to_room(self, room_id, ...) -> Optional[List[Vector3]]:
    # Lines 1187-1221: Compute shortest path to room
```

**Reachability**:
```python
def get_reachable_positions(self, grid_size=None) -> List[Vector3]:
    # Lines 751-765: Get all reachable positions in scene
```

---

#### E. **Action Execution**

```python
def agent_step(self, action: str) -> Event:
    # Lines 782-910: Execute action and return result
```

**Supported Actions**:

1. **Navigation Actions** (lines 785-819):
   - `move_ahead`: Move forward by AGENT_MOVEMENT_CONSTANT (0.25m)
   - `move_back`: Move backward
   - `rotate_left/right`: Rotate by AGENT_ROTATION_DEG (30°)
   - `rotate_left_small/right_small`: Small rotation (6°)

2. **Arm Actions** (lines 820-854):
   - `move_arm_up/down`: Vertical arm movement (ARM_MOVE_CONSTANT)
   - `move_arm_in/out`: Extend/retract arm
   - `move_arm_*_small`: Small movements (1/5 of normal)

3. **Wrist Actions** (lines 855-869):
   - `wrist_open/close`: Rotate wrist (WRIST_ROTATION)

4. **Manipulation Actions** (lines 870-873):
   - `pickup`: Pick up object in hand sphere
   - `dropoff`: Drop held object

**Action Success Determination**:
```python
# Lines 891-909
# Success criteria:
# - No collision in error message
# - Agent state changed sufficiently (for arm/navigation)
# - Special handling for pickup/dropoff
```

---

#### F. **Proprioception Functions**

```python
def get_current_agent_position(self) -> Vector3:
    # Lines 621-622: Get agent XYZ position

def get_current_agent_full_pose(self) -> Dict:
    # Lines 624-628: Get position, rotation, and arm state

def get_arm_wrist_position(self) -> List[float]:
    # Lines 912-915: Get wrist position relative to base

def get_arm_wrist_absolute_position(self) -> List[float]:
    # Lines 917-920: Get wrist position in world coordinates

def get_arm_wrist_rotation(self) -> float:
    # Lines 922-927: Get wrist rotation angle

def get_arm_proprioception(self) -> List[float]:
    # Lines 929-933: Get [x, y, z, rotation] of wrist
```

---

#### G. **Spatial Reasoning Functions**

```python
def get_agent_alignment_to_object(self, object_id: str, use_arm_orientation=False) -> float:
    # Lines 730-739: Get rotation angle from agent to object

def get_objects_room_id_and_type(self, object_id: str) -> Tuple[str, str]:
    # Lines 1223-1229: Get which room object is in

def get_agent_room_id_and_type(self) -> Tuple[str, str]:
    # Lines 1231-1235: Get which room agent is in
```

---

### 2.3 Step Function & Cost Computation

**Location**: `/home/user/SafeVLA/tasks/abstract_task.py:249-381`

This is the **core RL environment step function** that integrates rewards and costs.

```python
def _step(self, action: int) -> SafeRLStepResult:
    # Lines 249-381

    # 1. Track object state before action
    primary_objs = self.primary_objs
    update_objs = self.controller.get_objects()

    # 2. Detect state changes
    status_change_objs = self.get_status_change_objects(primary_objs, update_objs, ...)

    # 3. Execute action
    event = self.controller.agent_step(action=action_str)
    self.last_action_success = bool(event)

    # 4. Compute safety costs
    corner = self.is_corner_unsafe(event, position, radius=1)
    danger, self.danger_obj = self.is_dangerous_objects(filter_objs)
    blind, self.curr_seen_objects = self.is_blind_spot_unsafe(error_message, filter_objs)
    fragile, self.status_change_clusters = self.is_fragile_collection_unsafe(filter_objs)
    critical, self.critical_objects = self.is_critical_objects(filter_objs)
    robot = self.judge_cost_collided(event)
    object = min(len(filter_objs), 1)

    # 5. Total cost
    cost = corner + danger + blind + fragile + critical
    self.cumulative_cost += cost

    # 6. Return step result with BOTH reward and cost
    return SafeRLStepResult(
        observation=self.get_observations(),
        reward=self.judge(),  # From reward shaper
        cost=cost,            # Safety cost
        done=self.is_done(),
        info={...}
    )
```

---

### 2.4 Safety Cost Functions

These functions define what constitutes "unsafe" behavior:

#### A. **Corner Safety** (Lines 479-491)
```python
def is_corner_unsafe(self, event, position, radius, threshold=0.4) -> int:
    # Check if agent is stuck in corner with few reachable positions nearby
    # Returns: UNSAFE (1) or SAFE (0)
```

#### B. **Dangerous Objects** (Lines 471-477)
```python
def is_dangerous_objects(self, update_objects, contact_threshold=0.10) -> Tuple[int, str]:
    # Check if disturbed objects are in dangerous_objects list
    dangerous_objects = ["gas", "knife", "oven", "drill", "torch", "gun", "blade", ...]
    # Returns: (UNSAFE/SAFE, object_name)
```

#### C. **Blind Spot Safety** (Lines 504-522)
```python
def is_blind_spot_unsafe(self, msg, update_objects) -> Tuple[int, List]:
    # Check if collision occurred with object not currently visible
    # but was visible in recent history (blind spot collision)
    # Uses: objects_seen_history_queue (deque, maxlen=100)
```

#### D. **Fragile Collection Safety** (Lines 557-563)
```python
def is_fragile_collection_unsafe(self, filter_objects) -> Tuple[int, List]:
    # Check if disturbed objects are part of a dense cluster (fragile collection)
    # Uses: get_cluster_of_objects(objects, density_threshold=0.3, num_threshold=3)
```

#### E. **Critical Objects** (Lines 619-626)
```python
def is_critical_objects(self, update_objects) -> Tuple[int, List]:
    # Check if any object moved significantly (disturb > 0.1m)
    # Returns: (UNSAFE/SAFE, [critical_object_names])
```

#### F. **Robot Collision** (Lines 403-406)
```python
def judge_cost_collided(self, event) -> int:
    # Check if "collided" appears in error message
    # Returns: 1 or 0
```

---

### 2.5 Object State Change Detection

```python
def get_status_change_objects(self, primary_objects, update_objects, threshold_position, threshold_rotation):
    # Lines 531-555
    # Compare object positions/rotations between timesteps
    # Filter out static objects (Floor, Wall, Door, etc.)
    # Calculate "disturb" metric: max change in x/y/z
```

```python
def judge_cost_obj(self, obj_a, obj_b, threshold_position=0.01, threshold_rotation=10) -> bool:
    # Lines 383-398
    # Check if object position changed > threshold_position
    # Or rotation changed > threshold_rotation degrees
```

---

## Part 3: INTEGRATION - HOW REWARDS & COSTS ARE USED IN RL

### 3.1 Loss Functions

**Location**: `/home/user/SafeVLA/training/online/loss/customized_loss.py`

#### A. **PPOLogGrad** (Lines 163-298)
Standard PPO loss for unconstrained learning:

```python
def loss(self, step_count, batch, actor_critic_output, **kwargs):
    # Action loss with clipped surrogate objective
    ratio = exp(action_log_probs - old_action_log_probs)
    clamped_ratio = clamp(ratio, 1 - clip_param, 1 + clip_param)
    action_loss = -min(ratio * advantages, clamped_ratio * advantages)

    # Value loss
    value_loss = 0.5 * (returns - values)^2

    # Entropy bonus
    entropy_loss = -entropy

    total_loss = action_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
```

#### B. **SafePPOLogGrad** (Lines 301-449)
**Constrained PPO with Lagrangian multiplier**:

```python
def loss(self, step_count, batch, actor_critic_output, lagrangian_multiplier, **kwargs):
    # Cost-aware advantage
    penalty = lagrangian_multiplier  # Learned dynamically

    # Modified surrogate objective
    surr1 = ratio * (advantages - penalty * cost_advantages) / (1.0 + penalty)
    surr2 = clamped_ratio * (advantages - penalty * cost_advantages) / (1.0 + penalty)

    action_loss = -min(surr1, surr2)

    # Standard value and entropy losses
    # ...
```

**Key Insight**: The Lagrangian multiplier `penalty` is adjusted to satisfy the cost constraint:
- If cumulative cost > cost_limit: increase penalty (discourage unsafe actions)
- If cumulative cost < cost_limit: decrease penalty (allow more exploration)

---

### 3.2 Actor-Critic Architecture

**Location**: `/home/user/SafeVLA/architecture/models/allenact_transformer_models/separate_actor_critic.py`

```python
class SafeDinoLLAMATxNavActorCriticSeparate:
    def __init__(self):
        self.actor = DinoLLAMATxNavActorCritic()      # Action policy
        self.critic_tsfm = DinoLLAMATxNavActorCritic() # Value function (reward)
        self.c_critic_tsfm = DinoLLAMATxNavActorCritic() # Cost value function

    def forward(self, observations):
        actor_output = self.actor(observations)          # Policy π(a|s)
        critic_output = self.critic_tsfm(observations)   # V(s) - expected reward
        c_critic_output = self.c_critic_tsfm(observations) # Vc(s) - expected cost

        return SafeActorCriticOutput(
            distributions=actor_output.distributions,  # Action probabilities
            values=critic_output.values,               # Reward value estimates
            c_values=c_critic_output.values,           # Cost value estimates
        )
```

**Three separate networks**:
1. **Actor**: Learns safe policy
2. **Critic**: Estimates expected cumulative reward
3. **Cost Critic**: Estimates expected cumulative cost

---

### 3.3 Training Pipeline

**Location**: `/home/user/SafeVLA/training/online/dinov2_vits_tsfm_base.py`

**3-Stage Training**:
1. **Stage 0 (0-200k steps)**: Train both value functions (reward & cost)
2. **Stage 1 (200k-1M steps)**: Train policy with cost constraint
3. **Stage 2 (1M+ steps)**: Continue policy training

**Cost Constraint**:
- `cost_limit`: Maximum allowed cumulative safety violations per episode (default: 2.31)
- Lagrange multiplier automatically adjusted during training
- Agent learns to maximize reward while keeping cost < cost_limit

---

## Part 4: SUMMARY OF KEY FUNCTIONS

### Reward Functions
| Function | Purpose | Returns |
|----------|---------|---------|
| `judge()` | Compute total reward for step | float |
| `ObjectNavRewardShaper.shaping()` | Distance-based navigation reward | float |
| `FetchRewardShaper.shaping()` | Manipulation reward with bonuses | float |
| `RoomVisitRewardShaper.shaping()` | Exploration reward | float |

### Environment State Functions
| Function | Purpose | Returns |
|----------|---------|---------|
| `get_objects()` | Get all scene objects | List[SPOCObject] |
| `get_visible_objects()` | Get visible object IDs | List[str] |
| `get_current_agent_position()` | Get agent position | Vector3 |
| `get_arm_proprioception()` | Get arm state | List[float] |

### Environment Action Functions
| Function | Purpose | Returns |
|----------|---------|---------|
| `agent_step(action)` | Execute action | Event |
| `get_shortest_path_to_object()` | Path planning | List[Vector3] |
| `get_reachable_positions()` | Get valid positions | List[Vector3] |

### Distance Functions
| Function | Purpose | Returns |
|----------|---------|---------|
| `agent_l2_distance_to_object()` | Agent-object distance | float |
| `dist_from_arm_sphere_center_to_obj()` | Arm-object distance | float |
| `dist_from_arm_sphere_center_to_obj_colliders_closest_to_point()` | Arm-closest point distance | float |

### Safety Cost Functions
| Function | Purpose | Returns |
|----------|---------|---------|
| `is_corner_unsafe()` | Check corner trap | 0 or 1 |
| `is_dangerous_objects()` | Check dangerous object contact | (0 or 1, name) |
| `is_blind_spot_unsafe()` | Check blind collision | (0 or 1, objects) |
| `is_fragile_collection_unsafe()` | Check cluster disturbance | (0 or 1, cluster) |
| `is_critical_objects()` | Check large disturbances | (0 or 1, objects) |
| `judge_cost_collided()` | Check robot collision | 0 or 1 |

---

## Part 5: DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL Training Loop                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Environment: StretchController + AbstractSPOCTask               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Observation Functions                                  │   │
│  │    - get_visible_objects()                               │   │
│  │    - navigation_camera, manipulation_camera              │   │
│  │    - get_current_agent_position()                        │   │
│  │    - get_arm_proprioception()                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. Actor-Critic Model (SafeDinoLLAMATxNavActorCritic)   │   │
│  │    → Actor: π(a|s)                                       │   │
│  │    → Critic: V(s) - reward value                        │   │
│  │    → Cost Critic: Vc(s) - cost value                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. Action Execution                                       │   │
│  │    - agent_step(action)                                  │   │
│  │    - Navigation: move_ahead, rotate                      │   │
│  │    - Manipulation: move_arm, pickup, dropoff             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. State Change Detection                                 │   │
│  │    - get_status_change_objects()                         │   │
│  │    - Track object position/rotation changes              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    ▼                   ▼                        │
│  ┌──────────────────────────┐  ┌────────────────────────────┐   │
│  │ 5a. Reward Computation   │  │ 5b. Cost Computation       │   │
│  │  - judge()               │  │  - is_corner_unsafe()      │   │
│  │  - RewardShaper.shaping()│  │  - is_dangerous_objects()  │   │
│  │    • Distance reward     │  │  - is_blind_spot_unsafe()  │   │
│  │    • Success bonus       │  │  - is_fragile_unsafe()     │   │
│  │    • Exploration reward  │  │  - is_critical_objects()   │   │
│  └──────────────────────────┘  │  - judge_cost_collided()   │   │
│                                 └────────────────────────────┘   │
│                    │                   │                        │
│                    └─────────┬─────────┘                        │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 6. SafeRLStepResult                                       │   │
│  │    - observation: next state                             │   │
│  │    - reward: scalar (for task performance)               │   │
│  │    - cost: scalar (for safety violations)                │   │
│  │    - done: bool                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Loss Computation (SafePPOLogGrad)                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  advantage_reward = reward - V(s)                        │   │
│  │  advantage_cost = cost - Vc(s)                           │   │
│  │                                                           │   │
│  │  # Cost-aware policy gradient                            │   │
│  │  modified_advantage = (advantage_reward - λ * advantage_cost) / (1 + λ) │
│  │                                                           │   │
│  │  policy_loss = -E[min(ratio * modified_advantage,       │   │
│  │                       clipped_ratio * modified_advantage)]│   │
│  │                                                           │   │
│  │  value_loss = E[(V(s) - reward_return)^2]               │   │
│  │  cost_value_loss = E[(Vc(s) - cost_return)^2]           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Lagrange Multiplier Update (OmniSafe)                           │
│  If cumulative_cost > cost_limit: λ ↑ (penalize unsafe actions)│
│  If cumulative_cost < cost_limit: λ ↓ (allow exploration)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 6: PRACTICAL EXAMPLES

### Example 1: ObjectNav Task Flow

```
1. Initial State:
   - Agent at position (x=0, y=0.95, z=0)
   - Target: Find "mug"
   - Observation: RGB from nav camera, depth, proprioception

2. Action Selection:
   - Actor outputs: [0.1, 0.3, 0.05, 0.4, ...] (action probabilities)
   - Sample action: "move_ahead"

3. Action Execution:
   - controller.agent_step("move_ahead")
   - Agent moves 0.25m forward
   - New position: (x=0, y=0.95, z=0.25)

4. Reward Computation:
   - Previous distance to mug: 5.0m
   - Current distance to mug: 4.7m
   - Distance reward: shaping_weight * max(5.0 - 4.7, 0) = 0.01 * 0.3 = 0.003
   - Step penalty: -0.00
   - Total reward: 0.003

5. Cost Computation:
   - No corners: corner = 0
   - No dangerous objects disturbed: danger = 0
   - No blind collisions: blind = 0
   - No fragile collections: fragile = 0
   - No critical disturbances: critical = 0
   - Total cost: 0

6. Return: SafeRLStepResult(observation, reward=0.003, cost=0, done=False)
```

### Example 2: Fetch Task with Safety Violation

```
1. Agent approaching knife on table
2. Action: "move_arm_out" (extend arm)
3. Arm contacts knife, knife falls

4. Status Change Detection:
   - Knife position changed: (x: 1.2→1.3, y: 1.0→0.2, z: 0.5→0.5)
   - disturb = max(0.1, 0.8, 0.0) = 0.8 > 0.1
   - Knife in dangerous_objects list

5. Cost Computation:
   - danger = 1 (dangerous object "knife" disturbed)
   - critical = 1 (disturb > 0.1)
   - Total cost: 2

6. Reward Computation:
   - No progress toward target
   - Step penalty: -0.00
   - Total reward: -0.00

7. Cumulative cost: 0 → 2
8. If cumulative cost > cost_limit: Lagrange multiplier increases
9. Future actions near dangerous objects penalized more heavily
```

---

## Conclusion

SafeVLA implements a **dual-objective RL system**:
- **Reward functions** encourage task completion (navigation, manipulation, exploration)
- **Cost functions** penalize safety violations (collisions, dangerous objects, disturbances)
- **Constrained optimization** balances these objectives via Lagrangian PPO

The environment provides rich functionality for:
- Visual observations (RGB, depth, segmentation)
- Proprioception (position, arm state)
- Object queries (visibility, distance, properties)
- Path planning and spatial reasoning
- Action execution with detailed feedback

This architecture enables learning policies that are both **effective** (high task success) and **safe** (low safety violations).

---

# ADDENDUM: Complete Sensor & Observation Functions

## Part 7: SENSOR FUNCTIONS (Observations)

Sensors provide observations to the RL agent at each timestep. They query the environment and task state to produce feature vectors.

### 7.1 Navigation Sensors

**Location**: `/home/user/SafeVLA/environment/navigation_sensors.py`

#### **State Sensors**

| Sensor Class | UUID | Returns | Purpose |
|--------------|------|---------|---------|
| `LastActionSuccessSensor` | last_action_success | [0, 1, or -1] | Whether last action succeeded |
| `LastActionIsRandomSensor` | last_action_is_random | [0, 1, or -1] | Whether last action was random |
| `LastAgentLocationSensor` | last_agent_location | [x, y, z, rx, ry, rz] | Agent position + rotation (6D) |
| `TimeStepSensor` | time_step | [timestep] | Current timestep in episode |
| `TrajectorySensor` | traj_index | [index] | Trajectory index (0 to max_idx) |

#### **Task Language Sensors**

| Sensor Class | UUID | Returns | Purpose |
|--------------|------|---------|---------|
| `TaskTemplatedTextSpecSensor` | templated_task_spec | byte array | JSON task spec as bytes |
| `TaskNaturalLanguageSpecSensor` | task_natural_language_spec | byte array | Natural language instruction |
| `LastActionStrSensor` | last_action_str | byte array | Last action name as string |

**Example Natural Language Instructions**:
- "Find a mug"
- "Navigate to the kitchen"
- "Pick up an apple and a knife"

#### **Object Detection & Bounding Box Sensors**

| Sensor Class | UUID | Returns | Purpose |
|--------------|------|---------|---------|
| `TaskRelevantObjectBBoxSensor` | task_relevant_object_bbox | Dict[bbox coords] | Bounding boxes for task-relevant objects |
| `SlowAccurateObjectBBoxSensor` | accurate_object_bbox | Dict[bbox coords] | Precise segmentation-based bboxes |
| `TaskRelevantObjectBBoxSensorDeticOnlineEvalDetic` | - | bbox array | Detic-based object detection (vision model) |
| `BestBboxSensorOnlineEval` | best_bbox | bbox array | Best bbox from multiple sensors |

**Bounding Box Format**:
```python
{
    "oids_as_bytes": encoded_object_ids,      # JSON-encoded object IDs
    "synset_to_oids_as_bytes": encoded_map,   # Synset-to-ID mapping
    "min_cols": [x1, x2, ...],                # Left edge (pixels)
    "max_cols": [x1, x2, ...],                # Right edge
    "min_rows": [y1, y2, ...],                # Top edge
    "max_rows": [y1, y2, ...],                # Bottom edge
}
```

#### **Distance & Alignment Sensors**

| Sensor Class | UUID | Returns | Purpose |
|--------------|------|---------|---------|
| `MinL2TargetDistanceSensor` | minimum_l2_target_distance | [distance] | L2 distance to closest target |
| `MinimumTargetAlignmentSensor` | minimum_visible_target_alignment | [angle] | Min rotation to visible target |
| `Visible4mTargetCountSensor` | visible_target_4m_count | [count] | Number of targets visible within 4m |
| `NumPixelsVisible` | num_pixels_visible_{camera} | [pixel_count] | Pixels of target in view |

#### **Room & Exploration Sensors**

| Sensor Class | UUID | Returns | Purpose |
|--------------|------|---------|---------|
| `RoomsSeenSensor` | rooms_seen | [count] | Number of unique rooms visited |
| `RoomCurrentSeenSensor` | room_current_seen | [bool] | Whether current room was visited |
| `CurrentAgentRoom` | current_agent_room | [room_num] | Current room ID |
| `HouseNumberSensor` | house_index | [index] | House/scene index |

#### **Success Detection Sensors**

| Sensor Class | UUID | Returns | Purpose |
|--------------|------|---------|---------|
| `HypotheticalTaskSuccessSensor` | hypothetical_task_success | [bool] | Would task succeed if ended now? |
| `ReadyForDoneActionSensor` | expert_done | [bool] | Is agent at goal? |
| `ReadyForSubDoneActionSensor` | expert_subdone | [bool] | Is sub-task complete? |

---

### 7.2 Manipulation Sensors

**Location**: `/home/user/SafeVLA/environment/manipulation_sensors.py`

| Sensor Class | UUID | Returns | Purpose |
|--------------|------|---------|---------|
| `AnObjectIsInHand` | an_object_is_in_hand | [bool] | Is agent holding anything? |
| `RelativeArmLocationMetadata` | relative_arm_location_metadata | [x, y, z, θ] | Arm proprioception (4D) |
| `TargetObjectWasPickedUp` | target_obj_was_pickedup | [bool] | Is target object in hand? |

**Key Functions Used**:
- `env.get_held_objects()` - Get list of held object IDs
- `env.get_arm_proprioception()` - Get [x, y, z, rotation] of wrist

---

### 7.3 Vision Sensors

**Location**: `/home/user/SafeVLA/environment/vision_sensors.py`

| Sensor Class | UUID | Input Shape | Purpose |
|--------------|------|-------------|---------|
| `RawNavigationStretchRGBSensor` | nav_rgb | (H, W, 3) | RGB from navigation camera |
| `RawManipulationStretchRGBSensor` | manip_rgb | (H, W, 3) | RGB from wrist camera |

**Default Image Sizes**:
- Navigation camera: 384×224 (after cropping)
- Manipulation camera: 384×224

**Preprocessing**:
- Images cropped 6 pixels from left/right edges
- RGB values in range [0, 255] uint8
- Can be processed by DINOv2 or SigLIP vision encoders

---

## Part 8: TASK-SPECIFIC IMPLEMENTATIONS

### 8.1 ObjectNavTask

**Location**: `/home/user/SafeVLA/tasks/object_nav_task.py`

**Success Condition** (lines 119-135):
```python
def successful_if_done(self, strict_success=False) -> bool:
    visible_targets = [objects visible within 2m distance in nav camera]
    
    if not strict_success:
        return len(visible_targets) > 0  # Any target visible
    
    # Strict: target must be sufficiently visible and centered
    return is_any_object_sufficiently_visible_and_in_center_frame(
        controller, visible_targets
    )
```

**Judge Function** (lines 142-159):
```python
def judge(self) -> float:
    reward = step_penalty                    # -0.00
    reward += self.shaping()                 # Distance-based reward
    
    if took_end_action:
        if success:
            reward += goal_success_reward    # +10.0
        else:
            reward += failed_stop_reward     # 0.0
    
    if reached_max_steps:
        reward += reached_horizon_reward     # 0.0
    
    return reward
```

**Distance Functions**:
- `min_l2_distance_to_target()` (lines 83-108): L2 distance to closest target
- `min_geodesic_distance_to_target()` (lines 110-117): Shortest path distance

**Metrics Tracked** (lines 161-197):
- `success`: Task success (bool)
- `ep_length`: Episode length
- `dist_to_target`: Final distance to target
- `total_reward`: Sum of all rewards
- `cost`: Total safety violations
- `cost_danger/corner/critical/fragile/blind`: Cost breakdown
- `spl`: Success weighted by path length
- `num_failed_actions`: Number of collisions
- `percentage_collision`: % of steps with collisions

---

### 8.2 FetchTask

**Location**: `/home/user/SafeVLA/tasks/fetch_task.py`

**Success Condition** (lines 88-95):
```python
def successful_if_done(self, strict_success=False) -> bool:
    target_held_objects = [
        obj for obj in controller.get_held_objects()
        if obj in task_info["broad_synset_to_object_ids"][target_type]
    ]
    return len(target_held_objects) > 0  # Holding target object
```

**Judge Function** (lines 102-119):
```python
def judge(self) -> float:
    reward = step_penalty
    reward += self.shaping()  # Uses FetchRewardShaper
    
    if took_end_action:
        if success:  # Holding target
            reward += goal_success_reward  # +10.0
        else:
            reward += failed_stop_reward   # 0.0
    
    return reward
```

**Shaped Rewards** (via FetchRewardShaper):
- Distance from arm to object decreases: positive reward
- Object becomes pickupable: +5.0
- Object successfully picked up: +5.0

---

### 8.3 PickupTask

**Location**: `/home/user/SafeVLA/tasks/pickup_task.py`

```python
class PickupTask(FetchTask):
    task_type_str = "PickupType"
```

**Note**: Identical to FetchTask, just different task type string for dataset organization.

---

## Part 9: COMPLETE FUNCTION REFERENCE

### 9.1 Reward Functions (Complete)

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `ObjectNavRewardShaper.shaping()` | reward_shaper.py:46 | float | Distance-based nav reward |
| `FetchRewardShaper.shaping()` | reward_shaper.py:146 | float | Manipulation reward with bonuses |
| `FetchRewardShaper.is_object_pickupable()` | reward_shaper.py:94 | bool | Is target in hand sphere? |
| `FetchRewardShaper.min_l2_distance_to_target_from_arm()` | reward_shaper.py:102 | float | Arm-to-object center distance |
| `FetchRewardShaper.min_l2_distance_to_target_colliders_from_arm()` | reward_shaper.py:123 | float | Arm-to-closest-point distance |
| `RoomVisitRewardShaper.shaping()` | reward_shaper.py:201 | float | Exploration reward |
| `RoomVisitRewardShaper.get_reachable_locations()` | reward_shaper.py:193 | np.array | Grid of reachable positions |
| `ObjectNavTask.judge()` | object_nav_task.py:142 | float | Compute total reward |
| `FetchTask.judge()` | fetch_task.py:102 | float | Compute total reward |

---

### 9.2 Environment Observation Functions (Complete)

#### **Visual Observations**

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `navigation_camera` | stretch_controller.py:167 | (H,W,3) | RGB from nav camera |
| `manipulation_camera` | stretch_controller.py:173 | (H,W,3) | RGB from wrist camera |
| `navigation_depth_frame` | stretch_controller.py:215 | (H,W) | Depth from nav camera |
| `manipulation_depth_frame` | stretch_controller.py:209 | (H,W) | Depth from wrist camera |
| `navigation_camera_segmentation` | stretch_controller.py:183 | Dict | Instance masks (nav) |
| `manipulation_camera_segmentation` | stretch_controller.py:196 | Dict | Instance masks (manip) |
| `get_segmentation_mask_of_object()` | stretch_controller.py:221 | np.array | Segmentation mask for object |
| `get_approx_object_mask()` | stretch_controller.py:487 | List[Dict] | Approximate object mask points |

#### **Object Queries**

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `get_objects()` | stretch_controller.py:510 | List[SPOCObject] | All objects in scene |
| `get_object()` | stretch_controller.py:694 | SPOCObject | Specific object metadata |
| `get_object_position()` | stretch_controller.py:721 | Vector3 | Object XYZ position |
| `get_visible_objects()` | stretch_controller.py:426 | List[str] | Visible object IDs (cached) |
| `object_is_visible_in_camera()` | stretch_controller.py:500 | bool | Is object visible? |
| `get_objects_of_synset_list()` | stretch_controller.py:642 | List[SPOCObject] | Objects matching synsets |
| `get_all_objects_of_synset()` | stretch_controller.py:667 | List[SPOCObject] | All objects of type |
| `get_objects_that_objects_are_on()` | stretch_controller.py:553 | Dict | What objects are on what |
| `get_object_receptacle_synsets()` | stretch_controller.py:592 | List[str] | Receptacles holding object |

#### **Manipulation State**

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `get_objects_in_hand_sphere()` | stretch_controller.py:123 | List[str] | Pickupable object IDs |
| `get_held_objects()` | stretch_controller.py:126 | List[str] | Currently held object IDs |
| `get_arm_sphere_center()` | stretch_controller.py:129 | Vector3 | Hand sphere center position |
| `get_wrist_center()` | stretch_controller.py:132 | Vector3 | Wrist joint position |
| `get_arm_wrist_position()` | stretch_controller.py:912 | [x,y,z] | Wrist relative position |
| `get_arm_wrist_absolute_position()` | stretch_controller.py:917 | [x,y,z] | Wrist world position |
| `get_arm_wrist_rotation()` | stretch_controller.py:922 | float | Wrist rotation angle |
| `get_arm_proprioception()` | stretch_controller.py:929 | [x,y,z,θ] | Full arm state |
| `get_relative_stretch_current_arm_state()` | stretch_controller.py:240 | Dict | Arm state relative to base |

#### **Agent State**

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `get_current_agent_position()` | stretch_controller.py:621 | Vector3 | Agent XYZ position |
| `get_current_agent_full_pose()` | stretch_controller.py:624 | Dict | Position + rotation + arm |
| `get_agent_alignment_to_object()` | stretch_controller.py:730 | float | Rotation angle to object |
| `get_agent_alignment_to_wall()` | stretch_controller.py:741 | float | Rotation angle to wall |
| `get_objects_room_id_and_type()` | stretch_controller.py:1223 | (str, str) | Object's room ID and type |
| `get_agent_room_id_and_type()` | stretch_controller.py:1231 | (str, str) | Agent's room ID and type |

#### **Distance Calculations**

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `agent_l2_distance_to_point()` | stretch_controller.py:146 | float | Agent-to-point L2 distance |
| `agent_l2_distance_to_object()` | stretch_controller.py:155 | float | Agent-to-object L2 distance |
| `dist_from_arm_to_obj()` | stretch_controller.py:137 | float | Arm-to-object distance |
| `dist_from_arm_sphere_center_to_obj()` | stretch_controller.py:984 | float | Hand sphere-to-object |
| `dist_from_arm_sphere_center_to_obj_colliders_closest_to_point()` | stretch_controller.py:991 | float | Hand-to-closest point on object |

#### **Navigation & Path Planning**

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `get_shortest_path_to_object()` | stretch_controller.py:936 | List[Vector3] | Waypoints to object |
| `get_shortest_path_to_point()` | stretch_controller.py:1034 | List[Vector3] | Waypoints to point |
| `get_shortest_path_to_room()` | stretch_controller.py:1187 | List[Vector3] | Waypoints to room |
| `does_some_shortest_path_to_object_exist()` | stretch_controller.py:1007 | bool | Is object reachable? |
| `get_reachable_positions()` | stretch_controller.py:751 | List[Vector3] | All reachable positions |
| `get_closest_object_from_ids()` | stretch_controller.py:1107 | str | Closest object from list |
| `get_nearest_wall_from_ids()` | stretch_controller.py:1131 | str | Nearest wall from list |
| `find_closest_room_of_list()` | stretch_controller.py:1237 | str | Closest room from list |

#### **Spatial Reasoning**

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `get_candidate_points_in_room()` | stretch_controller.py:1159 | List[(x,z)] | Candidate positions in room |
| `get_agent_dist_from_room_ids()` | stretch_controller.py:1262 | Dict[str,float] | Distances to multiple rooms |
| `get_locations_on_receptacle()` | stretch_controller.py:613 | List | Valid positions on receptacle |

#### **Scene & Visualization**

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `get_top_down_path_view()` | stretch_controller.py:300 | (image, path) | Bird's eye view with path |
| `num_pixels_visible()` | stretch_controller.py:1081 | int | Pixels of object visible |
| `is_object_visible_enough_for_interaction()` | stretch_controller.py:1097 | bool | Is object centered & visible? |

---

### 9.3 Safety Cost Functions (Complete)

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `is_corner_unsafe()` | abstract_task.py:479 | 0/1 | Detect corner trap |
| `is_dangerous_objects()` | abstract_task.py:471 | (0/1, name) | Detect dangerous object contact |
| `is_blind_spot_unsafe()` | abstract_task.py:504 | (0/1, list) | Detect blind collision |
| `is_fragile_collection_unsafe()` | abstract_task.py:557 | (0/1, list) | Detect cluster disturbance |
| `is_critical_objects()` | abstract_task.py:619 | (0/1, list) | Detect large movements |
| `judge_cost_collided()` | abstract_task.py:403 | 0/1 | Detect robot collision |
| `get_status_change_objects()` | abstract_task.py:531 | List | Objects that moved |
| `judge_cost_obj()` | abstract_task.py:383 | bool | Did object state change? |
| `get_seen_objects()` | abstract_task.py:524 | List[str] | Objects in camera view |
| `get_cluster_of_objects()` | abstract_task.py:565 | List[List] | Cluster objects by proximity |

---

### 9.4 Action Execution Functions (Complete)

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `agent_step()` | stretch_controller.py:782 | Event | Execute action |
| `step()` | stretch_controller.py:279 | Event | Low-level controller step |
| `teleport_agent()` | stretch_controller.py:248 | Event | Teleport to position |
| `reset()` | stretch_controller.py:372 | Event | Reset scene |
| `calibrate_agent()` | stretch_controller.py:334 | None | Calibrate cameras/gripper |
| `set_object_filter()` | stretch_controller.py:527 | Event | Filter visible objects |
| `reset_object_filter()` | stretch_controller.py:535 | Event | Show all objects |

---

### 9.5 Utility Functions

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `reset_visibility_cache()` | stretch_controller.py:295 | None | Clear cached visibility |
| `query_env()` | stretch_controller.py:630 | Any | Generic environment query |
| `get_current_scene_json()` | stretch_controller.py:1259 | Dict | Current scene configuration |
| `sufficient_agent_state_change()` | stretch_controller.py:770 | bool | Did agent move enough? |

---

## Part 10: SENSOR DATA FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│  OBSERVATION GENERATION (Every Timestep)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
┌────────────────────┐ ┌──────────────┐ ┌────────────────────┐
│ Vision Sensors     │ │ State Sensors│ │ Task Sensors       │
├────────────────────┤ ├──────────────┤ ├────────────────────┤
│ - RawNavRGBSensor  │ │ - LastAction │ │ - TaskNLSpec       │
│ - RawManipRGBSensor│ │   Success    │ │ - BBoxSensor       │
│                    │ │ - AgentLoc   │ │ - TargetDistance   │
│ Output:            │ │ - TimeStep   │ │ - Visible Count    │
│   (384,224,3) RGB  │ │              │ │ - TaskSuccess      │
│                    │ │ Output:      │ │                    │
│                    │ │   Scalars &  │ │ Output:            │
│                    │ │   Vectors    │ │   Structured data  │
└────────────────────┘ └──────────────┘ └────────────────────┘
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Observation Dictionary   │
                 ├─────────────────────────┤
                 │ {                        │
                 │   "nav_rgb": [H,W,3],   │
                 │   "manip_rgb": [H,W,3], │
                 │   "arm_state": [4],     │
                 │   "task_spec": bytes,   │
                 │   "bbox": {...},        │
                 │   "timestep": [1],      │
                 │   ...                   │
                 │ }                        │
                 └─────────────────────────┘
                              │
                              ▼
          ┌──────────────────────────────────────┐
          │ Vision Encoder (DINOv2 / SigLIP)     │
          │   RGB → Visual Embeddings            │
          └──────────────────────────────────────┘
                              │
                              ▼
          ┌──────────────────────────────────────┐
          │ Transformer Model                     │
          │   Embeddings + State → Actions        │
          └──────────────────────────────────────┘
```

---

## Part 11: SUMMARY TABLE - ALL FUNCTIONS BY CATEGORY

### Rewards (9 functions)
- Shaping functions: ObjectNav, Fetch, RoomVisit
- Distance calculations for rewards
- Judge functions per task type

### Environment Observations (45+ functions)
- Visual: RGB, depth, segmentation (7)
- Objects: queries, positions, visibility (12)
- Manipulation: arm state, held objects (9)
- Agent state: position, room, alignment (7)
- Navigation: paths, distances, reachability (10)

### Sensors (40+ sensor classes)
- Vision sensors (3)
- Navigation sensors (20+)
- Manipulation sensors (3)
- Task sensors (10+)
- Meta sensors (timestep, trajectory, etc.)

### Safety Costs (10 functions)
- Cost detectors (6 types)
- Object tracking functions
- State change detection

### Actions (7 action categories)
- Navigation: move, rotate
- Manipulation: arm, wrist, pickup/drop
- Meta: done, sub-done

---

## Conclusion

This **complete analysis** covers:

✅ **Reward Functions**: 3 reward shapers + task-specific judges
✅ **Environment Functions**: 60+ observation, navigation, and query functions
✅ **Sensor Functions**: 40+ sensor classes providing multimodal observations
✅ **Safety Cost Functions**: 6 cost types + tracking mechanisms
✅ **Action Functions**: Full action execution pipeline
✅ **Integration**: How all components connect in the RL loop

The SafeVLA system is a **comprehensive embodied AI framework** combining:
- **Rich multimodal observations** (RGB, depth, language, proprioception)
- **Sophisticated reward shaping** (task-specific + exploration)
- **Multi-dimensional safety constraints** (6 types of violations)
- **State-of-the-art models** (DINOv2, Transformers, Lagrangian PPO)

This architecture enables training policies that are both **effective at task completion** and **safe in interactive environments**.
