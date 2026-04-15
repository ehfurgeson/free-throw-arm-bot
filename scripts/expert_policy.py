import numpy as np

def compute_expert_action(obs):
    # ---- Tunable parameters (edit while rendering) ----
    hover_height = 0.16                 # meters above ball center for hover
    hover_xy_tol = 0.012                # XY alignment tolerance (m)
    hover_z_tol = 0.02                  # Hover altitude tolerance (m)
    descend_z_tol = 0.008               # Z tolerance before grasp phase (m)
    grasp_wait_steps = 12               # Must be >= 10 for physics settle
    windup_x_offset = -0.30             # Backward offset from grasp point (m)
    windup_z_offset = -0.08             # Slight downward offset from grasp point (m)
    windup_pos_tol = 0.015              # Wind-up position tolerance (m)
    release_x_threshold = 1.50          # Open gripper once gripper_x >= this
    reset_jump_threshold = 0.15         # Detect episode reset by object jump

    # Per-phase gains / speed limits
    hover_kp = 6.0
    descend_kp = 8.0
    windup_kp = 7.0
    hover_speed_cap = 0.35
    descend_speed_cap = 0.25
    windup_speed_cap = 0.60

    o = obs["observation"]
    gripper_pos = o[0:3]
    object_pos = o[3:6]
    object_rel_pos = o[6:9]
    finger_state = o[9:11]  # opening widths, used for optional grasp checks/debug
    _ = finger_state  # keeps explicit index use while avoiding lints for now

    # Initialize persistent controller state on first call.
    if not hasattr(compute_expert_action, "_state"):
        compute_expert_action._state = "hover"
        compute_expert_action._grasp_wait_counter = 0
        compute_expert_action._follow_counter = 0
        compute_expert_action._windup_target = None
        compute_expert_action._prev_object_pos = object_pos.copy()

    # Detect episode resets robustly (object teleports on env.reset()).
    if (
        np.linalg.norm(object_pos - compute_expert_action._prev_object_pos)
        > reset_jump_threshold
    ):
        compute_expert_action._state = "hover"
        compute_expert_action._grasp_wait_counter = 0
        compute_expert_action._follow_counter = 0
        compute_expert_action._windup_target = None
    compute_expert_action._prev_object_pos = object_pos.copy()

    # Helper for bounded Cartesian velocity commands.
    def _bounded_xyz(delta, kp, cap):
        vel = kp * delta
        vel = np.clip(vel, -cap, cap)
        return np.clip(vel, -1.0, 1.0)

    state = compute_expert_action._state
    action = np.zeros(4, dtype=np.float32)

    if state == "hover":
        hover_target = object_pos + np.array([0.0, 0.0, hover_height], dtype=np.float32)
        delta = hover_target - gripper_pos
        action_xyz = _bounded_xyz(delta, hover_kp, hover_speed_cap)
        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], -1.0], dtype=np.float32)

        xy_ok = np.linalg.norm(object_rel_pos[0:2]) <= hover_xy_tol
        z_ok = abs(delta[2]) <= hover_z_tol
        if xy_ok and z_ok:
            compute_expert_action._state = "descend"

    elif state == "descend":
        # Strictly descend in Z; keep XY frozen for stable top-down approach.
        equator_target_z = object_pos[2]
        dz = equator_target_z - gripper_pos[2]
        z_vel = np.clip(descend_kp * dz, -descend_speed_cap, descend_speed_cap)
        action = np.array([0.0, 0.0, z_vel, -1.0], dtype=np.float32)

        if abs(dz) <= descend_z_tol:
            compute_expert_action._state = "grasp_wait"
            compute_expert_action._grasp_wait_counter = 0

    elif state == "grasp_wait":
        # Critical settle window: stay fully still while squeezing closed.
        action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        compute_expert_action._grasp_wait_counter += 1
        if compute_expert_action._grasp_wait_counter >= grasp_wait_steps:
            compute_expert_action._state = "windup"
            compute_expert_action._windup_target = gripper_pos + np.array(
                [windup_x_offset, 0.0, windup_z_offset], dtype=np.float32
            )

    elif state == "windup":
        target = compute_expert_action._windup_target
        delta = target - gripper_pos
        action_xyz = _bounded_xyz(delta, windup_kp, windup_speed_cap)
        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], 1.0], dtype=np.float32)

        if np.linalg.norm(delta) <= windup_pos_tol:
            compute_expert_action._state = "whip"

    elif state == "whip":
        # Maximum acceleration toward hoop direction while keeping grasp.
        action = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        if gripper_pos[0] >= release_x_threshold:
            compute_expert_action._state = "release"

    elif state == "release":
        # Instant release while preserving arm velocity profile.
        action = np.array([1.0, 0.0, 1.0, -1.0], dtype=np.float32)
        compute_expert_action._follow_counter += 1
        if compute_expert_action._follow_counter > 6:
            compute_expert_action._state = "follow_through"

    else:  # follow_through
        action = np.array([0.8, 0.0, 0.4, -1.0], dtype=np.float32)

    return np.clip(action, -1.0, 1.0).astype(np.float32)