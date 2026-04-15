import numpy as np

def compute_expert_action(obs):
    # ---- Tunable parameters (edit while rendering) ----
    preopen_steps = 12                  # stay still and command full-open before any motion
    approach_open_min_steps = 4         # keep opening while moving toward hover target
    approach_open_height_bias = 0.08    # extra Z bias during approach to avoid touching ball
    finger_open_sum_thresh = 0.012      # optional open confirmation, non-blocking with timeout
    hover_height = 0.16                 # meters above ball center for hover
    hover_xy_tol = 0.012                # XY alignment tolerance (m)
    hover_z_tol = 0.02                  # Hover altitude tolerance (m)
    open_wait_steps = 2                 # short hold-open at hover before descend
    ball_radius = 0.04                  # sphere radius from task definition
    descend_grasp_z_offset = -0.004     # go slightly below center so fingers wrap around sphere
    descend_xy_abort_tol = 0.018        # if drifted too far in XY, re-hover before contact
    descend_z_tol = 0.008               # Z tolerance before grasp phase (m)
    grasp_xy_tol = 0.010                # require tight XY centering before closing
    min_descend_steps = 8               # enforce some descend time before any close attempt
    grasp_wait_steps = 16               # Must be >= 10 for physics settle
    grip_lock_steps = 8                 # extra hard-close hold before lift check
    lift_check_steps = 10               # lift while closed to confirm object is captured
    lift_check_z_vel = 0.22             # upward velocity command during lift check
    lift_success_delta_z = 0.03         # object must rise this much to confirm grasp
    post_grasp_min_lift = 0.06          # keep gripper at least this much above ball center after grasp
    windup_x_offset = -0.30             # Backward offset from grasp point (m)
    windup_z_offset = 0.06              # Backward + upward wind-up (never down after grasp)
    windup_pos_tol = 0.015              # Wind-up position tolerance (m)
    release_x_threshold = 1.50          # Open gripper once gripper_x >= this
    reset_jump_threshold = 0.15         # Detect episode reset by object jump

    # Per-phase gains / speed limits
    hover_kp = 10.0
    descend_kp = 8.0
    windup_kp = 7.0
    hover_speed_cap = 1.0
    descend_speed_cap = 0.25
    descend_xy_kp = 5.0
    descend_xy_cap = 0.08
    windup_speed_cap = 0.60

    o = obs["observation"]
    gripper_pos = o[0:3]
    object_pos = o[3:6]
    object_rel_pos = o[6:9]
    finger_state = o[9:11]
    finger_open_sum = float(finger_state[0] + finger_state[1])

    # Initialize persistent controller state on first call.
    if not hasattr(compute_expert_action, "_state"):
        compute_expert_action._state = "pre_open"
        compute_expert_action._preopen_counter = 0
        compute_expert_action._approach_open_counter = 0
        compute_expert_action._grasp_wait_counter = 0
        compute_expert_action._grip_lock_counter = 0
        compute_expert_action._lift_check_counter = 0
        compute_expert_action._grasp_object_z_ref = object_pos[2]
        compute_expert_action._open_wait_counter = 0
        compute_expert_action._descend_counter = 0
        compute_expert_action._follow_counter = 0
        compute_expert_action._windup_target = None
        compute_expert_action._post_grasp_min_z = gripper_pos[2]
        compute_expert_action._prev_object_pos = object_pos.copy()
        compute_expert_action._prev_finger_open_sum = finger_open_sum
        compute_expert_action._last_gripper_cmd = -1.0
        compute_expert_action._open_sign = -1.0  # will auto-calibrate if env is inverted
        compute_expert_action._open_sign_locked = False

    # Detect episode resets robustly (object teleports on env.reset()).
    if (
        np.linalg.norm(object_pos - compute_expert_action._prev_object_pos)
        > reset_jump_threshold
    ):
        compute_expert_action._state = "pre_open"
        compute_expert_action._preopen_counter = 0
        compute_expert_action._approach_open_counter = 0
        compute_expert_action._grasp_wait_counter = 0
        compute_expert_action._grip_lock_counter = 0
        compute_expert_action._lift_check_counter = 0
        compute_expert_action._grasp_object_z_ref = object_pos[2]
        compute_expert_action._open_wait_counter = 0
        compute_expert_action._descend_counter = 0
        compute_expert_action._follow_counter = 0
        compute_expert_action._windup_target = None
        compute_expert_action._post_grasp_min_z = gripper_pos[2]
        compute_expert_action._prev_finger_open_sum = finger_open_sum
        compute_expert_action._last_gripper_cmd = -1.0
        compute_expert_action._open_sign_locked = False
    compute_expert_action._prev_object_pos = object_pos.copy()

    state = compute_expert_action._state

    # Auto-calibrate gripper sign only during early opening phases.
    # Freeze it once we enter closing/throw phases to prevent sign flip oscillation.
    finger_delta = finger_open_sum - compute_expert_action._prev_finger_open_sum
    prev_cmd = compute_expert_action._last_gripper_cmd
    calibration_states = {"pre_open", "approach_open", "hover", "descend"}
    allow_calibration = (
        (not compute_expert_action._open_sign_locked) and (state in calibration_states)
    )
    if allow_calibration and abs(prev_cmd) > 0.5 and abs(finger_delta) > 1e-4:
        # If prior command increased opening width, that command is the open sign.
        compute_expert_action._open_sign = float(prev_cmd if finger_delta > 0.0 else -prev_cmd)
    open_cmd = float(compute_expert_action._open_sign)
    close_cmd = float(-compute_expert_action._open_sign)

    # Helper for bounded Cartesian velocity commands.
    def _bounded_xyz(delta, kp, cap):
        vel = kp * delta
        vel = np.clip(vel, -cap, cap)
        return np.clip(vel, -1.0, 1.0)

    def _enforce_post_grasp_no_down(action_vec):
        # Never command negative Z after successful grasp/lift.
        action_vec[2] = max(0.0, float(action_vec[2]))
        # If we're below protected height, force a small upward correction.
        if gripper_pos[2] < compute_expert_action._post_grasp_min_z:
            action_vec[2] = max(action_vec[2], 0.35)
        return action_vec

    action = np.zeros(4, dtype=np.float32)

    if state == "pre_open":
        # Fully open while completely still before any approach motion.
        action = np.array([0.0, 0.0, 0.0, open_cmd], dtype=np.float32)
        compute_expert_action._preopen_counter += 1
        fingers_open_enough = float(finger_state[0] + finger_state[1]) >= finger_open_sum_thresh
        if (
            compute_expert_action._preopen_counter >= preopen_steps
            and (fingers_open_enough or compute_expert_action._preopen_counter >= (preopen_steps + 12))
        ):
            compute_expert_action._state = "approach_open"
            compute_expert_action._approach_open_counter = 0

    elif state == "approach_open":
        # Open while actively moving toward above-ball hover point.
        hover_target = object_pos + np.array(
            [0.0, 0.0, hover_height + approach_open_height_bias], dtype=np.float32
        )
        delta = hover_target - gripper_pos
        action_xyz = _bounded_xyz(delta, hover_kp, hover_speed_cap)
        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], open_cmd], dtype=np.float32)

        compute_expert_action._approach_open_counter += 1
        reached_hover = (
            np.linalg.norm(object_rel_pos[0:2]) <= hover_xy_tol
            and abs(delta[2]) <= hover_z_tol
        )
        finger_open_enough = float(finger_state[0] + finger_state[1]) >= finger_open_sum_thresh
        min_open_time_met = (
            compute_expert_action._approach_open_counter >= approach_open_min_steps
        )
        if reached_hover and min_open_time_met and (
            finger_open_enough or compute_expert_action._approach_open_counter >= (approach_open_min_steps + 10)
        ):
            compute_expert_action._state = "hover"
            compute_expert_action._open_wait_counter = 0

    elif state == "hover":
        hover_target = object_pos + np.array([0.0, 0.0, hover_height], dtype=np.float32)
        delta = hover_target - gripper_pos
        action_xyz = _bounded_xyz(delta, hover_kp, hover_speed_cap)
        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], open_cmd], dtype=np.float32)

        xy_ok = np.linalg.norm(object_rel_pos[0:2]) <= hover_xy_tol
        z_ok = abs(delta[2]) <= hover_z_tol
        if xy_ok and z_ok:
            compute_expert_action._open_wait_counter += 1
        else:
            compute_expert_action._open_wait_counter = 0

        if compute_expert_action._open_wait_counter >= open_wait_steps:
            compute_expert_action._state = "descend"
            compute_expert_action._descend_counter = 0

    elif state == "descend":
        # Controlled descend: keep tiny XY correction and stop slightly above center.
        compute_expert_action._descend_counter += 1
        xy_error = object_rel_pos[0:2]
        if np.linalg.norm(xy_error) > descend_xy_abort_tol:
            compute_expert_action._state = "hover"
            compute_expert_action._open_wait_counter = 0
            action = np.array([0.0, 0.0, 0.0, open_cmd], dtype=np.float32)
            return action

        grasp_target_z = object_pos[2] + descend_grasp_z_offset
        dz = grasp_target_z - gripper_pos[2]
        xy_vel = np.clip(descend_xy_kp * xy_error, -descend_xy_cap, descend_xy_cap)
        z_vel = np.clip(descend_kp * dz, -descend_speed_cap, descend_speed_cap)
        action = np.array([xy_vel[0], xy_vel[1], z_vel, open_cmd], dtype=np.float32)

        xy_ready = np.linalg.norm(xy_error) <= grasp_xy_tol
        z_ready = abs(dz) <= descend_z_tol
        descend_ready = compute_expert_action._descend_counter >= min_descend_steps
        if xy_ready and z_ready and descend_ready:
            compute_expert_action._state = "grasp_wait"
            compute_expert_action._open_sign_locked = True
            compute_expert_action._grasp_wait_counter = 0
            compute_expert_action._grasp_object_z_ref = object_pos[2]

    elif state == "grasp_wait":
        # Critical settle window: stay fully still while squeezing closed.
        action = np.array([0.0, 0.0, 0.0, close_cmd], dtype=np.float32)
        compute_expert_action._grasp_wait_counter += 1
        if compute_expert_action._grasp_wait_counter >= grasp_wait_steps:
            compute_expert_action._state = "grip_lock"
            compute_expert_action._grip_lock_counter = 0

    elif state == "grip_lock":
        # Keep squeezing shut for a few extra steps to maximize grasp force.
        action = np.array([0.0, 0.0, 0.0, close_cmd], dtype=np.float32)
        compute_expert_action._grip_lock_counter += 1
        if compute_expert_action._grip_lock_counter >= grip_lock_steps:
            compute_expert_action._state = "lift_check"
            compute_expert_action._lift_check_counter = 0
            compute_expert_action._grasp_object_z_ref = object_pos[2]

    elif state == "lift_check":
        # Lift upward while closed; only continue if object rises with gripper.
        action = np.array([0.0, 0.0, lift_check_z_vel, close_cmd], dtype=np.float32)
        compute_expert_action._lift_check_counter += 1
        lifted_enough = (object_pos[2] - compute_expert_action._grasp_object_z_ref) >= lift_success_delta_z
        timed_out = compute_expert_action._lift_check_counter >= lift_check_steps
        if lifted_enough or timed_out:
            compute_expert_action._post_grasp_min_z = max(
                gripper_pos[2],
                object_pos[2] + post_grasp_min_lift,
            )
            compute_expert_action._state = "windup"
            compute_expert_action._windup_target = gripper_pos + np.array(
                [windup_x_offset, 0.0, windup_z_offset], dtype=np.float32
            )

    elif state == "windup":
        target = compute_expert_action._windup_target
        delta = target - gripper_pos
        action_xyz = _bounded_xyz(delta, windup_kp, windup_speed_cap)
        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], close_cmd], dtype=np.float32)
        action = _enforce_post_grasp_no_down(action)

        if np.linalg.norm(delta) <= windup_pos_tol:
            compute_expert_action._state = "whip"

    elif state == "whip":
        # Maximum acceleration toward hoop direction while keeping grasp.
        action = np.array([1.0, 0.0, 1.0, close_cmd], dtype=np.float32)
        action = _enforce_post_grasp_no_down(action)
        if gripper_pos[0] >= release_x_threshold:
            compute_expert_action._state = "release"

    elif state == "release":
        # Instant release while preserving arm velocity profile.
        action = np.array([1.0, 0.0, 1.0, open_cmd], dtype=np.float32)
        action = _enforce_post_grasp_no_down(action)
        compute_expert_action._follow_counter += 1
        if compute_expert_action._follow_counter > 6:
            compute_expert_action._state = "follow_through"

    else:  # follow_through
        action = np.array([0.8, 0.0, 0.4, open_cmd], dtype=np.float32)
        action = _enforce_post_grasp_no_down(action)

    compute_expert_action._last_gripper_cmd = float(action[3])
    compute_expert_action._prev_finger_open_sum = finger_open_sum

    return np.clip(action, -1.0, 1.0).astype(np.float32)