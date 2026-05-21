STATE_KEYS = [
    "vx",
    "vy",
    "roll",
    "pitch",
    "yaw",
    "p",
    "q",
    "r",
    "omega_fl",
    "omega_fr",
    "omega_rl",
    "omega_rr",
]

CONTROL_KEYS = [
    "sw_angle",
    "steer_cmd",
    "throttle_cmd",
    "brake_cmd",
    "tau_drv_obs_fl",
    "tau_drv_obs_fr",
    "tau_drv_obs_rl",
    "tau_drv_obs_rr",
    "tau_brk_obs_fl",
    "tau_brk_obs_fr",
    "tau_brk_obs_rl",
    "tau_brk_obs_rr",
]

WHEEL_KEYS = ["fl", "fr", "rl", "rr"]

CONTEXT_KEYS = [
    "wheelbase",
    "track_front",
    "track_rear",
    "wheel_radius",
    "steering_ratio_nominal",
    "mass_nominal",
    "Ix_nominal",
    "Iy_nominal",
    "Iz_nominal",
    "cg_x_nominal",
    "cg_z_nominal",
    "tau_steer_nominal",
    "drive_is_fwd",
    "drive_is_rwd",
    "drive_is_awd",
    "brake_is_hydraulic_split",
    "brake_is_brake_by_wire",
]
