from dataclasses import dataclass


@dataclass(frozen=True)
class AutonomyConfig:
  loop_hz: float = 20.0
  max_forward_axis: float = 0.45
  max_turn_axis: float = 0.6
  stop_obstacle_distance_m: float = 0.7
  creep_target_distance_m: float = 1.5
  acquire_sound_file: str = "selfdrive/assets/sounds/engage.wav"
  lose_sound_file: str = "selfdrive/assets/sounds/disengage.wav"
  gaze_attend_threshold: float = 0.65
  eye_closed_threshold: float = 0.45
  attention_hold_s: float = 0.30
  inattentive_hold_s: float = 0.55
