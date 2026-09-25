"""Path-following course benchmarks for locomotion policies: one
interpretable score per named scenario. Run:
    ./run.sh courses --run runs/<name>

Where battery.py measures gait quality under *open-loop* velocity commands,
this measures whether the robot can actually walk a path. Each scenario is a
geometric course (waypoints) plus a commanded speed per segment; a frozen
pure-pursuit follower turns the robot's pose into the [vx, vy, wz, height]
command the policy tracks, and the scenario scores how faithfully the base
followed the course.

Deliberately NOT a sweep: every scenario varies exactly one thing off the
nominal (flat floor, model friction, 0.5 m/s, no disturbance), so a bad row
has a single interpretation. The catalogue lives in `families/`, one module
per family -- add a course by appending to the right family's COURSES:

  families/geometry_paths.py   straight_10m, arc_r3_90deg, circle_r2,
                               circle_r075, figure_eight_r15, square_3m,
                               slalom_05m, u_turn
  families/speed.py            straight_slow, straight_fast, circle_r2_fast,
                               speed_steps_straight
  families/floor.py            straight_slippery, circle_r1_slippery
  families/disturbance.py      straight_push, straight_push_fast
  families/spin.py             spin_left, spin_right, spin_slow, spin_fast --
                               rotate-in-place rows (SpinCourse): a held pure
                               wz command, completion = one full rotation,
                               scored on rotation-rate tracking, positional
                               drift, height, smoothness

The package splits by role: `geometry` (polyline builders), `spec` (the
Course dataclass and shared nominals), `follower` (the FROZEN pure-pursuit
constants and Pursuit), `rollout` (one course x seed in the measurement
env), `scoring` (physical normalizers -> sub-scores -> min), `runner`
(driver, plots, video, CLI).

Score. Each scenario's score is the WEAKEST of five sub-scores, each of them
a measured error divided into a *physical* reference -- so there is no
calibration file, no hand-tuned good/bad cutoff, and nothing to re-baseline:

    tracking    STANCE_HALFWIDTH_M / RMS cross-track error
    speed       mean commanded speed / RMS along-path speed error
    height      NOMINAL_HEIGHT_M / RMS height error
    grip        base distance travelled / total foot-slip distance
    smoothness  1 / vibration_index (>5 Hz joint-velocity power fraction)

A sub-score of 1.0 means that axis is exactly as bad as its physical
reference (off the path by a full stance half-width; speed error as large as
the command; height error equal to the whole standing height; feet sliding as
far as the body moved; all joint-velocity power above 5 Hz). Higher is always
better and there is no ceiling. `min` rather than a weighted average so one
bad axis cannot be diluted by four good ones -- every sub-score is reported
alongside the score, so the binding one is always visible.

Completion is a gate, not a sub-score: a fall or an unfinished course scores
0 and the rollout is abandoned there. (It has to be a gate -- a sub-score
capped at 1.0 would cap the whole `min` and destroy the unbounded scale.)

Each scenario runs `--seeds` rollouts (default 8); the table reports the
median and the worst seed, so a policy that only sometimes falls cannot pass
by luck.

Cost. The rollout is a single-env Python loop (like battery.rollout), so this
is minutes, not seconds: measured ~30 s per 2600-step course per seed on CPU,
i.e. roughly half an hour for all 20 scenarios x 8 seeds and up to an hour if
most of them time out instead of finishing. Use `--only NAME... --seeds 1`
while iterating and the full set for a verdict.
"""

from wojtek_rl.courses.families import course_catalogue
from wojtek_rl.courses.follower import (
    GOAL_MIN_PROGRESS_M,
    GOAL_RADIUS_M,
    K_YAW_SPIN,
    LOOKAHEAD_M,
    PROGRESS_WINDOW_M,
    RESAMPLE_DS_M,
    SPIN_ENTER_RAD,
    SPIN_EXIT_RAD,
    YAW_MAX,
    Pursuit,
)
from wojtek_rl.courses.geometry import LEAD_IN_M, arc, join, lead_in, line, sine_slalom
from wojtek_rl.courses.rollout import (
    MAX_COURSE_STEPS,
    STAND_STEPS,
    TIME_FACTOR,
    course_rollout,
    friction_geom_ids,
    spin_rollout,
    step_budget,
)
from wojtek_rl.courses.runner import (
    main,
    print_table,
    run_courses,
    write_path_plot,
)
from wojtek_rl.courses.scoring import (
    MIN_SCORE_STEPS,
    NOMINAL_HEIGHT_M,
    STANCE_HALFWIDTH_M,
    SUBSCORE_CAP,
    VIBRATION_CUTOFF_HZ,
    aggregate,
    seed_result,
    spin_seed_result,
)
from wojtek_rl.courses.spec import (
    HEIGHT_CMD,
    NOMINAL_SPEED,
    PUSH_VEL,
    SLIPPERY_FRICTION,
    Course,
    Scenario,
    SpinCourse,
)

__all__ = [
    "GOAL_MIN_PROGRESS_M", "GOAL_RADIUS_M", "HEIGHT_CMD", "K_YAW_SPIN",
    "LEAD_IN_M", "LOOKAHEAD_M", "MAX_COURSE_STEPS", "MIN_SCORE_STEPS",
    "NOMINAL_HEIGHT_M", "NOMINAL_SPEED", "PROGRESS_WINDOW_M", "PUSH_VEL",
    "RESAMPLE_DS_M", "SLIPPERY_FRICTION", "SPIN_ENTER_RAD", "SPIN_EXIT_RAD",
    "STANCE_HALFWIDTH_M", "STAND_STEPS", "SUBSCORE_CAP", "TIME_FACTOR",
    "VIBRATION_CUTOFF_HZ", "YAW_MAX", "Course", "Pursuit", "Scenario",
    "SpinCourse", "aggregate", "arc", "course_catalogue", "course_rollout",
    "friction_geom_ids", "join", "lead_in", "line", "main", "print_table",
    "run_courses", "seed_result", "sine_slalom", "spin_rollout",
    "spin_seed_result", "step_budget", "write_path_plot",
]
