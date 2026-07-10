# Overnight locomotion iterations (2026-07-05/06) — fbb_loco v1→v9

One policy: stand ↔ 4-beat walk ↔ diagonal trot scheduled by commanded
speed, plus commanded stance height (0.09–0.17 m), no-IMU actor obs.
Each version: 300M PPO steps (v9: 600M) on a 4×H100 node (~15 min),
scored by the fixed battery (`./run.sh battery`) — push-free rollouts:
three stand→1.0 m/s ramps (low/mid/tall stance), walk→trot switch,
stand with height steps.

## Scoreboard (push-free battery, hold-window stand metrics)

| ver | vel err (avg) | slip (avg) | stand h err | hold qvel | settle | walk corr | trot corr | falls |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.075 | 0.167 | 14.2mm | 0.082 | [150, 4] | 0.87¹ | 0.90 | 0 |
| v2 | 0.097 | 0.203 | 7.0mm | 0.115 | [5, 4] | -0.03 | 0.91 | 0 |
| v3 | 0.089 | 0.227 | 8.1mm | 0.147 | [7, 4] | 0.02 | 0.92 | 0 |
| v4 | 0.082 | 0.246 | 20.7mm | 0.160 | [150, 150] | 0.05 | 0.91 | 0 |
| v5 | 0.116 | 0.319 | 7.4mm | 0.042 | [6, 4] | 0.09 | 0.85 | 0 |
| v6 | 0.073 | 0.253 | 8.5mm | 0.145 | [150, 4] | -0.02 | 0.94 | 0 |
| v7 | 0.095 | 0.181 | 6.3mm | 0.101 | [7, 14] | 0.06 | 0.90 | 0 |
| **v8** | **0.080** | **0.230** | **6.8mm** | **0.128** | **[6, 14]** | **0.08** | **0.94** | **0** |
| v9 | 0.114 | 0.204 | 6.3mm | 0.159 | [6, 4] | -0.05 | 0.90 | 0 |

¹ v1's "walk" was a soft trot (diag-corr should be ~0 for a 4-beat walk);
its slip/vel numbers are not comparable — it never expressed the gait.

**Winner: v8** (`runs/fbb_loco_v8`, preset `locomotion_v8`) — near-best
velocity, v7-grade heights/settling, cleanest trot, true walk, no falls.

## Changelog & lessons

- **v2** duty-aware gait schedule (walk 70% stance) + `contact_match`
  reward; stronger stand/height terms. Fixed the fake walk; height err
  halved; velocity regressed.
- **v3** clock cap 3.6 Hz + tracking 2.0 + velocity-damped `stand_still`.
  Recovered most velocity.
- **v4** `stand_still` −2.5, zero_prob 0.25. Lesson: heavy anchor weight
  makes the policy obey the approximate height→ctrl table over the real
  height command (stand err 8→21 mm). Reverted.
- **v5** EMA action filter 0.5. Lesson: filter lag costs velocity and
  slip everywhere; the "standing tremble" it targeted turned out to be a
  **metric artifact** (height-transition motion averaged into the stand
  qvel; true hold stillness ≈0.15 rad/s, confirmed noise-on/noise-off).
  Battery now measures hold windows + settle time separately.
- **v6** v3 base + zero_prob 0.25, filter off. Best velocity of the fleet.
- **v7** slip −0.5, height 2.5. Slip −25%, best heights, crouch settling
  fixed — but velocity −20% (caution). A speed↔precision frontier.
- **v8** frontier midpoint (slip −0.35, tracking 2.5). Balanced; chosen.
- **v9** v8 × 600M steps. Lesson: training longer slides further down the
  same reward trade-off (reward ↑, battery velocity ↓) — the reward mix,
  not the step count, sets the balance point.

Process lessons: (1) fix the yardstick before tuning — pushes in the eval
env and transition-contaminated stand metrics burned two iterations;
(2) `eval/avg_episode_length` in training logs since the trot_v1
die-and-reset exploit; (3) one intent per iteration, always vs the
battery, never vs the reward number.

## Deploying / reproducing

- Train (cluster): `make hpc-train EXPERIMENT=locomotion_v8` (4×H100,
  ~15 min; remote `~/M/wojtek`).
- Train (local 3090): `./run.sh train +experiment=locomotion_v8 run_name=...`
  with the single-GPU scale (or `locomotion_3090` for the base config).
- Actor obs (no-IMU): joint_pos, joint_vel, last_act, command(vx,vy,wz,h),
  phase clock (8) — 48-D. Command height 0.09–0.17 m. No action filter.
- Videos: `videos/fbb_loco_v8/{transitions,stand_to_run_ramp,walk_to_trot}.mp4`.
