# NVIDIA Isaac vs. MuJoCo for Wojtek — features, fidelity, and the 2026 reality

**Date:** 2026-07-09
**Scope:** What NVIDIA Isaac (Isaac Sim + Isaac Lab; the deprecated Isaac Gym; the new Newton/Warp engine) gives over MuJoCo (+ MJX, MuJoCo Playground, MJWarp) for the Wojtek quadruped RL project.
**Situation this is written for:** proprioceptive quadruped policy; existing stack is MJX + JAX/brax-PPO + Hydra + wandb; local dev on a Mac (Apple Silicon); training on rented NVIDIA GPU boxes (vast.ai-style); MIT-Mini-Cheetah-class actuators (AK80-9 / MD80, ~18 Nm real peak); priorities in order = **(1) sim2real fidelity, (2) training throughput, (3) CLI/headless controls, (4) headless eval video**; sim/GUI polish explicitly *not* a priority.
**Method:** 6 parallel research agents (one per dimension) + 6 adversarial fact-check agents that re-verified each dimension's riskiest claims against primary sources. Vendor performance figures are flagged as such where methodology could not be independently confirmed.

---

## Bottom line up front

For **this** situation, Isaac does not give a meaningful sim2real or training advantage, and it costs the entire local workflow. The reason isn't that Isaac is weak — it's that **the physics engine you would switch to Isaac *for* is becoming MuJoCo.** NVIDIA's next-gen engine (Newton) runs the DeepMind MuJoCo solver (MuJoCo-Warp) as its primary backend; Google's Brax now tells users to drop its own physics and use MJX/MuJoCo-Warp. The two ecosystems are converging on *our* physics core.

What Isaac genuinely offers over MuJoCo is **tooling and turnkey scale**, not better physics: a ready-made actuator-model library, a first-class domain-randomization framework, photoreal RTX rendering, and documented multi-node training. Those are worth *borrowing the ideas from* — one of them (the actuator zoo) is directly relevant to closing the AK80-9/MD80 gap — but none justify re-platforming.

**Recommendation: stay on MJX/MuJoCo Playground; invest the sim2real effort in an actuator model + contact tuning + domain randomization; track MJWarp and `mjlab`.**

---

## The one fact that reframes the whole comparison

**Newton** (NVIDIA + Google DeepMind + Disney, governed by the Linux Foundation) reached **1.0 GA at GTC 2026** (March). Its primary solver is **MuJoCo-Warp (MJWarp)** — DeepMind's MuJoCo reimplemented on NVIDIA's Warp (Python→CUDA). Isaac Lab is adding Newton as a backend **alongside** PhysX (multi-backend architecture — PhysX is *not* being retired).

So the "which engine has more accurate physics" debate is collapsing: **MuJoCo's contact solver is the one everyone is standardizing on.** Staying on MuJoCo/MJX is not betting against the trend — it is standing at its source.

The catch: **Newton *inside Isaac Lab* for RL is still beta** ("not recommended for production RL training," flat-terrain locomotion examples only, breaking changes expected), and **Newton in Playground** is also beta. The *engine* is GA; the *RL integrations* on both sides are not yet. Don't bet a production pipeline on Newton-for-RL in mid-2026 — on either stack.

---

## Dimension 1 — Sim-to-real fidelity (priority #1)

**Physics accuracy.** MuJoCo's convex soft-constraint solver (penetration-free, proper friction cone, stable at large timesteps) is the most-respected engine for contact-rich legged locomotion. PhysX 5 (TGS solver, reduced-coordinate Featherstone articulations) is fast and stable at massive scale but is generally regarded as less physically faithful for contact.

> *Honest caveat on the accuracy gap:* the classic "PhysX omits Coriolis/centrifugal terms" critique traces to a **peer-reviewed source** (Erez, Tassa, Todorov, *Simulation tools for model-based robotics*, ICRA 2015) plus NVIDIA's own PhysX issue #394 (Coriolis/centrifugal only modeled for reduced-coordinate articulations) — but that is a decade old and predates PhysX 5. The *PhysX-5-specific* comparative-accuracy claims circulating today are blog-level and unverified for current versions. Treat "MuJoCo is more accurate" as well-founded but not freshly benchmarked.

**Where Isaac genuinely helps — the actuator zoo.** This is Isaac Lab's most portable, most relevant advantage. `isaaclab.actuators` ships drop-in classes:

| Class | What it models | Relevance to AK80-9/MD80 |
|---|---|---|
| `DCMotor` | Velocity-dependent torque-speed saturation (four-quadrant) | **BLDC torque droop — the core AK80-9 nonlinearity** |
| `DelayedPDActuator` | `min_delay`/`max_delay` control & comms latency | Models the real control-loop / CAN latency |
| `RemotizedPDActuator` | Angle-dependent torque limits | Geared / linkage transmissions |
| `ImplicitActuator` / `IdealPDActuator` | PD folded into solver / PD + feedforward + saturation | Baseline PD control |
| `ActuatorNetMLP` / `ActuatorNetLSTM` | **Learned actuator networks** (Hwangbo et al., ANYmal) | The technique that first made blind quadruped sim2real work |

In MuJoCo/MJX you build all of this by hand (MuJoCo Playground ships the *patterns*, but there is no drop-in class library). **This is the single most valuable thing to steal from Isaac regardless of stack.**

**Domain randomization.** Isaac Lab has a first-class `EventManager` (startup/reset/interval modes; randomizes mass, CoM, friction, joint stiffness/damping, actuator gains, pushes, gravity, obs/action noise — per-env on GPU). MJX/Playground DR is code-based: you write a `randomize_fn` over `mjx.Model` fields and `vmap` it across envs. More flexible, but you own the taxonomy.

**Proof point (weakly documented).** NVIDIA blogs an ANYmal-D policy trained on Newton/MJWarp (rsl_rl, 4096 envs, IMU+encoder-only obs) deployed to real hardware by ETH — but it's an NVIDIA blog with no independent validation detail (trials, tuning, failure modes).

> **Verdict:** "Most real sim" = MuJoCo contact physics **either way**. The gap-closers are contact tuning (`solref`/`solimp`/`condim`/`frictionloss`), an actuator model for the AK80-9 (torque-speed saturation + latency, optionally an actuator net), and per-env DR — none of which require Isaac. Borrow Isaac's actuator-model *design*; keep MuJoCo's physics.

---

## Dimension 2 — Agent-driven RL training

| Stack | Verified throughput |
|---|---|
| **MuJoCo Playground** (MJX + brax PPO, single A100) | Go1 ~417k, Spot ~408k, Barkour ~386k PPO steps/s; flat-terrain Go1 trains **< 5 min on 2× RTX 4090** ✅ (Playground tech report) |
| **Isaac Lab** (single RTX 4090) | 94k step-FPS on `Isaac-Velocity-Rough-G1` — **but G1 is the Unitree *humanoid*, not a quadruped** ⚠️. Isaac Lab's official benchmark table has **no ANYmal/Go1/Spot entry at all** |
| **Isaac multi-node** | ~1.2M FPS on **16× L40 / 4 nodes** — scaling *GPU count* at a fixed 4096 envs, not env-count-per-GPU |

Two corrections the verifiers forced:
1. Any Isaac locomotion throughput figure available in NVIDIA's docs is a **humanoid proxy** — there is no apples-to-apples quadruped number published.
2. Locomotion benchmarks use **4096 envs**, manipulation 8192. The "tens of thousands of envs per GPU" framing is unsupported for locomotion; the >1M FPS number comes from scaling GPUs, not envs.

**MJWarp speedups** (vendor-claimed, methodology unstated): **152× locomotion / 313× manipulation vs MJX on RTX 4090** — use this as the realistic rented-GPU expectation. 252×/475× is best-case on the RTX PRO 6000 Blackwell workstation card. Both are NVIDIA headline figures, not third-party benchmarks; discount accordingly.

> **Verdict:** For a small quadruped, Playground gives comparable-or-faster *time-to-walking-gait* on the hardware we rent, in the framework we already run. Isaac's real training edge is **turnkey multi-node** (documented `torchrun --distributed` across rl_games/rsl_rl/skrl/SB3); MJX/Playground has **no documented multi-node recipe** (open, unanswered issue #45) — you'd hand-roll JAX `pmap`/sharding. That matters only if we outgrow single-box rentals.

Note: **Isaac Gym** (the standalone preview) is deprecated/unsupported; NVIDIA directs users to Isaac Lab. Not a candidate.

---

## Dimension 3 — CLI / cluster controls

**Isaac Lab** is more batteries-included: per-library `train.py` with `--task --headless --num_envs --checkpoint --video --video_length --video_interval`, genuine **Hydra** overrides, `--distributed` multi-node, and a documented SLURM+Singularity recipe.

Caveats from verification:
- The SLURM recipe was tested only on ETH Euler / IIT Franklin HPC clusters — **irrelevant to a vast.ai bare-Docker box**, where the plain `docker run --gpus all` path applies.
- Docker image is **~20 GB** (secondary-source estimate; NVIDIA's own docs state no size).
- A wandb-drops-metrics bug (#5252) exists but is pinned to **Isaac Lab 2.3.2 + skrl**; current release is **3.0 Beta 2 (Feb 2026)**, so treat it as version-stale, not necessarily live.

**MuJoCo/Playground** ships `train-jax-ppo` / `train-rsl-ppo` console scripts; you own the Hydra tree (our current setup). Install is `pip install mujoco mujoco-mjx "jax[cuda12]"` — **minutes on a fresh box, no container, no Singularity conversion.** Headless is native. wandb is one clean code path.

> **Verdict:** For SSH-driven ephemeral rented GPUs, **MuJoCo is smoother to operate** — pip vs a ~20 GB image, no GUI mode to suppress, and it's the stack we've already scripted. Isaac's scaffolding is real but targets teams needing turnkey HPC multi-node, which buys nothing on single-box rentals.

---

## Dimension 4 — Headless video on the GPU cluster

This dimension has a **hard gotcha that likely disqualifies Isaac on our rented hardware:**

- **Isaac Sim requires RT Cores.** Per current docs (Isaac Sim **5.1** — there is no "6.0" despite some pages implying it): minimum **RTX 4080, 16 GB VRAM**, and **"GPUs without RT Cores (A100, H100) are not supported."** That rules out the cheapest/most-available datacenter cards on vast.ai for *anything touching the renderer, including eval-video recording.* (NVIDIA's own docs are internally inconsistent — the stated floor ranges from T4-class to RTX 4080 across pages; see IsaacSim issue #311.) Isaac Lab's `--video` also needs `--enable_cameras`, has open memory-leak / H100-tiled-camera bugs, and pulls the full Omniverse renderer in even offscreen.
- **MuJoCo** renders offscreen via **EGL** (`MUJOCO_GL=egl`) on *any* CUDA GPU — including the cheap A100/H100 boxes Isaac refuses. The whole video pipeline is ~a dozen lines of Python (`renderer.update_scene` → `imageio.mimwrite`) in the existing eval loop.
  - *Caveat:* EGL doesn't always "just work" on minimal vast.ai images — nvidia-docker does not auto-provide an EGL ICD. Some templates need a one-time fix: add `/etc/glvnd/egl_vendor.d/10_nvidia.json` and set `NVIDIA_DRIVER_CAPABILITIES=graphics`.

Isaac's RTX/Replicator photoreal rendering is a **genuine differentiator only for vision** — synthetic camera data with domain-randomized perception (NVIDIA case study: object-detector AP 5%→87% after Replicator DR). **Wojtek's policy is proprioceptive**, so eval video is just to eyeball gait/foot-slip/torque — flat OpenGL is entirely sufficient.

> **Verdict:** MuJoCo wins decisively for this use. Isaac's rendering only pays off if we add camera observations.

---

## Dimension 5 — The Mac factor

**Isaac Sim/Lab has zero macOS path.** No local install, no viewer, no scene debugging — it requires an RTX GPU + Linux/Windows + proprietary drivers. The only options would be streaming the GUI from the remote box or blind headless SSH. Every local iteration step (author MJCF/USD, eyeball a contact, tweak a reward, re-render) is gone.

**MuJoCo is a first-class Apple Silicon citizen:** `pip install mujoco` gives a native arm64 binary; the interactive viewer runs via `mjpython` (required only because macOS forces Cocoa/rendering calls onto the main thread — a wrapper quirk, not a missing feature). You author/debug/visualize the *exact* MJCF used in training, locally, offline.

The one asterisk: **MJX on Mac is CPU-only.** JAX has no real Apple-GPU backend — `jax-metal` has been stale since Oct 2024 (still "experimental," not officially deprecated), and `jax-mps` (MLX-backed, June 2026) is too immature and has no MJX track record. Fine for tiny smoke tests; real training goes to the remote GPU anyway — which is already our workflow.

> **Verdict:** Choosing Isaac forfeits ~100% of local iteration ability. MuJoCo gives it for free on the machine we already use daily.

---

## What Isaac genuinely gives over MuJoCo (the honest feature list)

1. **Actuator-model library** — `DCMotor`, `DelayedPDActuator`, `ActuatorNetMLP/LSTM`. The one thing worth stealing for AK80-9 fidelity. *(Steal the design, not the stack.)*
2. **DomainRandomization `EventManager`** — structured, per-env-on-GPU; more turnkey than hand-rolled MJX DR.
3. **RTX ray-traced rendering + Omniverse Replicator** — photoreal synthetic data + annotation SDK. Only matters if we go vision-based.
4. **Turnkey multi-GPU/multi-node** — documented `torchrun --distributed`; MJX has none.
5. **Broader RL-library menu** — rl_games, rsl_rl, skrl, SB3 out of the box.
6. **Deepest published legged sim2real corpus** — legged_gym/rsl_rl → ANYmal/Spot/Go2 baselines to copy from.

**Costs of adopting Isaac:** no Mac, RT-Cores-only GPUs, ~20 GB Omniverse stack, PhysX/Omniverse EULA (vs MuJoCo's clean Apache-2.0), a full re-platform from JAX→PyTorch, and an RL layer that is itself mid-migration to Newton.

---

## MuJoCo-side alternatives worth knowing

- **MuJoCo Playground** — the de facto open GPU locomotion suite (RSS 2025 Outstanding Demo, 50+ envs incl. Go1/Barkour/Spot, single-GPU friendly). Our natural home; already MJWarp-backend-capable.
- **MJWarp / mujoco_warp** — GPU MuJoCo on Warp, co-built by NVIDIA + DeepMind, the tentpole of Newton. Path to Newton-class speed *without leaving MuJoCo semantics or the Apache license.*
- **mjlab** — reimplements **Isaac Lab's manager-based API** (observations/rewards/events) on top of MuJoCo-Warp. Literally "Isaac Lab ergonomics without Omniverse." Worth watching — it's the community porting Isaac's best ideas onto our stack. (NVIDIA GPU for training; macOS eval-only.)
- **Brax** — physics deprecated (docs point to MJX/MuJoCo-Warp); `brax/training` PPO still maintained, pairs with Playground.
- **Genesis** — Taichi-based, claims 10–80× throughput; real sim2real examples exist (Go2 walking, matched 50 Hz control, simulated action latency) but there is no large published quadruped sim2real corpus comparable to the Isaac/rsl_rl lineage or the fast-growing MJX corpus. Interesting, not proven.
- **ManiSkill / SAPIEN** (manipulation-oriented) and **Drake / Gazebo** (reference-grade contact / classical robotics) — not GPU-RL substrates for legged work; useful as cross-checks only.

---

## Recommendation for Wojtek

1. **Stay on MJX / MuJoCo Playground.** It's the same physics the whole field (Isaac included) is converging toward, keeps the Mac-local + rented-GPU workflow, and has zero migration cost.
2. **Spend the sim2real effort where it moves the needle:** a proper AK80-9/MD80 actuator model (torque-speed saturation + control latency, optionally a small actuator net), contact tuning, and per-env domain randomization. **Copy Isaac Lab's actuator-class designs as the spec.**
3. **Track MJWarp + `mjlab`.** When MJWarp's RL integration stabilizes, we get big speedups without changing physics semantics or license; `mjlab` offers Isaac-Lab-style ergonomics on MuJoCo if we ever want them.
4. **Only reconsider Isaac if** we pivot to vision-based policies (RTX/Replicator synthetic data) *or* need turnkey multi-node HPC — and even then, budget for losing all Mac-local dev.

---

## Numbers to re-verify at implementation time

These are version-pinned or vendor-claimed and drift fast:

- Isaac Sim GPU floor / RT-Cores requirement — re-check against whatever Isaac Sim version is actually pinned (docs currently inconsistent: T4-class → RTX 4080 depending on the page).
- MJWarp speedups (152×/313× on 4090; 252×/475× on Blackwell) — NVIDIA headline figures, methodology unstated.
- Isaac Lab Docker image size (~20 GB) — secondary-source estimate, not in NVIDIA docs.
- Newton-for-RL maturity — beta in both Isaac Lab and Playground as of mid-2026; re-check status before relying on it.
- `jax-metal` / `jax-mps` on Apple Silicon — both effectively unusable for MJX training today; re-check if Mac-GPU training ever becomes desirable.

---

## Primary sources

**Physics / Newton / actuators**
- NVIDIA blog — *Train a Quadruped Locomotion Policy … with Isaac Lab and Newton*: https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/
- NVIDIA blog — *Newton Adds Contact-Rich Manipulation and Locomotion …*: https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/
- Isaac Lab — Newton physics integration (experimental): https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html
- Isaac Lab — actuators API: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html
- MuJoCo-Warp: https://github.com/google-deepmind/mujoco_warp · https://mujoco.readthedocs.io/en/stable/mjwarp/index.html
- Newton engine: https://github.com/newton-physics/newton
- Erez, Tassa, Todorov, ICRA 2015 (PhysX/MuJoCo comparison): https://roboti.us/lab/papers/ErezICRA15.pdf
- PhysX Coriolis issue: https://github.com/NVIDIAGameWorks/PhysX/issues/394

**RL training**
- Isaac Lab performance benchmarks: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html
- MuJoCo Playground (arXiv 2502.08844): https://arxiv.org/html/2502.08844v1 · https://playground.mujoco.org/ · https://github.com/google-deepmind/mujoco_playground
- Isaac Gym deprecation: https://forums.developer.nvidia.com/t/isaac-gym-deprecation-transition-to-isaac-lab/322978

**CLI / cluster**
- Isaac Lab multi-GPU: https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html
- Isaac Lab Hydra: https://isaac-sim.github.io/IsaacLab/main/source/features/hydra.html
- Isaac Lab Docker / cluster: https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html · https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html
- Playground CLI training: https://deepwiki.com/google-deepmind/mujoco_playground/5.4-command-line-training
- Playground multi-GPU (open issue): https://github.com/google-deepmind/mujoco_playground/issues/45

**Video / rendering**
- Isaac Lab record video: https://isaac-sim.github.io/IsaacLab/main/source/how-to/record_video.html
- Isaac Sim requirements (5.1): https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
- Madrona-MJX batch renderer: https://github.com/shacklettbp/madrona_mjx
- Replicator sim2real case study (5%→87% AP): https://developer.nvidia.com/blog/closing-the-sim2real-gap-with-nvidia-isaac-sim-and-nvidia-isaac-replicator/
- EGL ICD in nvidia-docker: https://github.com/NVIDIA/nvidia-docker/issues/1520

**Mac / platform**
- MuJoCo MJX docs: https://mujoco.readthedocs.io/en/stable/mjx.html · Python/viewer: https://mujoco.readthedocs.io/en/stable/python.html
- jax-metal: https://developer.apple.com/metal/jax/ · https://pypi.org/project/jax-metal/
- jax-mps (MLX-backed): https://github.com/tillahoffmann/jax-mps
- Isaac Sim GPU-spec inconsistency: https://github.com/isaac-sim/IsaacSim/issues/311

**Ecosystem**
- mjlab: https://github.com/mujocolab/mjlab · https://arxiv.org/html/2601.22074v1
- Brax (physics deprecated → MJX/MuJoCo-Warp): https://github.com/google/brax
- legged_gym / rsl_rl lineage: https://github.com/leggedrobotics/legged_gym
