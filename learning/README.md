# Wojtek RL guide

`wojtek_rl_guide.ipynb`, a Google Colab notebook. Step by step: robot → environment → training → export → your policy against the keeper deployed on the robot. Every step ends with an inline MuJoCo video.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinekind/w01-tek/blob/main/learning/wojtek_rl_guide.ipynb)

1. Open the badge, choose a **GPU** runtime.
2. Optional Secrets: `HF_ORGANIZATION`, `HF_TOKEN` (published keepers are private).
3. Run the cells in order. Step 0 clones the repo and installs `training/`; restart the session once if the import cell fails.

Committed without outputs. Generated files land in the git-ignored `training/runs/` and `training/videos/guide/`. Nothing in the notebook deploys to or arms the robot.
