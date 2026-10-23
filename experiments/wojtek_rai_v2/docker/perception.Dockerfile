# RAI perception services (GroundingDINO detection + SAM2 segmentation) for
# the Wojtek experiment, on the laptop GPU. Built on top of the agent image so
# ROS 2, rai_interfaces and rai-core are already there; adds torch (CUDA) and
# rai_perception from the pinned upstream commit (0.3.0 is not on PyPI).
#
#   ./experiments/wojtek_rai_v2/run.sh perception build
#
# `wojtek_rai` is the agent service's image, wired in as a build context by
# compose.yaml (additional_contexts), so Compose builds it first.
FROM wojtek_rai

ARG RAI_GIT_SHA=6802d40
# torchvision <0.19 (rai_perception's pin) requires torch 2.3.1 exactly.
ARG TORCH_VERSION=2.3.1
ARG TORCHVISION_VERSION=0.18.1
ENV RAI_VENV=/opt/rai-venv

RUN "${RAI_VENV}/bin/pip" install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu121 && \
    "${RAI_VENV}/bin/pip" install --no-cache-dir \
        "rai-perception @ git+https://github.com/RobotecAI/rai.git@${RAI_GIT_SHA}#subdirectory=src/rai_extensions/rai_perception"

# rai.tools.ros2 imports its Nav2 tools unconditionally (see the agent
# Dockerfile); kept after the pip layer so a rebuild does not refetch torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-nav2-msgs ros-jazzy-nav2-simple-commander && \
    rm -rf /var/lib/apt/lists/*

# Model weights are fetched on first start into /root/.cache (a named volume
# in compose.yaml), not baked into the image.
WORKDIR /exp
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "rai_perception.scripts.run_perception_services"]
