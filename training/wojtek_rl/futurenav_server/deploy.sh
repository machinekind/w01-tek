#!/usr/bin/env bash
# Deploy the FutureNav action server to any GPU host reachable over SSH.
#
#   ./deploy.sh <ssh-host> [remote-dir]
#
#   ./deploy.sh user@HOST            # installs into ~/futurenav
#   ./deploy.sh gpu-box /opt/futurenav
#
# Idempotent: re-running updates server.py and skips completed steps
# (repo clone, venv, weights). Requires: ssh access, python3-venv and git on
# the host, a CUDA GPU with >=16 GB (bf16 config needs ~24 GB dedicated).
#
# Start afterwards:
#   ssh <host> 'cd <dir> && ./start.sh'          # foreground
#   ssh -f <host> 'cd <dir> && nohup ./start.sh >> server.log 2>&1'
set -euo pipefail

HOST=${1:?"usage: deploy.sh <ssh-host> [remote-dir]"}
DIR=${2:-'~/futurenav'}
HERE=$(cd "$(dirname "$0")" && pwd)

echo ">> syncing server files to $HOST:$DIR"
ssh "$HOST" "mkdir -p $DIR"
scp "$HERE/server.py" "$HERE/smoke_test.py" "$HERE/requirements.txt" "$HOST:$DIR/"

echo ">> provisioning (repo, venv, deps, weights)"
ssh "$HOST" "bash -s" <<EOF
set -euo pipefail
cd $DIR

if [ ! -d FutureNav ]; then
  git clone --depth 1 https://github.com/linglingxiansen/FutureNav.git
fi

if [ ! -d venv ]; then
  python3 -m venv venv
fi
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt

if [ ! -f weights/FutureNav-4B-Base/config.json ]; then
  ./venv/bin/hf download llxs/FutureNav --include "FutureNav-4B-Base/*" --local-dir weights
fi

cat > start.sh <<'START'
#!/usr/bin/env bash
# FutureNav action server on port ${FUTURENAV_PORT:-8100}. Env knobs: see README.md.
cd "\$(dirname "\$0")"
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export FUTURENAV_WEIGHTS=\${FUTURENAV_WEIGHTS:-\$PWD/weights/FutureNav-4B-Base}
export FUTURENAV_SRC=\${FUTURENAV_SRC:-\$PWD/FutureNav/src}
exec ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port "\${FUTURENAV_PORT:-8100}"
START
chmod +x start.sh
echo "DEPLOY_OK"
EOF

echo ">> done. start with: ssh $HOST 'cd $DIR && ./start.sh'"
