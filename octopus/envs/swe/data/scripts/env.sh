# Source this for any Harbor/udocker work on node-94. Everything lives on the NVMe home disk.
export N=/home/user/harbor-udocker
export PATH="$N/bin:/home/user/.local/bin:$PATH"
export PYTHONPATH="$N/code"
export UDOCKER_DIR="$N/udocker"
export UDOCKER_GOLDEN_DIR="$N/golden"
export UDOCKER_CONTAINERS="$N/containers"
export UDOCKER_IPC_INDEX="$N/ipc"
export UDOCKER_CREATE_SLOTS=4
export UDOCKER_THREADS=256
export OCTOPUS_HTTP_WORKERS=256
export HARBOR_SETUP_TIMEOUT=180
export HARBOR_EXEC_TIMEOUT=180
