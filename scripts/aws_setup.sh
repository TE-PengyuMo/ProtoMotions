#!/usr/bin/env bash
# AWS single-instance multi-GPU setup + launch for ProtoMotions
# steering / MaskedMimic-on-terrain training (g5.48xlarge = 8x A10G).
#
# ── ON YOUR LOCAL MACHINE FIRST ──────────────────────────────────────────────
#   AWS:  git clone https://github.com/TE-PengyuMo/ProtoMotions.git ~/ProtoMotions
#   Then sync the gitignored data from local -> AWS (keep the same subpaths):
#     rsync -avz --progress \
#       data/amass/motionlib_smpl/amass_smpl_train.pt \
#       <AWS>:~/ProtoMotions/data/amass/motionlib_smpl/
#     rsync -avz data/smpl/ <AWS>:~/ProtoMotions/data/smpl/
#     rsync -avz data/pretrained_models/motion_tracker/smpl-terrains/ \
#       <AWS>:~/ProtoMotions/data/pretrained_models/motion_tracker/smpl-terrains/
#
# ── ON THE AWS INSTANCE ──────────────────────────────────────────────────────
#   ./scripts/aws_setup.sh setup                  # container + deps + data check
#   docker exec -it isaac-lab wandb login         # one-time, interactive
#   ./scripts/aws_setup.sh train                  # both lines, 4+4 GPUs parallel
#   ./scripts/aws_setup.sh logs                   # tail both training logs
#
# Override defaults via env, e.g.:  NUM_ENVS=6144 BATCH=12288 ./scripts/aws_setup.sh train
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ProtoMotions}"
CONTAINER="${CONTAINER:-isaac-lab}"
IMAGE="${IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.2}"
PY="/isaac-sim/python.sh"
MOTION="data/amass/motionlib_smpl/amass_smpl_train.pt"
EXPERT="data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt"
NUM_ENVS="${NUM_ENVS:-4096}"   # per-GPU env count (A10G 24GB: 4096 safe)
BATCH="${BATCH:-8192}"         # per-GPU minibatch size

dexec() { docker exec -w /workspace/ProtoMotions -e PYTHONPATH=. "$@"; }

cmd_setup() {
  echo ">> docker pull $IMAGE"
  docker pull "$IMAGE"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo ">> starting container $CONTAINER (mount $REPO_DIR)"
    docker run -d --name "$CONTAINER" --gpus all --network host \
      -v "$REPO_DIR":/workspace/ProtoMotions "$IMAGE" bash -c "sleep infinity"
  else
    echo ">> container $CONTAINER exists; (re)starting"
    docker start "$CONTAINER" >/dev/null 2>&1 || true
  fi
  echo ">> installing protomotions + dm_control"
  dexec "$CONTAINER" $PY -m pip install -e .
  dexec "$CONTAINER" $PY -m pip install dm_control
  echo ">> data check (inside container):"
  for f in "$MOTION" "$EXPERT" "data/smpl/SMPL_NEUTRAL.pkl"; do
    if dexec "$CONTAINER" test -f "$f"; then echo "   OK       $f"; else echo "   MISSING  $f  <-- rsync this!"; fi
  done
  echo ">> NEXT: docker exec -it $CONTAINER wandb login   (one-time), then: $0 train"
}

# launch <name> <cuda_visible> <nproc> <master_port> <extra train_agent args...>
launch() {
  local name="$1" cuda="$2" nproc="$3" port="$4"; shift 4
  echo ">> launch $name  GPUs=$cuda  nproc=$nproc  port=$port  (log: ${name}.log)"
  docker exec -d -w /workspace/ProtoMotions -e PYTHONPATH=. \
    -e CUDA_VISIBLE_DEVICES="$cuda" "$CONTAINER" bash -c \
    "$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=$nproc --master_port=$port \
       protomotions/train_agent.py --robot-name smpl --simulator isaaclab \
       --motion-file $MOTION --num-envs $NUM_ENVS --batch-size $BATCH --ngpu $nproc --use-wandb $* \
       > ${name}.log 2>&1"
}

cmd_train() {
  # Line A: AMP steering on terrain (GPUs 0-3)
  launch steering_terrain "0,1,2,3" 4 29500 \
    --experiment-path examples/experiments/steering/mlp_terrain.py \
    --experiment-name steering_terrain
  # Line B: MaskedMimic distillation on terrain (GPUs 4-7), uses pretrained expert
  launch mm_distill_terrain "4,5,6,7" 4 29501 \
    --experiment-path examples/experiments/masked_mimic/transformer_terrain.py \
    --experiment-name mm_distill_terrain \
    --expert-model-path "$EXPERT"
  echo ">> both launched (4+4). watch:  $0 logs"
}

cmd_logs() {
  dexec "$CONTAINER" bash -c "tail -n 30 -f steering_terrain.log mm_distill_terrain.log"
}

case "${1:-}" in
  setup) cmd_setup ;;
  train) cmd_train ;;
  logs)  cmd_logs ;;
  *) echo "usage: $0 {setup|train|logs}"; exit 1 ;;
esac
