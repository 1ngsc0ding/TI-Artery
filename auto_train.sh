#!/bin/bash
# auto_train.sh — ฝึกโมเดลอัตโนมัติ หลุดก็ resume เองจนกว่าจะจบ
# วิธีใช้:
#   ./auto_train.sh wall     # ฝึก wall model (CUBS)
#   ./auto_train.sh plaque   # ฝึก plaque model (Ar-PlaqSegm1)
# รันเบื้องหลัง: nohup ./auto_train.sh wall > logs/auto_wall.out 2>&1 &
# ดู log: tail -f logs/wall_auto.log

set -u
cd "$(dirname "$0")"
mkdir -p logs models

MODE=${1:-wall}

if [ "$MODE" = "wall" ]; then
  TRAIN_CMD="python train_wall_unet.py --data data/cubs --epochs 50 --size 320 --batch 16 --lr 3e-4 --pos-weight 2.0 --out models/my_wall.pt --last models/last_wall.pt"
  EVAL_CMD="python eval_wall.py --model models/my_wall.pt --out eval_wall"
  LOG="logs/wall_auto.log"
elif [ "$MODE" = "plaque" ]; then
  TRAIN_CMD="python train_plaque_unet.py --data data/plaque_v2 --epochs 50 --size 320 --batch 16 --lr 1e-4 --pos-weight 2.0 --out models/plaque_v2.pt --last models/plaque_v2_last.pt"
  EVAL_CMD="python eval_plaque.py --model models/plaque_v2.pt --data data/plaque_v2/val --threshold 0.5 --out eval_plaque_val"
  LOG="logs/plaque_auto.log"
else
  echo "ใช้: $0 [wall|plaque]"
  exit 1
fi

echo "=== auto-train [$MODE] started $(date) ===" | tee -a "$LOG"

# ฝึกวนจนกว่าจะครบทุก epoch (ถ้าหลุดจะ resume เอง)
while true; do
  # shellcheck disable=SC2086
  python -c "import torch; print('cuda:', torch.cuda.is_available())" 2>/dev/null
  $TRAIN_CMD --resume >> "$LOG" 2>&1
  # เช็คว่าฝึกจบครบหรือยัง (สคริปต์จะพิมพ์ "best val Dice" ตอนจบ)
  if tail -5 "$LOG" | grep -q "best val Dice"; then
    echo "=== training complete ===" | tee -a "$LOG"
    break
  fi
  echo "=== interrupted, resuming in 15s... $(date) ===" | tee -a "$LOG"
  sleep 15
done

echo "=== running evaluation ===" | tee -a "$LOG"
# shellcheck disable=SC2086
$EVAL_CMD >> "$LOG" 2>&1
echo "=== all done $(date) ===" | tee -a "$LOG"
