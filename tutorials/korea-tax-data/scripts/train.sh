#!/usr/bin/env bash
# Fine-tune bge-reranker-v2-m3 on the generated trainset (FlagEmbedding encoder_only.base).
# Ported from the CE-trainset builder. Runs in an ISOLATED env (transformers<5) and never
# touches a live reranker server. See SPEC §5.5.
#
#   python -m korea_tax_data.cli build --config config/neo4j_crossencoder.yaml
#   python -m korea_tax_data.cli split --config config/neo4j_crossencoder.yaml
#   CUDA_VISIBLE_DEVICES=<free gpu> bash scripts/train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"                                   # point at the ce-train env python
DATA="${DATA:-$ROOT/output/train.jsonl}"
OUT="${OUT:-$ROOT/output/ft-bge-reranker-v2-m3}"
GROUP_SIZE="${GROUP_SIZE:-16}"                        # pos1 + neg(N-1); our rows are sibling-heavy

[ -f "$DATA" ] || { echo "missing $DATA — run cli build + split first"; exit 1; }
"$PY" -c "import FlagEmbedding" 2>/dev/null || { echo "FlagEmbedding not installed"; exit 1; }

"$PY" -m torch.distributed.run --nproc_per_node 1 \
  -m FlagEmbedding.finetune.reranker.encoder_only.base \
  --model_name_or_path BAAI/bge-reranker-v2-m3 \
  --train_data "$DATA" \
  --output_dir "$OUT" \
  --train_group_size "$GROUP_SIZE" \
  --query_max_len 128 \
  --passage_max_len 2048 \
  --learning_rate 6e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 \
  --dataloader_drop_last True \
  --logging_steps 20 \
  --save_steps 500

echo "checkpoint -> $OUT"
echo "next: python -m korea_tax_data.cli eval --config config/neo4j_crossencoder.yaml --split test --model $OUT"
