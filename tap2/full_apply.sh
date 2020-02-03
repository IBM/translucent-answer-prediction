#!/bin/bash

#stop at first error, unset variables are errors
set -o nounset
set -o errexit

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="$SCRIPTDIR/.."
MODELDIR="$SCRIPTDIR"
# path in the Docker image
export PYTORCH_PRETRAINED_BERT_CACHE="$SCRIPTDIR/bert_archive"

# expected in the working directory
INPUT=input.json

# check if GPU is available
if $(python "$SCRIPTDIR/check_cuda.py"); then
    GPU_FLAGS="--fp16"
    # we expect to run on a single GPU, if any
    export CUDA_VISIBLE_DEVICES=0
    echo "Using GPU"
else
    GPU_FLAGS=""
    echo "Using CPU"
fi

if [[ -d bot_preds ]]; then
    echo "Already have bot_preds!"
    exit 1
fi

# call bot_apply for each bottom_machine model
for MODELFILE in `ls "$MODELDIR/bot_models"`
do
python "$SCRIPTDIR/bottom_machine/bot_apply.py" \
  --bert_model bert-large-uncased \
  --dev_data "$INPUT" \
  --cache_dir botCache \
  --load_model "$MODELDIR/bot_models/$MODELFILE" \
  --prediction_file "bot_preds/${MODELFILE}_predictions.tsv" \
  --no_blank $GPU_FLAGS
done

if [[ ! -d bot_preds ]]; then
    echo "Failed to produce bottom predictions"
    exit 1
fi

# call predictions2rc_data
python "$SCRIPTDIR/top_machine/predictions2rc_data_simple.py" \
  --predictions bot_preds \
  --data "$INPUT" \
  --output rc_data.jsonl \
  --qid2sfs_used qid2sfs_used.json \
  --thresholds 0,0,0.1

if [[ -d top_preds ]]; then
    echo "Already have top_preds!"
    exit 1
fi

# call top_apply for each top_machine model
for MODELFILE in `ls "$MODELDIR/top_models"`
do
python "$SCRIPTDIR/top_machine/top_apply.py" \
  --max_seq_length 448 --train_batch_size 4 \
  --bert_model bert-large-uncased \
  --load_model "$MODELDIR/top_models/$MODELFILE" \
  --dev_file rc_data.jsonl \
  --cache_dir topCache \
  --prediction_file "top_preds/${MODELFILE}_predictions.tsv" $GPU_FLAGS
done

if [[ ! -d top_preds ]]; then
    echo "Failed to produce top predictions"
    exit 1
fi

# use answer selection to inform what sentences are supporting facts
python "$SCRIPTDIR/top_machine/boost_helpful_sfs.py" \
  --qid2sfs_used qid2sfs_used.json \
  --answers top_preds \
  --data "$INPUT" \
  --output qid2sid2bonus.json

# call make_prediction_file
python "$SCRIPTDIR/make_prediction_file.py" \
  --facts bot_preds \
  --qid2sid2bonus qid2sid2bonus.json \
  --thresholds 0.31,0.38,0.41 \
  --answers top_preds \
  --output pred.json