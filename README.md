# TAP 2.0

TAP (Translucent Answer Prediction), is a system to identify answers and evidence (in the form of supporting facts) in an RCQA task that requires multi-hop reasoning. TAP comprises two loosely coupled networks, called Local and Global Interaction eXtractor (LoGIX) and the Answer Predictor (AP).

To train the LoGIX model mixed precision is needed to fit in 16GB GPU memory.
For this we use FP16_Optimizer, part of an older version of [apex](https://github.com/NVIDIA/apex).
Instructions for installing this on top of a pytorch environment are provided below.

### Using older apex with Env_PowerAI_Py3.6
```bash
# create and activate pyt environment
conda env create -n pyt -f=Env_PowerAI_Py3.6.yml
source activate pyt
# install apex (older version)
git clone https://github.com/NVIDIA/apex
cd apex
git checkout cfb628ba2e46c9b8dc1368d6429031bbfa7a6f10
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# install other dependencies
pip install ujson tqdm requests sklearn numpy
```

### Layout
* pytorch_pretrained_bert contains a subset of an older version of [pytorch-pretrained-bert](https://github.com/huggingface/transformers)
* util and torch_util contain general utilities
* evaluation contains the evaluation script
* tap2 is the system code
  * bottom_machine has the code for the Local and Global Interaction eXtractor (LoGIX)
  * top_machine has the code for the Answer Predictor (AP)

### Setup
* Set aside some directory you can write to, with plenty of space, hereafter '/somepath'
* From [HotpotQA](https://hotpotqa.github.io/) download the train and dev files for HotpotQA distractor setting.
* We trained our models starting from two Span Selection Pretraining models
  * large_100Msspt_model.bin
  * large_100M_i_model.bin
* [Download](FIXME) these and put them in /somepath/sspt_models


### Distributed Training

The models are best trained on multiple GPUs with data parallelism. We give the commands to run the training in terms of 
a 'bdist' script.  It's behavior is to create a job description file that will be sent to the [LSF](https://www.ibm.com/support/knowledgecenter/SSETD4/product_welcome_platform_lsf.html?view=embed) job scheduler.
This calls mpirun on a script that in turn sets up the environment for each process.

```text
#BSUB -q excl
#BSUB -e /somepath/logs/train_sf2.py_12:40:54.647454.log
#BSUB -o /somepath/logs/train_sf2.py_12:40:54.647454.log
#BSUB -J train_sf2.py_12:40:54.647454
#BSUB -R "span[ptile=4]"
#BSUB -R "rusage[ngpus_shared=4]"
#BSUB -W 2000
#BSUB -n 8
mpirun -disable_gpu_hooks /usr/bin/bash /somepath/scripts/mpi_train_sf2.py.sh
```

Most of these lines are constant, but the -W and -n lines correspond to 
the maximum number of minutes the (--BDmins in bdist) 
and the number of nodes/processes/GPUs (--BDnodes in bdist).

/somepath/scripts/mpi_train_sf2.py.sh
```bash
#!/usr/bin/env bash
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK
export MASTER_ADDR="$(echo $LSB_MCPU_HOSTS | cut -d' ' -f 1)"
export MASTER_PORT=29500
python train_sf2.py \
'--bert_model' 'bert-large-uncased' \
'--train_data' '/somepath/hotpot_train_v1.1.json' \
'--cache_dir' '/somepath/bottomMachineCache' \
'--save_model' '/somepath/bm/model.bin' \
'--fp16'
```

The above script illustrates the environment variables to set for distributed training:
* WORLD_SIZE = number of GPUs = number of processes
* RANK = the rank of this process
* MASTER_ADDR = host that is rank 0
* MASTER_PORT = 29500 (probably no need to change this)

Although the above shows how it is done through LSF, a different job scheduler can accomplish the same thing. 
It can even be done manually by starting each process with the environment set up by hand.

### Training

* run train_sf2 to train the bottom machine
```bash
bdist --BDnodes 8 --BDmins 600 train_sf2.py \
  --bert_model bert-large-uncased \
  --train_data /somepath/HotpotQA/hotpot_train_v1.1.json \
  --cache_dir /somepath/HotpotQA/bottomMachineCacheLarge \
  --save_model /somepath/HotpotQA/TAP/bm/large_out/sspt_model.bin \
  --pretrained_model /somepath/sspt_models/large_100Msspt_model.bin \
  --fp16
  
bdist --BDnodes 4 --BDmins 600 train_sf2.py \
  --bert_model bert-large-uncased \
  --dev_data /somepath/HotpotQA/hotpot_dev_distractor_v1.json \
  --cache_dir /somepath/HotpotQA/bottomMachineCacheLarge \
  --load_model /somepath/HotpotQA/TAP/bm/large_out/sspt_model.bin \
  --prediction_file /somepath/HotpotQA/TAP/bm/large_out/sspt_predictions.tsv \
  --fp16
```
* use tune_sf_thresholds to find a threshold that gives 90-95% recall (0,0,0.1 is good)
```bash
bsingle tune_sf_thresholds.py \
  --predictions /somepath/HotpotQA/TAP/bm/large_out/plain_predictions.tsv,/somepath/HotpotQA/TAP/bm/large_out/sspt_predictions.tsv \
  --data /somepath/HotpotQA/hotpot_dev_distractor_v1.json
```
* use predictions2rc_data to create validation data for the top_machine
```bash
bsingle predictions2rc_data.py \
  --predictions /somepath/HotpotQA/TAP/bm/large_out/plain_predictions.tsv,/somepath/HotpotQA/TAP/bm/large_out/sspt_predictions.tsv \
  --data /somepath/HotpotQA/hotpot_dev_distractor_v1.json \
  --output /somepath/HotpotQA/TAP/dev_pred_95_rc_data.jsonl \
  --qid2sfs_used /somepath/HotpotQA/TAP/dev_pred_95_qid2sfs_used.json \
  --thresholds 0,0,0.1
```
* preprocess the HotpotQA dataset into rc_data format using hotpotqa2rc_data
```bash
bsingle hotpotqa2rc_data.py --data /somepath/HotpotQA/hotpot_train_v1.1.json  --output /somepath/HotpotQA/train_rc_data.jsonl
bsingle hotpotqa2rc_data.py --data /somepath/HotpotQA/hotpot_dev_distractor_v1.json  --output /somepath/HotpotQA/dev_rc_data.jsonl
```
* under the ai-c-complex-e2e-qa/examples directory run train_rc for the top machine
```bash
bdist --BDnodes 4 --BDmins 600 train_rc.py \
--train_file /somepath/HotpotQA/TAP/train_rc_data.jsonl \
--results_dir /somepath/HotpotQA/TAP/results \
--max_seq_length 448 --train_batch_size 16 --learning_rate 3e-5 --fp16 \
--bert_model bert-large-uncased  --num_epochs 4 \
--load_model /somepath/sspt_models/large_100M_i_model.bin \
--dev_file /somepath/HotpotQA/TAP/dev_pred_95_rc_data.jsonl \
--cache_dir /somepath/HotpotQA/TAP/large_pred_95_cache \
--experiment_name large_sspt_i_pred95_multi_2layer_answer \
--save_model /somepath/HotpotQA/TAP/tm/large_out/sspt_i_95_model.bin \
--prediction_file /somepath/HotpotQA/TAP/tm/large_out/sspt_i_95_predictions.json
```
* use make_prediction_file to create the ensemble merger from different top and bottom machine predictions
```bash
bsingle make_prediction_file.py \
  --facts /somepath/HotpotQA/TAP/bm/large_out/sspt_i_predictions.tsv,/somepath/HotpotQA/TAP/bm/large_out/plain_predictions.tsv,/somepath/HotpotQA/TAP/bm/large_out/sspt_predictions.tsv \
  --thresholds 0.41 \
  --answers /somepath/HotpotQA/TAP/tm/large_out/sspt_i_union_95_predictions.json,/home/mrglass/gpfs/NIR/HotpotQA/TAP/tm/large_out/sspt_i_95_predictions.json,/somepath/HotpotQA/TAP/tm/large_out/sspt_i_cross_apply_95_predictions.json,/somepath/HotpotQA/TAP/tm/large_out/sspt_95_predictions.json \
  --data /somepath/HotpotQA/hotpot_dev_distractor_v1.json \
  --output /somepath/HotpotQA/TAP/predictions.json
```

Training with predicted supporting facts for answer selection (recommended)
* run five folds of train_sf2 passing 0-4 as the --fold
```bash
bdist --BDnodes 8 --BDmins 600 train_sf2.py \
  --bert_model bert-large-uncased \
  --train_data /somepath/HotpotQA/hotpot_train_v1.1.json \
  --cache_dir /somepath/HotpotQA/bottomMachineCacheLarge \
  --prediction_file /somepath/HotpotQA/TAP/bm/large_folds/predictions0.tsv \
  --fp16 --save_model /somepath/HotpotQA/TAP/bm/large_folds/model0.bin --fold 0
```
* cat the predictions[0-4].tsv together
```bash
cd /somepath/HotpotQA/TAP/bm/large_folds
cat predictions*.tsv > predictions_all_folds.tsv
```
* use predictions2rc_data to create training data for the top_machine
```bash
bsingle predictions2rc_data.py \
--predictions /somepath/HotpotQA/TAP/bm/large_folds/predictions_all_folds.tsv \
--data /somepath/HotpotQA/hotpot_train_v1.1.json \
--output /somepath/HotpotQA/TAP/train_pred_95_rc_data.jsonl \
--thresholds 0,0,0.1
```
* train_rc with the generated training data
```bash
bdist --BDnodes 4 --BDmins 600 train_rc.py \
--train_file /somepath/HotpotQA/TAP/train_pred_95_rc_data.jsonl \
--results_dir /somepath/HotpotQA/TAP/results \
--max_seq_length 448 --train_batch_size 16 --learning_rate 3e-5 --fp16 \
--bert_model bert-large-uncased  --num_epochs 4 \
--load_model /somepath/sspt_models/large_100M_i_model.bin \
--dev_file /somepath/HotpotQA/TAP/dev_pred_95_rc_data.jsonl \
--cache_dir /somepath/HotpotQA/TAP/large_cross_apply_pred_95_cache \
--experiment_name large_sspt_i_cross_apply_pred95 \
--save_model /somepath/HotpotQA/TAP/tm/large_out/sspt_i_cross_apply_95_model.bin \
--prediction_file /somepath/HotpotQA/TAP/tm/large_out/sspt_i_cross_apply_95_predictions.json
```

### Dataset
* [HotpotQA](https://hotpotqa.github.io/)

### Results


|Method | Answer F1, EM | Support F1, EM | Joint F1, EM |
|----|----|----|----|
| Leader (test) | 76.36, 63.29 | 85.60, 58.25 | 67.92, 41.39 |
| TAP2 (dev)   | 80.02, 66.46 | 86.75, 57.97 | 71.05, 41.77 |

[//]: # (em=63.29	f1=76.36	sp_em=58.25	sp_f1=85.60	joint_em=41.39	joint_f1=67.92
)
[//]: # (metrics = {'em': 0.6645509790681972, 'f1': 0.8002364225361368, 'prec': 0.825315292405096, 'recall': 0.8117287534657562,
 'sp_em': 0.5797434166103984, 'sp_f1': 0.8674649410908941, 'sp_prec': 0.8820070094209277, 'sp_recall': 0.8794114337159635,
 'joint_em': 0.4176907494935854, 'joint_f1': 0.7104526464560095, 'joint_prec': 0.7425128751259779, 'joint_recall': 0.7299789366234362}
)
