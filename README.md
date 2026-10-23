# The proposed EENet-IPS: code and run logs.....

#  PLEASE CONTACT QUTAIBA ALASAD for more information and help at qutaibaeng@tu.edu.iq).

#  The code has been implemented and run on the SPINTRONIC Server.



Note that this folder contains the code that produced every table and figure of the EENet-IPS
manuscript, together with the logs and result files of the reported runs.



## THE Data................

Download the following public files and place them in the folder `raw/` (set in
`prep.py`, variable `RAW`):



|Dataset|  File  | Source |
|---|---|---|
| CSE-CIC-IDS2018 | `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | https://www.unb.ca/cic/datasets/ids-2018.html (AWS bucket `cse-cic-ids2018`, folder *Processed Traffic Data for ML Algorithms*) |
| UNSW-NB15 | `UNSW_NB15_testing-set.csv` | https://research.unsw.edu.au/projects/unsw-nb15-dataset (*CSV Files / Training and Testing Sets*) |
| CICIoT2023 | `Merged01.csv` | https://www.unb.ca/cic/datasets/iotdataset-2023.html (*CSV / MERGED_CSV*) |


## The utilized Environment ...........

Python 3.13, PyTorch 2.14 (CPU), scikit-learn 1.9, imbalanced-learn 0.14, shap 0.53,
captum 0.9, pandas, numpy, matplotlib, psutil (`pip install -r requirements.txt`).
All reported runs used a 2-vCPU x86-64 Linux machine without GPU, one thread per job
(`THREADS=1`; `common.py` also limits the BLAS/OpenMP thread pools).

## The Pipeline stages: ...


```bash
python prep.py cse unsw ciciot            # leakage-free preprocessing -> data/*.npz
python prep_pickle.py cse unsw ciciot     # stores the fitted scaler (for latency timing)
# UNSW-NB15: batch size / learning rate selection on the validation split (seed 1)
for bs in 512 256 128 64; do for lr in 0.002 0.001; do python sweep.py unsw $bs $lr 1; done; done
python select_unsw.py                     # writes the selected setting into train.py
# centralized EENet-IPS, three seeds
for d in unsw cse ciciot; do for s in 1 2 3; do python train.py $d eenet $s; done; done
# federated training (seed 1): plain FedAvg, FedProx, EENet-FL
for d in unsw cse ciciot; do for c in fedavg fedprox eenetfl; do python fl.py $d $c 1; done; done
# deep ablations (seed 1) and classical baselines (seeds 1-3; baselines.py also stores, via tree_fl_cost.py,
# the communication a histogram-based federated GBDT would need to rebuild the seed-1 gradient-boosting model)
for d in unsw cse ciciot; do for k in noexit cnn bilstm transformer; do python train.py $d $k 1; done; python baselines.py $d; done
for k in noexit cnn bilstm transformer; do python train.py unsw $k 1 512; done; python select_ablation.py
# explainers, latency and flow-volume test (run on an otherwise idle machine, one job at a time;
# run_timing.sh fixes the glibc allocator thresholds, which keeps batch timings stable)
./run_timing.sh
# tables and figures
python report.py
```



`run_queue.sh` runs the training jobs two at a time (one thread each), skipping jobs
whose result file already exists. On the 2-vCPU machine used (two jobs in parallel,
one thread each), a centralized EENet-IPS run took about 1-5 hours and a federated run 7 minutes to 2 hours.

## The regenerating the tables and figures without retraining  ....

All of the given result files of the reported runs have been carefully included,  so the tables, in-text values and figures can be
regenerated without training:     


```bash
python prep.py cse unsw ciciot      # needs the raw files in raw/ (about one minute)
# extract EENet-IPS_predictions_cse.zip and EENet-IPS_predictions_unsw_ciciot.zip into results/
python report.py                    # writes generated/*.tex, generated/values.json and figs/*.png
```


Starting from the raw files, this reproduces the preprocessed data exactly, as well as every table and
in-text value of the manuscript. The SHAP figures are written by `xai.py` (part of `run_timing.sh`).


## The used Protocol (summary) ....


*  Identifier columns removed (UNSW-NB15 `id`; CSE-CIC-IDS2018 `Timestamp`, `Dst Port`);
  exact duplicate flows and label-conflicting feature vectors removed before splitting.
 * Stratified 80/20 train/test split, 10% of the training split for validation (seed 42).
*   Median imputation, constant/correlated-feature removal, log + min-max scaling, one-hot
  encoding and SMOTE-Tomek (UNSW-NB15 only) are fitted on the training split only. After
  training on the resampled UNSW-NB15 data, the log ratio of the validation and training class
  priors is added to the exit logits (prior correction).
* The training: Adam, ReduceLROnPlateau (factor 0.5, patience 1), early stopping on the validation
  loss (patience 4 and at most 25 epochs; patience 8 and at most 60 epochs for UNSW-NB15). For UNSW-NB15,
  batch size and learning rate were selected on the validation split with `sweep.py`.
 * The backbone: CNN stage -> BiLSTM stage -> Transformer stage, one exit after each stage,
  joint loss = mean of the three exit cross-entropies; per-exit temperature scaling on
  the validation split; routing thresholds 0.95 (Exit 1) and 0.90 (Exit 2).
* The federated: 5 clients (Dirichlet alpha = 0.5 label split), 4 per round, 1 local epoch
  capped at 50 steps, 15 rounds (60 for UNSW-NB15). Label skew: local SMOTE-Tomek on
  UNSW-NB15 and logit adjustment (local prior in training) on CSE-CIC-IDS2018 and
  CICIoT2023; the server adds the log class prior of the validation split. The global model with the lowest validation loss is
  kept. EENet-FL clips updates to L2 norm 5, keeps the top 10% entries with error feedback
  and adds Gaussian noise (sigma = 0.01) to sent values.

* The classical baseline techniques (logistic regression, random forest, gradient boosting) use the same splits
  and training data and are trained with seeds 1-3, like EENet-IPS.



## License

The code has been released under the MIT License (see `LICENSE`). The datasets are not included
in this repository and remain subject to the terms of their providers.

## The files .........

* `results/*.json` - metrics, histories, routing statistics and settings of every run
* `results/sweep/*.json` - validation results of the UNSW-NB15 batch-size / learning-rate selection
* `results/*_baselines.json` - mean, standard deviation and the three runs of each classical model, and
  (`hgb_fl_cost`) the communication of a histogram-based federated GBDT that rebuilds the seed-1 gradient-boosting model
* `results/*_info.json` - dataset statistics before and after cleaning (also written to `data/` by `prep.py`)
* `results/unsw_<model>_bs512_s1.*` (and `results/unsw_<model>_bs128_s1.*` where batch size 512 was selected) - the
  candidates of each UNSW-NB15 ablation model; the selected one is stored as `results/unsw_<model>_s1.*` (`select_ablation.py`)
    * `results/*_pred.npz` - test-set probabilities of every exit and routing decisions
  (distributed in the separate archives `EENet-IPS_predictions_*.zip`; extract them into `results/`)
 * `results/models.zip` - trained model weights (`*.pt`); extract with `unzip results/models.zip -d results/` before running `latency.py`, `xai.py` or `select_ablation.py` without retraining
  * `results/logs.zip` - console logs of every job (extracts to `results/logs/`)
