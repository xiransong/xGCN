This is an implementation for our TheWebConf 2023 paper: *xGCN: An Extreme Graph Convolutional Network for Large-scale Social Link Prediction*.

# Note

**The current code is our original raw scripts. We release a refactored new version with improved functionality and detailed documentation:** https\://xgcn.readthedocs.io/en/latest/index.html

We provide examples and scripts to run the models: https://xgcn.readthedocs.io/en/latest/user_guide/usage_examples/example_pokec_livejournal.html

# 1. Install dependencies

The Python version is 3.8.12, other dependencies are listed in `requirements.txt`.

# 2. Prepare the data

* The .sh scripts to process datasets and run models are in the `script` directory.
* You are expected to specify two paths: (1) `PROJECT_ROOT`, which is the root of this project (e.g. `/home/xxx/code/xgcn`); (2) `ALL_DATA_ROOT`, which is going to hold all the data including datasets and models' outputs (e.g. `/home/xxx/data/xgcn`).
* The processed Pokec and LiveJournal datasets can be downloaded from here: https://drive.google.com/file/d/1lv-JkPmswYQ-YfKfRpNpoVh_N2XCiRQg/view?usp=drive_link. The original datasets are from http://snap.stanford.edu/data/index.html.
* The data (.txt files) should be placed in the directory `$ALL_DATA_ROOT'/datasets/instance_[DATASET_NAME]'` as follows (e.g. the pokec dataset):

```
instance_pokec
├── train.txt       # graph for training
├── validation.txt  # validation set
├── test.txt        # test set
```

* Dataset format
  * Each dataset contains 3 files: `train.txt`, `validation.txt` and `test.txt`.
  * Node ids start from 0, and are continuous integers.
  * `train.txt` contains the directed edges for training. Each line represents an edge: `src dst`.
  * `validation.txt` contains the edges for validation. Each line represents an edge: `src dst`.
  * `test.txt` contains the edges for test. Each line represents one edge or mutiple edges: `src dst1 dst2 ...`.
* To process the raw .txt files, `cd script` and run the script `process_data.sh` (`bash prcess_data.sh` e.g. the dataset is pokec). After doing so, some new files for caching will be saved in directory `instance_[DATASET_NAME]`:

```
instance_pokec
├── train.txt
├── validation.txt
├── test.txt
├── info.yaml
├── train_csr_indptr.pkl
├── train_csr_indices.pkl
├── train_csr_src_indices.pkl
├── train_undi_csr_indptr.pkl
├── train_undi_csr_indices.pkl
├── train_undi_csr_src_indices.pkl
├── validation.pkl
├── test.pkl
```

# 3. Run models

* All the training settings and hyper-parameter configurations are in `config` and `script/run_model` (The cmd arguments in .sh files will overwrite those in .yaml config files).
* To run a model, modfiy `PROJECT_ROOT`, `ALL_DATA_ROOT`, `DEVICE`, `DATASET` and `MODEL` in `script/run_model.sh`. `cd script` and `bash run_model.sh`. The outputs (including training log, test results, ...) will be saved in `$ALL_DATA_ROOT'/model_outputs/gnn_'$DATASET'/'$MODEL`.
