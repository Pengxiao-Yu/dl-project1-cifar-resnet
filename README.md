# CIFAR ResNet under five million parameters

Deep Learning Project 1, Fall 2026. Pengxiao Yu (individual project).

**Status: the full 80-epoch local experiment and executed notebook are complete. Code is published at https://github.com/Pengxiao-Yu/dl-project1-cifar-resnet. Kaggle scoring is pending authentication and competition availability. The report transparently records the unverified score.**

## Measured result

| Quantity | Verified value |
| --- | --- |
| Selected checkpoint | Epoch 77 of 80 |
| Fixed validation accuracy | **95.10% (4,755 / 5,000)** |
| Trainable parameters | 4,368,690 |
| Training / validation examples | 45,000 / 5,000 |
| Unlabeled competition predictions | 2,000 rows, sample ID order verified |
| Epoch training + validation time | 60.64 minutes, Apple M1 Pro / MPS, 16 GiB |
| Kaggle hidden-test result | **Not verified; validation accuracy is separate** |

`results/main/` contains the actual configuration, 80-epoch history, selected-checkpoint summary, validation diagnostics, and figures. `results/verification.json` records the artifact and data-integrity audit. The four code cells in `Project1.ipynb` were executed without errors, including checkpoint inference; its regenerated prediction CSV matched the prior run exactly.

This project uses a CIFAR-adapted ResNet-18 with base width 40, channel widths 40/80/160/320, and two basic blocks per stage. The 32 x 32 input passes through a 3 x 3 stride-one stem without max pooling. Projection skips connect resolution transitions; global average pooling feeds a ten-class linear classifier. There are **4,368,690 trainable parameters**. Construction asserts the five-million-parameter budget. All weights start from random initialization.

## Reproduce

Obtain `data.zip` from the course Kaggle competition or the instructor's November 1 Brightspace announcement. Do not substitute another dataset or use hidden test labels. The archive and cached images are excluded from Git.

```bash
python -m pip install -r requirements.txt
python code/checks.py
python code/prepare_data.py /path/to/data.zip --out data
python -u code/train.py --data data/competition.npz --out results/main --epochs 80
python code/predict.py --device mps
python code/analyze.py --device mps
python code/execute_notebook.py
```

Use `--device cuda` on an allocated CUDA GPU. Training chooses CUDA, MPS, or CPU automatically unless specified. Prediction and analysis default to CPU. On NYU HPC, use an allocated Cloud Bursting GPU session with the course account, rather than a login node or the Torch cluster.

Resume the same code, data, seed, batch size, and schedule with:

```bash
python -u code/train.py --out results/main --epochs 80 --resume
```

Data preparation tests every ZIP entry, validates image shapes and labels, checks test IDs against the sample submission, records SHA256 hashes, and saves an explicit class-stratified 10% validation split with seed 2026. The cached dataset preserves CSV order. Channel normalization uses only the remaining training partition. Validation never updates the optimizer or batch-normalization statistics. No downloaded weights or outside datasets are used.

## Training protocol

| Setting | Value |
| --- | --- |
| Optimizer | SGD with Nesterov momentum 0.9 |
| Weight decay | 0.0005 |
| Batch size | 128 |
| Learning rate | 0.1, five-epoch warm-up, then cosine decay to 0.0001 |
| Epochs | 80 |
| Augmentation | Reflect padding 4, independent random crop, horizontal flip with probability 0.5 |
| Objective | Cross entropy with label smoothing 0.05 |
| Checkpoint selection | Highest validation accuracy; ties use lower ordinary validation cross entropy |

Training accuracy uses augmentation, and training loss includes smoothing; neither is exactly the same measurement as the validation objective.

Each completed epoch writes a JSON record and atomic `last.pt`; `best.pt` records the selected checkpoint. Resume preserves optimizer and Torch/device RNG state and checks the hashes of training sources. Identical seeds do not guarantee bitwise agreement across devices. Synthetic engineering checks are stored separately from real experiments. Runs with `--max-batches` are explicitly marked incomplete and must never be reported as formal results.

## Evaluation and artifacts

Validation accuracy is separate from Kaggle test accuracy. The hidden-test score remains unknown until a scored competition submission is observed. `predict.py` preserves sample ID order and records checkpoint and CSV hashes. A leaderboard score, rank, or successful submission is never inferred from local validation. Diagnostics and example errors use the fixed validation partition only.

The executed notebook checks parameter count, prints actual metrics, re-runs inference from the saved checkpoint, and displays learning curves, the validation confusion matrix, and the first twelve validation errors in fixed index order. Run it from the repository root. `execute_notebook.py` uses a temporary kernel specification and does not register a permanent user kernel.

Large checkpoints and competition data are excluded from Git. Train from scratch to reproduce them. Actual environment versions are recorded in `results/environment.json`. Prediction CSVs and logits are also excluded from publication; the code regenerates them from the competition data and checkpoint.

## Sources and assistance

- He, Zhang, Ren, and Sun. *Deep Residual Learning for Image Recognition*. CVPR, 2016. https://arxiv.org/abs/1512.03385
- Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. 2009. https://www.cs.toronto.edu/~kriz/cifar.html
- Instructor-linked background implementation: https://github.com/kuangliu/pytorch-cifar. This project's model and data/training pipeline were written for this task with AI assistance.
- Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS, 2019. https://arxiv.org/abs/1912.01703

OpenAI Codex assisted implementation, experiment orchestration, and report drafting. Numerical outcomes must come from saved executed artifacts; synthetic checks are not presented as training results.
