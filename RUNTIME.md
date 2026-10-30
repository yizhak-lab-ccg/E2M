# Run time

Time to train one E2M mutation model (no cross-validation) and to predict, for the dataset sizes of the manuscript. Data loading and preprocessing are not timed. Prediction is of all training samples and includes matching their genes to the model's features.

| Dataset | Samples | Genes | Targets | Train, GPU | Train, CPU | Predict, GPU | Predict, CPU |
|---|---|---|---|---|---|---|---|
| TCGA-BRCA | 788 | 19,938 | 16 | 3.8 s (9 epochs) | 6.0 s (9 epochs) | 6.4 s | 6.2 s |
| TCGA pan-cancer | 8,797 | 19,938 | 116 | 14.9 s (11 epochs) | 90.1 s (17 epochs) | 11.5 s | 11.1 s |
| Replogle K562 (single-cell) | 75,767 | 8,207 | 155 | 61.2 s (18 epochs) | 86.2 s (16 epochs) | 4.3 s | 3.7 s |

GPU: NVIDIA GeForce RTX 4080 SUPER (16 GB). CPU: Intel Core i9-14900KF, limited to its 16 efficiency cores (16 threads). E2M commit `b201b22`, PyTorch 2.13 with CUDA 12.6. Epochs is the number run before early stopping, which differs slightly between devices. Prediction time is mostly the matching of genes to the model's features, so it is similar on both.

Settings: the TCGA rows use [`configs/manuscript.yaml`](configs/manuscript.yaml) (hidden layers 512 and 256, batch size 64, up to 100 epochs with early stopping); the network has about 10.4 million parameters. The K562 row uses the manuscript's single-cell settings (batch size 1024, learning rate 1.5e-4, inputs already log-normalized, no standardization; about 4.4 million parameters).

The benchmark script is `Runs/Benchmark/timing.py` in [E2M-analyses](https://github.com/asafpinhasitechnion/E2M-analyses) (`python Runs/Benchmark/timing.py [--device cpu]`).
