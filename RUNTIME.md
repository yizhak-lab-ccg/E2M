# Run time

Time to train one E2M mutation model and then predict, for three example datasets: 

| Dataset | Samples | Genes | Targets | Train, GPU | Train, laptop CPU | Predict, GPU | Predict, laptop CPU |
|---|---|---|---|---|---|---|---|
| TCGA-BRCA | 788 | 19,938 | 16 | 3.8 s (9 epochs) | 11.8 s (9 epochs) | 6.4 s | 7.8 s |
| TCGA pan-cancer | 8,797 | 19,938 | 116 | 14.9 s (11 epochs) | 260.7 s (17 epochs) | 11.5 s | 18.0 s |
| Replogle K562 (single-cell) | 75,767 | 8,207 | 155 | 61.2 s (18 epochs) | 233.6 s (16 epochs) | 4.3 s | 7.0 s |

GPU: NVIDIA GeForce RTX 4080 SUPER (16 GB), PyTorch 2.13 with CUDA 12.6. Laptop CPU: Intel Core i7-1360P (12 cores, 16 threads), 32 GB RAM, no GPU, PyTorch 2.13 (CPU). E2M commit `83edb38`. Epochs is the number run before early stopping. Prediction time is mostly the matching of genes to the model's features, so it differs less between devices than training time.

Settings: the TCGA rows use [`configs/manuscript.yaml`](configs/manuscript.yaml) (hidden layers 512 and 256, batch size 64, up to 100 epochs with early stopping); the network has about 10.4 million parameters. The K562 row uses the manuscript's single-cell settings (batch size 1024, learning rate 1.5e-4, inputs already log-normalized, no standardization; about 4.4 million parameters).

The benchmark script is `Runs/Benchmark/timing.py` in [E2M-analyses](https://github.com/asafpinhasitechnion/E2M-analyses) (`python Runs/Benchmark/timing.py [--device cpu]`).
