# Adaptive Transformer-Aided Semantic Communication with Multi-Resolution Encoding

Code for the TinyImageNet experiments of

> M. Mortaheb, M. A. Khojastepour, and S. Ulukus, "Adaptive Transformer-Aided Semantic Communication with Multi-Resolution Encoding," *npj Wireless Technology* (2026).

A vision transformer (DINO ViT-S/8) computes the attention of the classification token to each image patch. The attention map drives (i) a binary patch selection with a random-selection parameter alpha and (ii) a budget-feasible multi-resolution allocation (Algorithm 2) over learned per-patch encoder-decoder pairs. Rates are counted bit by bit, including the side information, and reported in bits per native pixel.

## Installation

```bash
conda create -n semcomm python=3.11 -y && conda activate semcomm
pip install torch torchvision          # choose the wheel that matches your CUDA version
pip install -r requirements.txt
```

Download TinyImageNet (http://cs231n.stanford.edu/tiny-imagenet-200.zip), unzip it, copy
`configs/example_my.yaml` to `configs/my.yaml`, and set `data.root`.

Downloaded automatically at run time: the DINO ViT-S/8 weights, the downstream classifier
(Hugging Face `tzhao3/vit-tiny-imagenet`), and the WITT code (https://github.com/KeYang8/WITT),
which is cloned by `scripts/s7_witt.py` and is not redistributed here.

## Reproducing the results

```bash
bash scripts/run_all.sh --config ../configs/my.yaml
```

runs the whole pipeline (several days on one GPU; the decoder training of `s1_binary.py` is the
longest step). Each step can also be run alone from `scripts/`, e.g.
`python s6_baselines.py --config ../configs/my.yaml --set data.num_test=2000`.
Put further overrides in `my.yaml` rather than on the command line (a second `--set` replaces the first).
Every script has a `--smoke` option that runs the pipeline end to end on tiny synthetic data on a CPU.

| Script | Output | Paper |
|---|---|---|
| `s0_finetune_vit.py` | supervised fine-tuning of the DINO ViT-S/8 at 320x480 | Methods |
| `s0c_check_classifier.py` | label mapping and clean accuracy of the downstream classifier | Methods |
| `s2_train_codecs.py`, `s2b_fit_entropy.py` | per-resolution patch codecs and their entropy models | Methods |
| `s1_binary.py` | binary scheme: masked MSE, full-image PSNR/MS-SSIM, alpha sweep | Figs. 2, 4 |
| `s3n_native_mr.py` | single- and multi-resolution transmission, level-set ablation (native patches) | Figs. 5, 6; Table I |
| `s5_rate_table.py` | bit-level rate accounting | Table II |
| `s9_complexity.py` | parameters, GFLOPs, latency | Table III |
| `s3_mr_eval.py` | budget violations of the original allocation rule | Table IV |
| `s6_baselines.py` | uniform-rate coding, IAQ, JPEG/JPEG2000/WebP, native-resolution transmission | Fig. 7 |
| `s7_witt.py` | WITT at an equal number of channel uses over AWGN | Fig. 8 |
| `s8_robustness.py` | Rayleigh block fading, imperfect rate information, packet erasures | Fig. 9 |
| `s10_mr_stdcodec.py` | importance-guided pre-filtering for WebP/JPEG and the WebP digital reference | Results text |
| `make_figures.py` | all figures (PDF/PNG) and LaTeX tables in `outputs/paper/` | |

`s4_level_ablation.py` (level-set ablation with the codecs on the upsampled image) and
`s0b_train_classifier.py` (an alternative ResNet classifier) are optional.
`tests/test_allocation.py` checks that Algorithm 2 never exceeds the budget.

## Licence

MIT (see `LICENSE`). `semcomm/dino_vit.py` is adapted from facebookresearch/dino (Apache-2.0, `semcomm/DINO_LICENSE`).

## Citation

```bibtex
@article{mortaheb2026adaptive,
  title   = {Adaptive Transformer-Aided Semantic Communication with Multi-Resolution Encoding},
  author  = {Mortaheb, Matin and Khojastepour, Mohammad A. and Ulukus, Sennur},
  journal = {npj Wireless Technology},
  year    = {2026}
}
```
