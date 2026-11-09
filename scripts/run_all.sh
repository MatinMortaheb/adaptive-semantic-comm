#!/usr/bin/env bash
# Full pipeline used for the TinyImageNet results of the paper.
# Usage:  bash scripts/run_all.sh --config ../configs/my.yaml
# (my.yaml sets data.root and, optionally, other overrides; see README).
# Each step writes to outputs/ and can be re-run independently.
set -euo pipefail
cd "$(dirname "$0")"
C=("$@")
python s0_finetune_vit.py "${C[@]}" --set train.vit_epochs=5         # DINO ViT-S/8, supervised fine-tuning at 320x480
python s0c_check_classifier.py "${C[@]}"                             # downstream HF ViT classifier: label mapping + clean accuracy
python s2_train_codecs.py --domain upsampled --sizes 12 24 48 "${C[@]}" --set train.codec_epochs=10
python s2_train_codecs.py --domain native --sizes 24 48 "${C[@]}" --set train.codec_epochs=30
python s2_train_codecs.py --domain native --sizes 12 "${C[@]}" --set train.codec_epochs=30 mr.lr=1e-4
python s2b_fit_entropy.py --domain upsampled "${C[@]}"
python s2b_fit_entropy.py --domain native "${C[@]}"
python s1_binary.py "${C[@]}" --set "binary.rates=[1.0,0.75,0.5,0.25]"   # Figs. 2 and 4
python s3_mr_eval.py "${C[@]}"                                       # Table IV (budget violations of the original rule)
python s3n_native_mr.py "${C[@]}"                                    # Figs. 5 and 6, Table I
python s5_rate_table.py "${C[@]}"                                    # Table II
python s6_baselines.py "${C[@]}" --set data.num_test=2000            # Fig. 7
python s7_witt.py "${C[@]}"                                          # Fig. 8 (clones the WITT code at run time)
python s8_robustness.py "${C[@]}"                                    # Fig. 9
python s9_complexity.py "${C[@]}"                                    # Table III
python s10_mr_stdcodec.py "${C[@]}" --set data.num_test=2000         # WebP digital reference and pre-filtering experiment
python make_figures.py "${C[@]}"                                     # figures and tables -> outputs/paper/
