**Table 5. Probability quality and end-to-end false alarms**

| Model | Log loss | Brier score | ECE | Noise captures with a non-NOISE label |
|---|---|---|---|---|
| RF (tuned) | 0.747 ± 0.088 | 0.246 ± 0.002 | **0.0780 ± 0.0070** | **0 ± 0** |
| XGBoost | 0.614 ± 0.025 | 0.246 ± 0.012 | 0.0896 ± 0.0104 | **0 ± 0** |
| LightGBM | 0.751 ± 0.018 | 0.243 ± 0.006 | 0.106 ± 0.005 | **0 ± 0** |
| CatBoost | 0.690 ± 0.034 | 0.276 ± 0.013 | 0.119 ± 0.009 | **0 ± 0** |
| Ensemble (calibrated) | 1.31 ± 0.33 | 0.238 ± 0.003 | 0.0820 ± 0.0065 | **0 ± 0** |
| Ensemble (uncalibrated) | **0.527 ± 0.008** | **0.234 ± 0.005** | 0.0813 ± 0.0040 | **0 ± 0** |
| RF (deployed) | 5.33 ± 0.12 | 0.670 ± 0.005 | 0.157 ± 0.004 | **0 ± 0** |
