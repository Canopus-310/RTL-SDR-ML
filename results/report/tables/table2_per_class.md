**Table 2. Per-class precision / recall / F1 (best per class in bold)**

| Class | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| FM_BROADCAST | RF (tuned) | 0.998 ± 0.003 | 0.746 ± 0.004 | 0.854 ± 0.003 |
| FM_BROADCAST | XGBoost | 0.994 ± 0.005 | **0.777 ± 0.013** | **0.872 ± 0.009** |
| FM_BROADCAST | LightGBM | 0.998 ± 0.003 | 0.757 ± 0.004 | 0.861 ± 0.003 |
| FM_BROADCAST | CatBoost | **1.00 ± 0.00** | 0.767 ± 0.009 | 0.868 ± 0.006 |
| FM_BROADCAST | Ensemble (calibrated) | 0.998 ± 0.003 | 0.772 ± 0.005 | 0.871 ± 0.003 |
| FM_BROADCAST | Ensemble (uncalibrated) | 0.998 ± 0.003 | 0.758 ± 0.007 | 0.862 ± 0.005 |
| FM_BROADCAST | RF (deployed) | 0.511 ± 0.002 | 0.564 ± 0.002 | 0.536 ± 0.002 |
| DRONE_TELEMETRY | RF (tuned) | 0.970 ± 0.012 | **0.736 ± 0.035** | **0.836 ± 0.019** |
| DRONE_TELEMETRY | XGBoost | 0.932 ± 0.037 | 0.710 ± 0.036 | 0.805 ± 0.020 |
| DRONE_TELEMETRY | LightGBM | 0.959 ± 0.028 | 0.710 ± 0.038 | 0.815 ± 0.018 |
| DRONE_TELEMETRY | CatBoost | **0.972 ± 0.009** | 0.693 ± 0.028 | 0.808 ± 0.018 |
| DRONE_TELEMETRY | Ensemble (calibrated) | 0.939 ± 0.014 | 0.728 ± 0.022 | 0.819 ± 0.011 |
| DRONE_TELEMETRY | Ensemble (uncalibrated) | 0.962 ± 0.024 | 0.712 ± 0.032 | 0.817 ± 0.014 |
| DRONE_TELEMETRY | RF (deployed) | 0.249 ± 0.011 | 0.355 ± 0.028 | 0.293 ± 0.017 |
| DRONE_VIDEO | RF (tuned) | 0.991 ± 0.009 | 0.780 ± 0.002 | 0.873 ± 0.002 |
| DRONE_VIDEO | XGBoost | 0.982 ± 0.007 | 0.870 ± 0.028 | 0.922 ± 0.018 |
| DRONE_VIDEO | LightGBM | 0.982 ± 0.006 | 0.875 ± 0.012 | **0.926 ± 0.010** |
| DRONE_VIDEO | CatBoost | **0.996 ± 0.003** | 0.793 ± 0.018 | 0.883 ± 0.010 |
| DRONE_VIDEO | Ensemble (calibrated) | 0.988 ± 0.010 | 0.862 ± 0.021 | 0.921 ± 0.017 |
| DRONE_VIDEO | Ensemble (uncalibrated) | 0.983 ± 0.005 | 0.867 ± 0.024 | 0.921 ± 0.015 |
| DRONE_VIDEO | RF (deployed) | 0.469 ± 0.012 | **1.00 ± 0.00** | 0.638 ± 0.011 |
| INTERFERENCE | RF (tuned) | 0.821 ± 0.032 | 0.881 ± 0.077 | 0.848 ± 0.043 |
| INTERFERENCE | XGBoost | 0.816 ± 0.028 | 0.917 ± 0.062 | 0.862 ± 0.028 |
| INTERFERENCE | LightGBM | 0.821 ± 0.038 | **0.941 ± 0.012** | **0.876 ± 0.024** |
| INTERFERENCE | CatBoost | 0.817 ± 0.033 | 0.914 ± 0.048 | 0.862 ± 0.033 |
| INTERFERENCE | Ensemble (calibrated) | **0.827 ± 0.029** | 0.872 ± 0.036 | 0.848 ± 0.018 |
| INTERFERENCE | Ensemble (uncalibrated) | 0.823 ± 0.035 | 0.926 ± 0.053 | 0.870 ± 0.030 |
| INTERFERENCE | RF (deployed) | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| NOISE | RF (tuned) | 0.568 ± 0.015 | **1.00 ± 0.00** | 0.724 ± 0.012 |
| NOISE | XGBoost | 0.630 ± 0.026 | 0.999 ± 0.002 | 0.772 ± 0.019 |
| NOISE | LightGBM | **0.630 ± 0.020** | 0.999 ± 0.002 | **0.773 ± 0.015** |
| NOISE | CatBoost | 0.577 ± 0.002 | 0.999 ± 0.002 | 0.731 ± 0.002 |
| NOISE | Ensemble (calibrated) | 0.603 ± 0.021 | 0.999 ± 0.002 | 0.752 ± 0.017 |
| NOISE | Ensemble (uncalibrated) | 0.618 ± 0.019 | 0.999 ± 0.002 | 0.763 ± 0.014 |
| NOISE | RF (deployed) | 0.279 ± 0.007 | 0.0696 ± 0.0035 | 0.111 ± 0.005 |
| LORA | RF (tuned) | 0.995 ± 0.004 | 0.865 ± 0.033 | 0.925 ± 0.018 |
| LORA | XGBoost | 0.998 ± 0.002 | 0.855 ± 0.025 | 0.921 ± 0.015 |
| LORA | LightGBM | **0.998 ± 0.002** | 0.874 ± 0.026 | **0.932 ± 0.015** |
| LORA | CatBoost | 0.997 ± 0.002 | 0.871 ± 0.025 | 0.929 ± 0.013 |
| LORA | Ensemble (calibrated) | **0.998 ± 0.002** | 0.862 ± 0.018 | 0.925 ± 0.010 |
| LORA | Ensemble (uncalibrated) | **0.998 ± 0.002** | 0.872 ± 0.017 | 0.931 ± 0.010 |
| LORA | RF (deployed) | 0.879 ± 0.007 | **0.955 ± 0.004** | 0.915 ± 0.005 |
