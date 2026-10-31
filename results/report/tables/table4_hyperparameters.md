**Table 4. Tuned hyperparameters (30-trial random search, 5-fold stratified CV on the seed-0 training set)**

| Model | Best hyperparameters | CV macro-F1 | Tuning time (s) |
|---|---|---|---|
| RF (tuned) | max_depth=24, max_features=0.500, min_samples_leaf=1, n_estimators=228 | 0.990 ± 0.008 | 40.6 |
| XGBoost | colsample_bytree=0.912, learning_rate=0.0276, max_depth=3, min_child_weight=2, n_estimators=365, reg_lambda=1.19, subsample=0.903 | 0.988 ± 0.006 | 18.0 |
| LightGBM | colsample_bytree=0.933, learning_rate=0.165, min_child_samples=5, n_estimators=438, num_leaves=43, reg_lambda=3.97, subsample=0.785, subsample_freq=1 | 0.989 ± 0.006 | 24.6 |
| CatBoost | depth=4, iterations=455, l2_leaf_reg=6.00, learning_rate=0.267 | **0.990 ± 0.008** | 703 |
| Ensemble (calibrated) | members=xgb + lgbm + rf (tuned params), calibration=isotonic, 5-fold, voting=soft | 0.989 ± 0.009 | 12.9 |
| Ensemble (uncalibrated) | members=xgb + lgbm + rf (tuned params), calibration=none, voting=soft | 0.989 ± 0.008 | **2.74** |
