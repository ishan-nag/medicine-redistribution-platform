# Hyperparameter Comparison

Grid search used 3-fold stratified cross-validation on the training split, scored by macro F1. Each selected configuration was then evaluated once on the held-out test split.

| Model | Grid | Best parameters | CV F1 macro | Test F1 macro | Test accuracy |
|---|---|---|---:|---:|---:|
| Decision Tree | max_depth=[8, 12, None], min_samples_leaf=[2, 5, 10] | max_depth=8, min_samples_leaf=2 | 0.9199 | 0.9127 | 0.9125 |
| Random Forest | max_depth=[None, 12], min_samples_leaf=[1, 2, 3] | max_depth=12, min_samples_leaf=3 | 0.9270 | 0.9218 | 0.9213 |
| Extra Trees | max_depth=[None, 12], min_samples_leaf=[1, 2, 3] | max_depth=12, min_samples_leaf=1 | 0.9014 | 0.9025 | 0.9038 |
| Gradient Boosting | n_estimators=[100, 200], learning_rate=[0.1], max_depth=[2, 3] | learning_rate=0.1, max_depth=3, n_estimators=100 | 0.9232 | 0.9079 | 0.9075 |
| Hist Gradient Boosting | max_iter=[100, 200], max_leaf_nodes=[15, 31] | max_iter=200, max_leaf_nodes=31 | 0.9154 | 0.9081 | 0.9087 |
| Logistic Regression | estimator__C=[0.1, 1, 10] | estimator__C=10 | 0.9090 | 0.8943 | 0.8962 |
