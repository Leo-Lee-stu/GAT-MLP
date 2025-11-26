# V2 What's New?

Below is a comprehensive list of improvements introduced in **Version 2
(v2)** of the project.\
These enhancements focus on stability, reproducibility, modularity, and
large-scale batch experimentation.

## 1. Cache Isolation

A new directory `cached_graphs_v2/` is created to fully isolate v2
caches from v1, preventing cross-contamination that could cause training
crashes due to mismatched feature dimensions.

## 2. Robust Global Feature Handling

`global_feats.view(batch_size, -1)` is explicitly added inside the
model's `forward()`, eliminating implicit broadcasting issues triggered
by batch_size = 1 or variable-length tensors.

## 3. Centralized Hyperparameter Management

All hyperparameters (`lr`, `hidden_dim`, `dropout`, `patience`,
`test_size`, etc.) are consolidated into a single `params` dictionary,
enabling zero-hardcoded grid search.

## 4. Unified Data Splitting Logic

`train_test_split` is defined once and reused everywhere with the same
`seed` and `test_size`.

## 5. Standardized Class Weight Computation

`compute_class_weight` is always calculated based on the entire dataset.

## 6. Lightweight Early Stopping

Removed logging and plotting side effects; early stopping now retains
only best-model saving and counters.

## 7. Streamlined Code Structure

Removed deprecated functions, comments, and unused imports.

## 8. Stateless Batch Experiment Interface

`run_once_for_batch_v2()` depends only on local variables and `params`.

## 9. Standardized Naming

All APIs include a `_v2` suffix to coexist with v1 safely.

## 10. Independent Model Save Paths

v2 models are saved under `GNN_models_v2/`.

## 11. Explicit Self-loop Behavior

`add_self_loops=True` is documented to maintain compatibility across PyG
versions.

## 12. Edge Index Continuity Check

Added assertion `edge_index.max() < num_nodes` to prevent CUDA memory
issues.

## 13. Fixed Feature Scaling Pipeline

Scaler is saved with graph data to ensure consistency during inference.

## 14. Consistent Dropout Configuration

`self.dropout` ensures grid search affects all dropout layers
consistently.

## 15. Gradient Accumulation Fix

`zero_grad()` is placed before `backward()` to avoid accumulation.

## 16. DataLoader Multithreading Freeze Fix

Explicitly sets `num_workers=0` and `persistent_workers=False` for
Windows compatibility.

## 17. Complete Global Seed Control

Adds cudnn deterministic settings for reproducible GPU results.
