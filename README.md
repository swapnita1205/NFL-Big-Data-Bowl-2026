# NFL Big Data Bowl 2026 — Player Trajectory Prediction

This repository contains my end-to-end machine learning system for the NFL Big Data Bowl 2026 (Prediction Track).
The goal is to forecast multi-agent player trajectories (x, y positions) for several seconds after the quarterback releases the ball — a challenging problem requiring temporal modeling, spatial reasoning, and multi-player interaction understanding.

This solution is built as a fully modular pipeline covering data normalization, feature engineering, graph construction, temporal-sequence modeling, training, inference, and evaluation.

## What This Project Does

This project implements a player-centric, interaction-aware forecasting model that learns how offensive and defensive players move as a coordinated system.

Core capabilities include:
### Field Direction Normalization (High Impact)

A robust normalization pipeline ensures all plays follow a consistent orientation:

Offense always moves left → right

Y and X axes flipped when required

Coordinates recentered relative to the quarterback

Continuous features smoothed and standardized

This dramatically reduces variance and helps the model learn field semantics.

### Player Interaction Graphs

Each frame builds a K-Nearest Neighbor (KNN) interaction graph, capturing:

Spatial relationships (distance, angles, direction)

Matchups (WR–DB, RB–LB)

Local motion context (relative velocity/acceleration)

This enables the model to reason about strategic behavior, not just trajectories.

### Temporal Transformer Encoder

A lightweight transformer models each player’s temporal history, learning:

Momentum

Direction changes

Cuts and acceleration bursts

Defensive pursuit patterns

### Graph Attention Network (GAT)

Graph attention layers learn how players influence each other, providing:

Contextual awareness

Adaptive interaction weighting

Better modeling of real-time positioning

### Role-Specific Adapters

Dedicated modules for positions (WR, RB, DB, LB, etc.) allow specialization:

Routes vs coverage patterns

Pocket behavior vs edge rush

Open-field vs tight coverage

### Huber Loss

A composite loss that enforces:

Smooth acceleration

Realistic velocity curves

Collision-avoidance penalties

Accurate multi-step predictions

This produces fluid, game-realistic motion paths.

## My Workflow

### 1. Preprocessing & Feature Engineering

- Loaded & optimized tracking data with Polars
- Fully normalized field orientation and coordinate system
- Engineered velocity, acceleration, speed-change metrics
- Constructed frame-by-frame interaction graphs
- Cached processed data into efficient parquet files

### 2. Model Architecture Development

- Built a Temporal Transformer for individual motion encoding
- Added Graph Attention Networks for interaction reasoning
- Implemented role-specific adapters for position-aware behavior
- Designed a multi-step prediction head
- Added a physics-aware loss function
- Trained using PyTorch with learning rate warmup, gradient clipping, early stopping & checkpointing

### 3. Training & Cross-Validation

Split data by week (1–14 train, 15–18 validation) to simulate future-play inference

Logged:

RMSE

smoothness regularization

collision penalties

Generated visual evaluations of:

predicted vs ground truth trajectories

route/coverage reconstruction

Saved best checkpoints per fold for stability

### 4. Inference

- The inference pipeline performs:
- Loading of normalized play data
- Player graph reconstruction per frame
- Multi-step prediction of future (x, y) positions

Generation of clean trajectory visualizations

Designed to run efficiently for Kaggle test-time constraints.
