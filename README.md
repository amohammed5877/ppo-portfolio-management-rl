# PPO-Based Stock Portfolio Management with News Sentiment (DRL)

## Overview
This project implements a Deep Reinforcement Learning (DRL) system for stock portfolio allocation using **Proximal Policy Optimization (PPO)**. The agent learns portfolio weights from historical stock market data and integrates **news sentiment** as an additional signal to support better decisions.

Included in this repository:
- Training pipeline (PPO)
- Evaluation pipeline (logs + plots)
- Data loading & sentiment feature processing
- Custom RL environment for portfolio allocation
- Example datasets and saved evaluation outputs

---

## Files in this Repository
- `train.py` — Train PPO agent
- `evaluate_model.py` — Evaluate model and generate logs/plots
- `env.py` — Custom portfolio RL environment
- `data_loader.py` — Load and prepare stock/news data
- `sentiment.py` — Sentiment processing from news
- `app.py` — Optional entry / demo script

### Data
- `stock_train_long.csv` — Training market data
- `stock_test_long.csv` — Test market data
- `stock_news_2025_news.csv` — News dataset for sentiment features

### Outputs
- `evaluation_log.csv`, `evaluation_log_3.csv` — Evaluation logs
- `evaluation_plot.png`, `evaluation_plot_3.png` — Evaluation plots

---

## How to Run

### 1) Create environment & install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

### 2) Train
python train.py

### 3) Evaluate
python evaluate_model.py

Results

Evaluation artifacts are saved in:
evaluation_log.csv / evaluation_log_3.csv
evaluation_plot.png / evaluation_plot_3.png

Notes

Trained model files are not included to keep the repository lightweight.
You can retrain the model using train.py.
pip install -r requirements.txt

Author
Abdul Bari Mohammed
