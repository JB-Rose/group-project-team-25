# RL CAPTCHA System

A reinforcement learning-based bot detection system that processes raw user telemetry using a PPO+LSTM agent. Events are grouped into **windows of 30** and encoded as statistical feature vectors (speed variance, timing regularity, path curvature, etc.). The 2-layer LSTM accumulates evidence across all windows, then makes a terminal decision on the **final window** only.

This package is fully standalone and does not import from TicketMonarch.

## Architecture

```
Raw telemetry events (mouse, click, keystroke, scroll)
        |
        v
Windowed Event Encoder (26-dim feature vector per 30-event window)
        |  Features: speed mean/var, path curvature, click/key timing,
        |  spatial diversity, scroll behavior, interaction quality
        |
        v
LSTM (256 hidden, 2 layers, dropout 0.1) -- accumulates evidence over windows
        |
        |--> Actor head (256 -> 128 -> 64 -> 7 logits) --> action
        +--> Critic head (256 -> 128 -> 64 -> 1 value) --> V(s)
```

### Two-Phase Episode Structure with Action Masking

Episodes have two distinct phases enforced by **action masking** (invalid actions get `-inf` logits):

1. **Observation phase** (all non-final windows): Only actions 0-1 are valid
   - The LSTM processes windows and accumulates evidence
   - Agent can deploy honeypots to gather more information

2. **Decision phase** (final window only): Only actions 2-6 are valid
   - The agent must make a terminal decision based on all accumulated evidence
   - No more observing — must choose: puzzle, allow, or block

This ensures the agent always processes the **entire session** before deciding, preventing shortcut strategies where it decides on window 1.

### Windowed Observation Encoding (26 dimensions)

| Dims | Feature | Discriminative power |
|------|---------|---------------------|
| 0-3 | Event type ratios (mouse/click/key/scroll) | Bot profiles have different event mixes |
| 4-6 | Mouse speed: mean, variance, acceleration | Bots have low speed variance |
| 7 | Path curvature (path length / displacement) | Bots move in straight lines (~1.0) |
| 8-10 | Inter-event timing: mean, variance, min | Bots have near-zero timing variance |
| 11-12 | Click timing: mean interval, variance | Bots click at regular intervals |
| 13-14 | Keystroke hold: mean duration, variance | Bots have mechanical uniform holds |
| 15-16 | Key-press interval: mean, variance | Typing rhythm regularity |
| 17-18 | Scroll: total distance, direction changes | Bots rarely scroll organically |
| 19-22 | Spatial: unique positions, x/y range | Bots visit fewer screen areas |
| 23 | Interactive click ratio | Bots may click non-interactive areas |
| 24 | Window duration (log-normalized) | Time span of the window |
| 25 | Event count / window size | How full the window is |

### Action Space (7 actions)

| Index | Action | Phase | Description |
|-------|--------|-------|-------------|
| 0 | `continue` | Observation | Keep watching (masked on final window) |
| 1 | `deploy_honeypot` | Observation | Deploy invisible trap (masked on final window) |
| 2 | `easy_puzzle` | Decision | 95% human pass, 40% bot pass (masked on non-final) |
| 3 | `medium_puzzle` | Decision | 85% human pass, 15% bot pass (masked on non-final) |
| 4 | `hard_puzzle` | Decision | 70% human pass, 5% bot pass (masked on non-final) |
| 5 | `allow` | Decision | Let user through (masked on non-final) |
| 6 | `block` | Decision | Block user (masked on non-final) |

### Reward Structure

| Outcome | Reward |
|---------|--------|
| Correctly allow human | +0.5 |
| Correctly block/puzzle bot | +1.0 |
| False positive (challenge human) | -1.0 |
| False negative (allow bot) | -0.8 |
| Honeypot catches bot | +0.3 |
| Per-window continue penalty | -0.001 |

## Project Structure

```
rl_captcha/
├── config.py                    # EventEnvConfig, PPOConfig, DBConfig
├── requirements.txt             # torch, gymnasium, numpy, scikit-learn
│
├── data/
│   └── loader.py                # Session dataclass, load_from_directory()
│                                # Supports: chrome extension, live_confirm, flat JSON
│
├── environment/
│   └── event_env.py             # Windowed Gymnasium env (26-dim obs, 7 actions)
│                                # EventEncoder + action masking
│
├── agent/
│   ├── ppo_lstm.py              # PPO algorithm with LSTM recurrence + action masks
│   ├── lstm_networks.py         # LSTMActorCritic (2-layer LSTM, 256 hidden)
│   ├── rollout_buffer.py        # On-policy buffer with GAE + mask storage
│   └── checkpoints/
│       └── ppo_run1/            # Trained model weights
│
└── scripts/
    ├── train_ppo.py             # Train PPO+LSTM agent
    ├── evaluate_ppo.py          # Evaluate (confusion matrix, F1, action distribution)
    ├── plot_training.py         # Visualize training.log (7 figures)
    └── plot_online.py           # Visualize online_training.log (5 figures)
```

## Setup

```bash
pip install -r rl_captcha/requirements.txt
```

Dependencies: PyTorch, Gymnasium, NumPy, scikit-learn, matplotlib.

## Data

Training data lives in `data/human/` (label=1) and `data/bot/` (label=0). All events from a session are used (no truncation).

**Supported file formats:**
- `session_*.json` — Live-confirm format from Dev Dashboard: `{ "sessionId": "...", "segments": [{ "mouse": [...], ... }] }`
- `telemetry_*.json` — Chrome extension export: `{ "<sessionId>": { "segments": [...], "pageMeta": [...] } }`

**Important:** Only include data from the TicketMonarch site (localhost). Data from external sites will pollute the training distribution.

## Training

All commands from the `src/TicketMonarch/` directory.

### 1. Collect Data

**Human data:** Browse the live site normally. Sessions are auto-saved to `data/human/` when online learning runs via the `/api/agent/confirm` endpoint.

**Bot data:** Run bots against the live site:
```bash
python bots/selenium_bot.py --runs 5 --type scripted
python bots/llm_bot.py --runs 3 --provider anthropic
```
Export telemetry to `data/bot/`.

### 2. Train

```bash
python -u -m rl_captcha.scripts.train_ppo \
    --data-dir data/ --total-timesteps 500000 2>&1 | Tee-Object -FilePath training.log
```

Saves checkpoint to `rl_captcha/agent/checkpoints/ppo_run1/`.

### 3. Evaluate

```bash
python -m rl_captcha.scripts.evaluate_ppo \
    --agent rl_captcha/agent/checkpoints/ppo_run1 \
    --data-dir data/
```

### 4. Visualize

```bash
python -m rl_captcha.scripts.plot_training --log training.log --out figures/
python -m rl_captcha.scripts.plot_online --log online_training.log --out figures/
```

## Live Integration

The trained agent is loaded by `TicketMonarch/backend/agent_service.py` for real-time use:

- **`evaluate_session()`** — Full evaluation with action masking: processes all windows through LSTM, applies observation mask on non-final windows and decision mask on the final window
- **`rolling_evaluate()`** — Lightweight polling: returns bot probability from the final window's action distribution
- **`online_learn()`** — PPO update after confirmed human/bot sessions (3 epochs, 60% learning rate) with proper action masking. Logs before/after comparison to `online_training.log`

All methods are thread-safe (wrapped with `threading.Lock`). Online and offline evaluation use identical action masking logic.

## Configuration

All hyperparameters in `config.py`:

- **EventEnvConfig** — Window size (30 events), obs dim (26), min events (10), reward weights, puzzle pass rates, action masking, normalization constants
- **PPOConfig** — Learning rate (3e-4), gamma (0.99), GAE lambda (0.95), clip ratio (0.2), entropy coefficient (0.01), LSTM (256 hidden, 2 layers)
