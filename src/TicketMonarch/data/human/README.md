# Human Telemetry Data

Real human browsing sessions. Treated as **label=1 (human)** by the training pipeline.

**Important:** Only include data from the TicketMonarch site (localhost). Data from external sites (Gmail, GitHub, Canvas, etc.) will pollute the training distribution and was removed during cleanup.

## How to Collect

### Option 1: Chrome Extension Export

1. Load the Chrome extension from `chrome-extension/` into Chrome
2. Click **"Start Recording"** in the extension popup
3. Browse the TicketMonarch site normally (Home -> Seats -> Checkout -> Purchase)
4. Click **"Export JSON"** in the extension popup
5. Save the file here as `.json`

### Option 2: Online Learning (via API)

1. Browse the site normally while the backend is running
2. Call `POST /api/agent/confirm` with `{ "session_id": "...", "true_label": 1 }`
3. Sessions are auto-saved here as `session_<uuid>.json`

Bot scripts call this endpoint automatically after each run. Sessions are saved and the agent does an online PPO update.

## JSON Formats

The data loader (`rl_captcha/data/loader.py`) supports two formats:

### Chrome Extension Format (`telemetry_*.json`)

One or more sessions keyed by session UUID:

```json
{
  "<session_id>": {
    "sessionId": "...",
    "startTime": 1234567890,
    "pageMeta": [...],
    "totalSegments": 3,
    "segments": [
      {
        "segmentId": 1,
        "url": "http://localhost:3000/",
        "mouse": [{ "x": 100, "y": 200, "t": 1234.5 }],
        "clicks": [{ "t": 1234.5, "x": 100, "y": 200, "button": "left", "dt_since_last": 500 }],
        "keystrokes": [{ "field": "card_number", "type": "down", "t": 1234.5, "key": null }],
        "scroll": [{ "t": 1234.5, "scrollX": 0, "scrollY": 500, "dy": 100 }]
      }
    ]
  }
}
```

### Live Confirm Format (`session_*.json`)

Single session with segments at the top level:

```json
{
  "sessionId": "abc-123",
  "segments": [
    {
      "mouse": [{ "x": 100, "y": 200, "t": 1234.5 }],
      "clicks": [...],
      "keystrokes": [...],
      "scroll": [...]
    }
  ]
}
```

Segments are split by idle gaps (3+ seconds of inactivity). For training, all segments within a session are merged into flat event lists, then grouped into 30-event windows for the windowed observation encoder.

## Usage

All `.json` files here are automatically loaded by:

```bash
python -m rl_captcha.scripts.train_ppo --data-dir data/
```

Note: JSON files are gitignored. Training data stays local only.
