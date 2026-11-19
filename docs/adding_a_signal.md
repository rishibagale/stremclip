# Adding a New Signal to streamclip

This guide demonstrates how to extend `streamclip` with a custom signal extractor (e.g. OCR kill-feed tracker, heart-rate monitor, emotion detector, or chat sentiment).

---

## 1. The Signal Contract

Every signal in `streamclip` adheres to the `SignalTrack` model:
- **Sampling Frequency**: Exactly 1.0 Hz (one value per second of video).
- **Track Length**: Exactly $\lceil \text{vod.duration\_s} \rceil$.
- **Values**: Raw, non-normalized values (e.g. event count, dB, intensity score). Normalization is handled centrally by the fusion engine.
- **Metadata**: A dictionary containing at minimum `activity_confidence: float` (in $[0.0, 1.0]$) to inform multimodal fusion when the signal is weak or uninformative.

---

## 2. Implement the `SignalExtractor` Protocol

Create your extractor under `src/streamclip/signals/` implementing `SignalExtractor` from `streamclip.interfaces`:

```python
# src/streamclip/signals/sentiment.py
from __future__ import annotations

import math
from typing import Any
import numpy as np

from streamclip.config import Config
from streamclip.interfaces import SignalExtractor
from streamclip.models import ChatMessage, SignalTrack, VodMeta


class ChatSentimentExtractor(SignalExtractor):
    """Computes per-second positive/negative sentiment polarity."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    @property
    def name(self) -> str:
        return "sentiment"

    def extract(self, vod: VodMeta, **kwargs: Any) -> SignalTrack:
        total_seconds = math.ceil(vod.duration_s)
        chat_messages: list[ChatMessage] = kwargs.get("chat_messages", [])

        if not chat_messages:
            return SignalTrack(
                name=self.name,
                hz=1.0,
                values=[0.0] * total_seconds,
                meta={"empty": True, "activity_confidence": 0.0},
            )

        # Aggregate sentiment values into 1-second bins
        bins = np.zeros(total_seconds, dtype=np.float64)
        for msg in chat_messages:
            sec = int(math.floor(msg.t))
            if 0 <= sec < total_seconds:
                # Compute arbitrary metric, e.g. text polarity
                polarity = 1.0 if "hype" in msg.text.lower() else 0.1
                bins[sec] += polarity

        return SignalTrack(
            name=self.name,
            hz=1.0,
            values=bins.tolist(),
            meta={"activity_confidence": min(1.0, len(chat_messages) / 100.0)},
        )
```

---

## 3. Register the Signal in Configuration

Add the signal to `FusionWeights` in `src/streamclip/config.py`:

```python
class FusionWeights(BaseModel):
    chat: float = 0.5
    audio: float = 0.3
    vision: float = 0.2
    sentiment: float = 0.1  # Add your new signal weight
```

---

## 4. Hook Into the Pipeline

Include your new signal extractor in `src/streamclip/pipeline.py` inside `extract_signals_parallel`:

```python
# In pipeline.py
sentiment_track = extract_sentiment_signal(vod, chat_messages=chat_messages)
signals.append(sentiment_track)
```

`fuse_signals` in `streamclip.fusion` will automatically detect the track by name, compute rolling z-scores, scale by its `activity_confidence`, and blend it according to `config.fusion_weights.sentiment`.

---

## 5. Add Tests & Verification

Verify your signal by adding a test file in `tests/test_signals_sentiment.py`:
1. **Length match**: Check `len(track.values) == math.ceil(vod.duration_s)`.
2. **Empty source fallback**: Verify an empty chat log emits zero values with `activity_confidence == 0.0`.
3. **Burst detection**: Verify that synthetic burst timestamps produce corresponding elevated peaks.
