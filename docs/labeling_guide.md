# Human Ground-Truth Labeling Guide for Livestream Highlights

This guide establishes the annotation protocol for human annotators creating ground-truth `labels.json` datasets from real livestream VODs. These labels are used to benchmark detection metrics (Recall@K, Precision@K) and optimize hyperparameter weights with `streamclip tune`.

---

## 1. What Constitutes a Highlight?

In gaming and livestream entertainment (specifically Grand Theft Auto VI and similar high-action streams), a highlight moment is an engaging segment that can stand on its own in a vertical short-form format.

### Key Types of Highlight Moments
1. **Dramatic Gameplay Action**: Major car chases, stunts, explosions, close escapes, unexpected missions, or intense gunfights.
2. **Humorous & Glitch Moments**: Physics bugs, funny NPC interactions, streamer slip-ups, or chaotic accidents.
3. **Streamer Reactions & Rants**: Loud laughter, sudden screaming, shocking reveals, or chat interaction punchlines.
4. **Chat In-Jokes & Hype Chains**: Moments that sparked an overwhelming spam flood of emotes (`LUL`, `KEKW`, `POG`, `OMEGALUL`).

---

## 2. Boundary Annotation Rules ($t_{start}$ and $t_{end}$)

Accurate clip boundaries are critical. Follow these criteria when marking start and end times:

### $t_{start}$ (Clip Beginning)
- Include sufficient **context / buildup** so viewers understand what is happening.
- Start before the critical event begins (typically 5 to 10 seconds before the peak action).
- Whenever possible, align $t_{start}$ to a natural pause in speech or the start of a sentence.

### $t_{end}$ (Clip Conclusion)
- Include the **immediate climax and reaction** (e.g. the crash, streamer laughter, chat response).
- End as soon as the streamer finishes their thought or the energy returns to neutral.
- Target a total interval length between **15 and 60 seconds**.

---

## 3. Ground Truth File Format (`labels.json`)

Ground-truth files must be strictly formatted as a JSON array of objects with `t_start` and `t_end` floats (in seconds relative to the beginning of the VOD):

```json
[
  {
    "t_start": 54.0,
    "t_end": 78.5,
    "description": "Helicopter crash near downtown Vinewood"
  },
  {
    "t_start": 172.0,
    "t_end": 204.0,
    "description": "Streamer reaction to surprise NPC dialogue"
  },
  {
    "t_start": 845.5,
    "t_end": 880.0,
    "description": "Police chase climax and narrow bridge jump"
  }
]
```

> [!NOTE]
> `streamclip eval` and `streamclip tune` require `t_start` and `t_end`. Additional fields such as `description` or `category` are ignored by the parser but encouraged for documentation.

---

## 4. Dataset Size & Sampling Guidelines

To prevent overfitting when tuning weights:
- **Duration**: Annotate at least **5 to 10 hours** of total livestream footage.
- **Diversity**: Sample across **3 or more distinct streamers** (different microphone setups, facecam positions, chat speeds, and playing styles).
- **Split**: Separate your labeled data into an 80/20 train/test split. Run `streamclip tune` on the training set and evaluate generalization on the holdout test set with `streamclip eval`.

---

## 5. Running Evaluation & Hyperparameter Tuning

### Step 1: Benchmark Baseline Accuracy
```bash
streamclip eval --labels labels.json --video vod.mp4 --chat chat.csv --top-k 5
```

A candidate detection is counted as a **hit** if:
$$\text{Overlap} = \frac{\text{Candidate} \cap \text{GroundTruth}}{\text{Candidate} \cup \text{GroundTruth}} \ge 0.50$$
or if the candidate's peak timestamp ($t_{peak}$) lies strictly inside $[t_{start}, t_{end}]$.

### Step 2: Optimize Fusion Weights & Window Bounds
Run grid or random search to maximize Recall@K while maintaining precision above a desired threshold:
```bash
streamclip tune \
  --labels labels_train.json \
  --video vod.mp4 \
  --chat chat.csv \
  --output configs/tuned.yaml \
  --method random \
  --trials 50 \
  --precision-floor 0.6 \
  --top-k 5
```

The resulting `configs/tuned.yaml` can then be passed to `streamclip run --config configs/tuned.yaml` for production highlight generation.
