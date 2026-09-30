<h1 align="center">[NeurIPS 2026] PhysVista: Benchmarking Physical Intelligence in VLMs via a Perception-Reasoning-Assessment Loop</h1>

<p align="center">
 &nbsp;
  <a href="https://huggingface.co/datasets/HelenPeng/PhysVista">🤗 Dataset</a>
  &nbsp; | &nbsp;
  <a href="YOUR_PAPER_URL">📄 Paper</a>
</p>

---

## Benchmark Overview

PhysVista covers both **real-world** and **generated videos** and evaluates physical intelligence across three stages:

| Stage | Tasks |
| --- | --- |
| **Seeing: Physical State Perception** | Spatial State Perception, Violation Localization, Camera Motion Recognition, Quantitative Scale Estimation, Uncertainty Awareness |
| **Reasoning: Physical Dynamics Reasoning** | Temporal Order Reconstruction, Physical Mechanism Reasoning, Physical Principle Violation Reasoning, Physical Dynamics Prediction, Counterfactual Reasoning, Physical Violation Critique |
| **Assessment: Physical Plausibility Assessment** | Physical Plausibility Scoring, Physical Plausibility Comparison |
---
We use the following evaluation metrics:

- **Micro-F1:** Physical Principle Violation Reasoning
- **SRCC / PLCC:** Physical Plausibility Scoring
- **Accuracy:** All remaining tasks

---

## Data Loading Guide

### 1. Data Sources

Different tasks use videos from different source datasets.

| Task | Data Source |
| --- | --- |
| Camera | `cam` |
| Critique | `videophy2_train` |
| Localization | `videophy2_train` |
| Violation | `videophy2_train` |
| Prediction | `WISA-80K`, `videophy2_train`, `videophy2_test` |
| Spatial, Scale, Uncertainty, Mechanism, Counterfactual, Order, Comparison, Score | `WISA-80K`, `videophy2_train` |

### 2. Field Descriptions

Each annotation item generally contains the following fields:

- `name`: Video filename.
- `source`: Data source of the video.
- `extract_frames`: Number of pre-extracted frames available for the video.
- `prefix`: Prompt prefix used to guide model output.
- `suffix`: Prompt suffix used to guide model output.
- `question`: Question sent to the model.
- `options`: Candidate answers sent to the model.
- `options_answer`: Ground-truth answer used **only for evaluation**. This field must **never be included in the model input**.
- `frame_select`: Used only for the **Order** task. It specifies the indices of the frames provided to the model.

The **Comparison** task uses separate fields for the two input videos, including `video1_name`, `video2_name`, `video1_source`, `video2_source`, `video1_extract_frames`, and `video2_extract_frames`.

### 3. Frame Path Format

All tasks directly use **pre-extracted frames**.

The original MP4 videos do not need to be decoded or re-sampled during inference.

For a standard single-video task, the frame path follows:

```text
/.../{source}/frames/{video_stem}_{frame_index:04d}.jpg
```

For example:

```text
source = "videophy2_train"
name = "videophy2_21.mp4"
video_stem = "videophy2_21"
frame_index = 0
```

corresponds to:

```text
/.../videophy2_train/frames/videophy2_21_0000.jpg
```

### 4. Task-Specific Frame Selection

Different tasks use different frame-selection strategies:

- **Most tasks:** use all `extract_frames` frames.
- **Comparison:** use all `video1_extract_frames` frames from the first video and all `video2_extract_frames` frames from the second video.
- **Order:** use only the frames specified by `frame_select` in the given order.
- **Prediction:** use only the first half of the extracted frames.

The following frame-loading functions are used for different tasks:

- `all_video_frames_1`: Spatial, Camera, Scale, Uncertainty, Mechanism, Violation, Counterfactual, Critique, Localization, and Score
- `all_video_frames_2`: Comparison
- `all_video_frames_3`: Order
- `all_video_frames_4`: Prediction

```python
import json
from pathlib import Path

FRAME_ROOT = Path("/.../")

def load_gt(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_frame_path(source, video_name, frame_index):
    video_stem = Path(video_name).stem
    return (
        FRAME_ROOT
        / source
        / "frames"
        / f"{video_stem}_{frame_index:04d}.jpg"
    )

# Most tasks
def all_video_frames_1(item):
    return [
        make_frame_path(item["source"], item["name"], index)
        for index in range(item["extract_frames"])
    ]

# Comparison
def all_video_frames_2(item):
    video1_frames = [
        make_frame_path(
            item["video1_source"],
            item["video1_name"],
            index,
        )
        for index in range(item["video1_extract_frames"])
    ]

    video2_frames = [
        make_frame_path(
            item["video2_source"],
            item["video2_name"],
            index,
        )
        for index in range(item["video2_extract_frames"])
    ]

    return video1_frames + video2_frames

# Order
def all_video_frames_3(item):
    return [
        make_frame_path(item["source"], item["name"], index)
        for index in item["frame_select"]
    ]

# Prediction
def all_video_frames_4(item):
    return [
        make_frame_path(item["source"], item["name"], index)
        for index in range(item["extract_frames"] // 2)
    ]
```

### 5. Model Input Prompt

The model input consists of the following elements in order:

1. Image frames selected according to the task-specific frame-selection rule.
2. `prefix`
3. `question`
4. `options`
5. `suffix`

The `options_answer` field must **not** be included in the model input. It should only be accessed after inference for evaluation.

The textual prompt can be constructed as follows:

```python
prompt = "\n".join(
    item[key].strip()
    for key in ("prefix", "question", "options", "suffix")
    if item[key].strip()
)
```


## Contact

For questions, please contact: **xg.pengv@gmail.com**

## Citation

```bibtex
@article{physvista2026,
  title   = {PhysVista: Benchmarking Physical Intelligence in VLMs via a Perception-Reasoning-Assessment Loop},
  author  = {...},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2026}
}
```