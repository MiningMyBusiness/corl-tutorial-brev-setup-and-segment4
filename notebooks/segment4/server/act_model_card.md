---
license: other
license_name: suturebot-authors-permission
license_link: LICENSE
library_name: pytorch
pipeline_tag: robotics
tags:
- robotics
- surgical-robotics
- imitation-learning
- act
- dvrk
datasets:
- nvidia/PhysicalAI-Robotics-Open-H-Embodiment
---

# SutureBot ACT (multi-task, dot-conditioned), CoRL 2026 tutorial copy

The ACT policy checkpoint used in Segments 3 and 4 of the CoRL 2026 tutorial *Robot Learning on the Path to Surgical Autonomy*, published here with the SutureBot authors' permission so participants can use it after the tutorial.

- **Model:** Action Chunking Transformer (ACT) with FiLM-conditioned EfficientNet-B3 image encoders and DistilBERT task embeddings, 122.7 M parameters, trained by the [SutureBot](https://suturebot.github.io) authors (epoch 6311).
- **File:** `dot_best_6311.ckpt` holds `model_state_dict` and `epoch` only. The optimizer and scheduler state of the original 1.4 GB training checkpoint were removed; predictions are identical.
- **Normalization:** `checkpoint_normalization.json` (position and jaw channels standardized; rotation channels pass through).

## Inputs and outputs
- **In:** three RGB images (left endoscope, left wrist, right wrist) resized to 480×360 with ImageNet normalization; the task name (`needle pickup`, `needle throw`, `knot tying`) as a DistilBERT `[CLS]` embedding; robot state input is zeros.
- **Out:** a 60-step chunk at 30 Hz, 20 values: per arm, position change, the first two columns of the relative rotation matrix, and jaw angle. Absolute pose = measured pose + change (the authors' deployment also subtracts the first predicted translation).

## Run it
Model code: the SutureBot repo, ACT branch (https://github.com/SutureBot/SutureBot/tree/ACT), `src/act/detr`. Two ready-made paths:
- the Segment 3 notebook (`OpenHTutorial_SutureBotACT_Inference.ipynb`), which loads this file unchanged;
- `notebooks/segment4/server/act_policy.py` in the tutorial repo, a small wrapper that also returns 16-D poses (xyz + quaternion xyzw + jaw), like GR00T-H.

## Precomputed predictions (`predictions/act_<episode>.npz`)
For the held-out SutureBot clips 1441–1451 of [Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment), as used in the Segment 4 notebook. All arrays are (N, 16) absolute poses at 30 Hz.
- `open_loop`: the whole clip, GR00T-H's open-loop protocol: a new chunk from the real frames every 50 steps, first 50 rows kept.
- `chunk`: the 60-step chunk at `start_frame`.
- `dream`: the same receding-horizon rule from `start_frame`; it drives Cosmos-H-Dreams in Segment 4.

{rows}

## Caveats
- **Needle throw needs goal dots.** This checkpoint was trained with insertion and exit points drawn on the endoscope image for needle throw. Those annotations are not in Open-H, so on the needle-throw clips (1441–1444) the model runs without them, and its results there are diagnostic only. Knot tying (1445–1451) needs no dots.
- **Training split not recorded.** The SutureBot training split behind this checkpoint was not recorded with it, so it may include some Open-H test-split clips. Treat comparisons on 1441–1451 as a demonstration, not a benchmark.
- These are offline action comparisons, not the paper's physical suturing results.

## Citation
Haworth et al., *SutureBot: A Precision Framework & Benchmark For Autonomous End-to-End Suturing*, NeurIPS 2025 Datasets and Benchmarks ([arXiv:2510.20965](https://arxiv.org/abs/2510.20965)).
