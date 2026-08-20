# DepVidMood Coding Tasks — Solution

Solution to the Stage-2 take-home assignment (facial feature extraction, audio feature
extraction, multimodal fusion + evaluation without ground-truth labels).

## Files

- `face_features.py` — Task 1: face detection + landmark-based visual features (eye
  openness, mouth openness, head pose) via MediaPipe Face Mesh.
- `audio_features.py` — Task 2: pitch, energy, and speaking-rate/pause features via
  librosa, with a simple energy-based voice-activity detector.
- `multimodal_fusion.py` — Task 3: feature-level fusion of the two modalities, plus
  four label-free evaluation strategies (internal validity/clustering, bootstrap
  stability, cross-modal agreement, face-validity sanity checks).
- `run_pipeline.py` — orchestrator: point it at a folder of videos and it runs all
  three tasks end to end.
- `requirements.txt` — pinned dependencies.

## Setup

```bash
pip install -r requirements.txt
```

**Note on mediapipe:** the latest PyPI release (`1.0.x`) dropped the legacy
`mp.solutions` API this code relies on, in favor of a newer Tasks API that needs a
separately-downloaded model file. `requirements.txt` pins `mediapipe==0.10.14`, which
still ships `mp.solutions.face_mesh` and needs no extra downloads. If you `pip install
mediapipe` without the pin you will get an `AttributeError`.

`ffmpeg` must also be available on the system PATH (used by `audio_features.py` to
extract the audio track).

## Dataset

[DepVidMood: Facial Expression Video Dataset](https://www.kaggle.com/datasets/ziya07/depvidmood-facial-expression-video-dataset)
(Kaggle). Only the video files are used, per the task instructions.

To download it:

- **Via browser:** open the link above, sign in to Kaggle, and use the Download
  button. This gives you a `.zip` — extract it anywhere, e.g. `~/data/depvidmood/`.
- **Via Kaggle CLI** (if you have an API token set up at `~/.kaggle/kaggle.json`):
  ```bash
  pip install kaggle --break-system-packages
  kaggle datasets download -d ziya07/depvidmood-facial-expression-video-dataset -p ./data --unzip
  ```

The dataset may contain nested subfolders (e.g. per class/session); `run_pipeline.py`
searches recursively, so it doesn't matter how it's organized once extracted.

## Running

```bash
python run_pipeline.py /path/to/extracted_dataset_folder --out ./results
```

This writes `visual_features.csv`, `audio_features.csv`, and
`fused_evaluation_report.json` to the output folder. Fusion/evaluation is skipped
automatically if fewer than 3 videos have valid features (clustering isn't meaningful
below that).

For a quick sanity check before running the full dataset (which can take a while —
MediaPipe processes every sampled frame), use `--limit` to only process the first N
videos found, and `--every` to skip frames:

```bash
python run_pipeline.py /path/to/extracted_dataset_folder --out ./results_test --limit 5 --every 3
```

Once that runs cleanly and the CSVs look sensible, drop `--limit` (and reduce/drop
`--every`) for the full run.

Each module can also be run standalone on a single video, e.g.:

```bash
python face_features.py path/to/video.mp4 --every 2
python audio_features.py path/to/video.mp4
```

## Approach summary

**Task 1 (visual):** MediaPipe Face Mesh gives 468 landmarks per frame. From these,
Eye Aspect Ratio and Mouth Aspect Ratio are computed as simple, interpretable
openness proxies, and head pose (yaw/pitch/roll) is estimated with `solvePnP` against
a generic 3D face model (camera intrinsics are approximated since the dataset has no
calibration — standard practice for uncalibrated video). Per-frame values are
aggregated to mean and variance per video; variance captures expressivity, not just
average state.

**Task 2 (audio):** the audio track is extracted with ffmpeg, then librosa computes
pitch (pYIN, voiced frames only), RMS energy, and — via a simple adaptive
energy-threshold voice-activity detector — speech ratio, pause count/duration, and a
coarse speaking-rate proxy from onset density. The raw frame-level VAD mask is noisy
(brief energy dips inside continuous speech, or breath noise, flip the label for 1-2
frames at a time), so it's smoothed with a median filter before pause statistics are
computed — verified against a synthetic tone/silence test signal where the unsmoothed
mask produced ~15 spurious "pauses" for what should have been 2.

**Task 3 (fusion + evaluation):** the two video-level feature vectors are
concatenated and z-score standardized (early/feature-level fusion — chosen over a
learned joint embedding because there are no labels to train one against, and
concatenation keeps every fused dimension traceable to a named, interpretable
signal). For evaluation without ground-truth labels, four label-free checks are
implemented: (1) internal validity via KMeans + silhouette score — does the fused
feature space have any coherent structure at all; (2) stability via bootstrap
resampling + Adjusted Rand Index — is that structure robust to which videos happen to
be sampled; (3) cross-modal agreement — do the independently-computed visual and
audio "expressivity" indices correlate, which would suggest both modalities are
picking up a shared real signal rather than each being noise; (4) face-validity
checks — are feature values in the ranges the literature/domain knowledge would
predict. A fifth strategy, robustness under input perturbation, is described in the
code's module docstring but not run, since it requires re-extracting features from
deliberately perturbed video/audio inputs.

## Assumptions and simplifications

- No ground-truth mood/affect labels are assumed to exist for this dataset, per the
  task description — all of Task 3's evaluation is designed around that constraint.
- The voice-activity detector is a simple adaptive-threshold method, not a trained
  model; it can misclassify quiet speech as silence or background noise as speech.
  This is a deliberate simplicity/interpretability trade-off, documented in
  `audio_features.py`.
- The speaking-rate feature is an onset-density proxy, not a true syllable or word
  rate (that would require ASR/transcription, which the task doesn't call for).
- Head pose uses generic (uncalibrated) camera intrinsics; angles are reliable for
  relative comparison across frames/videos but not metrically precise.
- Frames with no detected face are skipped (not imputed); `detection_rate` is
  reported per video as a data-quality flag so low-quality videos are visible rather
  than silently interpolated.
- Missing values in the fused feature table are median-imputed before clustering.

## Testing

The pipeline was validated end-to-end on synthetic data (a real face image from a
standard test corpus, animated into short clips, paired with synthetic tone/silence
audio tracks standing in for speech) rather than the actual DepVidMood dataset, which
requires Kaggle credentials not available in the development environment. This caught
and fixed two real bugs: the mediapipe API-version issue above, and the VAD
smoothing issue described in Task 2. It does not substitute for running against real
DepVidMood videos before the interview.
