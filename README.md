# DepVidMood Coding Tasks — Solution

My solution for the Stage-2 take-home: face features, audio features, and combining
the two + checking if it's any good without labels.

## Files

- `face_features.py` — Task 1. Gets face landmarks with MediaPipe, then works out eye
  openness, mouth openness, and head pose from them.
- `audio_features.py` — Task 2. Pitch, energy, speaking rate/pauses, using librosa.
- `multimodal_fusion.py` — Task 3. Combines both, then checks if the result means
  anything using four different label-free checks.
- `run_pipeline.py` — runs all three on a whole folder of videos at once.
- `requirements.txt` — what to install.

## Setup

```bash
pip install -r requirements.txt
```

Heads up on mediapipe: the newest version on PyPI (1.0.x) removed the API this code
uses, so `requirements.txt` pins it to `0.10.14` instead. If you install mediapipe
without that pin you'll get an `AttributeError`.

You also need `ffmpeg` installed and on your PATH — that's what pulls the audio out
of each video.

## Dataset

[DepVidMood on Kaggle](https://www.kaggle.com/datasets/ziya07/depvidmood-facial-expression-video-dataset).


Doesn't matter how the folders are nested inside — `run_pipeline.py` searches through
subfolders automatically.

## Running

```bash
python run_pipeline.py /path/to/dataset --out ./results
```

Writes out `visual_features.csv`, `audio_features.csv`, and
`fused_evaluation_report.json`. If fewer than 3 videos come through okay, it skips
the fusion/clustering step since that needs more than 2-3 data points to mean
anything.

Worth testing on a handful of videos first rather than the whole dataset blind:

```bash
python run_pipeline.py /path/to/dataset --out ./results_test --limit 5 --every 3
```

Check the numbers look sane, then drop `--limit` for the real run.

Each file also works on its own, on a single video:

```bash
python face_features.py path/to/video.mp4 --every 2
python audio_features.py path/to/video.mp4
```

## How each part works

**Task 1, visual:** MediaPipe gives 468 points on the face per frame. From those I
calculate eye openness and mouth openness as simple ratios, and head pose (which way
the head is turned) using OpenCV's solvePnP against a generic face shape, since we
don't have camera calibration info for this dataset. For each video I take the
average and the variance of these over all frames — variance because how much
something moves matters as much as its average value.

One thing worth knowing: eye/mouth openness use normal average/variance, but head
pose angles don't — they use what's called circular variance instead. Angles wrap
around (180° and -180° are basically the same direction), and normal variance
doesn't know that. On the real dataset, a couple of videos had a head-pose variance
number in the tens of thousands because of this, while all the others were single
digits. Switched to circular stats and it fixed itself.

**Task 2, audio:** pull the audio out with ffmpeg, then get pitch (librosa's pYIN,
only counting frames where someone's actually voicing), energy (RMS), and speaking
rate/pauses from a simple loudness-based voice detector. That detector is noisy on
its own — quiet moments mid-word get misread as pauses — so I smooth it with a
median filter before counting pauses. Without that smoothing, a test clip with 2 real
silences came out as ~15 "pauses."

**Task 3, fusion + evaluation without labels:** stick the visual and audio features
together into one vector per video, scale everything so no single feature dominates
just because of its units, then reduce it with PCA to cut down redundancy. No
trained/learned fusion model, since there's nothing to train it against without
labels — just combining features and keeping them traceable back to what they mean.

For checking if any of this is actually meaningful without ground truth, four things:
cluster the videos and see if there's any real structure (silhouette score);
re-cluster on random subsets and see if you get the same answer each time (bootstrap
+ ARI); check the feature values look plausible (pitch in normal human range, etc.);
and check if visual "expressiveness" and audio "expressiveness" agree with each
other across videos. A fifth idea — re-running everything on slightly altered
video/audio to see if the features hold steady — is written up but not run, since it
needs a second pass through Tasks 1/2.

## Assumptions / things I simplified

- Assuming there really are no mood labels for this dataset, per the task brief —
  all of Task 3 is built around that.
- The voice-activity detector is a simple threshold, not a trained model, so it can
  get confused by quiet speech or background noise.
- Speaking rate is a rough proxy (counting energy onsets), not real syllables — that
  would need speech-to-text, which wasn't asked for.
- Head pose angles are good for comparing across frames/videos, not for exact
  degrees, since there's no camera calibration.
- If no face is found in a frame, that frame is just skipped, not guessed at. Each
  video reports what fraction of frames it actually found a face in.
- Any missing values in the final combined table get filled with the median before
  clustering.

## Testing

Tried it first on made-up data (a face photo turned into a short video, with a fake
tone-and-silence audio track standing in for speech), then ran it for real on the
whole DepVidMood dataset — 240 videos, all processed fine.

## Bugs I hit and fixed

- **mediapipe broke immediately.** Newest version dropped the API I needed. Pinned
  it to `0.10.14` in requirements.txt and it works.
- **Pauses were way overcounted.** The raw voice detector flickers on/off for a
  frame or two at a time even mid-speech. Added smoothing before counting pauses —
  went from ~15 false pauses down to the correct 2 on my test clip.
- **Head pose variance was way too high on a couple of real videos.** Turned out to
  be the 180°/-180° wraparound problem described above. Fixed with circular
  variance instead of normal variance.
- **Couldn't tell what the PCA components actually meant.** Added a small function
  that prints which original features drive each component the most, so it's not
  just a black box.
- **numpy/pandas broke on my Windows/Anaconda setup.** Mixing pip installs into an
  existing Anaconda environment caused a version mismatch error. Made a fresh
  virtual environment instead and installed everything clean there — fixed it.

## What I actually found (240 videos)

- PCA got the 22 features down to 13 without losing much (kept ~92% of the
  variation). The single biggest factor turned out to be video length, not
  anything about the person — pointing that out honestly rather than hiding it.
- Clustering found some structure, but it's weak (silhouette ≈ 0.14). Not clean,
  separate groups.
- That weak structure held up reasonably well when I reran it on random subsets
  (~63% agreement) — so it's weak, but not just noise either.
- Visual expressiveness and audio expressiveness didn't line up with each other at
  all (correlation ≈ 0.05, not statistically meaningful). Genuine negative result,
  not spinning it as a win.
- Feature values mostly look sane — eye/mouth openness and speech ratio all
  reasonable, pitch reasonable for ~96% of videos. The rest showed pitch way too
  high, most likely the pitch detector locking onto a harmonic rather than an
  actual bug.
