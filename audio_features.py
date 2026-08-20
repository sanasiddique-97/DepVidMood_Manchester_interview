"""
Task 2: Audio Feature Extraction (Speech Modality)

Pipeline per video:
    1. Extract the audio track with ffmpeg (mono, 16 kHz -- standard
       for speech-analysis libraries).
    2. Compute standard, interpretable speech features with librosa:
        - Pitch (F0): via the pYIN algorithm (probabilistic YIN),
          restricted to voiced frames only.
        - Energy: frame-wise RMS.
        - Speaking-rate / pause statistics: a simple energy-based
          voice-activity detector (VAD) is used to mark each frame as
          speech / silence, from which we derive:
              * speech ratio (fraction of the clip that is speech)
              * number of pauses
              * mean pause duration
              * a coarse "syllables-per-second" proxy from energy-onset
                peaks within speech segments (NOT a true ASR-based
                speaking rate -- documented as an approximation, since
                no transcript/ASR is required by the task).
    3. Aggregate frame-level pitch/energy into mean & variance to align
       with the video-level granularity and report the
       scalar pause/speaking-rate statistics alongside them.

"""

import subprocess
import tempfile
import os
import numpy as np
import librosa


def extract_audio(video_path, sr=16000):
    """Extract mono audio from a video file into a temp .wav using ffmpeg."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-loglevel", "error",
        tmp_wav.name,
    ]
    subprocess.run(cmd, check=True)
    return tmp_wav.name


def voice_activity_mask(y, sr, frame_length=1024, hop_length=256, energy_percentile=40,
                         min_run_ms=80):
   
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.percentile(rms, energy_percentile)
    raw_mask = rms > threshold

    frame_duration_ms = (hop_length / sr) * 1000
    min_run_frames = max(1, int(round(min_run_ms / frame_duration_ms)))
    speech_mask = _smooth_min_run(raw_mask, min_run_frames)

    return speech_mask, rms, hop_length


def _smooth_min_run(mask, min_run_frames):
    
    mask = np.asarray(mask).astype(int)
    if len(mask) == 0:
        return mask.astype(bool)

    window = min_run_frames * 2 + 1  # must be odd for scipy median_filter
    if window <= 1:
        return mask.astype(bool)

    from scipy.ndimage import median_filter
    smoothed = median_filter(mask, size=window, mode="nearest")
    return smoothed.astype(bool)


def pause_statistics(speech_mask, sr, hop_length):
    
    frame_duration = hop_length / sr
    speech_ratio = float(np.mean(speech_mask)) if len(speech_mask) else np.nan

    
    pauses = []
    in_pause = False
    pause_len = 0
    started_speech = False
    for is_speech in speech_mask:
        if is_speech:
            started_speech = True
            if in_pause and pause_len > 0:
                pauses.append(pause_len)
            in_pause = False
            pause_len = 0
        else:
            if started_speech:
                in_pause = True
                pause_len += 1

    n_pauses = len(pauses)
    mean_pause_dur = float(np.mean(pauses) * frame_duration) if pauses else 0.0
    return {
        "speech_ratio": speech_ratio,
        "n_pauses": n_pauses,
        "mean_pause_duration_s": mean_pause_dur,
    }


def speaking_rate_proxy(y, sr, speech_mask, hop_length):
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

    speech_frames = np.sum(speech_mask)
    speech_time_s = speech_frames * hop_length / sr
    if speech_time_s <= 0:
        return np.nan
    return float(len(onsets) / speech_time_s)


def extract_audio_features(video_path, sr=16000):
    wav_path = extract_audio(video_path, sr=sr)
    try:
        y, sr = librosa.load(wav_path, sr=sr, mono=True)

        # --- Pitch (F0) via pYIN, voiced frames only ---
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr,
        )
        f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])

        # --- Energy (RMS) ---
        rms = librosa.feature.rms(y=y)[0]

        # --- VAD + pause / speaking-rate statistics ---
        speech_mask, rms_full, hop_length = voice_activity_mask(y, sr)
        pauses = pause_statistics(speech_mask, sr, hop_length)
        rate = speaking_rate_proxy(y, sr, speech_mask, hop_length)

        features = {
            "video": str(video_path),
            "duration_s": float(len(y) / sr),
            "pitch_mean_hz": float(np.nanmean(f0_voiced)) if len(f0_voiced) else np.nan,
            "pitch_var_hz": float(np.nanvar(f0_voiced)) if len(f0_voiced) else np.nan,
            "energy_mean_rms": float(np.mean(rms)),
            "energy_var_rms": float(np.var(rms)),
            "speech_ratio": pauses["speech_ratio"],
            "n_pauses": pauses["n_pauses"],
            "mean_pause_duration_s": pauses["mean_pause_duration_s"],
            "speaking_rate_proxy_hz": rate,
        }
        return features
    finally:
        os.remove(wav_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract audio features from a video.")
    parser.add_argument("video_path", help="Path to a single video file")
    args = parser.parse_args()

    feats = extract_audio_features(args.video_path)
    for k, v in feats.items():
        print(f"{k}: {v}")
