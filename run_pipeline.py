"""
End-to-end orchestrator: given a folder of videos, run Task 1 (visual),
Task 2 (audio), and Task 3 (fusion + label-free evaluation), and write
all intermediate + final outputs to an output folder.

Usage:
    python run_pipeline.py /path/to/videos --out ./results

Searches the given folder RECURSIVELY (Kaggle downloads are often nested in
class/label subfolders), so it works regardless of the exact folder layout
inside the downloaded dataset zip. Produces:
    results/visual_features.csv
    results/audio_features.csv
    results/fused_evaluation_report.json
"""

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from face_features import extract_video_features
from audio_features import extract_audio_features
from multimodal_fusion import run_full_evaluation


def find_videos(folder, ext=(".mp4", ".mov", ".avi", ".mkv", ".webm")):
    videos = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(ext):
                videos.append(os.path.join(root, f))
    return sorted(videos)


def run(video_folder, out_folder, sample_every_n_frames=1, limit=None):
    os.makedirs(out_folder, exist_ok=True)
    videos = find_videos(video_folder)
    if not videos:
        print(f"No video files found under {video_folder} (searched recursively)", file=sys.stderr)
        sys.exit(1)

    if limit is not None:
        videos = videos[:limit]

    print(f"Found {len(videos)} video(s). Extracting features...")

    visual_rows, audio_rows = [], []
    for vp in videos:
        print(f"  [visual] {vp}")
        try:
            _, v_feats = extract_video_features(vp, sample_every_n_frames=sample_every_n_frames)
            visual_rows.append(v_feats)
        except Exception as e:
            print(f"    FAILED visual extraction for {vp}: {e}", file=sys.stderr)
            traceback.print_exc()

        print(f"  [audio]  {vp}")
        try:
            a_feats = extract_audio_features(vp)
            audio_rows.append(a_feats)
        except Exception as e:
            print(f"    FAILED audio extraction for {vp}: {e}", file=sys.stderr)
            traceback.print_exc()

    visual_df = pd.DataFrame(visual_rows)
    audio_df = pd.DataFrame(audio_rows)

    visual_csv = os.path.join(out_folder, "visual_features.csv")
    audio_csv = os.path.join(out_folder, "audio_features.csv")
    visual_df.to_csv(visual_csv, index=False)
    audio_df.to_csv(audio_csv, index=False)
    print(f"Saved: {visual_csv}")
    print(f"Saved: {audio_csv}")

    if len(visual_df) >= 3 and len(audio_df) >= 3:
        print("Running fusion + label-free evaluation...")
        report = run_full_evaluation(visual_df, audio_df)
        report_path = os.path.join(out_folder, "fused_evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved: {report_path}")
    else:
        print("Skipping fusion/evaluation: need at least 3 videos with successful "
              "visual + audio extraction for clustering to be meaningful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full DepVidMood feature pipeline.")
    parser.add_argument("video_folder", help="Folder containing video files")
    parser.add_argument("--out", default="./results", help="Output folder (default: ./results)")
    parser.add_argument("--every", type=int, default=1, help="Sample every N frames for visual extraction")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N videos found (useful for a quick sanity check "
                              "before running the full dataset)")
    args = parser.parse_args()

    run(args.video_folder, args.out, sample_every_n_frames=args.every, limit=args.limit)
