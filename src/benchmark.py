"""Benchmark YuNet face detection across input resolutions.

Measures, on a real video:
  - FPS: frames processed per second (hardware-dependent, honest measurement)
  - faces/frame: mean detections per frame (descriptive — the sample video ships
    without ground-truth labels, so we report counts, not accuracy)
  - mean confidence of detections

Run:  python src/benchmark.py --save-sample-frame
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors import SUPPORTED_SIZES, YuNetDetector, draw_faces


def benchmark_video(detector, video_path: Path, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = 0
    frames_with_faces = 0
    total_faces = 0
    confidences = []

    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1
        if max_frames and total_frames >= max_frames:
            break

        faces = detector.detect(frame)
        if faces:
            frames_with_faces += 1
            total_faces += len(faces)
            confidences.extend(f.confidence for f in faces)

    elapsed = time.perf_counter() - t0
    cap.release()

    stats = {
        "detector": detector.name,
        "frames_processed": total_frames,
        "fps": round(total_frames / elapsed, 2),
        "mean_faces_per_frame": round(total_faces / total_frames, 3),
        "frames_with_detection": f"{frames_with_faces}/{total_frames}",
        "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else None,
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/face-walking.mp4")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-sample-frame", action="store_true")
    args = parser.parse_args()

    video = Path(args.video)
    results = {}

    for size in SUPPORTED_SIZES:
        detector = YuNetDetector(input_size=size)
        print(f"\n=== {detector.name} ===")
        stats = benchmark_video(detector, video, max_frames=args.max_frames)
        print(json.dumps(stats, indent=2))
        results[detector.name] = stats

        if args.save_sample_frame:
            # find a frame with the most faces for a nice annotated demo shot
            cap = cv2.VideoCapture(str(video))
            best_frame, best_faces = None, []
            for _ in range(300):
                ok, frame = cap.read()
                if not ok:
                    break
                faces = detector.detect(frame)
                if len(faces) > len(best_faces):
                    best_frame, best_faces = frame, faces
            cap.release()
            if best_frame is not None and best_faces:
                annotated = draw_faces(best_frame, best_faces, show_confidence=True)
                slug = f"sample_frame_{size[0]}x{size[1]}.png"
                cv2.imwrite(str(PROJECT_ROOT / "images" / slug), annotated)
                print(f"  sample frame ({len(best_faces)} face(s)) -> images/{slug}")

    (PROJECT_ROOT / "reports" / "benchmark.json").write_text(
        json.dumps(results, indent=2)
    )
    print("\nSaved reports/benchmark.json")

    # speed/accuracy trade-off headline
    names = list(results)
    if len(names) >= 2:
        fastest, slowest = names[0], names[-1]
        ratio = results[fastest]["fps"] / results[slowest]["fps"]
        print(f"\nHeadline: {fastest} runs {ratio:.1f}x faster than {slowest} "
              f"({results[fastest]['mean_faces_per_frame']} vs "
              f"{results[slowest]['mean_faces_per_frame']} faces/frame)")


if __name__ == "__main__":
    main()
