# 😀 Real-Time Face Detection System

Real-time face detection with **YuNet** — OpenCV's current deep-learning face detector (ONNX) —
including a full benchmark of the **speed vs sensitivity trade-off** across input resolutions,
plus a live webcam application.

- **Modern stack:** OpenCV 5.x · YuNet (ONNX, 232 KB) · no legacy Haar cascades
- **Benchmark:** 732-frame real video, honest FPS + detection stats
- **Demo:** real-time webcam detection with FPS overlay

---

## 🚀 Quick start

```bash
# 1. install deps
pip install -r requirements.txt

# 2. download the YuNet model (232 KB)
bash scripts/download_assets.sh

# 3a. run the benchmark
python src/benchmark.py --save-sample-frame

# 3b. OR run the live webcam demo
python scripts/webcam_detect.py --size 320   # press q / ESC to quit
```

---

## 📈 Benchmark results (speed vs sensitivity)

Tested on a 732-frame sample video (1920×1080, people walking). Measured on this machine:

| Config | FPS | Mean faces/frame | Frames with detection | Mean confidence |
|---|---|---|---|---|
| YuNet **160×160** | **339** | 0.13 | 75/732 | 0.76 |
| YuNet **320×320** | 258 | 0.50 | 221/732 | 0.80 |
| YuNet **640×640** | 162 | **0.59** | **273/732** | **0.83** |

![tradeoff](images/tradeoff.png)

### What this means
1. **Bigger internal resolution catches ~4.7× more faces** — small or distant faces survive — with higher confidence.
2. **The cost is real:** 640×640 runs at **2.1× slower** FPS than 160×160.
3. **For real-time webcam (30 FPS budget), 320×320 is the sweet spot**: 258 FPS of headroom with near-best sensitivity.
4. This is an **engineering decision, not a "best model"**: mobile → 160, desktop real-time → 320, accuracy-critical → 640.

![detection demo](images/demo_frame.png)

---

## 📁 Project layout

| Path | What |
|---|---|
| `src/detectors.py` | YuNet wrapper (resolution control, box rescaling) + annotation utils |
| `src/benchmark.py` | Video benchmark → `reports/benchmark.json` |
| `scripts/webcam_detect.py` | Real-time webcam app with FPS overlay |
| `scripts/download_assets.sh` | Fetches the YuNet ONNX model |
| `notebooks/01-benchmark.ipynb` | Executed benchmark walkthrough |
| `models/` | YuNet ONNX model (232 KB) |
| `assets/` | Sample test video (people walking) |
| `images/` | Annotated frames + trade-off plot |

---

## 🔑 Interview talking points

1. **Modern CV:** OpenCV 5 removed legacy Haar/Caffe paths — I rebuilt the project around **YuNet
   (FaceDetectorYN)**, the current standard, so the stack is future-proof.
2. **Honest benchmarking:** FPS is measured end-to-end (decode → inference → post-processing);
   no ground-truth labels are shipped with the sample video, so I report detection counts and
   confidence rather than fake accuracy numbers.
3. **Trade-off thinking:** quantified how input resolution buys sensitivity at the cost of speed,
   and justified the 320×320 default with a 30 FPS budget.
4. **Engineering hygiene:** reproducible setup script, parameterized detector, serialized benchmark
   results (`reports/benchmark.json`).

---

## 📚 References

- OpenCV Zoo — [face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- Sample video: [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos)
