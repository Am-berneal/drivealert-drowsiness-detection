# Model card – DriveAlert driver drowsiness detection

*Format based on "Model Cards for Model Reporting" (Mitchell et al., 2019). Numbers are from our
own training run (`train_model.py`, checked with `check_accuracy.py`) on the 218 held-out test
images of each task. Rows still in square brackets were not measured yet.*

## Model details
- **Developed by:** Divyam Chopra (1024030995) and Sarthak Duvedi (1024031013), 2026 – final-year student project.
- **System:** MediaPipe Face Mesh landmarks (EAR, MAR, head pose) + CNN eye-state and yawn
  classifiers (MobileNetV2 transfer learning; 4 other CNNs and 11 classical models compared) +
  optional temporal model (LSTM/GRU) + rule-based alarm engine with 3 alarm levels.
- **Night model:** separate eye CNN trained on infrared images (MRL Eye Dataset).
- **Licence / use:** educational prototype.

## Intended use
- Warn a driver of signs of drowsiness (long eye closure, nodding, yawning, PERCLOS) and
  distraction (looking away, phone in hand) in real time, on a laptop, phone or Raspberry Pi.
- Help researchers and students compare ML/DL approaches for eye-state classification.

**Out of scope:** it is *not* a certified automotive safety system (not tested to EU GSR / ADDW,
AIS-184 or any standard); it must never be relied on instead of rest, and must not be used to
make decisions about a driver's employment, insurance or legal responsibility.

## Training data
| Model | Data | Notes |
|---|---|---|
| Eye / yawn CNNs, classical models | Kaggle Yawn-Eye Dataset (serenaraju) | few subjects; random stratified 70/15/15 split (seed 42) |
| Night eye model | MRL Eye Dataset – 84,898 infrared eye images, 37 people | split by person with `--by-subject` |
| Temporal models | UTA Real-Life Drowsiness Dataset (videos) | split by video |

## Evaluation
| Test | Result |
|---|---|
| Eye CNN (MobileNetV2) – 218 held-out images | **accuracy 97.7%**, precision 97.3%, recall 98.2%, F1 0.977, ROC-AUC 0.999; 5 wrong (3 false alarms, 2 missed) |
| Yawn CNN (MobileNetV2) – 218 held-out images | **accuracy 74.3%**, precision 72.6%, recall 78.0%, F1 0.752, ROC-AUC 0.856; 56 wrong (32 false yawns, 24 missed) |
| Training | eye: 16 epochs (10 head + 6 fine-tune), 106 s on CPU · yawn: 16 epochs, 84 s · 2,259,265 parameters, 21.9 MB each |
| Best classical model (earlier run) | eye: SVM (RBF) 95.4%, F1 0.954 · yawn: Extra Trees 97.2%, F1 0.972 (HOG features) |
| CNN vs SVM, eye (earlier run, CNN 97.2%) | McNemar p = 0.42 – not significant; 95% CI CNN 95.0–99.1%, SVM 92.2–97.7% |
| Your own clips (`evaluate.py`) | microsleeps caught [..]%, false alarms [..] / hour |
| Day model on infrared images | accuracy [..]% → night model [..]% (`train_night.py`) |
| Usability (SUS, n = [..]) | [..] / 100 ([grade]) |

## Known limitations and failure modes
- **Same people in train and test** for the Kaggle split → image accuracy is optimistic;
  the clip test (`evaluate.py`) and the person-wise MRL split are the honest numbers.
- **Darkness:** a normal webcam cannot see without light – the app reports "too dark" and
  switches to voice check-ins; real night use needs an IR camera.
- **Sunglasses / masks:** eye signals are ignored when the eyes are hidden; head pose and
  yawning still work.
- **Failure analysis** (`failure_gallery.py`, earlier eye model): mistakes were blurry (3),
  half-closed / borderline (3), low-contrast (2) and dark (1) images – see `results/eye_failures.md`.
- **Yawn CNN is weak (74.3%)**: one image cannot tell a yawn from talking or eating, so the live
  system never trusts it alone – a yawn also needs MAR > 0.6 for 1 s. Training longer
  (`--epochs 15 --fine-tune-epochs 10`) or using Extra Trees for yawns is recommended.
- **Fairness:** the datasets have few people and may not represent all ages, skin tones or
  eye shapes; per-group accuracy was not measured. Personal calibration and false-alarm
  learning reduce, but do not remove, this risk.
- **Phone detection** uses general COCO object detectors, not trained on in-car images.

## Privacy and ethics
- Video frames are processed in memory and never stored or uploaded; only events
  (time, reason, scores, optional location) are saved locally in `data/`.
- Emergency alerts and AI reports send data to Telegram / Google Gemini only if the user
  enables them; API keys are never sent to the browser.
- The driver can see, export and delete all their data (Session History, Trips, Settings).
