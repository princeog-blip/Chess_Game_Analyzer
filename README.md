# Chess_Game_Analyzer
This is a model that analyzes chess games played by 2 people and determines which player is in a better spot to win based on piece positioning and vulnerability.

## Modules

### 1. Computer Vision & Calibration (`main/v1_chess`)
Runs the OpenCV camera calibration and corner detection pipeline:
- Reads an image via `cv2.imread()` and converts it to grayscale.
- Finds chessboard inner corners using `cv2.findChessboardCorners()`.
- Refines corner accuracy using `cv2.cornerSubPix()`.
- Visualizes verified corners via `cv2.drawChessboardCorners()`.
- Estimates camera intrinsics and 3D board pose with `cv2.calibrateCamera()` and `cv2.solvePnP()`.

Run:
```bash
python main/v1_chess [optional_path_to_image]
```

### 2. Feature Extraction, Machine Learning & Interactive Viewer (`main/v2_chess`)
Extracts 5 core strategic features (Material, Vulnerability, Center Control, Mobility, and King Safety), trains a Random Forest evaluation regressor, and launches an interactive step-by-step game analyzer featuring:
- Full move navigation (**⏮ Start**, **◀ Back**, **Next ▶**, **End ⏭**, and arrow keys).
- A real-time sliding evaluation bar showing White vs. Black win probabilities and engine centipawn scores.

Run:
```bash
python main/v2_chess
```