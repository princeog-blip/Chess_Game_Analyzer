What is this?
The Chess Game Analyzer is a two-part computer vision and supervised machine learning system that tracks physical, over-the-board (OTB) chess games via a live webcam and predicts which player holds the winning advantage. Expanding on our original static-screenshot proposal, the final deliverable pairs an interactive real-time webcam tracker with a 7-feature supervised evaluation pipeline (Version 4).
•	Live Camera Chess Tracker: Calibrates a physical chessboard via 4-corner mouse clicks, warps it to an 800×800 top-down grid, detects moves via frame differencing (cv2.absdiff), validates legal moves with python-chess, and renders a side-by-side procedural digital chessboard UI with real-time win verdicts (Material, Center Control, Mobility, and Check Pressure).
•	Version 4 Supervised ML Pipeline: Extracts 7 strategic and tactical features from Lichess master/evaluated positions, splits data cleanly by game_id (GroupShuffleSplit), and trains a HistGradientBoostingRegressor to predict continuous pawn scores ([-15, +15]), logistic win probabilities, MSE/RMSE/R² metrics, and diagnostic charts (chess_ml_results.png).

How did we do it?
Stack: Python 3, OpenCV (cv2), python-chess, scikit-learn (HistGradientBoostingRegressor, StandardScaler, GroupShuffleSplit, permutation_importance), joblib, pandas, NumPy, and matplotlib.
Part A: Data Ingestion & Version 4 ML Pipeline
•	1. Multi-Source Data Loading: Evaluated Leela Chess Zero (~2.5M games) and Lichess archives (Nov 2020 Standard Rated Games & Hugging Face Stockfish evals). The loader prioritizes NEWlichess_elite_2020-08.pgn (up to 100k mid/endgame positions, blending positional heuristics with game outcomes), with automatic fallbacks to local Stockfish CSVs, live Lichess API games, or a deterministic master-opening generator.
•	2. Leakage-Free Split & Scaling: Uses GroupShuffleSplit(test_size=0.2) grouped by game_id so positions from the same match never leak across train and test sets, followed by Z-Score Normalization (StandardScaler fit strictly on X_train).
•	3. Model Training & Diagnostics: Trains a tuned HistGradientBoostingRegressor (max_iter=300, learning_rate=0.05, max_depth=7, l2_regularization=0.1) and saves chess_eval_model.joblib and chess_eval_scaler.joblib. Evaluates unseen games via MSE, RMSE, R², and permutation_importance (exported to chess_ml_results.png).
•	4. Win Probability Conversion: LiveGameEvaluator converts predicted pawn scores into win odds using a logistic curve: P(White Win) = 1 / (1 + 10^(-score / 4.0)).
Part B: Real-Time Webcam Tracker & Digital UI
•	1. Calibration & Warping: User clicks the 4 outer board corners (TL, TR, BR, BL); order_points() and warp_board() flatten the feed via cv2.getPerspectiveTransform() into an 800×800 reference frame.
•	2. Move Detection: Pressing [SPACEBAR] after a physical move triggers crop_all_squares() (64 squares with an 8px margin) and detect_changed_squares() (grayscale cv2.absdiff) to rank the top 4 visually altered squares.
•	3. Legal Move Resolution: resolve_move() tests directed pairs among the top 4 squares against board.legal_moves in python-chess, accurately resolving standard moves, captures, queen promotions, and castling.

Future Works: 
•	Unify V4 ML Model with Live Camera: Load chess_eval_model.joblib and chess_eval_scaler.joblib directly into the webcam loop so the side-by-side UI displays live 7-feature pawn scores and win probabilities.
•	Hands-Free Move Detection: Replace the manual [SPACEBAR] press with frame-stability detection to automatically ignore moving hands and register settled moves.
•	CNN / YOLO Piece Recognition: Enable recognition from arbitrary mid-game board states or static screenshots without needing the initial 32-piece setup.
•	Web / Mobile Deployment: Package into a standalone app with an animated evaluation bar, blunder alerts, and PGN export.

•	4. Live UI Rendering: evaluate_position() scores the updated board, and draw_digital_board() procedurally draws all 6 piece silhouettes with OpenCV polygons/circles, displaying the digital board beside the webcam via np.hstack().
