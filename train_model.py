"""
train_model.py
==============
CSCI 270 - Chess Position Win Predictor Model Training

Pipeline:
  1. Loads 'game_positions.csv' extracted by filter_chess_games.py
  2. Extracts the 5 strategic features from each board FEN:
     - Feature 1: Material Advantage
     - Feature 2: Piece Vulnerability (hanging/undefended pieces)
     - Feature 3: Center Control (attacks on d4, e4, d5, e5)
     - Feature 4: Piece Mobility (White legal moves - Black legal moves)
     - Feature 5: King Safety (enemy danger in the king's zone)
  3. Trains a LogisticRegression / RandomForest classification model
  4. Evaluates accuracy and classification report on test positions
  5. Serializes the trained model to 'chess_model.pkl' for cam_v2.py
"""

import os
import sys
import chess
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Directory handling to work whether executed from workspace root or Project 2/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(CURRENT_DIR, "game_positions.csv")
MODEL_OUTPUT = os.path.join(CURRENT_DIR, "chess_model.pkl")

# Chess piece standard values
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


def extract_features_from_fen(fen_string):
    """
    Extracts the 5 numerical features from White's perspective:
    (Positive = White advantage, Negative = Black advantage).
    """
    try:
        board = chess.Board(fen_string)
    except Exception:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # --- FEATURE 1: Material Advantage ---
    white_material = 0
    black_material = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_material += val
        else:
            black_material += val
    material_advantage = white_material - black_material

    # --- FEATURE 2: Piece Vulnerability ---
    white_vulnerable = 0
    black_vulnerable = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.KING:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                if board.is_attacked_by(chess.BLACK, square) and not board.is_attacked_by(chess.WHITE, square):
                    white_vulnerable += val
            else:
                if board.is_attacked_by(chess.WHITE, square) and not board.is_attacked_by(chess.BLACK, square):
                    black_vulnerable += val
    vulnerability_score = black_vulnerable - white_vulnerable

    # --- FEATURE 3: Center Control ---
    white_center = sum(len(board.attackers(chess.WHITE, sq)) for sq in CENTER_SQUARES)
    black_center = sum(len(board.attackers(chess.BLACK, sq)) for sq in CENTER_SQUARES)
    center_control = white_center - black_center

    # --- FEATURE 4: Piece Mobility ---
    original_turn = board.turn
    board.turn = chess.WHITE
    white_mobility = board.legal_moves.count()
    board.turn = chess.BLACK
    black_mobility = board.legal_moves.count()
    board.turn = original_turn
    mobility_advantage = white_mobility - black_mobility

    # --- FEATURE 5: King Safety ---
    def get_king_danger(king_color, enemy_color):
        king_sq = board.king(king_color)
        if king_sq is None:
            return 0
        king_zone = board.attacks(king_sq)
        return sum(len(board.attackers(enemy_color, sq)) for sq in king_zone)

    white_king_danger = get_king_danger(chess.WHITE, chess.BLACK)
    black_king_danger = get_king_danger(chess.BLACK, chess.WHITE)
    king_safety_advantage = black_king_danger - white_king_danger

    return [
        float(material_advantage),
        float(vulnerability_score),
        float(center_control),
        float(mobility_advantage),
        float(king_safety_advantage)
    ]


def train_model(csv_path=DEFAULT_CSV, output_model_path=MODEL_OUTPUT, max_samples=40000):
    """
    Loads extracted positions, creates feature matrices, trains a model,
    and serializes the classifier to disk.
    """
    if not os.path.exists(csv_path):
        print(f"[Error] Dataset not found at: '{csv_path}'")
        print("Please run 'filter_chess_games.py' first to generate 'game_positions.csv'.")
        return False

    print("=" * 65)
    print("CHESS WIN-PREDICTOR MODEL TRAINING")
    print("=" * 65)
    print(f"Reading dataset: {csv_path}")
    
    # Read positions CSV
    df = pd.read_csv(csv_path)
    print(f"Total raw positions loaded: {len(df):,}")

    # Map game results:
    # 1: White Win ("1-0"), -1: Black Win ("0-1"), 0: Draw ("1/2-1/2")
    result_mapping = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
    df = df[df["result"].isin(result_mapping.keys())].copy()
    df["target"] = df["result"].map(result_mapping)

    # Sample if the dataset is very large to speed up training
    if len(df) > max_samples:
        print(f"Subsampling {max_samples:,} positions for fast, high-quality training...")
        df = df.sample(n=max_samples, random_state=42)

    print("Extracting 5-dimensional feature vectors from FEN strings...")
    features_list = [extract_features_from_fen(fen) for fen in df["fen_after_move"]]
    X = np.array(features_list)
    y = df["target"].values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]:,} samples | Testing set: {X_test.shape[0]:,} samples")
    print("Training Logistic Regression classifier with class balancing...")
    
    # Multi-class Logistic Regression with class-weight balance
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    target_names = ["Black Win (-1)", "Draw (0)", "White Win (1)"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Feature weights inspection
    feature_names = ["Material", "Vulnerability", "Center_Control", "Mobility", "King_Safety"]
    print("--- Learned Feature Coefficients (White Win Perspective) ---")
    white_class_idx = list(clf.classes_).index(1)
    for name, weight in zip(feature_names, clf.coef_[white_class_idx]):
        print(f"  {name:<16}: {weight:>8.4f}")

    # Serialize model to disk
    joblib.dump(clf, output_model_path)
    print(f"\nTrained model saved successfully to: {output_model_path}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    train_model(csv_arg)
