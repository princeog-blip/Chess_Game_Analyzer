# ==============================================================================
# CSCI 270 - Project 2: Chess Game Analyzer (Group 3)
# Module: Feature Engineering & Machine Learning Evaluation Pipeline
# ==============================================================================

import chess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import random

# ==============================================================================
# STEP 1: FEATURE ENGINEERING (Using python-chess)
# Extracts the exact 5 features promised in the Group 3 Proposal:
#   1. Material Advantage
#   2. Piece Vulnerability
#   3. Center Control
#   4. Piece Mobility
#   5. King Safety
# ==============================================================================

# Standard chess piece values (Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9)
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# The 4 central squares on a chessboard (d4, e4, d5, e5)
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

def extract_board_features(fen_string):
    """
    Takes a digital chessboard (FEN string) and returns a list of 5 numerical
    features from White's perspective (Positive = White advantage, Negative = Black advantage).
    """
    board = chess.Board(fen_string)
    
    # --- FEATURE 1: Material Advantage ---
    # Sum the point values of White's pieces minus Black's pieces
    white_material = 0
    black_material = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += val
            else:
                black_material += val
    material_advantage = white_material - black_material

    # --- FEATURE 2: Piece Vulnerability ---
    # Count how many of a player's pieces are currently under attack and undefended ("hanging")
    white_vulnerable = 0
    black_vulnerable = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            if piece.color == chess.WHITE:
                # Attacked by Black and NOT defended by White
                if board.is_attacked_by(chess.BLACK, square) and not board.is_attacked_by(chess.WHITE, square):
                    white_vulnerable += PIECE_VALUES[piece.piece_type]
            else:
                # Attacked by White and NOT defended by Black
                if board.is_attacked_by(chess.WHITE, square) and not board.is_attacked_by(chess.BLACK, square):
                    black_vulnerable += PIECE_VALUES[piece.piece_type]
    # Net vulnerability score (positive means Black has more vulnerable pieces, favoring White)
    vulnerability_score = black_vulnerable - white_vulnerable

    # --- FEATURE 3: Center Control ---
    # Count how many times White attacks the 4 center squares minus Black's attacks
    white_center = sum(len(board.attackers(chess.WHITE, sq)) for sq in CENTER_SQUARES)
    black_center = sum(len(board.attackers(chess.BLACK, sq)) for sq in CENTER_SQUARES)
    center_control = white_center - black_center

    # --- FEATURE 4: Piece Mobility ---
    # Count total number of legal moves available to White vs. Black
    original_turn = board.turn
    
    board.turn = chess.WHITE
    white_mobility = board.legal_moves.count()
    
    board.turn = chess.BLACK
    black_mobility = board.legal_moves.count()
    
    board.turn = original_turn # Restore original turn
    mobility_advantage = white_mobility - black_mobility

    # --- FEATURE 5: King Safety ---
    # Measure how many enemy pieces are attacking the 8 squares immediately surrounding the King
    def get_king_danger(king_color, enemy_color):
        king_sq = board.king(king_color)
        if king_sq is None:
            return 0
        # Get the squares adjacent to the King
        king_zone = board.attacks(king_sq)
        danger = sum(len(board.attackers(enemy_color, sq)) for sq in king_zone)
        return danger

    white_king_danger = get_king_danger(chess.WHITE, chess.BLACK)
    black_king_danger = get_king_danger(chess.BLACK, chess.WHITE)
    # Positive means Black's king is in more danger than White's king
    king_safety_advantage = black_king_danger - white_king_danger

    return [
        material_advantage,
        vulnerability_score,
        center_control,
        mobility_advantage,
        king_safety_advantage
    ]


# ==============================================================================
# STEP 2: DATASET GENERATION / LOADING (Lichess FEN + Stockfish Evaluations)
# Note: The full Lichess HuggingFace dataset is 15+ GB. 
# Below, we generate 1,500 realistic random chess positions so your code runs 
# immediately out-of-the-box. If you download a Lichess CSV sample, just 
# uncomment the pd.read_csv() block!
# ==============================================================================

def create_dataset(num_samples=1500):
    print(f"Generating {num_samples} chess positions and extracting 5 features...")
    random.seed(42)
    np.random.seed(42)
    
    feature_rows = []
    evaluations = []
    
    for _ in range(num_samples):
        board = chess.Board()
        # Play between 6 and 45 random legal half-moves to get diverse mid-game boards
        num_moves = random.randint(6, 45)
        for _ in range(num_moves):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
            
        fen = board.fen()
        feats = extract_board_features(fen)
        
        # Simulate Stockfish Centipawn Evaluation (or use real 'cp' column from Lichess CSV)
        # Weighting the features + slight positional noise to mimic engine depth evaluation
        simulated_stockfish_eval = (
            feats[0] * 1.00 +  # Material weight
            feats[1] * 0.45 +  # Vulnerability weight
            feats[2] * 0.25 +  # Center control weight
            feats[3] * 0.08 +  # Mobility weight
            feats[4] * 0.35 +  # King safety weight
            np.random.normal(0, 0.6)
        )
        
        feature_rows.append(feats)
        evaluations.append(simulated_stockfish_eval)
        
    columns = [
        'Material_Advantage', 
        'Piece_Vulnerability', 
        'Center_Control', 
        'Piece_Mobility', 
        'King_Safety'
    ]
    df = pd.DataFrame(feature_rows, columns=columns)
    df['Stockfish_Eval'] = evaluations
    return df

# --- HOW TO LOAD REAL LICHESS CSV DATA INSTEAD ---
# If you download a subset of the Lichess dataset as 'lichess_sample.csv' with columns ['fen', 'cp']:
# raw_df = pd.read_csv('lichess_sample.csv').dropna().head(5000)
# features_list = [extract_board_features(fen) for fen in raw_df['fen']]
# df = pd.DataFrame(features_list, columns=['Material_Advantage', 'Piece_Vulnerability', 'Center_Control', 'Piece_Mobility', 'King_Safety'])
# df['Stockfish_Eval'] = raw_df['cp'].values / 100.0  # Convert centipawns to pawns

df = create_dataset(1500)
print("\nSample of Engineered Chess Features:")
print(df.head())


# ==============================================================================
# STEP 3: PREPROCESSING & MACHINE LEARNING TRAINING
# ==============================================================================

X = df.drop('Stockfish_Eval', axis=1)
y = df['Stockfish_Eval']

# Split 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler (Z-Score Normalization from Lecture 4.0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor (from Lecture 11.0)
print("\nTraining Random Forest Evaluation Model...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model on unseen test positions
y_pred = rf_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL PERFORMANCE RESULTS ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Accuracy Score:        {r2:.4f}")


# ==============================================================================
# STEP 4: SINGLE BOARD PREDICTION (Predicting Who Wins from a New Position)
# ==============================================================================

# Example test position (White has a strong center and material lead)
sample_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
sample_features = pd.DataFrame([extract_board_features(sample_fen)], columns=X.columns)
sample_scaled = scaler.transform(sample_features)
predicted_score = rf_model.predict(sample_scaled)[0]

print("\n--- LIVE GAME ANALYZER TEST ---")
print(f"Input FEN: {sample_fen}")
print(f"Predicted Evaluation Score: {predicted_score:+.2f}")
if predicted_score > 0.5:
    print("Prediction: WHITE is most likely to win.")
elif predicted_score < -0.5:
    print("Prediction: BLACK is most likely to win.")
else:
    print("Prediction: Game is roughly EQUAL / DRAWISH.")


# ==============================================================================
# STEP 5: VISUALIZATIONS FOR FINAL PRESENTATION
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Actual vs Predicted Evaluation Scores
axes[0].scatter(y_test, y_pred, alpha=0.4, color='royalblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
axes[0].set_xlabel('Actual Evaluation Score (Pawns)', fontsize=11)
axes[0].set_ylabel('ML Predicted Score (Pawns)', fontsize=11)
axes[0].set_title(f'Chess Position Evaluation (R² = {r2:.2f})', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Feature Importance Bar Chart
importances = rf_model.feature_importances_
indices = np.argsort(importances)
axes[1].barh(range(len(indices)), importances[indices], color='seagreen', align='center')
axes[1].set_yticks(range(len(indices)))
axes[1].set_yticklabels([X.columns[i] for i in indices], fontsize=10)
axes[1].set_xlabel('Relative Importance in Predicting Winner', fontsize=11)
axes[1].set_title('Chess Feature Importances (Random Forest)', fontsize=13)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('chess_ml_results.png')
print("\nSaved 'chess_ml_results.png' successfully!")
plt.show()