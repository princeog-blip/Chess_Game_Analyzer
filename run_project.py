"""
run_project.py
==============
CSCI 270 - Real-Time Chess Vision & Win Prediction System

Master Runner to execute any stage of the project pipeline:
  [1] Filter Lichess Games (filter_chess_games.py)
  [2] Train Win-Prediction ML Model (train_model.py)
  [3] Launch Real-Time Chess Camera Tracker (cam_v2.py)
"""

import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def print_banner():
    print("\n" + "=" * 65)
    print(" CSCI 270: REAL-TIME CHESS TRACKER & WIN PREDICTOR")
    print("=" * 65)
    print(" Pipeline Stages:")
    print("   [1] Run Data Filter  -> Process lichess_games.pgn.zst")
    print("   [2] Train ML Model   -> Train classifier on game positions")
    print("   [3] Launch Camera    -> Real-time board vision & win prediction")
    print("   [4] Quick Demo       -> Test feature extraction & win predictor")
    print("   [q] Quit")
    print("=" * 65)

def main():
    while True:
        print_banner()
        choice = input("Select an option (1-4, q): ").strip().lower()

        if choice == "1":
            script = os.path.join(PROJECT_DIR, "filter_chess_games.py")
            print(f"\n[Running] {script}...")
            subprocess.run([sys.executable, script])

        elif choice == "2":
            script = os.path.join(PROJECT_DIR, "train_model.py")
            print(f"\n[Running] {script}...")
            subprocess.run([sys.executable, script])

        elif choice == "3":
            script = os.path.join(PROJECT_DIR, "cam_v2.py")
            print(f"\n[Launching Camera] {script}...")
            subprocess.run([sys.executable, script])

        elif choice == "4":
            print("\n--- Quick Demo: Testing Starting vs Scholar's Mate Position ---")
            import chess
            from train_model import extract_features_from_fen
            
            # Starting position
            fen_start = chess.STARTING_FEN
            feats_start = extract_features_from_fen(fen_start)
            print(f"Starting FEN Features (Mat, Vuln, Ctr, Mob, KingSafe): {feats_start}")

            # Scholar's mate (White Queen on f7 checkmating Black)
            board = chess.Board()
            for move in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
                board.push_san(move)
            feats_mate = extract_features_from_fen(board.fen())
            print(f"Scholar's Mate FEN: {board.fen()}")
            print(f"Scholar's Mate Features: {feats_mate}")
            print(f"Checkmate state: {board.is_checkmate()} (White wins!)\n")

        elif choice == "q":
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please choose 1, 2, 3, 4, or q.")

if __name__ == "__main__":
    main()
