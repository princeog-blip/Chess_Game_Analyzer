"""
filter_chess_games.py
======================
CSCI 270 - Chess Game Analyzer Project

Streams a compressed Lichess PGN file (lichess_games.pgn.zst) without loading
the entire archive into memory.

Filters applied:
  1. Standard chess variant (ignores Crazyhouse, Chess960, etc.)
  2. Game speed: Rapid or Classical
  3. Both players must have an Elo rating >= 2000
  4. Completed games only (Result is 1-0, 0-1, or 1/2-1/2)
  5. At least 20 full moves played (>= 40 half-moves/plies)

Outputs:
  - strong_player_games.csv: Metadata, ratings, result, and full move notation for 1,000 games.
  - game_positions.csv: FEN notation and metadata for every board position after each move.
"""

import os
import sys
import io
import csv
import time
import chess
import chess.pgn
import zstandard as zstd


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DEFAULT_INPUT_FILE = "lichess_games.pgn.zst"
OUTPUT_GAMES_CSV = "strong_player_games.csv"
OUTPUT_POSITIONS_CSV = "game_positions.csv"

MIN_ELO = 2000
MIN_FULL_MOVES = 20           # 20 full moves = at least 40 half-moves (plies)
TARGET_MATCHED_GAMES = 1000
PROGRESS_INTERVAL = 50        # Log progress every 50 matching games


def is_valid_event(event_str):
    """
    Checks if the game event is Standard Rapid or Classical.
    Excludes Blitz, Bullet, UltraBullet, and non-standard variants.
    """
    if not event_str:
        return False
    event_lower = event_str.lower()
    
    # Must be Rapid or Classical
    is_rapid_or_classical = ("rapid" in event_lower) or ("classical" in event_lower)
    
    # Exclude blitz / bullet if accidentally tagged
    is_speed_chess = ("bullet" in event_lower) or ("blitz" in event_lower)
    
    return is_rapid_or_classical and not is_speed_chess


def parse_elo(elo_str):
    """Safely parse Elo rating integer, returning 0 if missing or invalid."""
    try:
        return int(elo_str)
    except (ValueError, TypeError):
        return 0


def filter_and_process_games(input_filepath):
    """
    Streams lichess_games.pgn.zst and exports matching games and FEN positions.
    """
    if not os.path.isfile(input_filepath):
        print(f"\n[ERROR] Input file not found: '{input_filepath}'")
        print("Please place 'lichess_games.pgn.zst' in this directory or pass its path:")
        print("    python filter_chess_games.py path/to/lichess_games.pgn.zst\n")
        sys.exit(1)

    file_size_mb = os.path.getsize(input_filepath) / (1024 * 1024)
    print("=" * 70)
    print(" Chess Game Analyzer - High-Elo Game & Position Filter")
    print("=" * 70)
    print(f" Input File:          {input_filepath} ({file_size_mb:.2f} MB)")
    print(f" Target Matches:      {TARGET_MATCHED_GAMES:,} games")
    print(f" Minimum Rating:      {MIN_ELO}+ Elo (Both White & Black)")
    print(f" Speed Categories:    Standard Rapid / Classical")
    print(f" Minimum Move Count:  {MIN_FULL_MOVES} full moves (>= {MIN_FULL_MOVES * 2} plies)")
    print(f" Games Output CSV:    {OUTPUT_GAMES_CSV}")
    print(f" Positions Output CSV:{OUTPUT_POSITIONS_CSV}")
    print("=" * 70)

    start_time = time.time()
    scanned_count = 0
    matched_count = 0
    total_positions_count = 0

    valid_results = {"1-0", "0-1", "1/2-1/2"}

    # Open CSV files for streaming writes
    with open(OUTPUT_GAMES_CSV, mode="w", newline="", encoding="utf-8") as games_file, \
         open(OUTPUT_POSITIONS_CSV, mode="w", newline="", encoding="utf-8") as positions_file:

        games_writer = csv.writer(games_file)
        positions_writer = csv.writer(positions_file)

        # Write Headers
        games_writer.writerow([
            "game_id",
            "event",
            "site",
            "white",
            "black",
            "white_elo",
            "black_elo",
            "result",
            "termination",
            "time_control",
            "full_moves",
            "total_plies",
            "moves_san"
        ])

        positions_writer.writerow([
            "game_id",
            "ply",
            "move_number",
            "turn",
            "move_san",
            "fen",
            "game_result"
        ])

        # Streaming decompression via zstandard
        dctx = zstd.ZstdDecompressor()
        with open(input_filepath, "rb") as compressed_fp:
            with dctx.stream_reader(compressed_fp) as reader:
                # Text stream with replace error-handling to prevent unicode decode halts
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

                while matched_count < TARGET_MATCHED_GAMES:
                    try:
                        game = chess.pgn.read_game(text_stream)
                        if game is None:
                            print("\nReached end of PGN archive.")
                            break

                        scanned_count += 1
                        headers = game.headers

                        # 1. Standard Variant check
                        variant = headers.get("Variant", "Standard")
                        if variant.lower() != "standard":
                            continue

                        # 2. Rapid or Classical check
                        event = headers.get("Event", "")
                        if not is_valid_event(event):
                            continue

                        # 3. Completed Game check
                        result = headers.get("Result", "*")
                        if result not in valid_results:
                            continue

                        # Exclude abandoned games
                        termination = headers.get("Termination", "Normal")
                        if "abandoned" in termination.lower():
                            continue

                        # 4. Both players Elo >= 2000
                        white_elo = parse_elo(headers.get("WhiteElo", "0"))
                        black_elo = parse_elo(headers.get("BlackElo", "0"))
                        if white_elo < MIN_ELO or black_elo < MIN_ELO:
                            continue

                        # 5. Move count >= 20 full moves (>= 40 plies)
                        moves = list(game.mainline_moves())
                        total_plies = len(moves)
                        full_moves = total_plies // 2

                        if full_moves < MIN_FULL_MOVES:
                            continue

                        # Match confirmed!
                        matched_count += 1
                        game_id = matched_count

                        # Replay moves on a chess.Board to extract FENs and SAN notation
                        board = game.board()
                        san_moves = []

                        for ply_idx, move in enumerate(moves, start=1):
                            move_number = (ply_idx + 1) // 2
                            turn_str = "White" if (ply_idx % 2 == 1) else "Black"
                            
                            san = board.san(move)
                            san_moves.append(san)
                            board.push(move)
                            
                            fen_after_move = board.fen()
                            total_positions_count += 1

                            # Save position row
                            positions_writer.writerow([
                                game_id,
                                ply_idx,
                                move_number,
                                turn_str,
                                san,
                                fen_after_move,
                                result
                            ])

                        # Save game row
                        games_writer.writerow([
                            game_id,
                            event,
                            headers.get("Site", ""),
                            headers.get("White", "Unknown"),
                            headers.get("Black", "Unknown"),
                            white_elo,
                            black_elo,
                            result,
                            termination,
                            headers.get("TimeControl", ""),
                            full_moves,
                            total_plies,
                            " ".join(san_moves)
                        ])

                        # Periodically flush CSV buffers to ensure data safety
                        if matched_count % 10 == 0:
                            games_file.flush()
                            positions_file.flush()

                        # Progress Report
                        if matched_count % PROGRESS_INTERVAL == 0 or matched_count == TARGET_MATCHED_GAMES:
                            elapsed = time.time() - start_time
                            rate = scanned_count / elapsed if elapsed > 0 else 0
                            print(
                                f"[Progress] Matched: {matched_count:4d}/{TARGET_MATCHED_GAMES} games | "
                                f"Scanned: {scanned_count:6,d} | "
                                f"Positions: {total_positions_count:6,d} | "
                                f"Speed: {rate:4.0f} games/s | "
                                f"Elapsed: {elapsed:5.1f}s"
                            )

                    except Exception as err:
                        # Log parsing warning and continue with subsequent games
                        print(f"[Warning] Error parsing game #{scanned_count}: {err}")
                        continue

    total_elapsed = time.time() - start_time
    print("=" * 70)
    print(" Processing Complete!")
    print(f" Total Games Scanned:       {scanned_count:,}")
    print(f" Total Games Saved:         {matched_count:,} -> {OUTPUT_GAMES_CSV}")
    print(f" Total Positions Saved:     {total_positions_count:,} -> {OUTPUT_POSITIONS_CSV}")
    print(f" Total Elapsed Time:        {total_elapsed:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    # Support optional command-line argument for input file path
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    filter_and_process_games(input_file)

