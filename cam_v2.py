# ==============================================================================
# CSCI 270 - Real-Time Chess Game Tracker & Position Evaluator
# Pipeline:
#   1. Detect the chessboard's 7x7 inner-corner pattern in the live camera feed
#   2. Perspective warp the board into an 800x800 top-down square
#   3. Initialize standard chess board (chess.Board)
#   4. After a player makes a move and presses Spacebar:
#        - Detect changed squares via image differencing (cv2.absdiff)
#        - Resolve and validate the exact legal move with python-chess
#        - Update internal board state and FEN
#        - Evaluate and predict who is in position to win
# ==============================================================================

import os
import joblib
import cv2
import numpy as np
import chess

# ------------------------------------------------------------------------------
# 1. PERSPECTIVE TRANSFORMATION & WARPING
# ------------------------------------------------------------------------------
def order_points(pts):
    """
    Orders 4 coordinates into:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts, dtype="float32")
    
    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest diff, bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def warp_board(image, corners, size=800):
    """
    Warps the quadrilateral defined by 4 corners into an 800x800 square image.
    """
    rect = order_points(corners)
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    return warped

def detect_chessboard(frame, rows=7, cols=7):
    """Detect a chessboard, draw its corners, and outline its projected boundary."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern_size = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
    else:
        sb_flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, sb_flags
        )

    if not found:
        return frame, None

    cv2.drawChessboardCorners(frame, pattern_size, corners, found)

    grid_x, grid_y = np.meshgrid(
        np.arange(1, cols + 1, dtype=np.float32),
        np.arange(1, rows + 1, dtype=np.float32),
    )
    object_points = np.zeros((rows * cols, 3), dtype=np.float32)
    object_points[:, :2] = np.column_stack(
        (grid_x.ravel(), grid_y.ravel())
    )

    height, width = gray.shape
    focal_length = float(max(width, height))
    camera_matrix = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((5, 1), dtype=np.float64)
    pose_found, rotation, translation = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not pose_found:
        return frame, None

    board_points = np.array(
        [
            [0, 0, 0],
            [cols + 1, 0, 0],
            [cols + 1, rows + 1, 0],
            [0, rows + 1, 0],
        ],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(
        board_points, rotation, translation, camera_matrix, distortion
    )
    outline = np.round(projected).astype(np.int32)
    cv2.polylines(frame, [outline], True, (0, 255, 0), 3, cv2.LINE_AA)
    return frame, outline.reshape(4, 2)

# ------------------------------------------------------------------------------
# 2. SQUARE CROPPING & DIFFERENCE DETECTION
# ------------------------------------------------------------------------------
def crop_all_squares(warped_img, margin=8):
    """
    Slices the 800x800 board into 64 individual square crops.
    Returns: dict {'a1': crop, 'a2': crop, ..., 'h8': crop}
    Assumes standard orientation: White on ranks 1-2 (bottom), Black on ranks 7-8 (top).
    """
    sq_size = warped_img.shape[0] // 8
    squares = {}
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']  # Row 0 is Rank 8 (top of board)
    
    for r_idx, rank in enumerate(ranks):
        for c_idx, file in enumerate(files):
            x1 = c_idx * sq_size + margin
            y1 = r_idx * sq_size + margin
            x2 = (c_idx + 1) * sq_size - margin
            y2 = (r_idx + 1) * sq_size - margin
            squares[f"{file}{rank}"] = warped_img[y1:y2, x1:x2]
            
    return squares

def detect_changed_squares(prev_warped, curr_warped, top_n=4):
    """
    Compares the current frame with the previous move frame to detect
    which squares experienced the highest pixel intensity changes.
    """
    prev_sq = crop_all_squares(prev_warped)
    curr_sq = crop_all_squares(curr_warped)
    
    diff_scores = {}
    for sq_name in prev_sq:
        g1 = cv2.cvtColor(prev_sq[sq_name], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(curr_sq[sq_name], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g1, g2)
        diff_scores[sq_name] = float(np.mean(diff))
        
    sorted_sq = sorted(diff_scores.items(), key=lambda item: item[1], reverse=True)
    return [sq for sq, score in sorted_sq[:top_n]]

# ------------------------------------------------------------------------------
# 3. MOVE RESOLUTION VIA PYTHON-CHESS
# ------------------------------------------------------------------------------
def resolve_move(board, changed_squares):
    """
    Tests combinations of changed squares against legal moves in python-chess.
    Resolves standard moves, captures, pawn promotions, and castling.
    """
    legal_moves = list(board.legal_moves)
    
    # 1. Try pairs of changed squares (from_square -> to_square)
    for sq1 in changed_squares:
        for sq2 in changed_squares:
            if sq1 == sq2:
                continue
            
            # Standard move or capture
            candidate_uci = f"{sq1}{sq2}"
            move = chess.Move.from_uci(candidate_uci)
            if move in legal_moves:
                return move
                
            # Pawn promotion attempt (defaults to Queen)
            promo_move = chess.Move.from_uci(f"{candidate_uci}q")
            if promo_move in legal_moves:
                return promo_move

    # 2. Check for Castling (King moves 2 squares)
    castling_moves = [
        chess.Move.from_uci("e1g1"), chess.Move.from_uci("e1c1"),  # White King/Queen side
        chess.Move.from_uci("e8g8"), chess.Move.from_uci("e8c8")   # Black King/Queen side
    ]
    for c_move in castling_moves:
        if c_move in legal_moves:
            king_from = chess.square_name(c_move.from_square)
            king_to = chess.square_name(c_move.to_square)
            if king_from in changed_squares and king_to in changed_squares:
                return c_move
                
    return None

# ------------------------------------------------------------------------------
# 4. POSITION & WIN ADVANTAGE EVALUATION (ML Model + 5 Strategic Features)
# ------------------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

# Load trained ML model if available
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_model.pkl")
ml_model = None
if os.path.exists(MODEL_FILE):
    try:
        ml_model = joblib.load(MODEL_FILE)
        print(f"[Model Loaded] Successfully loaded ML predictor from: {MODEL_FILE}")
    except Exception as e:
        print(f"[Model Warning] Could not load ML model: {e}")

def extract_board_features(fen_string):
    """
    Extracts the 5 numerical features from White's perspective:
    (Positive = White advantage, Negative = Black advantage).
    """
    board = chess.Board(fen_string)

    # 1. Material Advantage
    white_material = 0
    black_material = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_material += val
        else:
            black_material += val
    material_advantage = white_material - black_material

    # 2. Piece Vulnerability
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

    # 3. Center Control
    white_center = sum(len(board.attackers(chess.WHITE, sq)) for sq in CENTER_SQUARES)
    black_center = sum(len(board.attackers(chess.BLACK, sq)) for sq in CENTER_SQUARES)
    center_control = white_center - black_center

    # 4. Piece Mobility
    original_turn = board.turn
    board.turn = chess.WHITE
    white_mobility = board.legal_moves.count()
    board.turn = chess.BLACK
    black_mobility = board.legal_moves.count()
    board.turn = original_turn
    mobility_advantage = white_mobility - black_mobility

    # 5. King Safety
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

def evaluate_position(board):
    """
    Predicts who is in position to win. Uses the trained ML model if
    'chess_model.pkl' is present; otherwise falls back to weighted heuristics.
    """
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"CHECKMATE! {winner} Wins!", (100.0 if winner == "White" else -100.0)

    if board.is_game_over():
        return "DRAW / STALEMATE", 0.0

    features = extract_board_features(board.fen())

    # --- 1. ML-Based Prediction (if model is loaded) ---
    if ml_model is not None:
        try:
            probs = ml_model.predict_proba([features])[0]
            classes = list(ml_model.classes_)
            
            p_white = probs[classes.index(1)] if 1 in classes else 0.33
            p_black = probs[classes.index(-1)] if -1 in classes else 0.33
            p_draw = probs[classes.index(0)] if 0 in classes else 0.33

            score = (p_white - p_black) * 10.0

            if p_white >= 0.55:
                verdict = f"White Winning (W:{p_white*100:.0f}% D:{p_draw*100:.0f}% B:{p_black*100:.0f}%)"
            elif p_black >= 0.55:
                verdict = f"Black Winning (B:{p_black*100:.0f}% D:{p_draw*100:.0f}% W:{p_white*100:.0f}%)"
            elif p_draw >= 0.45:
                verdict = f"Draw Favored (D:{p_draw*100:.0f}% W:{p_white*100:.0f}% B:{p_black*100:.0f}%)"
            else:
                verdict = f"Even Position (W:{p_white*100:.0f}% D:{p_draw*100:.0f}% B:{p_black*100:.0f}%)"

            return verdict, score
        except Exception as err:
            print(f"[ML Prediction Error, using fallback]: {err}")

    # --- 2. Heuristic Fallback ---
    mat, vuln, center, mob, king_safe = features
    total_score = (
        1.0 * mat +
        0.5 * vuln +
        0.3 * center +
        0.05 * mob +
        0.4 * king_safe
    )

    if total_score >= 2.0:
        verdict = f"White is Winning (+{total_score:.2f})"
    elif total_score >= 0.7:
        verdict = f"White has Slight Advantage (+{total_score:.2f})"
    elif total_score <= -2.0:
        verdict = f"Black is Winning ({total_score:.2f})"
    elif total_score <= -0.7:
        verdict = f"Black has Slight Advantage ({total_score:.2f})"
    else:
        verdict = f"Equal Position / Balanced ({total_score:.2f})"

    return verdict, total_score

def draw_digital_board(board, height, evaluation):
    """Render the tracked chess position in a panel beside the camera feed."""
    panel_width = max(380, int(height * 0.85))
    panel = np.full((height, panel_width, 3), (35, 35, 35), dtype=np.uint8)
    cv2.putText(
        panel, "CURRENT POSITION", (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2
    )
    turn = "White" if board.turn == chess.WHITE else "Black"
    cv2.putText(
        panel, f"{turn} to move | Move {board.fullmove_number}", (16, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1
    )
    cv2.putText(
        panel, evaluation[:48], (16, 78),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 230, 120), 1
    )

    margin = 30
    board_size = min(panel_width - 2 * margin, height - 115)
    if board_size < 8:
        return panel
    square_size = board_size // 8
    board_size = square_size * 8
    left = (panel_width - board_size) // 2
    top = 92 + max(0, (height - 115 - board_size) // 2)

    light_square = (238, 216, 181)
    dark_square = (181, 136, 99)

    def draw_piece_icon(piece, center, size):
        x, y = center
        fill = (250, 250, 250) if piece.color else (25, 25, 25)
        edge = (25, 25, 25) if piece.color else (250, 250, 250)
        scale = size
        thickness = max(1, size // 24)

        def point(px, py):
            return (int(x + px * scale), int(y + py * scale))

        def polygon(points):
            vertices = np.array([point(px, py) for px, py in points], dtype=np.int32)
            cv2.fillPoly(panel, [vertices], fill, lineType=cv2.LINE_AA)
            cv2.polylines(
                panel, [vertices], True, edge, thickness, cv2.LINE_AA
            )

        def circle(px, py, radius, color, line_width=-1):
            cv2.circle(
                panel, point(px, py), max(1, int(radius * scale)),
                color, line_width, cv2.LINE_AA
            )

        #Ever coordinate here is just python drawing the pieces instead of using images
        def base():
            polygon([(-0.25, 0.18), (0.25, 0.18), (0.22, 0.30), (-0.22, 0.30)])

        if piece.piece_type == chess.PAWN:
            circle(0, -0.20, 0.09, fill)
            circle(0, -0.20, 0.09, edge, thickness)
            polygon([(-0.09, -0.09), (0.09, -0.09), (0.16, 0.20), (-0.16, 0.20)])
            base()
        elif piece.piece_type == chess.ROOK:
            polygon([
                (-0.22, -0.27), (-0.11, -0.27), (-0.11, -0.19),
                (-0.04, -0.19), (-0.04, -0.27), (0.04, -0.27),
                (0.04, -0.19), (0.11, -0.19), (0.11, -0.27),
                (0.22, -0.27), (0.20, -0.10), (0.15, 0.20),
                (-0.15, 0.20), (-0.20, -0.10)
            ])
            base()
        elif piece.piece_type == chess.KNIGHT:
            polygon([
                (-0.20, 0.20), (-0.20, 0.08), (-0.10, -0.01),
                (-0.19, -0.15), (-0.16, -0.28), (-0.04, -0.24),
                (0.04, -0.15), (0.16, -0.14), (0.23, -0.06),
                (0.17, 0.02), (0.12, 0.06), (0.17, 0.20)
            ])
            circle(-0.08, -0.16, 0.018, edge)
            base()
        elif piece.piece_type == chess.BISHOP:
            polygon([
                (-0.21, 0.21), (-0.16, 0.12), (-0.11, 0.02),
                (-0.16, -0.04), (-0.10, -0.20), (0, -0.30),
                (0.10, -0.20), (0.16, -0.04), (0.11, 0.02),
                (0.16, 0.12), (0.21, 0.21)
            ])
            cv2.line(panel, point(-0.045, -0.22), point(0.055, -0.10),
                     edge, thickness, cv2.LINE_AA)
            base()
        elif piece.piece_type == chess.QUEEN:
            polygon([
                (-0.22, -0.13), (-0.18, 0.03), (-0.10, -0.18),
                (-0.04, 0.02), (0, -0.28), (0.04, 0.02),
                (0.10, -0.18), (0.18, 0.03), (0.22, -0.13),
                (0.17, 0.18), (-0.17, 0.18)
            ])
            for px, py in ((-0.10, -0.18), (0, -0.28), (0.10, -0.18)):
                circle(px, py, 0.035, fill)
                circle(px, py, 0.035, edge, thickness)
            base()
        elif piece.piece_type == chess.KING:
            polygon([
                (-0.05, -0.30), (0.05, -0.30), (0.05, -0.21),
                (0.13, -0.21), (0.13, -0.13), (0.05, -0.13),
                (0.05, -0.06), (0.13, 0.02), (0.16, 0.20),
                (-0.16, 0.20), (-0.13, 0.02), (-0.05, -0.06),
                (-0.05, -0.13), (-0.13, -0.13), (-0.13, -0.21),
                (-0.05, -0.21)
            ])
            base()

    for row in range(8):
        rank = 7 - row
        for col in range(8):
            x1 = left + col * square_size
            y1 = top + row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            color = light_square if (row + col) % 2 == 0 else dark_square
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)

            piece = board.piece_at(chess.square(col, rank))
            if piece is not None:
                center = (x1 + square_size // 2, y1 + square_size // 2)
                draw_piece_icon(piece, center, square_size)

    return panel

# ------------------------------------------------------------------------------
# 5. MAIN CAMERA LOOP
# ------------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(1)   #RIGHT HERE TO CHANGE DEFAULT CAMERA (0, 1, 2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  #lowered the resolution to 640x480 for better frame rate
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not access webcam. Check permissions and connection.")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")

    window_name = "Real-Time Chess Tracker"
    cv2.namedWindow(window_name)

    # Initialize standard chess board starting position
    board = chess.Board()
    previous_warped = None
    last_evaluation = "Game Started - White to move."

    print("=" * 65)
    print("REAL-TIME CHESS TRACKER & WIN PREDICTOR")
    print("=" * 65)
    print("Show the full chessboard with all 7x7 inner corners visible.")
    print("Make a move, take hands away, then press [SPACEBAR].")
    print("Press 'r' to reset the game | Press 'q' to quit.")
    print("=" * 65 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        display, board_corners = detect_chessboard(display)
        warped = None
        if board_corners is not None:
            warped = warp_board(frame, board_corners)
            if previous_warped is None:
                previous_warped = warped.copy()
                print("Board calibrated! Initial frame captured. White moves first.\n")

        info_text = (
            "Chessboard detected | Press Space to record move"
            if board_corners is not None
            else "Show full chessboard (7x7 inner corners)"
        )
        cv2.putText(display, info_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        digital_board = draw_digital_board(board, display.shape[0], last_evaluation)
        combined_view = np.hstack((display, digital_board))
        cv2.imshow(window_name, combined_view)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            previous_warped = None
            board.reset()
            last_evaluation = "Game Reset."
            print("\nGame has been reset. Show the chessboard to recalibrate.")

        elif key == 32:  # SPACEBAR
            if warped is None or previous_warped is None:
                print("Warning: Show the full chessboard before recording a move.")
                continue

            # Detect top changed squares
            changed = detect_changed_squares(previous_warped, warped, top_n=4)
            print(f"\nChanged squares detected: {changed}")

            # Check if this matches a valid legal move
            move = resolve_move(board, changed)
            if move is not None:
                board.push(move)
                previous_warped = warped.copy()

                # Run win prediction model
                verdict, score = evaluate_position(board)
                last_evaluation = verdict

                print("-" * 50)
                print(f"MOVE REGISTERED : {move.uci()} ({'White' if board.turn == chess.BLACK else 'Black'})")
                print(f"WIN PREDICTION  : {verdict}")
                print(f"FEN             : {board.fen()}")
                print("-" * 50)
                print(board)
                print("-" * 50 + "\n")
            else:
                print("Could not identify a valid legal move from the changed squares.")
                print(f"Current player's legal moves: {[m.uci() for m in board.legal_moves][:8]}...")
                print("Tip: Ensure your hand is completely out of frame and press Spacebar again.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()