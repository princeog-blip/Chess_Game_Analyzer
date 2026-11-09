# ==============================================================================
# CSCI 270 - Real-Time Chess Game Tracker & Position Evaluator
# Pipeline:
#   1. Capture live camera feed and calibrate 4 board corners (interactive click)
#   2. Perspective warp the board into an 800x800 top-down square
#   3. Initialize standard chess board (chess.Board)
#   4. After a player makes a move and presses Spacebar:
#        - Detect changed squares via image differencing (cv2.absdiff)
#        - Resolve and validate the exact legal move with python-chess
#        - Update internal board state and FEN
#        - Evaluate and predict who is in position to win
# ==============================================================================

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

def detect_changed_squares(prev_warped, curr_warped, top_n=6):
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
        
        # Blur to reduce minor camera sensor noise
        g1 = cv2.GaussianBlur(g1, (5, 5), 0)
        g2 = cv2.GaussianBlur(g2, (5, 5), 0)
        
        diff = cv2.absdiff(g1, g2)
        diff_scores[sq_name] = float(np.mean(diff))
        
    sorted_sq = sorted(diff_scores.items(), key=lambda item: item[1], reverse=True)
    return [sq for sq, score in sorted_sq[:top_n]], sorted_sq[:top_n]

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
# 4. POSITION & WIN ADVANTAGE EVALUATION
# ------------------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.2,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# Center squares bonus (d4, e4, d5, e5)
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

def evaluate_position(board):
    """
    Evaluates who is in position to win based on:
      1. Material balance (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9)
      2. Board center control (d4, e4, d5, e5)
      3. Piece mobility (number of legal move options)
      4. King safety / Checks
    Returns formatted evaluation verdict string and numeric score.
    """
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"CHECKMATE! {winner} Wins!", (100.0 if winner == "White" else -100.0)
    
    if board.is_game_over():
        return "DRAW / STALEMATE", 0.0

    # 1. Material score
    material_score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            material_score += val if piece.color == chess.WHITE else -val

    # 2. Center control bonus
    center_score = 0.0
    for sq in CENTER_SQUARES:
        piece = board.piece_at(sq)
        if piece:
            bonus = 0.3
            center_score += bonus if piece.color == chess.WHITE else -bonus

    # 3. Mobility score
    white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    board.push(chess.Move.null())  # Null move to peek at opponent mobility
    black_mobility = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.pop()
    mobility_score = 0.05 * (white_mobility - black_mobility)

    # 4. Check pressure
    check_score = 0.0
    if board.is_check():
        # Negative impact for whichever player is in check
        check_score = -0.5 if board.turn == chess.WHITE else 0.5

    # Total advantage score (> 0 means White advantage, < 0 means Black advantage)
    total_eval = material_score + center_score + mobility_score + check_score

    if total_eval >= 2.0:
        verdict = f"White is Winning (+{total_eval:.2f})"
    elif total_eval >= 0.7:
        verdict = f"White has Slight Advantage (+{total_eval:.2f})"
    elif total_eval <= -2.0:
        verdict = f"Black is Winning ({total_eval:.2f})"
    elif total_eval <= -0.7:
        verdict = f"Black has Slight Advantage ({total_eval:.2f})"
    else:
        verdict = f"Equal Position / Balanced ({total_eval:.2f})"

    return verdict, total_eval

# ------------------------------------------------------------------------------
# 5. MAIN INTERACTIVE CAMERA LOOP
# ------------------------------------------------------------------------------
clicked_corners = []

def mouse_callback(event, x, y, flags, param):
    global clicked_corners
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_corners) < 4:
        clicked_corners.append((x, y))
        print(f"Corner {len(clicked_corners)} captured at: ({x}, {y})")

def main():
    global clicked_corners
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam. Check permissions and connection.")
        return

    window_name = "Real-Time Chess Tracker"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Initialize standard chess board starting position
    board = chess.Board()
    previous_warped = None
    last_evaluation = "Game Started - White to move."

    print("=" * 65)
    print("REAL-TIME CHESS TRACKER & WIN PREDICTOR")
    print("=" * 65)
    print("1. Set up all 32 pieces in their starting positions.")
    print("2. Click the 4 OUTER CORNERS of the board on the camera window.")
    print("   (Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
    print("3. Make a physical move on the board, take hands away.")
    print("4. Press [SPACEBAR] to detect the move & update win prediction.")
    print("   Press 'r' to reset board and corners | Press 'q' to quit.")
    print("=" * 65 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Draw red dots for clicked corners
        for idx, pt in enumerate(clicked_corners, 1):
            cv2.circle(display, pt, 6, (0, 0, 255), -1)
            cv2.putText(display, str(idx), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        warped = None
        if len(clicked_corners) == 4:
            # Draw green outline polygon connecting corners
            pts = np.array(clicked_corners, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Warp to 800x800 square
            warped = warp_board(frame, clicked_corners)

            # Auto-save initial board frame once 4 corners are selected
            if previous_warped is None:
                previous_warped = warped.copy()
                print("\n>>> Board calibrated! Initial frame captured. White moves first.\n")

            # Draw current evaluation on warped view
            warped_display = warped.copy()
            
            # Overlay an 8x8 grid with square names for easy alignment verification
            sq_size = warped.shape[0] // 8
            files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
            for r_idx, r in enumerate(ranks):
                for c_idx, f in enumerate(files):
                    cv2.rectangle(warped_display, 
                                  (c_idx * sq_size, r_idx * sq_size), 
                                  ((c_idx + 1) * sq_size, (r_idx + 1) * sq_size), 
                                  (50, 50, 50), 1)
                    if (r_idx == 7) or (c_idx == 0):
                        cv2.putText(warped_display, f"{f}{r}", 
                                    (c_idx * sq_size + 4, (r_idx + 1) * sq_size - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            cv2.putText(warped_display, last_evaluation, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("Top-Down Warped View", warped_display)

        # Onscreen instructions
        info_text = f"Corners: {len(clicked_corners)}/4 | Focus window & Press SPACE for Move"
        cv2.putText(display, info_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            clicked_corners.clear()
            previous_warped = None
            board.reset()
            last_evaluation = "Game Reset."
            print("\nGame and corners have been reset. Set pieces and click 4 corners again.")

        elif key == 32:  # SPACEBAR
            print("\n[Spacebar Pressed] Processing move detection...")
            if warped is None or previous_warped is None:
                print("Warning: Please click all 4 board corners first before making a move!")
                continue

            # Detect top changed squares
            changed, scores = detect_changed_squares(previous_warped, warped, top_n=6)
            print(f"Top changed squares detected: {scores}")

            # Check if this matches a valid legal move
            move = resolve_move(board, changed)
            if move is not None:
                board.push(move)
                previous_warped = warped.copy()

                # Run win prediction model
                verdict, score = evaluate_position(board)
                last_evaluation = f"{move.uci().upper()} | {verdict}"

                print("-" * 50)
                print(f"MOVE REGISTERED : {move.uci()} ({'White' if board.turn == chess.BLACK else 'Black'})")
                print(f"WIN PREDICTION  : {verdict}")
                print(f"FEN             : {board.fen()}")
                print("-" * 50)
                print(board)
                print("-" * 50 + "\n")
            else:
                print(f"Could not identify a legal move among changed squares: {changed}")
                print(f"White/Black turn: {'White' if board.turn == chess.WHITE else 'Black'}")
                print(f"Sample legal moves: {[m.uci() for m in board.legal_moves][:10]}")
                print("Tip: Check that rank 1 & 2 (White) are at the bottom of the warped view!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

