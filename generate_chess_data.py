import os
import random

def generate_chess_dataset(filename, num_samples=5000):
    """Generates a synthetic dataset of chess Q&A."""
    
    questions = [
        ("How does the pawn move?", "The pawn moves forward exactly one square, but captures diagonally. On its first move, it can advance two squares."),
        ("Explain the en passant rule.", "En passant is a special pawn capture that can only occur immediately after a pawn makes a move of two squares from its starting position."),
        ("What is a fork in chess?", "A fork is a tactic where a single piece makes two or more direct attacks simultaneously."),
        ("How does the knight move?", "The knight moves in an L-shape: two squares vertically and one square horizontally, or two squares horizontally and one square vertically."),
        ("What is a pin?", "A pin is a situation where a piece cannot move because doing so would expose a more valuable piece to capture."),
        ("How does the bishop move?", "The bishop moves diagonally any number of squares."),
        ("What is castling?", "Castling is a special move involving the king and a rook, moving the king two squares towards the rook and the rook to the square the king crossed."),
        ("When does a stalemate happen?", "A stalemate occurs when the player whose turn it is to move is not in check but has no legal moves. The game ends in a draw."),
        ("How does the king move?", "The king moves exactly one square horizontally, vertically, or diagonally. A special move is castling."),
        ("What is a discovered attack?", "A discovered attack is a direct attack revealed when one piece moves out of the way of another."),
        ("How does the queen move?", "The queen can move any number of vacant squares diagonally, horizontally, or vertically."),
        ("What does checkmate mean?", "Checkmate occurs when a king is placed under direct attack (check) and has no legal move to escape."),
        ("What is a skewer?", "A skewer is an attack upon two pieces in a line, similar to a pin. The more valuable piece is in front, and forced to move."),
        ("How does the rook move?", "The rook moves horizontally or vertically any number of squares."),
        ("Explain pawn promotion.", "When a pawn reaches the eighth rank, it is exchanged for the player's choice of a queen, rook, bishop, or knight of the same color.")
    ]

    # Add variations to questions to make the model robust
    variations = {
        "How does the pawn move?": ["What is a pawn move?", "Tell me about the pawn.", "Pawn movement rules?"],
        "Explain the en passant rule.": ["What is en passant?", "How does en passant work?", "En passant explained."],
        "What is a fork in chess?": ["Define a chess fork.", "Explain fork tactic.", "What does fork mean?"],
        "How does the knight move?": ["Knight movement?", "What is the knight's move?", "Tell me how a knight moves."],
        "What is a pin?": ["Explain the pin tactic.", "Define a pin.", "What is pinning?"],
        "How does the bishop move?": ["Bishop movement?", "What is a bishop's move?", "Tell me about the bishop."],
        "What is castling?": ["Explain castling.", "How do you castle?", "What are castling rules?"],
        "When does a stalemate happen?": ["What is stalemate?", "Explain a draw by stalemate.", "How does stalemate occur?"],
        "How does the king move?": ["King movement?", "What is the king's move?", "Tell me about the king."],
        "What is a discovered attack?": ["Define discovered attack.", "Explain discovered check.", "What is a discovery?"],
        "How does the queen move?": ["Queen movement?", "What is the queen's move?", "Tell me how the queen moves."],
        "What does checkmate mean?": ["What is checkmate?", "Explain checkmate.", "Define checkmate in chess."],
        "What is a skewer?": ["Define a skewer.", "Explain skewer tactic.", "What is skewering?"],
        "How does the rook move?": ["Rook movement?", "What is a rook's move?", "Tell me about the rook."],
        "Explain pawn promotion.": ["What is pawn promotion?", "How do you promote a pawn?", "Pawn promotion rules?"]
    }

    # Format clearly as Q&A for the model to learn the pattern
    # Use a separator the model can learn to stop at
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Write the core definitions multiple times to engrain them
        for _ in range(num_samples):
            q, a = random.choice(questions)
            
            # 30% chance to use a varied question phrasing
            if random.random() < 0.3 and q in variations:
                q = random.choice(variations[q])
            
            # Write in a strict Q&A format
            f.write(f"Question: {q}\nAnswer: {a}\n\n")
            
    print(f"Generated {num_samples} chess QA pairs to {filename}")

if __name__ == "__main__":
    generate_chess_dataset("data/chess_qa.txt")
