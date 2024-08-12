# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random

def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    # If no previous play, make a random choice
    if prev_play == "":
        return random.choice(["R", "P", "S"])

    # Strat 1: Mirror move
    def mirror_strategy():
        if len(opponent_history) > 1: return opponent_history[-1] 
        else: return random.choice(["R", "P", "S"])

    # Strat 2: Beat common move
    def counter_most_common():
        if len(opponent_history) == 0:
            return random.choice(["R", "P", "S"])
        common = max(set(opponent_history), key=opponent_history.count)
        if common == "R":
            return "P"
        elif common == "P":
            return "S"
        else:
            return "R"

    # Strategy 3: Cycle order
    def cycle_strategy():
        mv = ["R", "P", "S"]
        if len(opponent_history) == 0:
            return random.choice(["R", "P", "S"])
        idx = mv.index(opponent_history[-1])
        return mv[(idx + 1) % 3]

    # Decide which strategy to use
    if len(opponent_history) < 10:
        guess = mirror_strategy()
    elif len(opponent_history) < 50:
        guess = counter_most_common()
    else:
        guess = cycle_strategy()

    return guess
