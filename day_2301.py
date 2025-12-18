# =========================
# POLICY (Step 2)
# =========================

POLICY = {
    "block_high_risk": True,
    "min_gap_length": 2
}

# =========================
# UTILITIES (Step 1)
# =========================

def detect_gaps(road):
    gaps = []
    start_index = None

    for i in range(len(road)):
        if road[i] == 0 and start_index is None:
            start_index = i
        elif road[i] == 1 and start_index is not None:
            gaps.append((start_index, i - start_index))
            start_index = None

    if start_index is not None:
        gaps.append((start_index, len(road) - start_index))

    return sorted(gaps, key=lambda x: x[1], reverse=True)


def risk_track(gaps):
    predictions = []
    for start, length in gaps:
        if length >= 3:
            risk = "low"
        elif length == 2:
            risk = "medium"
        else:
            risk = "high"

        predictions.append({
            "start": start,
            "length": length,
            "risk": risk
        })
    return predictions


# =========================
# SYSTEM OBJECT (Step 4)
# =========================

class TrafficController:
    def __init__(self, road, policy):
        self.road = road
        self.policy = policy
        self.history = []

    def log_action(self, action, details=None):
        self.history.append({
            "action": action,
            "details": details or {}
        })

    def feedback_guard(self):
        if len(self.history) < 2:
            return True

        last_two = self.history[-2:]
        blocked = sum(1 for h in last_two if h["action"] == "blocked")

        return blocked < 2

    def decide_best_action(self):
        gaps = detect_gaps(self.road)
        if not gaps:
            return None

        predictions = risk_track(gaps)
        best = predictions[0]

        if self.policy["block_high_risk"] and best["risk"] == "high":
            return None

        if best["length"] < self.policy["min_gap_length"]:
            return None

        return best["start"]

    def control_loop(self, steps=3):
        for step in range(steps):
            print(f"\n--- Step {step + 1} ---")

            if not self.feedback_guard():
                print("System cooling down")
                self.log_action("blocked", {"reason": "feedback_guard"})
                continue

            position = self.decide_best_action()
            if position is not None:
                self.road[position] = 1
                self.log_action("car inserted", {"position": position})
                print(f"Car inserted at position {position}")
            else:
                print("No safe insertion")
                self.log_action("blocked", {"reason": "policy"})

            print("Road state:", self.road)

        return self.road


# =========================
# STEP 4 — OBJECT CALL
# =========================

road = [1, 1, 0, 0, 1, 0, 0, 0, 1, 0]

controller = TrafficController(road, POLICY)

final_road = controller.control_loop(steps=3)

print("\nFinal road:", final_road)
print("\nHistory:", controller.history)