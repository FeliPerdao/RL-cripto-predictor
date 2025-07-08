from itertools import combinations
import json

ALL_FEATURES = [
    "return", "volume", "rsi", "ema_9", "ema_21",
    "macd", "macd_signal", "macd_diff",
    "boll_upper", "boll_lower",
    "d_price", "dd_price", "volume_change"
]

def generate_all_combinations():
    combos = []
    for r in range(1, len(ALL_FEATURES) + 1):
        for combo in combinations(ALL_FEATURES, r):
            combos.append(list(combo))

    with open("feature_combinations.json", "w") as f:
        json.dump(combos, f, indent=2)

    print(f"✅ Guardadas {len(combos)} combinaciones en feature_combinations.json")

if __name__ == "__main__":
    generate_all_combinations()
