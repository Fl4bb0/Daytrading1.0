# kvant — quantitative trading library

# One-way brokerage fee as a fraction (0.0008 = 0.08 %).
# Round-trip cost is 2 * BROKERAGE_FEE.
BROKERAGE_FEE: float = 0.0008

# Minimum predicted-class probability required to enter a trade.
# Predictions below this threshold are treated as HOLD.
CONFIDENCE_THRESHOLD: float = 0.55
