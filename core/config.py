"""Configuration loading.

Every tunable parameter lives in config.yaml at the project root.
Modules receive plain dictionaries; nothing is hard-coded elsewhere.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Instruments traded as crypto pairs (fractional quantities, 24/7 market,
# higher assumed costs). Everything else is treated as an equity/ETF.
CRYPTO_SYMBOLS = {"BTC/USD"}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load and lightly validate the YAML configuration."""
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    for section in ("account", "risk", "costs", "data", "backtest", "strategies"):
        if section not in config:
            raise KeyError(f"config.yaml is missing the '{section}' section")
    return config


def is_crypto(symbol: str) -> bool:
    """True if the symbol trades on the crypto side of the Alpaca account."""
    return symbol in CRYPTO_SYMBOLS or "/" in symbol


def cost_pct_for(symbol: str, costs: dict) -> float:
    """Per-side slippage+fee assumption for a symbol, as a fraction (not %)."""
    pct = costs["crypto_cost_pct_per_side"] if is_crypto(symbol) else costs["etf_cost_pct_per_side"]
    return pct / 100.0
