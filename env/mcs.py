"""Discrete PHY rate mapping used by the Co-SR environment."""
import numpy as np

import config


def sinr_linear_to_db(sinr):
    return 10.0 * np.log10(max(float(sinr), 1e-15))


def sinr_to_mcs_rate_mbps(sinr):
    """Map linear SINR to the highest supported discrete MCS rate."""
    sinr_db = sinr_linear_to_db(sinr)
    rate = 0.0
    for threshold_db, candidate_rate in zip(
        config.COSR_MCS_SINR_THRESHOLDS_DB,
        config.COSR_MCS_RATES_MBPS,
    ):
        if sinr_db < threshold_db:
            break
        rate = float(candidate_rate)
    return rate


def validate_mcs_table():
    thresholds = config.COSR_MCS_SINR_THRESHOLDS_DB
    rates = config.COSR_MCS_RATES_MBPS
    if len(thresholds) != len(rates) or not thresholds:
        raise ValueError("Co-SR MCS thresholds and rates must have equal non-zero length")
    if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError("Co-SR MCS SINR thresholds must be strictly increasing")
    if any(b <= a for a, b in zip(rates, rates[1:])):
        raise ValueError("Co-SR MCS rates must be strictly increasing")
