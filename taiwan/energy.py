"""Convert recorder-hours into energy and battery days (AudioMoth, Hill et al. 2018).

Lead with the percentage of recording energy saved: it does not depend on battery capacity.
Battery days depend on config.BATTERY_WH, an assumption the paper does not give.
"""
from taiwan import config as C


def mwh_per_night(sampled_hours_recorded: float) -> float:
    """Energy for one 24 h day when `sampled_hours_recorded` hours each record
    MINUTES_PER_SAMPLED_HOUR minutes; the rest of the day the device sleeps."""
    rec_h = sampled_hours_recorded * C.MINUTES_PER_SAMPLED_HOUR / 60
    return rec_h * C.RECORD_MW + (24 - rec_h) * C.SLEEP_MW


def battery_days(fraction_of_schedule: float) -> float:
    """Days of operation, recording every night, when only `fraction_of_schedule`
    of the baseline sampled hours are recorded."""
    per_day = mwh_per_night(fraction_of_schedule * C.SAMPLED_HOURS_PER_NIGHT)
    return C.BATTERY_WH * 1000 / per_day


def baseline_days() -> float:
    return battery_days(1.0)
