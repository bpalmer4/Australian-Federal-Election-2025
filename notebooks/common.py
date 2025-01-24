"""Common constants, etc."""

import sys
from typing import Any

# --- CONSTANTS


VOTING_INTENTION = "voting-intention"
ATTITUDINAL = "attitudinal"
NSW = "NSW"
VIC = "VIC"
QLD = "QLD"
SA = "SA"
WA = "WA"
TAS = "TAS"
NT = "NT"
ACT = "ACT"

MIDDLE_DATE = "Mean Date"
FIRST_DATE = "First Date"
LAST_DATE = "Last Date"
# Note: next line is a list and not a tuple because this what pandas expects
ALL_DATES = [FIRST_DATE, MIDDLE_DATE, LAST_DATE]

COLOR_COALITION = "royalblue"
COLOR_LABOR = "crimson"


# --- FUNCTIONS


def ensure(condition: Any, message: str = "", exit_code: int = 1) -> None:
    """Check a conditional and generate a system exit if that conditional is False.
    Much like a Python assert statement, but this function cannot be turned off
    in production code (although the resulting exception could be caught)."""

    if condition is None or not bool(condition):
        exit_arg: int | str = message if message else exit_code
        sys.exit(exit_arg)
