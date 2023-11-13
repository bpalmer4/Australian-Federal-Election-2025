"""Common constants, etc."""
import sys

# --- CONSTANTS


VOTING_INTENTION = "voting-intention"
ATTITUDINAL = "attitudinal"

MIDDLE_DATE = "Mean Date"
FIRST_DATE = "First Date"
LAST_DATE = "Last Date"
# Note: next line is a list and not a tuple because this what pandas expects
ALL_DATES = [FIRST_DATE, MIDDLE_DATE, LAST_DATE]

COLOR_COALITION = "royalblue"
COLOR_LABOR = "crimson"


# --- FUNCTIONS


def confirm(condition: bool, message: str = "", exit_code: int = -1) -> None:
    """Check a conditional and generate a system exit if that conditional is False.
    Much like a Python assert statement, but this function cannot be turned off
    in production code."""

    if not condition:
        if message:
            print(message)
        sys.exit(exit_code)
