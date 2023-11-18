"""Functions for capturing and cleaning data from Wikipedia."""

import re
from datetime import date
from io import StringIO
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from common import ALL_DATES, MIDDLE_DATE, ensure

# --- [VERY SIMPLE] WEB BASED DATA CAPTURE --


def get_url(url: str) -> str:
    """Get the text found at a URL."""

    headers = {
        "Cache-Control": "no-cache, must-revalidate, private, max-age=0",
        "Pragma": "no-cache",
    }
    timeout = 15  # seconds
    response = requests.get(url.format(rn=time()), headers=headers, timeout=timeout)
    ensure(response.status_code == 200)  # successful retrieval
    return response.text


def get_table_list(url: str) -> list[pd.DataFrame]:
    """Return a list of tables found at a URL. Tables
    are returned in pandas DataFrame format."""

    html = get_url(url)
    df_list = pd.read_html(StringIO(html))
    ensure(df_list)  # check we have at least one table
    return df_list


def get_combined_table(
    df_list: list[pd.DataFrame], 
    table_list: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """Get selected tables (by int in table_list) from Wikipedia page.
    Return a single merged table for the selected tables.
    NOTE: Wikipedia has calandar year tables. Consequerntly,
          the table_list argument will need to be updated
          each year."""

    if not table_list:
        return None
    selected: list[pd.DataFrame] = [df_list[i] for i in table_list]
    combined: Optional[pd.DataFrame] = None
    for table in selected:
        table = table.copy()  # preserve original
        flat = flatten_col_names(table.columns)
        table.columns = flat
        if combined is None:
            combined = table.copy()
        else:
            # check table headers align ...
            combined_set = set(combined.columns)
            table_set = set(table.columns)
            ensure(combined_set.issuperset(table_set), "Column name mismatch")
            combined = pd.concat((combined, table))

    return combined


# --- DATA CLEANING ---

# Common unicode symbols
ENDASH = "\u2013"
EMDASH = "\u2014"
HYPHEN = "\u002D"
MINUS = "\u2212"
TILDE = "~"
COMMA = ","


def remove_event_rows(t: pd.DataFrame) -> pd.DataFrame:
    """Remove the event marker rows."""

    t = t.loc[t[t.columns[0]] != t[t.columns[1]]]
    t = t.loc[t[t.columns[1]] != t[t.columns[2]]]
    t = t[t[t.columns[1]].notna()]
    return t


def drop_empty(t: pd.DataFrame) -> pd.DataFrame:
    """Remove all empty rows and columns."""

    t = t.dropna(axis=0, how="all")
    t = t.dropna(axis=1, how="all")
    return t


def fix_numerical_cols(t: pd.DataFrame) -> pd.DataFrame:
    """Convert selected columns from strings to numeric data type."""

    fixable_cols = (
        "Primary vote",
        "2pp vote",
        "Sample",
        "Preferred Prime Minister",
        "Satisfied",
        "Dissatisfied",
        "Don't Know",
        "Net",
    )
    for col in t.columns:
        if not any(x in col for x in fixable_cols):
            continue
        t[col] = (
            t[col]
            .str.replace(r"\[.+\]", "", regex=True)  # remove footnotes
            .str.replace("%", "")
            .str.replace(TILDE, "")
            .str.replace(ENDASH, HYPHEN)
            .str.replace(EMDASH, HYPHEN)
            .str.replace(MINUS, HYPHEN)
            .str.replace(r"^\s*-+\s*$", "", regex=True)
            .str.replace("n/a", "")
            .str.replace("?", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.strip()
            .replace("", np.nan)  # NaN empty lines
            .astype(float)
        )
    return t


def fix_column_names(t: pd.DataFrame) -> pd.DataFrame:
    """Replace 'Unnamed' column names with ''."""

    replacements = {}
    for c in t.columns:
        if "Unnamed" in c[1]:
            replacements[c[1]] = ""
    if replacements:
        t = t.rename(columns=replacements, level=1)
    return t


def remove_footnotes(t: pd.DataFrame) -> pd.DataFrame:
    """Remove Wikipedia footnote references from the Brand column"""

    branding_labels = ["Brand", "Firm"]

    for brand in branding_labels:
        if brand not in t.columns.get_level_values(0):
            continue
        col = t.columns[t.columns.get_level_values(0) == brand]
        ensure(len(col) == 1)
        t.loc[:, col] = (
            t.loc[:, col]  # returns a single column DataFrame
            .pipe(lambda x: x[x.columns[0]])  # make as Series
            .str.replace(r"\[.*\]", "", regex=True)  # remove footnotes
            .str.strip()  # remove any leading/trailing whitespaces
        )
    return t


def get_dates(tokens: list[str]) -> pd.Series:
    """Extract the first, middle and last date from a list of date tokens."""

    last_day = None
    day, month, year = None, None, None
    remember = tokens.copy()
    while tokens:
        token = tokens.pop()

        if re.match(r"[0-9]{4}", token):
            year = token
        elif re.match(r"[A-Za-z]+", token):
            month = token
        elif re.match(r"[0-9]{1,2}", token):
            day = token
        else:
            print(
                f"WARNING: {token} not recognised in get_mean_date()"
                f"with these date tokens {remember}"
            )

        if (
            last_day is None
            and day is not None
            and month is not None
            and year is not None
        ):
            last_day = pd.Timestamp(f"{year} {month[:3]} {day}")

    if month is None:
        print(f"WARNING: missing month in these tokens? {remember}")

    # sadly we have cases of this ...
    if not last_day:
        if day is None:
            day = 1  # assume first of month
        last_day = pd.Timestamp(f"{year} {month[:3]} {day}")

    # get the middle date
    first_day = pd.Timestamp(f"{year} {month[:3]} {day}")
    if first_day > last_day:
        print(
            "CHECK there may be a problem with these dates in get_dates(): "
            f"{first_day} {last_day} with these tokens {remember}"
        )
    middle_day = first_day + ((last_day - first_day) / 2)
    return pd.Series(
        data=(first_day.date(), middle_day.date(), last_day.date()),
        index=ALL_DATES,
    )


def tokenise_dates(dates: pd.Series) -> pd.Series:
    """Return the date as a list of tokens."""

    return (
        dates.str.strip()
        .str.replace(r"\[.+\]", "", regex=True)  # footnotes
        .str.replace(ENDASH, HYPHEN)
        .str.replace(EMDASH, HYPHEN)
        .str.replace(MINUS, HYPHEN)
        .str.replace("c. ", "", regex=False)
        .str.split(r"[\-,\s\/]+")
    )


def get_relevant_dates(t: pd.DataFrame) -> pd.DataFrame:
    """Get the first, middle and last date in the date range,
    into the columns FIRST_DATE, MIDDLE_DATE and LAST_DATE."""

    # get the Date column name
    cols = [col for col in t.columns if "Date" in col]
    ensure(len(cols) == 1)

    # assumes dates in strings are ordered from first to last
    tokens = tokenise_dates(t[cols[0]])
    t[ALL_DATES] = tokens.apply(get_dates).astype(("datetime64[ns]"))
    return t


def clean(table: pd.DataFrame) -> pd.DataFrame:
    """Clean the extracted data tables."""

    t = table.copy()
    t = remove_event_rows(t)
    t = drop_empty(t)
    t = fix_numerical_cols(t)
    t = fix_column_names(t)
    t = remove_footnotes(t)
    t = get_relevant_dates(t)
    t = t.sort_values(MIDDLE_DATE, ascending=True)
    t = t.reset_index(drop=True)  # Ensure a unique index
    return t


def flatten_col_names(columns: pd.Index) -> list[str]:
    """Flatten the hierarchical column index."""

    ensure(columns.nlevels >= 2)
    flatter = [
        " ".join(col).strip() if col[0] != col[1] else col[0] for col in columns.values
    ]
    pattern = re.compile(r"\[.+\]")
    flat = [re.sub(pattern, "", s) for s in flatter]  # remove footnotes
    return flat


# --- Validation ---


def row_addition_check(
    table: pd.DataFrame,
    pattern_list: list[str],
    target: int | float = 100.0,
    tolerance: int | float = 1.01,
) -> Optional[pd.DataFrame]:
    """Check that the columns that regex match with a pattern in the
    pattern list all add across to the target plus/minus the specified
    tolerance. Returns a DataFrame with the rows of concern."""

    problems = {}
    for pattern in pattern_list:
        c_pattern = re.compile(pattern)
        cols = [c for c in table.columns if re.match(c_pattern, c)]
        row_sum = table[cols].sum(axis=1)
        problematic = table[cols].notna().sum(axis=1).astype(bool) & (
            (row_sum > target + tolerance) | (row_sum < target - tolerance)
        )
        if problematic.any():
            problems[pattern] = row_sum[table[problematic].index]
    if problems:
        r_table = table.copy()
        for pattern, rows in problems.items():
            r_table[pattern] = np.nan
            r_table.loc[rows.index, pattern] = rows
        return r_table[r_table[problems.keys()].any(axis=1)]
    return None


def distribute_undecideds(
    table: pd.DataFrame,
    undec_col: str,
    col_pattern_list: list[str],
    target: int | float = 100,
    tolerance: Optional[int | float] = None,
) -> pd.DataFrame:
    """Distribute the undecided vote column among those columns that
    match the col_pattern, where the other columns do not add to
    the target (plus or minus the tolerance).
    Return an updated table."""

    undecideds = table.loc[table[undec_col].notna(), undec_col]
    for col_pattern in col_pattern_list:
        columns = [c for c in table.columns if col_pattern in c and c != undec_col]
        tol = tolerance if tolerance is not None else 0.2 if len(columns) <= 2 else 1.01
        row_sum = table.loc[undecideds.index, columns].sum(axis=1)
        row_sum = row_sum[(row_sum < target - tol) | (row_sum > target + tol)]
        affected_rows = row_sum.index
        share = table.loc[affected_rows, columns].div(row_sum, axis=0)
        allocation = share.mul(undecideds[affected_rows], axis=0)
        table.loc[affected_rows, columns] += allocation

    return table


def methodology(
    data: dict[str, pd.DataFrame],
    effective_date: pd.Period,
    change_from: str,
    change_to: str,
) -> dict[str, pd.DataFrame]:
    """Mark methodological changes in the data by changing
    the firm/brand of the pollster in the data."""

    for label, df in data.items():
        branding = "Brand", "Firm"
        for brand in branding:
            if brand not in df.columns:
                continue
            mask = df[brand].str.contains(change_from) & (
                df[MIDDLE_DATE] >= effective_date
            )
            df.loc[mask, brand] = change_to
            data[label] = df
    return data


# --- DATA STORAGE ---
DATA_DIR = "../data/"
FILE_TYPE = ".csv"


def _init_common_storage(
    data_dir: str,
) -> str:
    """Initialisation: Path creation and today getting."""

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    today = str(date.today()).replace("-", "")
    return today


def store(
    dictionary: dict[str, pd.DataFrame],
    data_dir: str = DATA_DIR,
) -> None:
    """Save the captured data to file."""

    today = _init_common_storage(data_dir)
    for label, table in dictionary.items():
        filename = f"{DATA_DIR}{label}-{today}{FILE_TYPE}"
        table.to_csv(filename)


def retrieve(
    capture_date: Optional[str] = None,  # format YYYYMMDD
    data_dir: str = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Retrieve today's captured data from file. Return a
    dictionary of DataFrames. Return an empty dictionary if
    data has not been captured for today."""

    data = {}
    today = _init_common_storage(data_dir)
    capture_date = today if capture_date is None else capture_date
    directory = Path(data_dir)
    for file in directory.glob(f"*{capture_date}{FILE_TYPE}"):
        name = file.name.replace(f"-{capture_date}{FILE_TYPE}", "")
        df = pd.read_csv(file, index_col=0)
        for column in ALL_DATES:
            if column not in df.columns:
                continue
            df[column] = pd.PeriodIndex(df[column], freq="D")
        data[name] = df

    return data
