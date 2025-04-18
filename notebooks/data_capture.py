"""Functions for capturing and cleaning data from Wikipedia."""

import re
from datetime import date
from io import StringIO
from pathlib import Path
from time import time
from typing import Sequence

import numpy as np
import pandas as pd
import requests
from IPython.display import display
from common import ALL_DATES, MIDDLE_DATE, ATTITUDINAL, ensure


# --- Constants

UNDECIDED_COLUMN: str = "Primary vote UND"
PRIM_OTHERS_TO_SUM: str = "Primary vote (GRN|UAP|ONP|OTH|DEM|DLP)"


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


def flatten_col_names(columns: pd.Index) -> list[str]:
    """Flatten the hierarchical column index."""

    ensure(columns.nlevels >= 2)
    flatter = [
        " ".join(col).strip() if col[0] != col[1] else col[0] for col in columns.values
    ]
    pattern = re.compile(r"\[.+\]")
    flat = [re.sub(pattern, "", s) for s in flatter]  # remove footnotes
    return flat


def get_combined_table(
    df_list: list[pd.DataFrame],
    table_list: Sequence[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """Get selected tables (by int in table_list) from Wikipedia page.
    Return a single merged table for the selected tables.
    NOTE: Wikipedia has calandar year tables. Consequerntly,
          the table_list argument will need to be updated
          each year."""

    if table_list is None or not table_list:
        if verbose:
            print("No tables selected.")
        return None
    combined: pd.DataFrame | None = None
    for table_num in table_list:
        table = df_list[table_num].copy()
        if verbose:
            print("DEBUG:", table.head())
        flat = flatten_col_names(table.columns)
        table.columns = pd.Index(flat)
        if combined is None:
            combined = table
        else:
            table_set = set(table.columns)
            combined_set = set(combined.columns)
            problematic = table_set.difference(combined_set)
            if problematic:
                print(f"WARNING: with table {table_num}, "
                + f"{problematic} not in combined table.") 
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
        "Preferred prime minister",
        "Satisfied",
        "Dissatisfied",
        "Don't Know",
        "Net",
    )
    fixable_cols_lower = [x.lower() for x in fixable_cols]
    for col in t.columns:
        if not any(x in col.lower() for x in fixable_cols_lower):
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
    for col in t.columns:
        if col in branding_labels:
            t[col] = t[col].str.replace(r"\[.*\]", "", regex=True).str.strip()

    return t


def get_dates(tokens: list[str]) -> pd.Series:
    """Extract the first, middle and last date from a list of date tokens."""

    last_day: pd.Timestamp | None = None
    day: int | None = None
    month: str | None = None
    year: int | None = None
    remember = tokens.copy()
    while tokens:
        token = tokens.pop()

        if re.match(r"[0-9]{4}", token):
            year = int(token)
        elif re.match(r"[A-Za-z]+", token):
            month = token
        elif re.match(r"[0-9]{1,2}", token):
            day = int(token)
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
        month = "Jan"  # randomly default to January, why not?
        print(f"--> assuming month is {month}")

    if year is None:
        print(f"WARNING: missing year in these tokens? {remember}")
        year = 2000  # randomly default to 200 - will be obvious if wrong

    # sadly we have cases of this ...
    if day is None:
        day = 1
    if last_day is None:
        last_day = pd.Timestamp(f"{year} {month[:3]} {day}")

    # get the middle date
    first_day = pd.Timestamp(f"{year} {month[:3]} {day}")
    year = first_day.year
    if first_day > last_day:
        print(
            "CHECK there may be a problem with these dates in get_dates(): "
            f"{first_day} {last_day} with these tokens {remember}"
        )
        first_day = pd.Timestamp(f"{int(year)-1} {month[:3]} {day}")
        print(f"--> assuming first day is {first_day}")
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


# --- Validation and corrective manipulations ---


def row_sum_check(
    table: pd.DataFrame,
    pattern_list: list[str],
    target: int | float = 100.0,
    tolerance: int | float = 2.01,  # Focus on most egregious issues
) -> pd.DataFrame | None:
    """Check that the columns that regex match with a pattern in the
    pattern list all add across to the target plus/minus the specified
    tolerance. Returns None or a DataFrame with the rows of concern."""

    problems = {}
    for pattern in pattern_list:
        c_pattern = re.compile(pattern)
        cols = [c for c in table.columns if re.match(c_pattern, c)]
        row_sum = table[cols].sum(axis=1)
        problematic = (row_sum - target).abs() > tolerance
        if problematic.any():
            problems[pattern] = row_sum[table.loc[problematic].index]

    if problems:
        r_table = table.copy()  # preserve original data
        for pattern, rows in problems.items():
            r_table[pattern] = np.nan
            r_table.loc[rows.index, pattern] = rows
        return r_table[r_table[problems.keys()].any(axis=1)]
    return None


def distribute_undecideds(
    table: pd.DataFrame,
    col_pattern_list: Sequence[str],
    undec_col: str = UNDECIDED_COLUMN,
    target: int | float = 100,  # per cent
) -> pd.DataFrame:
    """Distribute the undecided vote column among those columns that
    match the col_pattern, IFF that undecideds get the total for the
    columns closer to the target than the sum without the undecideds.
    Return an updated table."""

    table = table.copy()  # preserve the original data
    undecideds: pd.Series = table.loc[table[undec_col].notna(), undec_col]
    for pattern in col_pattern_list:
        # raw are the column calculations EXCLUDING the undecideds
        c_pattern = re.compile(pattern)
        raw_cols = [
            c for c in table.columns if re.match(c_pattern, c) and c != undec_col
        ]
        raw_sum = table.loc[undecideds.index, raw_cols].sum(axis=1)
        raw_closeness_to_target = (raw_sum - target).abs()

        # cooked are the column calculations INCLUDING the undecideds
        cooked_cols = raw_cols + [undec_col]
        _cooked_sum = table.loc[undecideds.index, cooked_cols].sum(axis=1)
        cooked_closeness_to_target = (raw_sum - target).abs()

        # we only redistribute the undecideds for the rows where
        # the cooked calculations are closer to the target than raw
        should_redistribute = (
            cooked_closeness_to_target < raw_closeness_to_target
        ).index
        amount_to_share = table.loc[should_redistribute, undec_col]
        shares = table.loc[should_redistribute, raw_cols].div(
            raw_sum[should_redistribute], axis=0
        )
        allocation = shares.mul(amount_to_share, axis=0)
        table.loc[should_redistribute, raw_cols] += allocation
        print(
            f"For {pattern} distributed undecideds over "
            f"{len(should_redistribute)/len(table) * 100:.2f}% of rows."
        )

    return table


def normalise(
    data: dict[str, pd.DataFrame],
    checkables: dict[str, list[str]],
    force_total_to: float = 100.0,  # selected columns normally row-sum to 100%
    tolerance: float = 0.01,  # report most, but don't report anything too minor
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """For the key, list of regex-patterns pairs in the checkables
    dict: normalise all the columns in the DataFrame data[key],
    that match the each consecutive pattern in check_list.
    Because this is an aggressive treatment, you should keep
    verbose reporting and check through the output."""

    fixed = {}
    for name, check_list in checkables.items():
        df = data[name].copy()
        if verbose:
            print(f"Checking normalisation for table: {name}")

        for check in check_list:
            columns = [x for x in df.columns if re.match(check, x)]
            if verbose:
                print(f"In {name} checking row-additions for {columns}")
            row_sum = df[columns].astype(float).sum(axis=1, skipna=True)

            # we will ignore rows that are all NAN
            na_ignore = (df[columns].isna().astype(int).sum(axis=1)) == len(columns)
            # we will ignore rows in ATTitudINAL data that are not complete.abs
            complete = (df[columns].notna().astype(int).sum(axis=1) == len(columns)) | (
                name != ATTITUDINAL
            )

            problematic = (
                ~na_ignore
                & complete
                & (
                    (row_sum < (force_total_to - tolerance))
                    | (row_sum > (force_total_to + tolerance))
                )
            )
            count = problematic.astype(int).sum()
            print(
                f"{count / len(row_sum) * 100:.2f}% of rows need normalisation - for {name}, {check}."
            )
            if verbose and count:
                tmp_column_name = f"Normalisation totals {check}"
                df[tmp_column_name] = row_sum
                display(df.loc[problematic])
                df = df.drop(tmp_column_name, axis=1)
            df[columns] = df[columns].div(row_sum, axis=0) * force_total_to

        fixed[name] = df
    return fixed


def methodology(
    data: dict[str, pd.DataFrame],
    effective_date: str | pd.Period,
    change_from: str,
    change_to: str,
) -> dict[str, pd.DataFrame]:
    """Mark methodological changes in the data by changing
    the firm/brand of the pollster in the data."""

    data = data.copy()  # preserve the original
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
    capture_date: str | None = None,  # format YYYYMMDD
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
