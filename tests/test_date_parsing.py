"""Mixed-precision registry dates must all survive parsing.

CT.gov publishes "2015-10" for older records and "2022-10-21" for newer ones.
`pd.to_datetime` without an explicit format infers one from the first non-null
value and coerces the rest to NaT, and `parse_dates` drops what it nulls — which
silently deleted 44.9% of P1, 51.7% of P2 and 46.6% of P3, the half chosen by
whichever precision the API happened to return first.

These tests fail if that behaviour ever comes back, in BOTH orderings, because
the ordering is what decided which half was lost.
"""
import pandas as pd
import pytest

from backend.preprocessing.cleaner import parse_dates

SHORT_FIRST = ["2015-10", "2022-10-21", "2004-08", "2019-03-15"]
LONG_FIRST = ["2022-10-21", "2015-10", "2019-03-15", "2004-08"]


def _frame(starts):
    return pd.DataFrame({
        "Start Date": starts,
        "Primary Completion Date": ["2026-01-15"] * len(starts),
    })


@pytest.mark.parametrize("starts,label", [(SHORT_FIRST, "year-month first"),
                                          (LONG_FIRST, "full-date first")])
def test_no_row_is_lost_to_mixed_precision(starts, label):
    out = parse_dates(_frame(starts))
    assert len(out) == len(starts), (
        f"{label}: {len(starts) - len(out)} of {len(starts)} rows dropped")


@pytest.mark.parametrize("starts", [SHORT_FIRST, LONG_FIRST])
def test_parsed_values_are_correct_not_merely_present(starts):
    """Surviving a parse is not the same as parsing correctly."""
    out = parse_dates(_frame(starts))
    got = {d.strftime("%Y-%m-%d") for d in out["Start Date"]}
    assert got == {"2015-10-01", "2022-10-21", "2004-08-01", "2019-03-15"}


def test_ordering_does_not_change_the_result():
    a = parse_dates(_frame(SHORT_FIRST))["Start Date"].sort_values().tolist()
    b = parse_dates(_frame(LONG_FIRST))["Start Date"].sort_values().tolist()
    assert a == b


def test_genuinely_unparseable_dates_are_still_dropped():
    """The fix must not turn the guard off — junk still has to go."""
    out = parse_dates(_frame(["2015-10", "not a date", "", "2022-10-21"]))
    assert len(out) == 2
