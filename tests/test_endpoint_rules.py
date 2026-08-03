"""Endpoint classification rules, on real registry phrasings.

The EVENT_COMPOSITE clause `incidence of ... events` used to fire before SAFETY
was ever reached, so the most common safety phrasing in the registry landed in
the composite bucket. These cases are lifted verbatim from trials in the corpus,
with the NCT id kept so the label is checkable against the source.
"""
import pytest

from backend.preprocessing.endpoints import (ARCHETYPES, archetype_flags,
                                             classify_measure, classify_primary)

# Real primary outcome text -> the archetype it must NOT be, and must be.
SAFETY_PHRASINGS = [
    ("NCT06356389", "Incidence of treatment-emergent adverse events (TEAEs)"),
    ("NCT03408132", "Incidence of treatment-emergent adverse events"),
    ("NCT05278663",
     "Incidence of Treatment-emergent Adverse Events (TEAEs) and Serious Adverse Events (SAEs)"),
    ("NCT03051256",
     "Incidence of Treatment Emergent Adverse Events (AEs) [Safety and Tolerability]"),
    ("NCT05417126", "Type, severity, and incidence of ocular and systemic adverse events (AEs)"),
    ("NCT04666012", "Incidence of solicited adverse events(AEs)"),
    ("NCT04052737", "Incidence of PMZ-1620 related adverse events"),
]

# Genuine composites, which must survive the fix.
COMPOSITE_PHRASINGS = [
    "Major adverse cardiac events (MACE)",
    "Composite endpoint of death, myocardial infarction or stroke",
    "Time to first occurrence of the composite outcome",
    "Hospitalisation for heart failure",
    "Incidence of cardiovascular events",
    "Incidence of thrombotic episodes",
]


@pytest.mark.parametrize("nct,text", SAFETY_PHRASINGS)
def test_adverse_event_incidence_is_safety_not_composite(nct, text):
    assert classify_measure(text) == "SAFETY", (
        f"{nct}: {text!r} classified as {classify_measure(text)}")


@pytest.mark.parametrize("text", COMPOSITE_PHRASINGS)
def test_genuine_composites_still_classify_as_composite(text):
    assert classify_measure(text) == "EVENT_COMPOSITE"


def test_flags_are_multi_label_across_primary_outcomes():
    """A trial carrying both an AE and a PK endpoint must light both flags."""
    measures = ("Incidence of Treatment Emergent Adverse Events|"
                "Maximum observed plasma concentration (Cmax)")
    flags = archetype_flags(measures)
    assert flags["endpoint_has_SAFETY"] == 1
    assert flags["endpoint_has_PK_PD"] == 1
    assert flags["endpoint_has_EVENT_COMPOSITE"] == 0


def test_classify_primary_falls_through_abstentions():
    measures = "Something entirely unparseable|Overall survival"
    assert classify_primary(measures) == "SURVIVAL"


def test_every_rule_label_is_in_the_closed_vocabulary():
    for text in [t for _, t in SAFETY_PHRASINGS] + COMPOSITE_PHRASINGS:
        assert classify_measure(text) in ARCHETYPES
