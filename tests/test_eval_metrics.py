"""M2a: alarm-segment merging and evaluation metrics, checked on small hand-made examples (including no alarms, all alarms, no attack events)."""

import math

import numpy as np
import pytest

from src.eval.metrics import alarms_from_scores, event_metrics, point_metrics
from src.eval.segments import overlaps, to_segments

NAN = float("nan")


# ---------- to_segments ----------

def test_to_segments_merges_only_consecutive_by_default():
    assert to_segments([0, 1, 1, 0, 1, 0, 0, 1]) == [(1, 2), (4, 4), (7, 7)]
    assert to_segments(np.array([True, True, False])) == [(0, 1)]


def test_to_segments_merge_gap():
    b = [0, 1, 1, 0, 1, 0, 0, 1]
    assert to_segments(b, merge_gap_s=1) == [(1, 4), (7, 7)]
    assert to_segments(b, merge_gap_s=2) == [(1, 7)]


def test_to_segments_edge_cases():
    assert to_segments([]) == []
    assert to_segments([0, 0, 0]) == []
    assert to_segments([1, 1, 1]) == [(0, 2)]
    assert to_segments([1]) == [(0, 0)]
    with pytest.raises(ValueError):
        to_segments([1, 0], merge_gap_s=-1)
    with pytest.raises(ValueError):
        to_segments(np.zeros((2, 2)))


def test_overlaps_is_inclusive():
    assert overlaps((1, 3), (3, 5)) and overlaps((3, 5), (1, 3)) and overlaps((2, 2), (0, 9))
    assert not overlaps((1, 3), (4, 5))


# ---------- Point-wise metrics ----------

SCORES = np.array([NAN, 0.1, 0.9, 0.8, 0.2, 0.95])
LABELS = np.array([1, 0, 1, 0, 1, 0])


def test_point_metrics_hand_example():
    m = point_metrics(SCORES, LABELS, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 2, 1, 1)
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.4)
    assert m["auroc"] == pytest.approx(0.5)  # 3 of the 6 positive/negative pairs are ordered correctly
    # The NaN second (label 1) is excluded from every metric and only counted
    assert (m["n_points"], m["n_unscored"], m["n_positive"], m["n_positive_unscored"]) == (5, 1, 2, 1)


def test_point_metrics_no_point_adjust():
    """A 5-second event with only 1 alarm second: recall is 1/5, not the 1 that point-adjust would give."""
    labels = np.array([0, 1, 1, 1, 1, 1, 0])
    scores = np.array([0, 0, 0, 1.0, 0, 0, 0])
    assert point_metrics(scores, labels, 0.5)["recall"] == pytest.approx(0.2)


def test_point_metrics_threshold_is_strict():
    assert point_metrics(np.array([0.5, 0.6]), np.array([1, 1]), 0.5)["tp"] == 1


def test_point_metrics_no_alarms():
    m = point_metrics(SCORES, LABELS, 10.0)
    assert math.isnan(m["precision"]) and m["recall"] == 0 and m["f1"] == 0
    assert m["auroc"] == pytest.approx(0.5)  # AUROC does not depend on the threshold


def test_point_metrics_all_alarms():
    m = point_metrics(SCORES, LABELS, -1.0)
    assert m["precision"] == pytest.approx(2 / 5) and m["recall"] == 1 and m["f1"] == pytest.approx(4 / 7)


def test_point_metrics_no_positives():
    m = point_metrics(SCORES, np.zeros(6, dtype=int), 0.5)
    assert m["precision"] == 0 and m["fp"] == 3
    assert math.isnan(m["recall"]) and math.isnan(m["f1"]) and math.isnan(m["auroc"])


def test_point_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        point_metrics(np.zeros(3), np.zeros(4), 0.5)


def test_unscored_seconds_never_alarm():
    assert alarms_from_scores(np.array([NAN, 2.0, 0.0]), 1.0).tolist() == [False, True, False]


# ---------- Event-level metrics ----------

def make(n, alarm_idx, event_ranges, nan_head=0):
    scores, labels = np.zeros(n), np.zeros(n, dtype=int)
    scores[list(alarm_idx)] = 1.0
    scores[:nan_head] = NAN
    for a, b in event_ranges:
        labels[a:b + 1] = 1
    return scores, labels


def test_event_metrics_hand_example():
    # Events (5,9), (14,16); alarm segments (3,3) false alarm, (7,8) hits the first event, (18,19) false alarm; first 2 seconds unscored
    pair = make(20, [3, 7, 8, 18, 19], [(5, 9), (14, 16)], nan_head=2)
    summary, events, alarms = event_metrics([pair], 0.5)
    assert (summary["n_events"], summary["n_detected"], summary["n_missed"]) == (2, 1, 1)
    assert summary["event_recall"] == 0.5
    assert summary["delay_median_s"] == 2 and summary["delay_mean_s"] == 2  # Only detected events are counted
    assert (summary["n_alarm_segments"], summary["n_false_alarm_segments"]) == (3, 2)
    # Normal duration: 12 seconds with label 0, 2 of them unscored -> 10 seconds
    assert summary["normal_hours"] == pytest.approx(10 / 3600)
    assert summary["false_alarms_per_hour"] == pytest.approx(720)
    assert [(e["start"], e["end"], e["detected"], e["delay_s"]) for e in events] == [(5, 9, True, 2), (14, 16, False, None)]
    assert [(a["start"], a["end"], a["overlaps_event"]) for a in alarms] == [(3, 3, False), (7, 8, True), (18, 19, False)]


def test_event_delay_zero_when_alarm_started_before_event():
    summary, events, alarms = event_metrics([make(12, [4, 5, 6], [(5, 9)])], 0.5)
    assert events[0]["delay_s"] == 0 and summary["n_false_alarm_segments"] == 0 and alarms[0]["overlaps_event"]


def test_event_metrics_no_alarms():
    summary, events, alarms = event_metrics([make(10, [], [(2, 4)])], 0.5)
    assert summary["event_recall"] == 0 and summary["n_missed"] == 1 and alarms == []
    assert math.isnan(summary["delay_median_s"]) and math.isnan(summary["delay_mean_s"])
    assert summary["false_alarms_per_hour"] == 0


def test_event_metrics_all_alarms():
    summary, _, alarms = event_metrics([make(10, range(10), [(2, 4), (7, 8)])], 0.5)
    assert summary["event_recall"] == 1 and summary["delay_median_s"] == 0
    assert len(alarms) == 1 and summary["n_false_alarm_segments"] == 0  # A single alarm segment overlapping an event is not a false alarm


def test_event_metrics_no_events():
    summary, events, _ = event_metrics([make(3600, [10, 11, 50], [])], 0.5)
    assert events == [] and summary["n_events"] == 0 and math.isnan(summary["event_recall"])
    assert summary["n_false_alarm_segments"] == 2 and summary["false_alarms_per_hour"] == pytest.approx(2)


def test_events_and_alarms_do_not_cross_files():
    a = make(5, [4], [(3, 4)])
    b = make(5, [0], [(0, 1)])
    summary, events, alarms = event_metrics([a, b], 0.5)
    assert summary["n_events"] == 2 and summary["n_alarm_segments"] == 2
    assert [e["file_index"] for e in events] == [0, 1] and [e["delay_s"] for e in events] == [1, 0]


def test_merge_gap_changes_false_alarm_count_only_through_shared_function():
    pair = make(3600, [10, 12, 14], [])
    assert event_metrics([pair], 0.5, merge_gap_s=0)[0]["n_false_alarm_segments"] == 3
    assert event_metrics([pair], 0.5, merge_gap_s=1)[0]["n_false_alarm_segments"] == 1


def test_event_fully_unscored_counts_as_missed():
    summary, events, _ = event_metrics([make(10, [0, 1, 2], [(0, 2)], nan_head=3)], 0.5)
    assert summary["n_missed"] == 1 and events[0]["n_unscored_s"] == 3


# ---------- Agreed metric conventions (CLAUDE.md main-line decision 6b) ----------

def test_convention_no_alarms_with_attacks_counts_as_zero():
    """No alarms but attacks within the evaluation range: precision is NaN, recall and F1 are 0 (all attacks missed), not NaN."""
    m = point_metrics(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 1, 0]), 5.0)
    assert math.isnan(m["precision"]) and m["recall"] == 0.0 and m["f1"] == 0.0
    summary, _, _ = event_metrics([(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 1, 0]))], 5.0)
    assert summary["event_recall"] == 0.0 and summary["n_missed"] == 1


def test_convention_auroc_depends_only_on_scores():
    scores, labels = np.array([0.1, 0.9, 0.4, 0.7, NAN]), np.array([0, 1, 0, 1, 1])
    values = [point_metrics(scores, labels, t)["auroc"] for t in (-1.0, 0.5, 5.0)]  # All alarms, some alarms, no alarms
    assert values == [1.0, 1.0, 1.0]
    # NaN only when all labels within the evaluation range (scored seconds) are of one class
    assert math.isnan(point_metrics(scores, np.array([0, 0, 0, 0, 1]), 0.5)["auroc"])
    assert math.isnan(point_metrics(scores, np.ones(5, dtype=int), 0.5)["auroc"])
