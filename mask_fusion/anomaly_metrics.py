"""Binary defect judgments from explicit yes/no answers, without probabilities.

The question must ask whether a defect is present: yes means gt=1 (abnormal),
no means gt=0 (normal). These metrics assess the judgment, not the subsequent
description. They are not segmentation metrics and cannot produce AUROC.
"""

from collections import defaultdict
from collections.abc import Mapping
import re


_PUNCTUATION = ".,;:!?。，；：！？"
_ANSWER = re.compile(
    r"^(?:(?P<marker>\*\*|__|\*|_|`)(?P<decorated>yes|no)"
    r"(?P<inner_punctuation>[.,;:!?。，；：！？]?)(?P=marker)|(?P<plain>yes|no))"
    r"(?P<tail>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def validate_anomaly_target(record):
    """Return explicit integer gt=0/1; never infer it from text or paths.

    JSON booleans, floats, numeric strings, missing gt, and null are rejected.
    """
    if not isinstance(record, Mapping):
        raise ValueError("Anomaly evaluation requires a record with integer gt=0 or gt=1")
    target = record.get("gt")
    if type(target) is not int or target not in (0, 1):
        raise ValueError("Anomaly evaluation requires explicit integer gt=0 (normal) or gt=1 (abnormal)")
    return target


def extract_anomaly_prediction(text):
    """Parse an opening standalone yes/no, optionally followed by a description.

    Accept end of text, punctuation, or a newline after the answer. Paired bold,
    italic, and inline-code markers are allowed around that answer (and an
    optional punctuation mark). Plain whitespace followed by prose is rejected:
    ``No doubt ...`` and ``No defect ...`` are not standalone answers. The
    parser does not search later sentences or assign semantics to A/B choices.
    """
    if not isinstance(text, str):
        return None
    match = _ANSWER.fullmatch(text.strip())
    if match is None:
        return None
    answer = (match.group("decorated") or match.group("plain")).lower()
    tail = match.group("tail")
    # "Yes, and no" / "No, or yes" are explicitly ambiguous judgments.
    alternative = re.match(r"^[\s,;:]*(?:and|or)\s+(yes|no)\b", tail, re.IGNORECASE)
    if alternative and alternative.group(1).lower() != answer:
        return None
    remainder = tail.lstrip(" \t")
    if (remainder and not match.group("inner_punctuation")
            and remainder[0] not in _PUNCTUATION + "\r\n"):
        return None
    return 1 if answer == "yes" else 0


def _binary_counts(rows):
    counts = {name: 0 for name in ("tp", "tn", "fp", "fn", "unparsed_positive", "unparsed_negative")}
    for target, prediction in rows:
        if prediction is None:
            name = "unparsed_positive" if target == 1 else "unparsed_negative"
        elif target == 1:
            name = "tp" if prediction == 1 else "fn"
        else:
            name = "tn" if prediction == 0 else "fp"
        counts[name] += 1
    positive = counts["tp"] + counts["fn"] + counts["unparsed_positive"]
    negative = counts["tn"] + counts["fp"] + counts["unparsed_negative"]
    count = positive + negative
    positive_recall = counts["tp"] / positive if positive else None
    negative_recall = counts["tn"] / negative if negative else None
    return {
        "count": count,
        "accuracy": (counts["tp"] + counts["tn"]) / count if count else None,
        **counts,
        "unparseable_anomaly": counts["unparsed_positive"] + counts["unparsed_negative"],
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "balanced_accuracy": ((positive_recall + negative_recall) / 2
                              if positive_recall is not None and negative_recall is not None else None),
    }


def summarize_anomaly_predictions(predictions):
    """Summarize QA-level binary judgments using gt and anomaly_prediction.

    Input rows have gt, anomaly_prediction (0/1/None), and optional category.
    A supplied ``correct`` field is ignored: correctness is recomputed. Missing
    or null gt is unscored; any other invalid gt raises. The inference CLI
    separately requires a valid gt for every row before generation.

    Accuracy includes every scored row; an unparsed prediction is incorrect.
    Positive/negative recall also include unparsed rows of that true class in
    their denominators. Confusion counts describe only parsed rows, so each
    class denominator adds its separate unparsed count. Balanced accuracy is
    defined only when both true classes occur. Category macro accuracy averages
    equally over categories containing scored rows, without image deduplication.
    """
    rows = list(predictions)
    scored = []
    buckets = defaultdict(list)
    for record in rows:
        if record.get("gt") is None:
            continue
        target = validate_anomaly_target(record)
        prediction = record.get("anomaly_prediction")
        if prediction is not None and (type(prediction) is not int or prediction not in (0, 1)):
            raise ValueError("anomaly_prediction must be integer 0/1 or None for an unparsed answer")
        category = record.get("category") or "__unspecified__"
        if not isinstance(category, str):
            raise ValueError("category must be a string when supplied")
        scored.append((target, prediction))
        buckets[category].append((target, prediction))
    totals = _binary_counts(scored)
    categories = {name: _binary_counts(values) for name, values in sorted(buckets.items())}
    count = totals.pop("count")
    return {
        "qa_records": len(rows), "scored_anomaly": count, **totals,
        "macro_category_accuracy": (sum(values["accuracy"] for values in categories.values()) / len(categories)
                                    if categories else None),
        "by_category": categories,
    }
