"""Synthetic citizen requests and preference pairs for PoC experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from .labels import DEPARTMENTS, Department


@dataclass(frozen=True)
class LabeledRequest:
    text: str
    label: str


@dataclass(frozen=True)
class PreferencePair:
    text: str
    chosen: str
    rejected: str


_TEMPLATES: dict[str, list[str]] = {
    Department.STREETS.value: [
        "Pothole on {st} near {cross}, dangerous for bikes.",
        "Street light out on {st}, block is dark at night.",
        "Sidewalk lifted by tree roots on {st}, tripping hazard.",
        "Storm drain clogged on {st}, flooding after rain.",
        "Missing manhole cover on {st}, needs urgent cover replacement.",
    ],
    Department.PARKS.value: [
        "Broken swing at {park}, safety concern for children.",
        "Trash overflow at {park} picnic area on weekend.",
        "Dog off leash at {park} near playground despite rules.",
        "Irrigation leak at {park}, muddy patch growing.",
        "Bench damaged at {park}, splintered wood reported.",
    ],
    Department.UTILITIES.value: [
        "Water pressure low on {st} entire morning.",
        "Sewer odor from grate on {st} after heavy rain.",
        "Hydrant leaking at corner of {st} and {cross}.",
        "Brown water from tap on {st}, started yesterday.",
        "Power flicker on {st}, brief outages twice today.",
    ],
    Department.CODE_ENFORCEMENT.value: [
        "Unpermitted shed visible from alley behind {st}.",
        "Construction debris piled on {st} sidewalk for three days.",
        "Fence over height limit on {st} property line dispute.",
        "Rental unit window boarded against code on {st}.",
        "Business sign larger than allowed on {st} storefront.",
    ],
    Department.NOISE.value: [
        "Loud generator running past midnight on {st}.",
        "Band rehearsal audible from {st} garage nightly.",
        "Delivery trucks idling loudly on {st} early mornings.",
        "Car alarm repeatedly on {st}, hours at a time.",
        "Construction before permitted hours on {st}.",
    ],
    Department.OTHER.value: [
        "Question about recycling schedule for {st} area.",
        "Request copy of permit filed for {st} address.",
        "Compliment for helpful staff who answered zoning question.",
        "General inquiry about city council meeting agenda.",
        "Need help navigating online portal for bulk pickup.",
    ],
}


_STREETS = ("Oak Ave", "Maple Rd", "Cedar Ln", "River St", "Hill Blvd")
_CROSS = ("4th", "Main", "Union", "Park", "Lake")
_PARKS = ("Riverside Park", "Hilltop Green", "Central Commons", "Harbor View")


def _fill_template(template: str, rng: random.Random) -> str:
    return template.format(
        st=rng.choice(_STREETS),
        cross=rng.choice(_CROSS),
        park=rng.choice(_PARKS),
    )


def generate_labeled_requests(
    n_per_class: int = 120,
    seed: int = 42,
) -> list[LabeledRequest]:
    rng = random.Random(seed)
    rows: list[LabeledRequest] = []
    for dept in DEPARTMENTS:
        templates = _TEMPLATES[dept]
        for i in range(n_per_class):
            t = templates[i % len(templates)]
            noise = rng.choice(["", " Please route quickly.", " Urgent citizen report.", ""])
            text = _fill_template(t, rng) + noise
            rows.append(LabeledRequest(text=text.strip(), label=dept))
    rng.shuffle(rows)
    return rows


def iter_preference_pairs(
    requests: list[LabeledRequest],
    mistake_rate: float = 0.22,
    seed: int = 7,
) -> Iterator[PreferencePair]:
    """
    Simulate human reviewers correcting a noisy baseline policy.
    chosen = true department; rejected = plausible wrong route.
    """
    rng = random.Random(seed)
    confusion = {
        Department.STREETS.value: (
            Department.UTILITIES.value,
            Department.NOISE.value,
        ),
        Department.PARKS.value: (Department.NOISE.value, Department.OTHER.value),
        Department.UTILITIES.value: (
            Department.STREETS.value,
            Department.CODE_ENFORCEMENT.value,
        ),
        Department.CODE_ENFORCEMENT.value: (
            Department.UTILITIES.value,
            Department.OTHER.value,
        ),
        Department.NOISE.value: (
            Department.STREETS.value,
            Department.PARKS.value,
        ),
        Department.OTHER.value: (Department.PARKS.value, Department.STREETS.value),
    }
    for lr in requests:
        if rng.random() > mistake_rate:
            continue
        wrong_a, wrong_b = confusion[lr.label]
        rejected = rng.choice((wrong_a, wrong_b))
        yield PreferencePair(text=lr.text, chosen=lr.label, rejected=rejected)
