"""Routing labels for synthetic 311-style requests."""

from enum import Enum


class Department(str, Enum):
    STREETS = "streets"
    PARKS = "parks"
    UTILITIES = "utilities"
    CODE_ENFORCEMENT = "code_enforcement"
    NOISE = "noise"
    OTHER = "other"


DEPARTMENTS = [d.value for d in Department]
