"""How truth object names ("Apple_1", "DiningTable|1|1", "Mug") are compared with map labels
("apple", "dining table", "cup"): normalise both, map synonyms to one canonical word, then compare.
The synonym table is data in the layout (``labels``)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SYNONYMS: list[list[str]] = [
    ["cup", "mug"],
    ["sofa", "couch"],
    ["television", "tv", "tvmonitor"],
    ["refrigerator", "fridge"],
    ["garbagecan", "trashcan", "bin", "wastebin", "dustbin"],
    ["cellphone", "phone", "mobilephone", "smartphone"],
    ["laptop", "notebook"],
    ["pottedplant", "houseplant", "plant"],
    ["remotecontrol", "remote"],
    ["sink", "sinkbasin"],
    ["bathtub", "tub", "bathtubbasin"],
    ["bed", "mattress"],
]

STEPS = [
    "keep the part before the first '|' (THOR ids: 'Mug|1|2' -> 'Mug')",
    "split camelCase into words, lower-case",
    "strip an instance suffix: trailing digits with an optional separator ('apple_1' -> 'apple')",
    "keep letters only ('dining table' -> 'diningtable')",
    "singularise: -ies -> -y, -ches/-shes/-xes/-sses/-zes/-oes -> drop 'es', else a final 's' "
    "(not 'ss') on words longer than 3",
    "map synonyms to the first word of their group",
    "agree when equal, or (suffix_match) when one ends with the other, the shorter having at "
    "least 3 letters ('coffeetable' ~ 'table')",
]


def singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses", "zes", "oes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def normalise(name: str) -> str:
    s = name.split("|")[0]
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).lower()
    s = re.sub(r"[\s_\-.]*\d+$", "", s)
    return singular(re.sub(r"[^a-z]", "", s))


@dataclass
class LabelRule:
    synonyms: list[list[str]] = field(default_factory=lambda: [g[:] for g in DEFAULT_SYNONYMS])
    suffix_match: bool = True

    def __post_init__(self) -> None:
        self._canon = {normalise(w): normalise(group[0]) for group in self.synonyms
                       for w in group}

    @classmethod
    def from_layout(cls, cfg: dict[str, Any]) -> LabelRule:
        return cls([list(g) for g in cfg["synonyms"]], bool(cfg["suffix_match"]))

    def canon(self, name: str) -> str:
        n = normalise(name)
        return self._canon.get(n, n)

    def same(self, a: str, b: str) -> bool:
        x, y = self.canon(a), self.canon(b)
        if not x or not y:
            return False
        if x == y:
            return True
        short, long_ = sorted((x, y), key=len)
        return self.suffix_match and len(short) >= 3 and long_.endswith(short)

    def describe(self) -> dict[str, Any]:
        return {"steps": STEPS, "synonyms": self.synonyms, "suffix_match": self.suffix_match,
                "examples": {n: self.canon(n) for n in
                             ("Apple_1", "Mug", "DiningTable|1|1", "cups", "couch")}}
