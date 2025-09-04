"""Definition of the :class:`~onlinerake.targets.Targets` dataclass.

The :class:`~onlinerake.targets.Targets` class captures the target
population margins for each demographic characteristic under study. It
provides a simple, typed container for passing these values into the
online raking algorithms. Each attribute represents the proportion of
the population belonging to the ``1`` category of a binary indicator.

Attributes
==========

``age``
    Proportion of the population that is in the ``1`` category for age
    (e.g., ``1`` might correspond to “old” and ``0`` to “young”).

``gender``
    Proportion of the population with ``gender=1`` (e.g., female).

``education``
    Proportion of the population with ``education=1`` (e.g., high school
    or more).

``region``
    Proportion of the population with ``region=1`` (e.g., rural).  If
    additional binary categories are needed, extend this class or
    replace it with a more general mapping.

These fields default to commonly used values (0.5 for binary splits and
typical values for education/region) but should be explicitly set to
reflect the true target margins for your application.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Targets:
    """Target population proportions for binary demographics.

    Each attribute represents the desired proportion of cases with
    indicator value ``1``. If your survey uses different definitions or
    more categories per characteristic, either extend this class with
    additional fields or refactor your raking logic accordingly.
    """

    age: float = 0.5
    gender: float = 0.5
    education: float = 0.4
    region: float = 0.3

    def as_dict(self) -> dict:
        """Return the targets as a plain dictionary.

        Useful for iterating over targets programmatically.
        """
        return {
            "age": self.age,
            "gender": self.gender,
            "education": self.education,
            "region": self.region,
        }
