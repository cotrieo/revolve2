"""Genotype class."""

from __future__ import annotations

import config
import numpy as np
from base import Base
from revolve2.core.database import HasId
from revolve2.core.optimization.ea import Parameters as GenericParameters


class Genotype(Base, HasId, GenericParameters):
    """SQLAlchemy model for a genotype that is a list of parameters."""

    __tablename__ = "genotype"

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Create a random genotype.

        :param rng: Random number generator.
        :returns: The created genotype.
        """
        return Genotype(tuple(p for p in rng.random(size=config.NUM_PARAMETERS)))