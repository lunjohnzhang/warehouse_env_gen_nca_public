"""pyribs-compliant emitters."""
import gin
import ribs

from env_search.emitters.map_elites_baseline_emitter import MapElitesBaselineWarehouseEmitter, MapElitesBaselineMazeEmitter, MapElitesBaselineManufactureEmitter
from env_search.emitters.random_emitter import RandomEmitter

__all__ = [
    "GaussianEmitter",
    "EvolutionStrategyEmitter",
    "MapElitesBaselineWarehouseEmitter",
    "MapElitesBaselineMazeEmitter",
    "MapElitesBaselineManufactureEmitter",
    "RandomEmitter",
]


@gin.configurable
class GaussianEmitter(ribs.emitters.GaussianEmitter):
    """gin-configurable version of pyribs GaussianEmitter."""

    def ask(self):
        # Return addition None to cope with parent sol API of
        # MapElitesBaselineWarehouseEmitter
        return super().ask(), None

@gin.configurable
class IsoLineEmitter(ribs.emitters.IsoLineEmitter):
    """gin-configurable version of pyribs IsoLineEmitter."""

    def ask(self):
        # Return addition None to cope with parent sol API of
        # MapElitesBaselineWarehouseEmitter
        return super().ask(), None


@gin.configurable
class EvolutionStrategyEmitter(ribs.emitters.EvolutionStrategyEmitter):
    """gin-configurable version of pyribs EvolutionStrategyEmitter."""

    def ask(self):
        # Return addition None to cope with parent sol API of
        # MapElitesBaselineWarehouseEmitter
        return super().ask(), None