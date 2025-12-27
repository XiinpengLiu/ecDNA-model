"""
ecDNA Copy-Number Kinetics Model - Cell State
==============================================
Defines the Cell class representing single-cell state z = (E, M, K, A, Y).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import config as cfg


@dataclass(eq=False)
class Cell:
    """
    Single-cell state z = (E, M, K, A, Y).
    
    Attributes:
        e: Environment/stage index
        c: Cell-cycle phase
        s: Senescence status
        x: Expression program
        k: ecDNA copy-number vector (shape: J,)
        a: Age (time since last division)
        y: Continuous phenotype (shape: P,)
        cell_id: Unique identifier
        parent_id: Parent cell ID (None for initial cells)
        last_update_time: Time of last state update (for lazy flow)
    """
    e: int
    c: int
    s: int
    x: int
    k: np.ndarray
    a: float
    y: np.ndarray
    cell_id: int = 0
    parent_id: int = None
    last_update_time: float = 0.0
    
    def __eq__(self, other):
        """Compare cells by cell_id only."""
        if not isinstance(other, Cell):
            return False
        return self.cell_id == other.cell_id
    
    def __hash__(self):
        return hash(self.cell_id)
    
    @property
    def m(self) -> int:
        """Flat discrete state index M."""
        return cfg.tuple_to_m(self.c, self.s, self.x)
    
    @property
    def type_index(self) -> Tuple[int, int, int, int, Tuple]:
        """Full type index i = (e, m, k)."""
        return (self.e, self.c, self.s, self.x, tuple(self.k))
    
    def copy(self) -> 'Cell':
        """Create a deep copy of this cell."""
        return Cell(
            e=self.e, c=self.c, s=self.s, x=self.x,
            k=self.k.copy(), a=self.a, y=self.y.copy(),
            cell_id=self.cell_id, parent_id=self.parent_id,
            last_update_time=self.last_update_time
        )
    
    def total_ecdna(self) -> int:
        """Total ecDNA copy number across all species."""
        return int(np.sum(self.k))


class CellPopulation:
    """
    Population of cells with event tracking.
    """
    def __init__(self, rng: np.random.Generator = None):
        self.cells = []  # list of Cell
        self.next_id = 0
        self.rng = rng if rng is not None else np.random.default_rng(cfg.RANDOM_SEED)
        
        # Event log
        self.events = []  # (time, event_type, cell_id, details)
        
    def add_cell(self, cell: Cell) -> Cell:
        """Add a cell to population, assigning ID."""
        cell.cell_id = self.next_id
        self.next_id += 1
        self.cells.append(cell)
        return cell
    
    def remove_cell(self, cell: Cell):
        """Remove a cell from population."""
        self.cells.remove(cell)
    
    def size(self) -> int:
        """Current population size."""
        return len(self.cells)
    
    def initialize(self, n: int = cfg.N_INIT):
        """Initialize population with n cells."""
        for _ in range(n):
            e, c, s, x, k, a, y = cfg.sample_initial_state(self.rng)
            cell = Cell(e=e, c=c, s=s, x=x, k=k, a=a, y=y)
            self.add_cell(cell)
    
    def log_event(self, time: float, event_type: str, cell_id: int, details: dict = None):
        """Record an event."""
        self.events.append((time, event_type, cell_id, details or {}))
    
    def get_summary(self) -> dict:
        """Get population summary statistics."""
        if not self.cells:
            return {"n": 0}
        
        k_total = [c.total_ecdna() for c in self.cells]
        ages = [c.a for c in self.cells]
        
        # State distributions
        cycle_dist = np.zeros(cfg.N_CYCLE)
        sen_dist = np.zeros(cfg.N_SEN)
        expr_dist = np.zeros(cfg.N_EXPR)
        
        for c in self.cells:
            cycle_dist[c.c] += 1
            sen_dist[c.s] += 1
            expr_dist[c.x] += 1
        
        return {
            "n": len(self.cells),
            "ecdna_mean": np.mean(k_total),
            "ecdna_std": np.std(k_total),
            "ecdna_max": np.max(k_total),
            "age_mean": np.mean(ages),
            "cycle_dist": cycle_dist / len(self.cells),
            "sen_dist": sen_dist / len(self.cells),
            "expr_dist": expr_dist / len(self.cells),
        }
