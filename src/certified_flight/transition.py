"""Geometry transition: every pair is reachable and morphs over a fixed 8 s linear blend."""
from dataclasses import dataclass


@dataclass
class Transition:
    source: int
    target: int
    started: float = 0.0
    duration: float = 8.0

    def fraction(self, t):
        return 1.0 if self.source == self.target else min(1.0, max(0.0, (t-self.started)/self.duration))

    def active(self, t):
        return self.source != self.target and t < self.started+self.duration-1e-10

    def settle(self, t):
        if not self.active(t):
            self.source = self.target

    def command(self, target, t):
        # Complete each pair before accepting another. Replanning every 3 s does
        # not restart the clock. Completion occurs at 8 s, not at the next decision.
        self.settle(t)
        if self.active(t):
            return False
        if int(target) != self.target:
            self.source, self.target, self.started = self.target, int(target), float(t)
            return True
        return False
