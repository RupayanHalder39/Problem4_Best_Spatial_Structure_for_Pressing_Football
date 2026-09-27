"""One explicit reference-pitch, period-aware direction configuration.

The default end calibration belongs to this 120s clip, not every football match.
No stadium-true dimensions or official body-point geometry are implied.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class PitchConfig:
    length_cm: float = 12000.0
    width_cm: float = 7000.0
    # (first frame of period, Team 0 own-goal is at far x end)
    periods: tuple = ((0, True),)
    scale_source: str = 'idealized reference pitch; stadium dimensions unknown'

    def __post_init__(self):
        if self.length_cm <= 0 or self.width_cm <= 0 or not self.periods or self.periods[0][0] != 0:
            raise ValueError('Positive dimensions and a period beginning at frame 0 required')
        if any(a[0] >= b[0] for a,b in zip(self.periods,self.periods[1:])):
            raise ValueError('Period boundaries must increase')

    def period(self, frame=0):
        if frame < 0: raise ValueError('Negative frame')
        return max(i for i, (start, _) in enumerate(self.periods) if start <= frame)

    def own_goal(self, team, frame=0):
        if team not in (0,1): raise ValueError('Unknown team')
        far = self.periods[self.period(frame)][1]
        return self.length_cm if (far if team == 0 else not far) else 0.0

    def attacking_sign(self, team, frame=0):
        return 1.0 if self.own_goal(team,frame) == 0 else -1.0

    def defending_sign(self, team, frame=0):
        """Motion toward own goal; a defensive step-up uses attacking_sign."""
        return -self.attacking_sign(team,frame)

    def depth(self, team, x, frame=0):
        return (x-self.own_goal(team,frame))*self.attacking_sign(team,frame)

    def x_from_depth(self, team, depth, frame=0):
        return self.own_goal(team,frame)+depth*self.attacking_sign(team,frame)

    def progress(self, team, start_x, end_x, frame=0):
        return (end_x-start_x)*self.attacking_sign(team,frame)

    def inside(self,x,y):
        return x is not None and y is not None and 0 <= x <= self.length_cm and 0 <= y <= self.width_cm

DEFAULT_PITCH = PitchConfig()
