import sys, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from target_rl import flight as F, guided as G
from target_rl.scenario import Scenario, Target
import mismatch2 as M2
NOM = Scenario('L0', 0, 0., 0., 1., 1.)
F.TargetFlight._deriv = M2._deriv; F.TargetFlight.__init__ = M2.init
CL = [.02, .03, .04, .05, .06, .08, .10]
base = {}
for name in ('M0 none', 'M1 mild', 'M2 moderate', 'M3 strong'):
    M2.MM.update(M2.LEVELS[name]); out = []
    for cl in CL:
        e = F.TargetFlight(NOM, Target('x', 0., 0.), True); e.advance(e.w['T'], G._hold(cl, 0.)); out.append((e.range_km(), e.x[1]))
    base.setdefault('M0', out) if name == 'M0 none' else None
    if name != 'M0 none':
        d = [(o[0]-b[0], o[1]-b[1]) for o, b in zip(out, base['M0'])]
        print('%-12s range change vs no mismatch [km]: %s | speed change [m/s]: %s' % (name, ' '.join('%+5.0f' % x[0] for x in d), ' '.join('%+5.0f' % x[1] for x in d)))
print('CL levels', CL)
