import sys, numpy as np
sys.path.insert(0, '/home/user/target-rl/src')
from target_rl import flight as F, config
from target_rl.scenario import Scenario, Target
NOM = Scenario('L0', 0, 0., 0., 1., 1.)
levels = F.world(True)['cl_levels']
print('fast-model CL levels', levels[:6], '...')
for idx in [int(a) for a in sys.argv[1:]]:
    for kg in (0., 1.):
        env = F.TargetFlight(NOM, Target('x', 0., 0.), True, kg=kg)
        while not env.done: env.step(F.encode(idx, 0))
        print('CL %.3f kg %.0f: s %.1f km, V %.1f m/s, h %.0f m, gamma %.4f, failed %s protect_frac %.2f' % (levels[idx], kg, env.range_km(), env.x[1], env.x[0], env.x[2], env.failed, env.summary()['protect_fraction']))
