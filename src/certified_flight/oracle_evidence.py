"""OCP evidence writers explicitly bound to one flight's output directory."""
from pathlib import Path

import numpy as np

from .serialization import write_json


def oracle_sinks(folder):
    """Return event/artifact sinks for this flight, including independently forked flights.

    Function closures are not rebound by ``deepcopy``. A certifier fork must call
    this factory again, otherwise its OCP attempts overwrite the parent's files.
    """
    folder = Path(folder)

    def event_sink(event):
        write_json(folder/'oracle'/f"a{event['attempt']:06d}"/'event.json', event)

    def artifact_sink(event, head, plan, arrays):
        target = folder/'oracle'/f"a{event['attempt']:06d}"/head
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target/'solver_plan.npz', alpha=plan.alpha, times=plan.times, x=plan.x,
                            dual_x=plan.dual.get('x', []), dual_g=plan.dual.get('g', []))
        for dt, values in arrays.items():
            np.savez_compressed(target/f'audit_dt{dt}.npz',
                                **{k: values[k] for k in ('x', 'samples', 'alpha', 'times')})

    return event_sink, artifact_sink
