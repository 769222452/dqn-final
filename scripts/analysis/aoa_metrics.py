"""Angle-of-attack oscillation metrics for executed flights (V16 proofs or gate-experiment logs).

The executed alpha of a flight is a sequence of 3 s steps, each holding six 0.5 s commands.
Metrics separate the 3 s replanning sawtooth from slow trajectory-following changes:

- reversal_rate: share of consecutive command changes larger than NOISE_DEG that reverse direction;
- intra_step_ptp: peak-to-peak alpha within each 3 s step;
- boundary_jump: |first command of a step - last command of the previous step|;
- sawtooth_opposition: share of steps whose boundary jump opposes the previous step's drift, over steps
  where both the jump and the drift exceed NOISE_DEG (None if no such step: no sawtooth to measure);
- mean_profile: average deviation of the six commands from their step mean;
- hf_rms: RMS of alpha minus its 6 s centred moving average;
- power_3s: share of detrended alpha variance at the 1/3 Hz replanning frequency and harmonics.
"""
import json
from pathlib import Path

import numpy as np

# Changes below this are solver/round-off level and carry no sawtooth information. Without a
# floor, reversal and opposition shares of a ~1e-3 deg flat trajectory are noise statistics.
NOISE_DEG = .01


def metrics(steps, alpha0=None):
    """steps: array-like [n_steps, 6] of executed alpha in rad; alpha0: alpha before the first step."""
    a = np.rad2deg(np.asarray(steps, float))
    if a.ndim != 2 or a.shape[1] != 6 or len(a) < 3:
        raise ValueError('Expected at least three steps of six alpha commands')
    seq = a.ravel()
    d = np.diff(seq)
    moving = d[np.abs(d) > NOISE_DEG]
    reversals = float(np.mean(np.sign(moving[1:]) != np.sign(moving[:-1]))) if len(moving) > 1 else 0.
    prev_last = np.r_[np.nan if alpha0 is None else np.rad2deg(alpha0), a[:-1, -1]]
    jump = a[:, 0]-prev_last
    drift = a[:, -1]-a[:, 0]
    valid = np.isfinite(jump[1:]) & (np.abs(jump[1:]) > NOISE_DEG) & (np.abs(drift[:-1]) > NOISE_DEG)
    oppose = np.sign(jump[1:]) != np.sign(drift[:-1])
    k = 12
    smooth = np.convolve(seq, np.ones(k)/k, mode='same')
    hf = (seq-smooth)[k:-k]
    detrended = seq-np.polyval(np.polyfit(np.arange(len(seq)), seq, 3), np.arange(len(seq)))
    spectrum = np.abs(np.fft.rfft(detrended))**2
    freq = np.fft.rfftfreq(len(seq), d=.5)
    harmonics = np.zeros(len(freq), bool)
    for m in range(1, 4):
        harmonics |= np.abs(freq-m/3.) <= .5/len(seq)/.5*1.5
    return dict(steps=int(len(a)), alpha_min_deg=float(seq.min()), alpha_max_deg=float(seq.max()),
                reversal_rate=reversals,
                intra_step_ptp_median_deg=float(np.median(np.ptp(a, axis=1))),
                intra_step_ptp_p95_deg=float(np.percentile(np.ptp(a, axis=1), 95)),
                boundary_jump_median_deg=float(np.nanmedian(np.abs(jump))),
                boundary_jump_p95_deg=float(np.nanpercentile(np.abs(jump), 95)),
                sawtooth_opposition=float(np.mean(oppose[valid])) if valid.any() else None,
                sawtooth_opposition_steps=int(valid.sum()), noise_floor_deg=NOISE_DEG,
                mean_profile_deg=np.round((a-a.mean(1, keepdims=True)).mean(0), 4).tolist(),
                hf_rms_deg=float(hf.std()) if len(hf) else None,
                power_3s_fraction=float(spectrum[harmonics].sum()/max(spectrum[1:].sum(), 1e-300)))


def load_v16_flight(run):
    """Executed steps of an archived V16 flight folder holding pools.jsonl and proofs/."""
    run = Path(run)
    infos = []
    for line in (run/'pools.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        passed = [i for i, ok in enumerate(row['mask']) if ok]
        if not passed:
            break
        order = np.argsort(-np.asarray(row['preference']))
        action = next(int(a) for a in order if row['mask'][a])
        infos.append(json.loads((run/'proofs'/(row['proof_ids'][action]+'.json')).read_text(encoding='utf-8'))['info'])
    return infos


def load_gate_log(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def flight_metrics(infos):
    steps = [s['plant']['alpha'] for s in infos]
    out = metrics(steps, infos[0]['post_context']['alpha_prev'])
    errors = np.array([abs(s['executed_CL_mean']-s['CL_cmd']) for s in infos])
    out.update(CL_mean_abs_error_median=float(np.median(errors)), CL_mean_abs_error_p95=float(np.percentile(errors, 95)),
               range_km=float(infos[-1]['range_m']/1000), t_end_s=float(infos[-1]['t_end_s']))
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', nargs='+', help='V16 run folder (pools.jsonl + proofs/) or gate steps.jsonl')
    a = p.parse_args()
    for src in a.source:
        path = Path(src)
        infos = load_gate_log(path) if path.is_file() else load_v16_flight(path)
        print(json.dumps(dict(source=str(path), **flight_metrics(infos))))
