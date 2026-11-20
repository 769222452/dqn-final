"""Study-specific no-acceptance patience; other stopping rules remain unchanged."""
from target_rl.train import stop_reason


def history(n):
    return [dict(accepted=False, positive_pairs=1, invalid=0) for _ in range(n)]


def test_default_no_acceptance_patience_is_unchanged():
    assert stop_reason(history(11), patience=4) is None
    assert stop_reason(history(12), patience=4) == 'no candidate accepted in 12 iterations'


def test_study_r_stops_after_twenty_unaccepted_iterations():
    assert stop_reason(history(19), patience=4, no_acceptance_iterations=20) is None
    assert stop_reason(history(20), patience=4, no_acceptance_iterations=20) == 'no candidate accepted in 20 iterations'


def test_acceptance_resets_the_twenty_iteration_window():
    rows = history(30)
    rows[10]['accepted'] = True
    assert stop_reason(rows, patience=4, no_acceptance_iterations=20) is None
    rows.append(history(1)[0])
    assert stop_reason(rows, patience=4, no_acceptance_iterations=20) == 'no candidate accepted in 20 iterations'


def test_clean_sweep_stop_is_unchanged():
    rows = [dict(accepted=False, positive_pairs=0, invalid=0) for _ in range(8)]
    assert stop_reason(rows, patience=4, no_acceptance_iterations=20) == (
        '2 clean sweeps (8 iterations): no positive pair, no violating branch, no acceptance')
