import numpy as np


class Pair:
    dt = .02

    def __init__(self):
        self.data = type("Data", (), {})()
        self.data.qpos, self.data.qvel, self.data.ctrl = np.zeros(3), np.zeros(3), np.zeros(2)
        self.last_action, self.phase, self.frequency = np.zeros(2), np.zeros(2), np.zeros(2)


def test_fall_watcher_keeps_the_states_before_a_single_fall_and_forgets_them_at_the_restart():
    from boxer.mirror_starts import FallWatcher
    pair, watcher = Pair(), FallWatcher()
    standing = dict(done=False, fallen=[False, False])
    for t in range(1, 251):                      # 5 s of boxing, the clock in qpos
        pair.data.qpos[0] = t * pair.dt
        watcher(t * pair.dt, pair, standing)
    watcher(251 * pair.dt, pair, dict(done=True, fallen=[False, True]))
    kept = sorted(round(s["before"], 2) for s in watcher.states)
    assert kept == [.52, 1.02, 1.52, 2.02, 2.52, 3.02]
    assert all(s["risen"] == 1 for s in watcher.states)
    assert all(abs(251 * pair.dt - s["qpos"][0] - s["before"]) < 1e-9 for s in watcher.states)
    # A double knockdown keeps nothing, and the restart starts a fresh history.
    for t in range(252, 300):
        watcher(t * pair.dt, pair, standing)
    watcher(300 * pair.dt, pair, dict(done=True, fallen=[True, True]))
    assert len(watcher.states) == 6 and not watcher.recent
