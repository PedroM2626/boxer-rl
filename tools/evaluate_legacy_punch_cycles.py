"""Measure retained Warp6/7 with fist-only impulse/travel/return criteria.

Reconstructed legacy engine/configuration, not a matched current-model gate.
Original 0.75 m spawns and a separate 2 m transfer condition are both retained.
"""
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from boxer import scene
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.impact_scorecard import ImpactScorecard
from boxer.striking_punch_trace import PunchTrace, describe_events
from boxer.warp_config import make_env
from boxer.warp_train import numpy_forward, warp_run
from boxer.warp_readback import WarpReadback
from tools.run_gpu_closed_loop import valid

ROOT = Path('/home/pedro/boxing-runs/legacy-punch-cycles-v1')
SOURCES = {'warp6': 'ed0c56e760bb74a63aaa35d8b5d5ed445fa745e66519e5008536548a17368937',
           'warp7': 'b5a7e7bc837ff3938fcf0dbfd96fcdd74b0c32f711a1da05f7cb919a3be200fd'}


class LegacyTraceAdapter:
    def __init__(self, env):
        self.env = env
        self.model = env._model
        self.data = mujoco.MjData(self.model)
        self.readback = WarpReadback(env._d)
        self.scorecard = ImpactScorecard()
        self.maps = [dict(side=side, root=env._sides[side]['free_adr']) for side in scene.SIDES]
        self.fists = np.array([[self.model.geom(side+hand).id for hand in scene.HAND_GEOMS] for side in scene.SIDES])
        self.targets = [{self.model.geom(side+target).id for target in ('head_collision', 'torso_collision')}
                        for side in scene.SIDES]
        self.trace = PunchTrace()
        self.first_fall = None
        self.steps = 0
        self.solver_flags = 0
        self.scored = np.zeros(2)
        env._advance = self.advance
        # Legacy code suppressed all warnings. Restore capacity warnings and
        # inspect every flag; repetitive solver text alone stays suppressed.
        env._m.opt.warn_overflow = ~int(mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS) & 4095

    def read(self):
        flags = self.env._d.overflow.numpy()
        mask = int(mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS)
        if np.any(flags & ~mask):
            raise FloatingPointError(f'legacy GPU overflow: {flags.tolist()}')
        self.solver_flags += int(np.count_nonzero(flags))
        self.readback.export(self.data, self.model)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('nonfinite legacy dynamics')
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied) or self.model.neq or np.any(self.model.body_gravcomp):
            raise ValueError('legacy comparison forbids support/assistance')

    def fallen(self):
        result = []
        for m in self.maps:
            up = self.data.site(m['side']+'imu_in_torso').xmat.reshape(3, 3)[2, 2]
            result.append(self.data.qpos[m['root']+2] < .55 or up < .6)
        return np.asarray(result)

    def advance(self):
        for _ in range(self.env.n_frames):
            self.read()
            positions = self.data.geom_xpos.copy()
            velocities = {}
            for geom in set(self.fists.ravel()) | self.targets[0] | self.targets[1]:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, int(geom), velocity, 0)
                velocities[geom] = velocity[3:].copy()
            torso = []
            for mapping in self.maps:
                site = self.model.site(mapping['side']+'imu_in_torso').id
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0)
                torso.append((self.data.site_xpos[site].copy(), velocity))
            self.trace.before_substep(self, positions, velocities, torso)
            self.steps += 1
            mjwarp.step(self.env._m, self.env._d)
            self.read()
            present = np.zeros((2, 2), bool)
            impulse, closing, advancing = (np.zeros((2, 2)) for _ in range(3))
            for index, contact in enumerate(self.data.contact):
                geoms = set(map(int, contact.geom))
                for corner in range(2):
                    for hand, fist in enumerate(self.fists[corner]):
                        targets = geoms & self.targets[1-corner]
                        if fist not in geoms or not targets: continue
                        target = next(iter(targets))
                        force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, index, force)
                        present[corner, hand] = True
                        impulse[corner, hand] += max(0., force[0])*self.model.opt.timestep
                        direction = positions[target]-positions[fist]
                        direction /= max(np.linalg.norm(direction), 1e-8)
                        closing[corner, hand] = max(closing[corner, hand], float((velocities[fist]-velocities[target])@direction))
                        origin, velocity = torso[corner]
                        own = velocity[3:]+np.cross(velocity[:3], positions[fist]-origin)
                        advancing[corner, hand] = max(advancing[corner, hand], float((velocities[fist]-own)@direction))
            fallen = self.fallen()
            self.scored += self.scorecard.update(present, impulse, closing, advancing,
                self.data.time, self.model.opt.timestep, both_up=not fallen.any())
            self.trace.after_substep(self, present, impulse, closing, advancing)
            if fallen.any():
                self.first_fall = float(self.data.time)
                break
        return None  # Preserve the legacy engine's own control-level scorecard.


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (ROOT/'.lock', Path('runs/striking_diagnostic_supervisor.lock'), Path('runs/unassisted_boxing.lock')):
            fcntl.flock(stack.enter_context(path.open('a')), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'state.json').exists(): raise ValueError('Existing comparison requires review before recovery')
        prior = json.loads(Path('/home/pedro/boxing-runs/gpu-closed-loop-v1/state.json').read_text())
        assert prior['status'] == 'completed'
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), episodes=[], new_controls=0,
            maximum_new_controls=7680, formal_before=prior['formal_total'], formal_ceiling=prior['formal_total']+7680,
            main_retained=476041216, diagnostic_ppo_retained=9504000, new_ppo_steps=0, gpu_world_substeps=0,
            limitation='Own reconstructed legacy engine; 4 ms integration, original rewards/termination and observations differ. Not a current complete gate or exact historical executable reproduction.')
        atomic_json(ROOT/'state.json', state)
        try:
            for name, expected in SOURCES.items():
                _, checkpoint, config = warp_run('runs/'+name)
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
                with checkpoint.open('rb') as handle:
                    params, _, _, norm, trained = pickle.load(handle)
                assert trained == 200212480 and config['action_size'] == 29 and config['n_frames'] == 5
                predict = numpy_forward(config, params, norm, False)
                for separation in (.75, 2.):
                    for seed in (7, 23, 31, 47):
                        adapter = None
                        episode_substeps_before = state['gpu_world_substeps']
                        env = make_env(config, nworld=1, spawn_distance=separation)
                        # Fresh initialized physical state, avoiding uninitialized
                        # host scratch transfer and episode warm-start leakage.
                        env._d = mjwarp.make_data(env._model, nworld=1, nconmax=384, njmax=2048)
                        obs = env.reset(seed)
                        adapter = LegacyTraceAdapter(env)
                        original_hits = np.zeros(2)
                        for control in range(480):
                            state['new_controls'] += 1
                            state['formal_total'] = state['formal_before']+state['new_controls']
                            assert state['new_controls'] <= 7680
                            atomic_json(ROOT/'state.json', state)
                            action = predict(obs[..., :config['obs_size']].reshape(2, config['obs_size'])).reshape(1, 2, env.action_size)
                            if not np.isfinite(action).all(): raise FloatingPointError('nonfinite legacy action')
                            obs, _, done, metrics = env.step(action)
                            state['gpu_world_substeps'] = sum(e['gpu_substeps'] for e in state['episodes'])+adapter.steps
                            original_hits += metrics['scored_by'][0]
                            if adapter.first_fall is not None or done[0]: break
                        file = ROOT/f'{name}-separation-{separation}-seed-{seed}.npz'
                        adapter.trace.save(file)
                        events = describe_events(adapter.trace.arrays(), native_timestep=env._model.opt.timestep)
                        cycles = [sum(e['corner'] == c and valid(e) for e in events) for c in (0, 1)]
                        result = dict(run=name, checkpoint_sha256=expected, trained_steps=int(trained), seed=seed,
                            separation=separation, controls=control+1, gpu_substeps=adapter.steps,
                            physics_dt=env._model.opt.timestep, own_engine_terminal=bool(done[0]),
                            config_sha256=hashlib.sha256((Path('runs')/name/'warp_config.json').read_bytes()).hexdigest(),
                            common_posture_first_fall=adapter.first_fall, legacy_scored_value=original_hits.tolist(),
                            fist_only_events=adapter.scored.tolist(), complete_cycles=cycles,
                            solver_flag_observations=adapter.solver_flags, file=file.name,
                            trace_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), events=events,
                            common_posture_intact_at_stop=adapter.first_fall is None,
                            completed_480_controls=control+1 == 480,
                            self_leg_contact_rule_comparable=False)
                        state['episodes'].append(result)
                        atomic_json(ROOT/'state.json', state)
                        print(json.dumps({k:v for k,v in result.items() if k != 'events'}), flush=True)
                        del adapter, env
            state.update(status='completed', completed_utc=utc(), decision='Compare both original-range and separated transfer evidence; no automatic adoption')
        except BaseException as error:
            if locals().get('adapter') is not None:
                state['gpu_world_substeps'] = episode_substeps_before+adapter.steps
            state.update(status='failed', error=repr(error)); raise
        finally:
            atomic_json(ROOT/'state.json', state)


if __name__ == '__main__':
    main()
