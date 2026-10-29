"""Current unassisted paired task on batched CUDA with native reward semantics.

The first fall ends each world's episode. Kernels submitted for other live worlds
may also compute discarded terminal-world substeps; those never enter histories,
rewards or terminal observations and are separately accounted. Resetting a world
never forwards or rewrites the surviving worlds in the live physics data.
"""
import jax
import jax.numpy as jp
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from boxer.batched_impact_scorecard import BatchedImpactScorecard
from boxer.curriculum_train import inference
from boxer.paired_foundation import PairedFoundation
from boxer.paired_warp_features import PairedFeatures
from boxer.paired_batch_metrics import PairedBatchMetrics, metric_snapshot
from boxer.paired_warp_physics import PairedWarpPhysics
from boxer.striking_contact_progress import ContactProgressMonitor
from boxer.striking_motion_reward import replace_contact_payment
from boxer.striking_contact_rebalance import correction
from boxer.striking_arm_effort import effort_cost


def copy_world_rows(destination, source, keys, rows, worlds):
    """Copy typed contiguous row views, without host round trips or active-row writes."""
    for key in keys:
        target, origin = getattr(destination, key), getattr(source, key)
        if target.shape != origin.shape or target.shape[0] != worlds:
            raise ValueError(f'invalid per-world copy shape: {key}')
        width = target.size//worlds
        for row in rows:
            wp.copy(target.flatten(), origin.flatten(), dest_offset=int(row)*width,
                    src_offset=int(row)*width, count=width)


class BatchedStrikingEnv:
    dt = .02
    action_size = 17
    observation_size = 148
    reward_version = 17
    reset_fields = ('qpos', 'qvel', 'ctrl', 'time', 'qacc', 'qacc_warmstart', 'sensordata',
        'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat', 'xpos', 'xquat', 'xmat', 'subtree_com', 'cvel')

    def __init__(self, foundation, params=None, *, worlds=8, motor=None,
                 engagement_distance=.45, engagement_reward_distance=.45,
                 account=None, capture_evidence=True):
        self.worlds = worlds
        self.pair = PairedFoundation(foundation)
        self.model = self.pair.model
        self.accounting = dict(world_controls=0, gpu_world_substeps=0, gpu_active_world_substeps=0,
            gpu_discarded_terminal_world_substeps=0, gpu_warmup_world_substeps=0,
            gpu_reset_world_forwards=0, native_reset_forwards=1)
        self.account_callback = account
        self.health_summary = dict(checks=0, solver_limit_world_checks=0,
            overflow_flags_seen=0, peak_contacts=0, peak_constraints=0)
        self.features = PairedFeatures(self.pair, engagement_distance)
        self.metrics = PairedBatchMetrics(self.pair)
        self.motor = motor if motor is not None else jax.jit(inference(foundation, params))
        self.engagement_reward_distance = engagement_reward_distance
        if not np.isfinite(engagement_reward_distance) or engagement_reward_distance <= 0:
            raise ValueError('positive finite engagement reward distance required')
        self.capture_evidence = capture_evidence
        self.substep_observer = None
        self.arm_indices = np.array([i for i in range(29) if any(x in foundation.mj_model.actuator(i).name
            for x in ('shoulder', 'elbow', 'wrist'))])
        if len(self.arm_indices) != 14 or self.pair.substeps != 10:
            raise ValueError('unchanged 14-arm/10-substep current task required')
        self.physics = PairedWarpPhysics(self.model, worlds)
        self.reset_physics = PairedWarpPhysics(self.model, worlds)
        for engine in (self.physics, self.reset_physics):
            # Preserve machine flags and all capacity warnings. Avoid printing
            # an expected original-solver limit once per world/substep.
            engine.model.opt.warn_overflow &= ~(mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS)
        self.scorecard = BatchedImpactScorecard(worlds)
        self.phase = np.zeros((worlds, 2, 2)); self.frequency = np.zeros((worlds, 2, 1))
        self.last_action = np.zeros((worlds, 2, 29)); self.previous = np.zeros((worlds, 2, 17))
        self.residual = np.zeros((worlds, 2, 14)); self.age = np.zeros(worlds, int)
        self.done = np.ones(worlds, bool); self.monitors = [None]*worlds
        self.previous_distance = np.zeros((worlds, 2))
        self.state = None
        self.reset(np.arange(worlds), np.arange(worlds), np.zeros(worlds, bool))
        self._account(gpu_world_substeps=worlds, gpu_warmup_world_substeps=worlds)
        self.physics.step(); self._health(); self.physics.capture_step()
        # Warm-up is discarded, including all solver/history state.
        self.reset(np.arange(worlds), np.arange(worlds), np.zeros(worlds, bool))

    def _account(self, **deltas):
        for key, value in deltas.items(): self.accounting[key] += int(value)
        if self.account_callback is not None: self.account_callback(dict(self.accounting))

    def _health(self):
        health = self.physics.health()
        self.health_summary['checks'] += 1
        self.health_summary['solver_limit_world_checks'] += health['solver_limit_worlds']
        for flags in health['overflow_flags']:
            self.health_summary['overflow_flags_seen'] |= flags
        self.health_summary['peak_contacts'] = max(self.health_summary['peak_contacts'], health['contacts'])
        self.health_summary['peak_constraints'] = max(self.health_summary['peak_constraints'], health['max_constraints'])
        if np.any(self.physics.data.qfrc_applied.numpy()) or np.any(self.physics.data.xfrc_applied.numpy()):
            raise FloatingPointError('external support is forbidden')
        return health

    def observation(self):
        return self.features.striking_observation(self.state, self.last_action, self.phase,
            self.previous, np.isfinite(self.scorecard.last_score).astype(np.float32))

    def reset(self, rows, seeds, swaps):
        rows, seeds, swaps = np.asarray(rows, int), np.asarray(seeds), np.asarray(swaps, bool)
        if (rows.ndim != 1 or seeds.shape != rows.shape or swaps.shape != rows.shape
                or len(set(rows.tolist())) != len(rows) or np.any(rows < 0) or np.any(rows >= self.worlds)):
            raise ValueError('unique valid world rows with corresponding seeds and swaps required')
        if not len(rows): return self.observation()
        all_rows = len(rows) == self.worlds and set(rows.tolist()) == set(range(self.worlds))
        if self.state is None:
            qpos = np.tile(self.model.qpos0, (self.worlds, 1))
            qvel = np.zeros((self.worlds, self.model.nv)); ctrl = np.zeros((self.worlds, self.model.nu))
            time = np.zeros(self.worlds)
        else:
            qpos, qvel = self.state['qpos'].copy(), self.state['qvel'].copy()
            ctrl = self.physics.data.ctrl.numpy().astype(float); time = self.state['time'].copy()
        for row, seed, swap in zip(rows, seeds, swaps):
            self._account(native_reset_forwards=1)
            self.pair.reset(seed=int(seed), swap=bool(swap))
            qpos[row] = self.pair.data.qpos; qvel[row] = self.pair.data.qvel
            ctrl[row] = self.pair.data.ctrl; time[row] = 0.
            self.phase[row] = self.pair.phase; self.frequency[row] = self.pair.frequency
            self.last_action[row] = 0.; self.previous[row] = 0.; self.residual[row] = 0.; self.age[row] = 0
            self.monitors[row] = ContactProgressMonitor()
            self.monitors[row].capture_completed_events = self.capture_evidence
        self.scorecard.reset(rows)
        self._account(gpu_reset_world_forwards=self.worlds)
        if all_rows:
            self.physics.load(qpos, qvel, ctrl, time)
        else:
            # A separate data object computes new derived observations. Forwarding
            # live survivors here would refresh their one-substep-lagged sensors.
            self.reset_physics.load(qpos, qvel, ctrl, time)
            self.reset_physics.health()
            if self.reset_physics.forward_graph is None: self.reset_physics.capture_forward()
            mask = np.zeros(self.worlds, bool); mask[rows] = True
            with wp.ScopedDevice(self.physics.device):
                mjwarp.reset_data(self.physics.model, self.physics.data, wp.array(mask))
                copy_world_rows(self.physics.data, self.reset_physics.data, self.reset_fields, rows, self.worlds)
        self._health()
        self.state = metric_snapshot(self.physics.data, gpu=True)
        self.previous_distance[rows] = self.metrics.before(self.state)['target_center_distance'][rows].min(axis=2)
        self.done[rows] = False
        return self.observation()

    def step(self, actions, *, optimized_steps=-1):
        actions = np.asarray(actions)
        if actions.shape != (self.worlds, 2, 17) or not np.isfinite(actions).all() or np.any(abs(actions) > 1.00001):
            raise ValueError('finite bounded paired actions required for every world')
        if self.done.any(): raise RuntimeError('reset terminated worlds before the next control')
        self._account(world_controls=self.worlds)
        command = self.features.commands(self.state, offsets=actions[..., :3])
        obs = self.features.motor_observation(self.state, command, self.last_action, self.phase)
        motor, _ = self.motor({'state': jp.asarray(obs.reshape(-1, 103)),
                              'privileged_state': jp.zeros((self.worlds*2, 216))}, jax.random.PRNGKey(0))
        motor = np.asarray(motor).reshape(self.worlds, 2, 29).copy()
        self.residual = .8*self.residual+.2*actions[..., 3:]
        motor[..., self.arm_indices] += .8/self.pair.action_scale*self.residual
        ctrl = np.zeros((self.worlds, self.model.nu), np.float32)
        for c, mapping in enumerate(self.pair.maps):
            ctrl[:, mapping['act']] = self.pair.pose+self.pair.action_scale*motor[:, c]
        self.physics.data.ctrl.assign(ctrl)
        active = np.ones(self.worlds, bool)
        scored, impulse_total = np.zeros((self.worlds, 2)), np.zeros((self.worlds, 2))
        fallen = np.zeros((self.worlds, 2), bool)
        terminal = {}
        for _ in range(10):
            if not active.any(): break
            pre_state = metric_snapshot(self.physics.data, gpu=True)
            before = self.metrics.before(pre_state)
            engaged = self.scorecard.engaged.copy(); previous_score = self.scorecard.last_score.copy()
            count = int(active.sum())
            self._account(gpu_world_substeps=self.worlds, gpu_active_world_substeps=count,
                          gpu_discarded_terminal_world_substeps=self.worlds-count)
            self.physics.step(); self._health()
            after = metric_snapshot(self.physics.data, gpu=True)
            if self.substep_observer is not None:
                self.substep_observer(self.physics)
            world, geometry, force = self.physics.normal_forces()
            contacts = self.metrics.contacts(before, world, geometry, force)
            flags = self.metrics.fallen(after, world, geometry)
            up = ~flags.any(axis=1)
            scored += self.scorecard.update(contacts['present'], contacts['impulse'],
                contacts['contact_closing'], contacts['contact_advancing'], after['time'], .002, up, active)
            impulse_total[active] += contacts['impulse'][active].sum(axis=2)
            for row in np.flatnonzero(active):
                record = {key: before[key][row].copy() for key in
                    ('local_position', 'local_velocity', 'elbow', 'toward_nearest_target', 'target_center_distance')}
                record.update(time_before=float(before['time_before'][row]), time_after=float(after['time'][row]),
                    engaged_before=engaged[row].copy(), event=self.scorecard.last_score[row] > previous_score[row],
                    window_impulse=self.scorecard.impulse[row].copy(), **{k: v[row].copy() for k, v in contacts.items()})
                if self.capture_evidence:
                    record.update(evidence_qpos_before=pre_state['qpos'][row].copy(),
                        evidence_qvel_before=pre_state['qvel'][row].copy(), evidence_ctrl=ctrl[row].astype(float),
                        policy_optimized_steps=np.int64(optimized_steps), applied_force_norm=np.array(0.))
                self.monitors[row].append_row(record)
                if not up[row]:
                    fallen[row] = flags[row]
                    terminal[row] = {key: value[row].copy() for key, value in after.items()}
            active &= up
        self.state = after
        for row, values in terminal.items():
            for key, value in values.items(): self.state[key][row] = value
        self.last_action = motor; self.previous = actions.copy(); self.age += 1
        self.phase = (self.phase+2*np.pi*self.dt*self.frequency+np.pi) % (2*np.pi)-np.pi
        self.done = fallen.any(axis=1) | (self.age >= 500)
        per_hand_distance = self.metrics.before(self.state)['target_center_distance']
        distances = per_hand_distance.min(axis=2)
        progress = np.clip(self.previous_distance-distances, -.1, .1)
        self.previous_distance = distances
        roots = [self.state['qpos'][:, m['root']:m['root']+2] for m in self.pair.maps]
        distance = np.linalg.norm(roots[1]-roots[0], axis=-1)
        facing = []
        for c, site in enumerate(self.features.imu):
            rotation = self.state['site_xmat'][:, site].reshape(-1, 3, 3)
            facing.append(np.sum(rotation[:, :2, 0]*(roots[1-c]-roots[c]), axis=-1)/np.maximum(distance, 1e-6))
        base = 2*scored-2*scored[:, ::-1]-5*fallen.astype(float)
        both_up = ~fallen.any(axis=1)
        base[both_up] += (.5*progress+self.dt*(.1*np.stack(facing, axis=1)
            +.15*np.exp(-((distance[:, None]-self.engagement_reward_distance)/.2)**2)
            -.03*np.mean(self.residual**2, axis=2)))[both_up]
        base = base.astype(np.float32)
        reward = np.zeros((self.worlds, 2)); cycles = np.zeros_like(reward)
        for row, monitor in enumerate(self.monitors):
            cycles[row], dense = monitor.consume(terminal=bool(self.done[row]), target_distances_after=per_hand_distance[row])
            outcome = dict(scored=scored[row], arm_residual=self.residual[row])
            for c in (0, 1):
                reward[row, c] = (replace_contact_payment(base[row, c:c+1], outcome, cycles[row], dense, c)[0]
                    +correction(monitor.last_components, c)-effort_cost(outcome, c, self.dt))
        return self.observation(), reward, self.done.copy(), dict(fallen=fallen, scored=scored,
            legal_contact_impulse=impulse_total, cycles=cycles, arm_residual=self.residual.copy(),
            applied_actions=motor.copy(), distance=distance, timed_out=self.age >= 500,
            elapsed=self.state['time'].copy(), base_reward=base)
