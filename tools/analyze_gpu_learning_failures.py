"""Classify stored first-fall/contact timing without new simulation or causality claims."""
import json
from pathlib import Path
import numpy as np
from boxer.checkpoint_io import atomic_json


def main():
    root = Path('docs/results/striking-pilot/gpu-learning-v1/gates')
    groups = []
    for label in ('baseline', 'step-1048576', 'fresh-final'):
        for role in ('neutral-0', 'neutral-1', 'selfplay'):
            report = json.loads((root/label/(role+'.json')).read_text()); rows = []
            for trial in report['trials']:
                if trial['survived']: continue
                contacts = [p['elapsed'] for p in trial['first_observed_contacts']
                    if any(g.startswith('red/') for g in p['geoms']) and any(g.startswith('blue/') for g in p['geoms'])]
                first = min(contacts) if contacts else None
                path = Path(trial['trace_file'])
                with np.load(path) as raw:
                    early = raw['time_after'] <= min(1., trial['elapsed'])
                    arm = np.sqrt(np.mean(raw['arm_residual'][early]**2, axis=(0,2)))
                    offsets = np.mean(np.abs(raw['action'][early,:,:3]), axis=0)
                rows.append(dict(seed=trial['seed'], swap=trial['swap'], first_fall=trial['elapsed'],
                    fallen=trial['fallen'], first_interbody_contact=first,
                    no_interbody_contact_before_stop=first is None or first > trial['elapsed'],
                    before_two_seconds=trial['elapsed'] < 2., terminal=trial['terminal_state']['corners'],
                    early_arm_residual_rms=arm.tolist(), early_mean_abs_command_offsets=offsets.tolist()))
            groups.append(dict(gate=label, role=role, failures=len(rows),
                before_two_seconds=sum(row['before_two_seconds'] for row in rows),
                no_prior_interbody_contact=sum(row['no_interbody_contact_before_stop'] for row in rows),
                learner_only_failures=sum(row['fallen']==([True,False] if role=='neutral-0' else [False,True]) for row in rows) if role!='selfplay' else None,
                conditions=rows))
    result = dict(scope='Stored native baseline, final-known and final-fresh first-fall timing',
        groups=groups, new_simulation_controls=0, new_renderer_calls=0,
        limitation='Temporal and orientation associations do not prove that an arm, command, collision or backend causes a fall.')
    atomic_json(Path('docs/results/striking-pilot/gpu-learning-v1/failure-timing.json'), result)
    print(json.dumps([{k:v for k,v in group.items() if k!='conditions'} for group in groups], indent=2))


if __name__ == '__main__': main()
