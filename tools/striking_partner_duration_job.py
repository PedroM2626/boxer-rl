"""Reviewed duration test: unchanged partner task, full physical checks each block.

No automatic main adoption. Stop on survival regression, two stagnant checks,
two complete consecutive phase passes, or the finite four-block allocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc


def summarize(report, corner):
    rows=report['trials']
    expected={(seed,swap) for seed in (7,23,31,47) for swap in (False,True)}
    if (len(rows)!=8 or {(x['seed'],x['swap']) for x in rows}!=expected or
            not all(x['finite'] for x in rows) or report['episode_seconds']!=10. or
            report['scorecard_version']!=1 or report['assistance'] or
            report.get('sampled_policy') or report.get('diagnostic_zero_command_offsets') or
            report.get('diagnostic_approach_gain') is not None or
            report['neutral_partner_learner_corner']!=corner):
        raise ValueError('complete unchanged deterministic physical trials required')
    alive=[x for x in rows if x['survived']]
    return dict(survivors=len(alive),
        own_surviving_scoring_seeds=sorted({x['seed'] for x in alive if x['scored_by_corner'][corner]>0}),
        own_surviving_events=sum(x['scored_by_corner'][corner] for x in alive),
        median_minimum_hand_distance=float(np.median([x['minimum_hand_target_distance'][corner] for x in alive])) if alive else None)


def review(current, source, best_distance, best_seeds, stagnant, streak):
    """Predeclared review uses complete role reports, never training event counts."""
    regression=any(current[i]['survivors']<source[i]['survivors'] for i in range(2))
    phase_pass=all(current[i]['survivors']>=max(7,source[i]['survivors']) and
        len(current[i]['own_surviving_scoring_seeds'])>=2 for i in range(2))
    distance=float(np.mean([r['median_minimum_hand_distance'] for r in current])) if all(
        r['median_minimum_hand_distance'] is not None for r in current) else float('inf')
    seeds={f'{i}:{s}' for i in range(2) for s in current[i]['own_surviving_scoring_seeds']}
    improved=distance<=best_distance-.01 or bool(seeds-set(best_seeds))
    stagnant=0 if improved else stagnant+1
    streak=streak+1 if phase_pass else 0
    reason=('survival_regression' if regression else 'two_complete_phase_passes' if streak>=2
            else 'two_stagnant_checks' if stagnant>=2 else None)
    return dict(stop_reason=reason,phase_pass=phase_pass,consecutive_passes=streak,
        stagnant_checks=stagnant,improved=improved,best_distance=min(best_distance,distance) if improved else best_distance,
        best_scoring_role_seeds=sorted(set(best_seeds)|seeds))


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=args.output/'duration.json'
        if state_path.exists():raise ValueError('duration job exists; inspect before recovery')
        source=Path('checkpoints/striking-validation/partner-practice-1179648')
        reports=Path('docs/results/striking-pilot/partner-practice-1048576/completed')
        manifest=json.loads((source/'latest.json').read_text())
        payload=(source/manifest['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:
            raise ValueError('duration source checksum mismatch')
        offset=manifest['consumed_steps'];world_offset=manifest['world_control_steps']
        source_roles=[]
        for corner in (0,1):
            report=json.loads((reports/('trained-neutral-'+str(corner)+'.json')).read_text())
            if report['striking_sha256']!=manifest['sha256']:raise ValueError('source evaluation mismatch')
            source_roles.append(summarize(report,corner))
        folder=args.output/'learner'; cp=folder/'checkpoint';cp.mkdir(parents=True,exist_ok=False)
        (cp/manifest['file']).write_bytes(payload);atomic_json(cp/'latest.json',manifest)
        atomic_json(folder/'config.json',json.loads((source/'config.json').read_text()))
        atomic_json(folder/'progress.json',dict(consumed_steps=offset,world_control_steps=world_offset))
        state=dict(status='running',phase='training',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),source_sha256=manifest['sha256'],source_offset=offset,
            source_world_offset=world_offset,main_consumed_retained=1048576,
            diagnostic_ppo_before_job=925952,formal_evaluation_world_controls_before_job=261701,
            maximum_new_learner_steps=524288,block_steps=131072,maximum_evaluation_world_controls=48000,
            diagnostic_consumed_steps=0,diagnostic_evaluation_world_controls=0,
            source_roles=source_roles,checks=[],policy_adoption='none',assistance=False,
            consecutive_passes=0,stagnant_checks=0,
            best_distance=float(np.mean([r['median_minimum_hand_distance'] for r in source_roles])),
            best_scoring_role_seeds=[])
        atomic_json(state_path,state)
        try:
            for block in range(1,5):
                ceiling=offset+131072*block
                state.update(phase='training',active_block=block,active_ceiling=ceiling,updated_utc=utc())
                atomic_json(state_path,state)
                with (folder/f'block-{block}-training.log').open('w') as out,(folder/f'block-{block}-errors.log').open('w') as err:
                    subprocess.run([sys.executable,'-u','tools/striking_partner_job.py','--train-child',
                        '--output',str(folder),'--ceiling',str(ceiling)],stdout=out,stderr=err,check=True)
                progress=json.loads((folder/'progress.json').read_text())
                delta=progress['consumed_steps']-offset
                if (progress['consumed_steps']!=ceiling or progress['optimized_steps']!=ceiling or
                        progress['world_control_steps']-world_offset!=delta):
                    raise ValueError('duration learner/world counters disagree')
                state.update(phase='evaluation',diagnostic_consumed_steps=delta,
                    diagnostic_training_world_controls=delta,updated_utc=utc());atomic_json(state_path,state)
                check_folder=args.output/f'check-{ceiling}';check_folder.mkdir(exist_ok=False)
                current=[]
                for corner in (0,1,None):
                    name='selfplay' if corner is None else 'neutral-'+str(corner)
                    command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(cp),
                        '--output',str(check_folder/(name+'.json'))]
                    if corner is not None:command+=['--neutral-partner',str(corner)]
                    with (check_folder/(name+'.log')).open('w') as out,(check_folder/(name+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    report=json.loads((check_folder/(name+'.json')).read_text())
                    state['diagnostic_evaluation_world_controls']+=report['simulated_world_transitions']
                    if corner is not None:current.append(summarize(report,corner))
                    else:
                        if (len(report['trials'])!=8 or not all(x['finite'] for x in report['trials']) or
                                {(x['seed'],x['swap']) for x in report['trials']} !=
                                {(s,w) for s in (7,23,31,47) for w in (False,True)} or
                                report['episode_seconds']!=10. or report['scorecard_version']!=1 or
                                report['assistance'] or report.get('sampled_policy') or
                                report.get('diagnostic_approach_gain') is not None or
                                report.get('diagnostic_zero_command_offsets') or
                                report['neutral_partner_learner_corner'] is not None):
                            raise ValueError('incomplete original selfplay screen')
                    atomic_json(state_path,state)
                decision=review(current,source_roles,state['best_distance'],state['best_scoring_role_seeds'],
                    state['stagnant_checks'],state['consecutive_passes'])
                snapshot=json.loads((cp/'latest.json').read_text())
                state['checks'].append(dict(ceiling=ceiling,checkpoint=snapshot,roles=current,review=decision))
                state.update({k:v for k,v in decision.items() if k!='improved'})
                atomic_json(check_folder/'review.json',state['checks'][-1])
                atomic_json(state_path,state)
                if decision['stop_reason']:break
            state.update(status='completed',phase='awaiting_review',stop_reason=state.get('stop_reason') or 'budget_reached',updated_utc=utc())
            atomic_json(state_path,state)
        except BaseException as error:
            progress=json.loads((folder/'progress.json').read_text())
            state.update(status='failed',error=str(error),diagnostic_consumed_steps=max(0,progress['consumed_steps']-offset),
                diagnostic_training_world_controls=max(0,progress['world_control_steps']-world_offset),updated_utc=utc())
            atomic_json(state_path,state);raise


if __name__=='__main__':main()
