"""Passive qualified-training-event windows; no physical or learner updates."""
import hashlib,io,json,pickle
from pathlib import Path
import numpy as np
from boxer.checkpoint_io import atomic_bytes,atomic_json


def save_completed_evidence(monitor,output,checkpoint_dir,*,consumed,world,learner_corner,reset_seed,swap,active_opponent,foundation_sha256):
    pending=monitor.take_completed_evidence()
    if not pending:return []
    output=Path(output);output.mkdir(parents=True,exist_ok=True);checkpoint_dir=Path(checkpoint_dir);files=[]
    for item in pending:
        arrays=item['arrays'];event=item['event']
        if not all(np.isfinite(v).all() for v in arrays.values()):raise FloatingPointError('nonfinite passive event evidence')
        policies=[]
        for point in np.unique(arrays['policy_optimized_steps']):
            if point<0:raise ValueError('event evidence requires actual rollout policy provenance')
            file=checkpoint_dir/f'step-{int(point)}.pkl'
            if not file.exists():file=checkpoint_dir/'initial.pkl'
            payload=file.read_bytes();saved=pickle.loads(payload)
            if saved['optimized_steps']!=int(point):raise ValueError('event checkpoint version mismatch')
            policies.append(dict(optimized_steps=int(point),file=str(file),sha256=hashlib.sha256(payload).hexdigest()))
        key=f'step-{consumed}-world-{world}-corner-{event["corner"]}-hand-{event["hand"]}-time-{round(event["event_time"]*1e9)}'
        native=output/(key+'.npz');manifest=output/(key+'.json')
        if native.exists() or manifest.exists():raise ValueError('never overwrite event evidence')
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);atomic_bytes(native,buffer.getvalue())
        atomic_json(manifest,dict(scope='qualified_training_cycle_pending_independent_policy_evaluation',event=event,
            controlled_by='learner' if event['corner']==learner_corner else 'opponent',learner_corner=int(learner_corner),
            consumed_at_recognition=int(consumed),world=int(world),reset_seed=int(reset_seed),swap=bool(swap),active_opponent=bool(active_opponent),
            foundation_sha256=foundation_sha256,policy_checkpoints_in_window=policies,
            native_file=str(native),native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            new_physical_controls_for_capture=0,new_optimizer_updates_for_capture=0,new_geometry_queries=0,
            main_adoption=False,limitation='A stochastic training event is not a deterministic held-out gate pass; the window may span PPO updates.'))
        files.append(str(manifest))
    return files
