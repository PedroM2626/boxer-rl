"""Compare saved failure/timeout windows across the three matched lateral studies."""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    roots={name:Path('docs/results/striking-pilot')/name for name in
           ('lateral-command-v5','lateral-isolation-v6','guard-step-v7')}
    rows=[]
    for name,root in roots.items():
        audit=json.loads((root/'audit.json').read_text())
        assert audit['phase']=='independently_reviewed'
        for row in audit['trials']:
            if name=='lateral-command-v5' and row['mode']!='lateral_plus':continue
            path=root/'full'/Path(row['full_trace_file']).name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['full_trace_sha256']
            with np.load(path) as a:
                assert all(np.isfinite(a[k]).all() for k in a.files)
                window=a['time_after']>=a['time_after'][-1]-.5
                active=window&a['jab_active'].astype(bool)
                c=row['corner']
                rows.append(dict(study=name,corner=c,seed=row['seed'],swap=row['swap'],survived=row['survived'],
                    terminal_phase=int(a['cycle_phase'][-1]),terminal_time=float(a['time_after'][-1]),
                    last_half_second_min_pelvis=float(a['pelvis_height'][window,c].min()),
                    last_half_second_min_upright=float(a['upright'][window,c].min()),
                    terminal_root_distance=float(a['root_distance_after'][-1]),
                    last_half_second_active_controls=int(active.sum()),
                    active_tracking_error_median=None if not active.any() else float(np.median(a['jab_target_error_m'][active])),
                    active_action_clipped_fraction=None if not active.any() else float(np.mean(a['jab_action_clipped'][active])),
                    full_trace_sha256=row['full_trace_sha256']))
    result=dict(status='reviewed_saved_arrays',new_physical_controls=0,new_cloned_forwards=0,
        new_surface_queries=0,new_optimizer_updates=0,trials=rows,
        limitation='Terminal phase is the last commanded phase, not a proof of the mechanical cause of a fall.')
    destination=roots['guard-step-v7']/'matched-failure-windows.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(trials=len(rows),new_physical_controls=0)))


if __name__=='__main__':main()
