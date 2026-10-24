"""Measure the scripted guard offset from archived motor-only hand positions."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('docs/results/striking-pilot/guard-control-v1')
    source=json.loads((root/'audit.json').read_text());review=json.loads((root/'analysis.json').read_text())
    assert source['phase']=='independently_reviewed' and review['status']=='reviewed'
    target=np.asarray(source['profile']['guard_local']);rows=[]
    for row in source['trials']:
        if row['mode']!='guard_only':continue
        key=(row['corner'],row['seed'],row['swap'])
        baseline=next(r for r in source['trials'] if r['mode']=='motor_only' and (r['corner'],r['seed'],r['swap'])==key)
        full=root/'full'/Path(row['full_trace_file']).name;native=root/'native'/Path(baseline['native_trace_file']).name
        assert sha(full)==row['full_trace_sha256'] and sha(native)==baseline['native_trace_sha256']
        with np.load(full) as a:
            n=int(np.flatnonzero(a['jab_active'])[0]);time=float(a['time_before'][n])
        with np.load(native) as a:
            i=n*10;assert abs(a['time_before'][i]-time)<1e-12
            point=a['local_position'][i,row['corner'],0].copy()
        assert np.isfinite(point).all()
        rows.append(dict(corner=row['corner'],seed=row['seed'],swap=row['swap'],entry_time=time,
            motor_fist_local=point.tolist(),scripted_guard_local=target.tolist(),
            offset_m=float(np.linalg.norm(target-point)),full_trace_sha256=sha(full),baseline_native_sha256=sha(native)))
    assert len(rows)==8
    result=dict(status='reviewed_array_measurement',rows=rows,minimum_offset_m=min(r['offset_m'] for r in rows),
        maximum_offset_m=max(r['offset_m'] for r in rows),new_physical_controls=0,new_cloned_evaluations=0,new_optimizer_updates=0,
        limitation='Measured target displacement is not an isolated causal explanation of guard failures.')
    (root/'learned-guard-offsets.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('minimum_offset_m','maximum_offset_m')}))


if __name__=='__main__':main()
