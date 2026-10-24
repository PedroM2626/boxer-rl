"""Descriptive extension motion from reviewed saved arrays; no new physics."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('docs/results/striking-pilot/motor-guard-cycle-v2')
    state=json.loads((root/'audit.json').read_text());assert state['phase']=='independently_reviewed'
    rows=[]
    for row in state['trials']:
        c=row['corner'];full=root/'full'/Path(row['full_trace_file']).name;native=root/'native'/Path(row['native_trace_file']).name
        assert sha(full)==row['full_trace_sha256'] and sha(native)==row['native_trace_sha256']
        with np.load(full) as a:
            mask=a['jab_active'].astype(bool);starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
            times=a['time_before'].copy();center_distance=float(a['hand_distance'][:,c].min())
            tracking=float(np.median(a['jab_target_error_m'][mask]));root_distance=float(a['root_distance_after'].min())
        pulses=[]
        with np.load(native) as a:
            active=np.repeat(mask,10);assert len(active)==len(a['time_before'])
            pos=a['local_position'][:,c,0];velocity=a['local_velocity'][:,c,0]
            for start in starts:
                first=start*10;end=first
                while end<len(active) and active[end]:end+=1
                pulses.append(dict(start_time=float(times[start]),peak_relative_speed_m_s=float(np.linalg.norm(velocity[first:end],axis=1).max()),
                    max_local_excursion_m=float(np.linalg.norm(pos[first:end]-pos[first],axis=1).max())))
            contacts=int(a['present'][:,c,0].sum())
        rows.append(dict(case=row['case'],pulses=pulses,extension_tracking_error_m=tracking,
            left_contact_substeps=contacts,minimum_root_distance_m=root_distance,
            minimum_any_own_hand_to_any_target_center_distance_m=center_distance,
            full_trace_sha256=sha(full),native_trace_sha256=sha(native)))
    result=dict(status='reviewed_array_measurement',trials=rows,new_physical_controls=0,new_cloned_evaluations=0,new_optimizer_updates=0,
        limitations=['Peak extension speed is not impact-onset speed and certifies no punch.',
            'Hand/target center distance includes either own hand; it is not a left-fist surface gap.',
            'Local excursion is measured only during each extension, not a full punch/guard certification.'])
    (root/'extension-motion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(trials=len(rows),pulses=sum(len(r['pulses']) for r in rows),contact_substeps=sum(r['left_contact_substeps'] for r in rows))))


if __name__=='__main__':main()
