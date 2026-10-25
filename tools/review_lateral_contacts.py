"""Summarize reviewed native contact provenance without new simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('docs/results/striking-pilot/lateral-command-v5');audit=json.loads((root/'audit.json').read_text())
    assert audit['phase']=='independently_reviewed';rows=[]
    for row in audit['trials']:
        path=root/'native'/Path(row['native_trace_file']).name;digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==row['native_trace_sha256']
        with np.load(path) as a:
            count=int(a['present'][:,row['corner'],0].sum())
        rows.append(dict(case=row['case'],native_trace_sha256=digest,acting_left_target_contact_substeps=count,
            all_events=row['events']))
    result=dict(status='reviewed_saved_arrays',new_physical_controls=0,new_cloned_forwards=0,new_surface_queries=0,new_optimizer_updates=0,
        trials=rows,total_acting_left_target_contact_substeps=sum(r['acting_left_target_contact_substeps'] for r in rows))
    (root/'contact-provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(trials=len(rows),acting_left_target_contact_substeps=result['total_acting_left_target_contact_substeps'])))


if __name__=='__main__':main()
