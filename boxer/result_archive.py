"""Copy immutable results across WSL/Windows without rewriting identical files."""
import time
from pathlib import Path
from boxer.checkpoint_io import atomic_bytes


def archive_tree(source, target):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob('*'):
        if not path.is_file() or path.name.endswith('.lock') or path.name == '.lock':
            continue
        destination = target/path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = path.read_bytes()
        if destination.exists() and destination.read_bytes() == payload:
            continue
        for attempt in range(12):
            try:
                atomic_bytes(destination, payload)
                break
            except OSError as error:
                if error.errno not in (13, 22) or attempt == 11: raise
                time.sleep(min(.05*2**attempt, 1.))
