"""Archiving must survive Windows locks without rewriting immutable evidence."""
import errno
from pathlib import Path
from boxer.result_archive import archive_tree
import boxer.result_archive as archive


def test_archive_retries_transient_access_failure_and_keeps_identical_files(tmp_path, monkeypatch):
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.mkdir(); destination.mkdir()
    (source/'same.npz').write_bytes(b'immutable source')
    (destination/'same.npz').write_bytes(b'immutable source')
    (source/'new.npz').write_bytes(b'new evidence')
    calls = []
    original = archive.atomic_bytes
    def locked_once(path, payload):
        calls.append(Path(path).name)
        if len(calls) == 1: raise OSError(errno.EACCES, 'simulated Windows reader')
        original(path, payload)
    monkeypatch.setattr(archive, 'atomic_bytes', locked_once)
    monkeypatch.setattr(archive.time, 'sleep', lambda _: None)
    archive_tree(source, destination)
    assert calls == ['new.npz', 'new.npz']
    assert (destination/'same.npz').read_bytes() == b'immutable source'
    assert (destination/'new.npz').read_bytes() == b'new evidence'
