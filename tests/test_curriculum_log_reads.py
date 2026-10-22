"""Transient log reads must not discard physics warnings or mask persistent faults."""
import errno
from io import BytesIO

import pytest

from boxer import curriculum_job as job


def test_transient_read_reopens_at_original_offset_without_skipping_warning(monkeypatch):
    data = b"historical\nnefc overflow - increase njmax\n"
    opened, delays = [], []
    class PartialFailure(BytesIO):
        def read(self, *args):
            super().read(5)  # A failed read may have consumed bytes internally.
            raise OSError(getattr(errno, "ENODATA", 61), "No data available")
    class Log:
        def open(self, mode):
            assert mode == "rb"
            reader = PartialFailure(data) if not opened else BytesIO(data)
            opened.append(reader)
            return reader
    monkeypatch.setattr(job.time, "sleep", delays.append)
    chunk, offset = job.read_log_chunk(Log(), len(b"historical\n"))
    assert chunk == b"nefc overflow - increase njmax\n"
    assert offset == len(data)
    assert len(opened) == 2 and delays == [.1]


@pytest.mark.parametrize("code,attempts", [(getattr(errno, "ENODATA", 61), 3), (errno.EACCES, 1)])
def test_persistent_and_unrelated_errors_fail_closed(monkeypatch, code, attempts):
    opened, delays = [], []
    class Log:
        def open(self, mode):
            opened.append(mode)
            raise OSError(code, "injected read failure")
    monkeypatch.setattr(job.time, "sleep", delays.append)
    with pytest.raises(OSError) as captured:
        job.read_log_chunk(Log(), 123)
    assert captured.value.errno == code
    assert len(opened) == attempts
    assert len(delays) == attempts - 1
