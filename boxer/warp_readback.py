"""Prefetch arrays for MuJoCo's supported GPU-to-native measurement export.

The upstream exporter issues many separate synchronous device reads. Stage its
read set into pinned host memory, synchronize once, then call the same exporter
through a read-only proxy. No array values, contacts or physics are changed.
"""
import inspect
import re

import mujoco_warp as mjwarp
import warp as wp


class _ArrayView:
    def __init__(self, original, host):
        self.original, self.host = original, host

    def numpy(self):
        return self.host.numpy()

    def __getattr__(self, name):
        return getattr(self.original, name)


class _View:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)


class WarpReadback:
    def __init__(self, data):
        self.data = data
        self.view = _View(data)
        self.copies = []
        source = inspect.getsource(mjwarp.get_data_into)
        paths = sorted(set(re.findall(r'\bd((?:\.[a-zA-Z_]\w*)+)\.numpy\(\)', source)))
        if len(paths) < 20:
            raise RuntimeError('Unrecognized upstream export implementation; inspect before use')
        for path in paths:
            original, view = data, self.view
            names = path[1:].split('.')
            for name in names[:-1]:
                original = getattr(original, name)
                if name not in view.__dict__:
                    setattr(view, name, _View(original))
                view = getattr(view, name)
            array = getattr(original, names[-1])
            host = wp.empty(array.shape, dtype=array.dtype, device='cpu', pinned=True)
            self.copies.append((array, host))
            setattr(view, names[-1], _ArrayView(array, host))

    def export(self, target, model):
        for device, host in self.copies:
            wp.copy(host, device)
        wp.synchronize_device(self.data.qpos.device)
        mjwarp.get_data_into(target, model, self.view)


class SelectedWarpReadback:
    """Persistent pinned views of an explicit read set, refreshed with one sync.

    Host views are overwritten on refresh. A caller retaining a snapshot must
    copy/convert its arrays before the next refresh. No physics fields are written.
    """
    def __init__(self, data, paths, extras=None):
        self.view = _View(data)
        self.device = data.qpos.device
        self.copies = []
        for path in paths:
            original, view = data, self.view
            names = path.split('.')
            for name in names[:-1]:
                original = getattr(original, name)
                if name not in view.__dict__: setattr(view, name, _View(original))
                view = getattr(view, name)
            array = getattr(original, names[-1])
            self._add(view, names[-1], array)
        for name, array in (extras or {}).items(): self._add(self.view, name, array)

    def _add(self, view, name, array):
        host = wp.empty(array.shape, dtype=array.dtype, device='cpu', pinned=True)
        self.copies.append((array, host))
        setattr(view, name, _ArrayView(array, host))

    def refresh(self):
        for device, host in self.copies: wp.copy(host, device)
        wp.synchronize_device(self.device)
        return self.view
