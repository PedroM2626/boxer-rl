"""Which checkpoint ``boxer.train`` loads weights from, and what it must never do."""

import pytest

from boxer import train


def _seed(root, *steps):
    """Writes the orbax layout a real run leaves behind."""
    directory = root / "checkpoint"
    for step in steps:
        (directory / step).mkdir(parents=True)
    return directory


def test_no_resume_and_no_seed_starts_random(tmp_path):
    args = train._parse_args(["--output", str(tmp_path)])
    assert train._restore_path(args, tmp_path / "checkpoint") is None


def test_init_from_names_the_newest_step_of_the_seed(tmp_path):
    seed = tmp_path / "donor"
    _seed(seed, "0000000100", "0000000900", "0000000500")
    args = train._parse_args(["--output", str(tmp_path / "learner"),
                              "--init-from", str(seed)])
    assert train._restore_path(args, tmp_path / "learner" / "checkpoint").endswith("0000000900")


def test_init_from_accepts_a_run_root(tmp_path):
    """A person types the run directory; the weights live one level deeper."""
    _seed(tmp_path / "donor", "0000000300")
    args = train._parse_args(["--output", str(tmp_path / "learner"),
                              "--init-from", str(tmp_path / "donor")])
    assert train._restore_path(args, tmp_path / "learner" / "checkpoint").endswith("0000000300")


def test_a_corner_that_already_has_weights_keeps_them(tmp_path, capsys):
    learner = tmp_path / "learner"
    _seed(learner, "0000002000")
    _seed(tmp_path / "donor", "0000000100")
    args = train._parse_args(["--output", str(learner), "--init-from", str(tmp_path / "donor")])
    assert train._restore_path(args, learner / "checkpoint").endswith("0000002000"), (
        "resetting to the seed every iteration would make a league loop retrain the "
        "same few steps for ever"
    )
    assert "already has weights" in capsys.readouterr().out


def test_a_relative_seed_is_resolved(tmp_path, monkeypatch):
    """orbax refuses a relative directory, and callers type relative paths.

    This is the failure mode that killed the first league run: the seed resolved
    fine as a step directory and then died inside orbax, hours of compile away
    from any training step.
    """
    _seed(tmp_path / "donor", "0000000300")
    monkeypatch.chdir(tmp_path)
    args = train._parse_args(["--output", "learner", "--init-from", "donor/checkpoint"])
    path = train._restore_path(args, tmp_path / "learner" / "checkpoint")
    assert path.endswith("0000000300")
    # Windows keeps the drive letter in os.path.isabs, so compare to the resolved
    # form instead of asking what a POSIX absolute path looks like.
    assert path == str((tmp_path / "donor" / "checkpoint" / "0000000300").resolve())


def test_resume_without_a_checkpoint_fails_loudly(tmp_path):
    args = train._parse_args(["--output", str(tmp_path), "--resume"])
    with pytest.raises(SystemExit):
        train._restore_path(args, tmp_path / "checkpoint")


def test_init_from_without_weights_to_take_fails_loudly(tmp_path):
    (tmp_path / "empty").mkdir()
    args = train._parse_args(["--output", str(tmp_path / "learner"),
                              "--init-from", str(tmp_path / "empty")])
    with pytest.raises(SystemExit):
        train._restore_path(args, tmp_path / "learner" / "checkpoint")
