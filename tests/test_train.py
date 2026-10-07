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


@pytest.mark.parametrize("total,evals,observed", [
    # (requested --total-steps, --num-evals) -> the last step each run actually
    # reached. All three are real runs from this project, which is how the
    # quantisation was found: the horizon is set by the eval count, not by the
    # number typed into --total-steps.
    (10_000_000, 20, 12_451_840),   # frozen arm
    (20_000_000, 20, 24_903_680),   # symmetric self-play arm
    (4_194_304, 16, 9_830_400),     # a league corner: asked 4.19M, ran 9.83M
])
def test_total_steps_is_quantised_by_the_eval_count(total, evals, observed):
    args = train._parse_args([
        "--total-steps", str(total), "--num-evals", str(evals),
        "--batch-size", "4096", "--unroll-length", "20", "--num-minibatches", "8",
    ])
    block, horizon = train.eval_schedule(args)
    assert block == 4096 * 20 * 8
    assert horizon == observed


def test_a_short_run_needs_fewer_evals_not_a_smaller_total(tmp_path):
    """Below one block per checkpoint, shrinking --total-steps does nothing."""
    args = train._parse_args([
        "--total-steps", "1000", "--num-evals", "20",
        "--batch-size", "4096", "--unroll-length", "20", "--num-minibatches", "8",
    ])
    block, horizon = train.eval_schedule(args)
    assert horizon == 19 * block, "the request is ignored entirely at this size"


def test_restore_path_rejects_a_missing_resume_target(tmp_path):
    args = train._parse_args(["--output", str(tmp_path), "--resume"])
    with pytest.raises(SystemExit):
        train._restore_path(args, tmp_path / "checkpoint")


def test_init_from_without_weights_to_take_fails_loudly(tmp_path):
    (tmp_path / "empty").mkdir()
    args = train._parse_args(["--output", str(tmp_path / "learner"),
                              "--init-from", str(tmp_path / "empty")])
    with pytest.raises(SystemExit):
        train._restore_path(args, tmp_path / "learner" / "checkpoint")
