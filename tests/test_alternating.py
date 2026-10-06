"""The alternating best-response league driver, without running any training."""

import json

import pytest

from boxer import alternating


def _fake_league(root, side, *steps):
    directory = root / side / "checkpoint"
    for step in steps:
        (directory / step).mkdir(parents=True)
    return directory


@pytest.fixture
def args(tmp_path):
    return alternating._parse_args([
        "--output", str(tmp_path), "--iterations", "2",
        "--steps-per-iter", "1000", "--worlds", "8",
        "--batch-size", "16", "--num-minibatches", "2",
    ])


def test_a_fresh_corner_neither_resumes_nor_fights_a_missing_checkpoint(args, tmp_path):
    argv = alternating.command(args, "red", 0)
    assert "--resume" not in argv, "iteration 0 has no weights to restore"
    assert argv[argv.index("--opponent") + 1] == "passive"


def test_a_corner_fights_the_other_nets_current_checkpoint_once_it_exists(args, tmp_path):
    _fake_league(tmp_path, "blue", "0000000100", "0000000300")
    argv = alternating.command(args, "red", 1)
    opponent = tmp_path / "blue" / "checkpoint"
    assert argv[argv.index("--opponent") + 1] == str(opponent)
    assert "--resume" not in argv, "red itself is still untrained"


def test_a_trained_corner_resumes(args, tmp_path):
    _fake_league(tmp_path, "red", "0000001000")
    argv = alternating.command(args, "red", 1)
    assert "--resume" in argv
    # Brax resets env_steps on restore, so the horizon is per invocation: the
    # same number every iteration, not a growing total.
    assert argv[argv.index("--total-steps") + 1] == "1000"


def test_each_corner_keeps_its_own_side_and_output(args, tmp_path):
    for side, expected in (("red", "red"), ("blue", "blue")):
        argv = alternating.command(args, side, 0)
        assert argv[argv.index("--learner-side") + 1] == expected
        assert argv[argv.index("--mode") + 1] == "frozen"
        assert argv[argv.index("--output") + 1] == str(tmp_path / side)


def test_roster_reports_the_newest_step_of_each_corner(tmp_path):
    _fake_league(tmp_path, "red", "0000000100", "0000000250")
    league = alternating.roster(tmp_path)
    assert [row["side"] for row in league] == ["red", "blue"]
    assert league[0]["steps"] == 250
    assert league[1]["steps"] is None


def test_a_seed_checkpoint_is_offered_to_every_untrained_corner(args, tmp_path):
    """Without weights of its own, a corner fighting a trained boxer just loses."""
    argv = alternating.command(args, "red", 0)
    assert "--init-from" not in argv
    seeded = alternating._parse_args([
        "--output", str(tmp_path), "--seed-checkpoint", str(tmp_path / "frozen"),
    ])
    argv = alternating.command(seeded, "red", 0)
    assert argv[argv.index("--init-from") + 1] == str(tmp_path / "frozen")


def test_main_refuses_a_batch_that_cannot_split_across_learners(tmp_path):
    argv = ["--output", str(tmp_path), "--worlds", "6", "--batch-size", "16",
            "--num-minibatches", "2", "--dry-run"]
    with pytest.raises(SystemExit):
        alternating.main(argv)


def test_dry_run_walks_every_corner_of_every_iteration(tmp_path, capsys):
    assert alternating.main([
        "--output", str(tmp_path), "--iterations", "3", "--worlds", "8",
        "--batch-size", "16", "--num-minibatches", "2", "--dry-run",
    ]) == 0
    printed = capsys.readouterr().out
    assert printed.count("=== iteration") == 6
    for iteration in range(3):
        for side in ("red", "blue"):
            assert f"iteration {iteration} / corner {side}" in printed
