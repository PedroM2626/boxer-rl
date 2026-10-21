import pytest

@pytest.fixture(scope="session")
def env():
    """One shared env: composing and tracing the scene takes about a minute."""
    from boxer.env import BoxingEnv
    return BoxingEnv()


@pytest.fixture(scope="session")
def obs_dim(env):
    return int(env.observation_size)


@pytest.fixture(scope="session")
def level():
    """A spawn-randomisation-free env.

    Tests that place the boxers at an exact distance need the default pose and
    the default separation, which the randomised spawn does not provide.
    """
    from boxer.env import BoxingEnv
    return BoxingEnv(spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
