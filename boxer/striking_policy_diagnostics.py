"""Pure read-only distribution measurements; no policy or optimizer mutation."""
import numpy as np


def gaussian_kl(mean_a, log_std_a, mean_b, log_std_b):
    """Elementwise KL(a || b); a bijective tanh leaves joint KL unchanged."""
    a, sa, b, sb = [np.asarray(x, dtype=np.float64) for x in (mean_a,log_std_a,mean_b,log_std_b)]
    if not all(np.isfinite(x).all() for x in (a,sa,b,sb)):
        raise FloatingPointError('nonfinite policy distribution')
    return np.maximum(0., sb-sa + .5*(np.exp(2*(sa-sb)) + (a-b)**2*np.exp(-2*sb)-1))
