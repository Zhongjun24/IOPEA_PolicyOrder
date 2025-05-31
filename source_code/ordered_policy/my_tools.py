import numpy as np 


def make_rng_factory(seed_seq):
    def next_rng():
        child_seed = seed_seq.spawn(1)[0]
        return np.random.default_rng(child_seed)
    return next_rng