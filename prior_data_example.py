from priors.doscm import get_batch

prior_ds, do_scm = get_batch(
    batch_size=1,
    seq_len=100,
    num_features=3,
    # hyperparams from example
    num_unobserved=1,
    seed=42,
    noise_std=0.01,
    exo_std=0.1,
    zero_one_treatment=True,
    graph=None,
    t_idx=None,
    y_idx=None,
    x_idcs=None,
    inference_cov="pre_interventional",
    test=False,
    # added
    noise_dist="gaussian",
    nonlins="mixed",
    binary_strategy="mean",
)

print(prior_ds)

print(do_scm)
