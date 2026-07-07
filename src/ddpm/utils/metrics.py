"""
Fréchet Inception Distance (FID): measures how close the distribution of
generated images is to real images, by comparing statistics (mean,
covariance) of features extracted from a pretrained InceptionV3 network.
Lower FID = generated images are statistically closer to real ones.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r @ Sigma_g))

InceptionV3 is used as a fixed, pretrained feature extractor rather than
reimplemented from scratch — it's a standard, borrowed tool (like
torchvision's CIFAR10 loader), not something specific to diffusion models.
"""

import numpy as np
from scipy.linalg import sqrtm


def compute_fid(real_features, fake_features):
    """
    Args:
        real_features, fake_features: (N, D) — feature vectors from a
            pretrained InceptionV3, one row per image
    Returns:
        A single float: the FID score (0 = identical distributions).
    """
    real_features = real_features.numpy()
    fake_features = fake_features.numpy()

    mean_r = np.mean(real_features, axis=0)
    mean_f = np.mean(fake_features, axis=0)

    cov_r = np.cov(real_features, rowvar=False)
    cov_f = np.cov(fake_features, rowvar=False)

    dist = np.sum((mean_r - mean_f) ** 2)

    # sqrtm can return tiny spurious imaginary components due to floating
    # point precision in its eigendecomposition; the true result should
    # always be real, so we discard the imaginary part explicitly.
    covmean = sqrtm(cov_r @ cov_f).real
    trace_term = np.trace(cov_r + cov_f - 2 * covmean)

    fid = dist + trace_term
    return float(fid)