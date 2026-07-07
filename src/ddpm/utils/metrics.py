"""
Fréchet Inception Distance (FID): measures how close the distribution of
generated images is to real images, by comparing statistics (mean,
covariance) of features extracted from a pretrained InceptionV3 network.
Lower FID = generated images are statistically closer to real ones.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r @ Sigma_g))

InceptionV3 is used as a fixed, pretrained feature extractor rather than
reimplemented from scratch — it's a standard, borrowed tool (like
torchvision's CIFAR10 loader), not something specific to diffusion models.

Note: FID requires a substantial sample size to be numerically stable and
meaningful (typically thousands of images for both real and generated
sets) — with too few samples, the covariance matrices become singular and
the result becomes unreliable.
"""

import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from ddpm.datasets.transforms import unnormalize_to_zero_to_one


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


def load_inception_model(device="cpu"):
    """
    Loads a pretrained InceptionV3, configured to output pooled features
    (2048-dim) rather than classification logits.
    """
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def get_inception_features(images, model, device="cpu"):
    """
    Args:
        images: (N, 3, H, W), values in [-1, 1] (DDPM's normalization)
    Returns:
        (N, 2048) feature tensor
    """
    images_01 = unnormalize_to_zero_to_one(images)
    images_resized = F.interpolate(images_01, size=(299, 299), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images_normalized = (images_resized - mean) / std

    with torch.no_grad():
        features = model(images_normalized)

    return features