import torch
from ddpm.utils.metrics import compute_fid, load_inception_model, get_inception_features


def test_inception_features_shape():
    device = "cpu"
    model = load_inception_model(device=device)

    images = torch.rand(4, 3, 32, 32) * 2 - 1  # random images in [-1, 1]
    features = get_inception_features(images, model, device=device)

    assert features.shape == (4, 2048)
    print(f"Feature shape: {features.shape}")
    print("[OK] InceptionV3 produces correctly shaped (N, 2048) feature vectors")


def test_full_fid_pipeline_same_images():
    # Same images through the SAME feature extractor should give FID ~0
    device = "cpu"
    model = load_inception_model(device=device)

    images = torch.rand(8, 3, 32, 32) * 2 - 1
    features_a = get_inception_features(images, model, device=device)
    features_b = get_inception_features(images.clone(), model, device=device)

    fid = compute_fid(features_a, features_b)
    print(f"FID (identical images through full pipeline): {fid:.6f}")
    assert fid < 1.0, "FID should be very small for identical images"
    print("[OK] Full pipeline (images -> features -> FID) works correctly")


if __name__ == "__main__":
    print("===== METRICS TEST (INCEPTION) =====")
    test_inception_features_shape()
    test_full_fid_pipeline_same_images()
    print("ALL TESTS PASSED")