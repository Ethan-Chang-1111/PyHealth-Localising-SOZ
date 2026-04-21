"""Synthetic SPES SOZ ablation study example.

This script runs a small ablation with synthetic SPES samples for:
- CNN ResNet (divergent), with/without distance features
- CNN ResNet (convergent), with/without distance features
- CNN Transformer (convergent), with/without distance features

The synthetic data block is intentionally placed at the start so the example is
fully runnable without external dataset downloads.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.models import SPESResNet, SPESTransformer
from pyhealth.trainer import Trainer

N_SAMPLES = 50
MIN_CHANNELS = 2
MAX_CHANNELS = 4
TIMESTEPS = 509
SEED = 2026
BATCH_SIZE = 2
EPOCHS = 5


def generate_synthetic_spes_samples(
    n_samples: int = 12,
    min_channels: int = 3,
    max_channels: int = 6,
    timesteps: int = 509,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create synthetic SPES samples with variable channel counts."""
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 to include both binary labels.")
    if min_channels < 1 or max_channels < min_channels:
        raise ValueError("Invalid channel bounds for synthetic SPES generation.")
    if timesteps < 200:
        raise ValueError("timesteps must be >= 200 for SPES model compatibility.")

    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []

    for idx in range(n_samples):
        n_channels = int(rng.integers(min_channels, max_channels + 1))
        signal = rng.normal(size=(n_channels, 2, timesteps)).astype(np.float32)
        signal[:, 0, 0] = rng.uniform(1.0, 60.0, size=(n_channels,)).astype(np.float32)
        samples.append(
            {
                "patient_id": f"patient-{idx}",
                "visit_id": f"visit-{idx // 2}",
                "spes_responses": signal.tolist(),
                "soz_label": int(idx % 2),
            }
        )
    return samples


def create_synthetic_spes_dataset(
    n_samples: int = 12,
    min_channels: int = 3,
    max_channels: int = 6,
    timesteps: int = 509,
    seed: int = 42,
    dataset_name: str = "synthetic_spes_ablation",
):
    """Build an in-memory SampleDataset for SPES ablation."""
    samples = generate_synthetic_spes_samples(
        n_samples=n_samples,
        min_channels=min_channels,
        max_channels=max_channels,
        timesteps=timesteps,
        seed=seed,
    )
    return create_sample_dataset(
        samples=samples,
        input_schema={"spes_responses": "tensor"},
        output_schema={"soz_label": "binary"},
        dataset_name=dataset_name,
    )


def get_spes_ablation_configs() -> List[Dict[str, Any]]:
    """Return SPES ablation model configurations."""
    return [
        {
            "name": "cnn_resnet_divergent_no_features",
            "model_type": "spes_resnet",
            "paradigm": "divergent",
            "include_distance": False,
        },
        {
            "name": "cnn_resnet_divergent_with_features",
            "model_type": "spes_resnet",
            "paradigm": "divergent",
            "include_distance": True,
        },
        {
            "name": "cnn_resnet_convergent_no_features",
            "model_type": "spes_resnet",
            "paradigm": "convergent",
            "include_distance": False,
        },
        {
            "name": "cnn_resnet_convergent_with_features",
            "model_type": "spes_resnet",
            "paradigm": "convergent",
            "include_distance": True,
        },
        {
            "name": "cnn_transformer_convergent_no_features",
            "model_type": "spes_transformer",
            "paradigm": "convergent",
            "include_distance": False,
        },
        {
            "name": "cnn_transformer_convergent_with_features",
            "model_type": "spes_transformer",
            "paradigm": "convergent",
            "include_distance": True,
        },
    ]


def build_spes_ablation_model(config: Dict[str, Any], dataset):
    """Instantiate SPES ablation model for one config."""
    include_distance = bool(config["include_distance"])
    if config["model_type"] == "spes_resnet":
        return SPESResNet(
            dataset=dataset,
            input_channels=4,
            noise_std=0.0,
            include_distance=include_distance,
        )
    if config["model_type"] == "spes_transformer":
        return SPESTransformer(
            dataset=dataset,
            mean=True,
            std=True,
            conv_embedding=True,
            mlp_embedding=True,
            num_layers=1,
            embedding_dim=16,
            random_channels=2,
            noise_std=0.0,
            include_distance=include_distance,
        )
    raise ValueError(f"Unsupported model_type: {config['model_type']}")


def run_synthetic_spes_ablation() -> List[Dict[str, float]]:
    """Run SPES ablation configurations on tiny synthetic data.

    Returns:
        A list of dicts containing configuration names and evaluation metrics.
    """
    # STEP 0: Generate a small synthetic dataset for this ablation.
    sample_dataset = create_synthetic_spes_dataset(
        n_samples=N_SAMPLES,
        min_channels=MIN_CHANNELS,
        max_channels=MAX_CHANNELS,
        timesteps=TIMESTEPS,
        seed=SEED,
    )

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.6, 0.2, 0.2], seed=SEED
    )
    train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    results: List[Dict[str, float]] = []
    for config in get_spes_ablation_configs():
        model = build_spes_ablation_model(config=config, dataset=sample_dataset)
        trainer = Trainer(model=model, enable_logging=False)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=EPOCHS,
            monitor=None,
        )
        metrics = trainer.evaluate(test_loader)

        # Convert values to float for concise tabular output.
        summary = {"config": config["name"]}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary[key] = float(value)
        results.append(summary)

    return results


if __name__ == "__main__":
    ablation_results = run_synthetic_spes_ablation()
    print("SPES Synthetic SOZ Classification Results")
    for row in ablation_results:
        print(row)

