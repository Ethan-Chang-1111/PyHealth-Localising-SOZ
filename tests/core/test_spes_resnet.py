import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SPESResNet


class TestSPESResNet(unittest.TestCase):
    """Test cases for the SPESResNet model."""

    def setUp(self):
        """Set up test data and model."""
        # Create minimal synthetic data: [C, 2, T+1] where T=155 so T+1=156
        # Distances are at index 0 of the time dimension.
        # We give one sample 5 channels and another 3 channels to test padding.
        
        sample0_tensor = torch.randn(5, 2, 156)
        sample1_tensor = torch.randn(3, 2, 156)
        
        # Inject positive distances so valid channels are identified
        sample0_tensor[:, 0, 0] = torch.abs(torch.randn(5)) + 1.0
        sample1_tensor[:, 0, 0] = torch.abs(torch.randn(3)) + 1.0

        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "spes_responses": sample0_tensor.tolist(),
                "soz_label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "spes_responses": sample1_tensor.tolist(),
                "soz_label": 0,
            },
        ]

        self.input_schema = {
            "spes_responses": "tensor",
        }
        self.output_schema = {"soz_label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_spes",
        )

        # Make input_channels smaller for fast tests since we only have max 5 channels
        self.model = SPESResNet(
            dataset=self.dataset,
            input_channels=2,
            noise_std=0.1
        )

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, SPESResNet)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("spes_responses", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "soz_label")
        self.assertEqual(self.model.input_channels, 2)
        self.assertEqual(self.model.mode, "binary")

    def test_model_forward(self):
        """Test that the model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_empty_distances_fallback(self):
        """Test the fallback mechanism when distances are missing (all zeros)."""
        # Create data with zero distances to ensure the std-based fallback trigger works.
        sample0_tensor = torch.randn(5, 2, 156)
        # Clear distances
        sample0_tensor[:, 0, 0] = 0.0
        
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "spes_responses": sample0_tensor.tolist(),
                "soz_label": 1,
            }
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_spes_empty",
        )
        model = SPESResNet(dataset=dataset, input_channels=2)
        train_loader = get_dataloader(dataset, batch_size=1, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)
            
        self.assertEqual(ret["logit"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
