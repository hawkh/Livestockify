#!/usr/bin/env python3
"""
Comprehensive tests for weight estimation components.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import components to test
from src.models.weight_estimation.distance_adaptive_nn import DistanceAdaptiveWeightNet
from src.models.weight_estimation.feature_extractor import FeatureExtractor
from src.models.weight_estimation.age_classifier import AgeClassifier
from src.models.weight_estimation.weight_validator import WeightValidator
from src.utils.distance.compensation_engine import DistanceCompensationEngine
from src.core.interfaces.weight_estimation import WeightEstimate
from src.core.interfaces.detection import BoundingBox


class TestDistanceAdaptiveWeightNet(unittest.TestCase):
    """Test distance-adaptive weight estimation neural network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = DistanceAdaptiveWeightNet(
            input_channels=3,
            num_age_groups=8,
            hidden_dim=256
        )
        self.batch_size = 4
        
        # Create test inputs
        self.test_image = torch.randn(self.batch_size, 3, 224, 224)
        self.test_distance = torch.rand(self.batch_size, 1) * 5 + 1  # 1-6 meters
        self.test_occlusion = torch.rand(self.batch_size, 1)
        self.test_age_group = torch.randint(0, 8, (self.batch_size,))
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsNotNone(self.model.feature_extractor)
        self.assertIsNotNone(self.model.distance_adapter)
        self.assertIsNotNone(self.model.weight_predictor)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        self.model.eval()
        
        with torch.no_grad():
            weight, bounds = self.model(
                self.test_image,
                self.test_distance,
                self.test_occlusion,
                self.test_age_group
            )
        
        # Check output shapes
        self.assertEqual(weight.shape, (self.batch_size, 1))
        self.assertEqual(bounds.shape, (self.batch_size, 2))
        
        # Check that weights are within bounds
        for i in range(self.batch_size):
            self.assertGreaterEqual(weight[i].item(), bounds[i, 0].item())
            self.assertLessEqual(weight[i].item(), bounds[i, 1].item())
    
    def test_distance_adaptation(self):
        """Test that model adapts to different distances."""
        self.model.eval()
        
        # Test with different distances
        distances = [1.0, 3.0, 5.0]  # Close, medium, far
        weights = []
        
        for dist in distances:
            distance_tensor = torch.full((1, 1), dist)
            
            with torch.no_grad():
                weight, _ = self.model(
                    self.test_image[:1],
                    distance_tensor,
                    self.test_occlusion[:1],
                    self.test_age_group[:1]
                )
            
            weights.append(weight.item())
        
        # Weights should vary with distance (not be identical)
        self.assertFalse(all(abs(w - weights[0]) < 0.01 for w in weights))
    
    def test_age_group_influence(self):
        """Test that age group influences weight prediction."""
        self.model.eval()
        
        # Test with different age groups
        age_groups = [0, 3, 7]  # Young, medium, old
        weights = []
        
        for age in age_groups:
            age_tensor = torch.tensor([age])
            
            with torch.no_grad():
                weight, _ = self.model(
                    self.test_image[:1],
                    self.test_distance[:1],
                    self.test_occlusion[:1],
                    age_tensor
                )
            
            weights.append(weight.item())
        
        # Weights should generally increase with age
        self.assertLess(weights[0], weights[2])  # Young < Old
    
    def test_occlusion_handling(self):
        """Test that model handles occlusion appropriately."""
        self.model.eval()
        
        # Test with different occlusion levels
        occlusion_levels = [0.0, 0.5, 0.9]  # No, medium, high occlusion
        bounds_widths = []
        
        for occlusion in occlusion_levels:
            occlusion_tensor = torch.tensor([[occlusion]])
            
            with torch.no_grad():
                _, bounds = self.model(
                    self.test_image[:1],
                    self.test_distance[:1],
                    occlusion_tensor,
                    self.test_age_group[:1]
                )
            
            width = bounds[0, 1].item() - bounds[0, 0].item()
            bounds_widths.append(width)
        
        # Higher occlusion should lead to wider uncertainty bounds
        self.assertLess(bounds_widths[0], bounds_widths[2])
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        self.model.train()
        
        # Forward pass
        weight, bounds = self.model(
            self.test_image,
            self.test_distance,
            self.test_occlusion,
            self.test_age_group
        )
        
        # Create dummy loss
        target_weight = torch.randn_like(weight)
        loss = nn.MSELoss()(weight, target_weight)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")


class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor(backbone='resnet50')
        self.test_image = torch.randn(2, 3, 224, 224)
    
    def test_feature_extraction(self):
        """Test feature extraction from images."""
        features = self.extractor(self.test_image)
        
        # Check output shape
        self.assertEqual(len(features.shape), 2)  # [batch_size, feature_dim]
        self.assertEqual(features.shape[0], 2)  # Batch size
        self.assertGreater(features.shape[1], 0)  # Feature dimension
    
    def test_different_backbones(self):
        """Test different backbone architectures."""
        backbones = ['resnet50', 'efficientnet_b0']
        
        for backbone in backbones:
            try:
                extractor = FeatureExtractor(backbone=backbone)
                features = extractor(self.test_image)
                self.assertIsNotNone(features)
            except Exception as e:
                self.fail(f"Failed to create extractor with {backbone}: {e}")
    
    def test_feature_consistency(self):
        """Test that same input produces same features."""
        self.extractor.eval()
        
        with torch.no_grad():
            features1 = self.extractor(self.test_image)
            features2 = self.extractor(self.test_image)
        
        # Should be identical for same input
        torch.testing.assert_close(features1, features2)


class TestAgeClassifier(unittest.TestCase):
    """Test age classification component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = AgeClassifier(num_classes=8)
        self.test_features = torch.randn(4, 512)  # Batch of feature vectors
    
    def test_age_classification(self):
        """Test age classification from features."""
        age_probs = self.classifier(self.test_features)
        
        # Check output shape
        self.assertEqual(age_probs.shape, (4, 8))  # [batch_size, num_classes]
        
        # Check that probabilities sum to 1
        prob_sums = torch.sum(age_probs, dim=1)
        torch.testing.assert_close(prob_sums, torch.ones(4), atol=1e-6)
    
    def test_age_prediction(self):
        """Test age group prediction."""
        age_groups = self.classifier.predict_age_group(self.test_features)
        
        # Check output shape and range
        self.assertEqual(age_groups.shape, (4,))
        self.assertTrue(torch.all(age_groups >= 0))
        self.assertTrue(torch.all(age_groups < 8))
    
    def test_confidence_scores(self):
        """Test confidence score calculation."""
        confidences = self.classifier.get_confidence(self.test_features)
        
        # Check output shape and range
        self.assertEqual(confidences.shape, (4,))
        self.assertTrue(torch.all(confidences >= 0))
        self.assertTrue(torch.all(confidences <= 1))


class TestWeightValidator(unittest.TestCase):
    """Test weight validation component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = WeightValidator()
    
    def test_age_based_validation(self):
        """Test age-based weight validation."""
        test_cases = [
            (7, 0.5, True),    # 1 week old, 0.5kg - reasonable
            (7, 2.0, False),   # 1 week old, 2kg - too heavy
            (42, 3.0, True),   # 6 weeks old, 3kg - reasonable
            (42, 0.5, False),  # 6 weeks old, 0.5kg - too light
        ]
        
        for age_days, weight, expected_valid in test_cases:
            is_valid = self.validator.validate_by_age(weight, age_days)
            self.assertEqual(is_valid, expected_valid, 
                           f"Age {age_days} days, weight {weight}kg should be {'valid' if expected_valid else 'invalid'}")
    
    def test_distance_based_validation(self):
        """Test distance-based weight validation."""
        # Closer objects should have more reliable weight estimates
        close_distance = 1.5  # meters
        far_distance = 5.0    # meters
        
        close_confidence = self.validator.get_distance_confidence(close_distance)
        far_confidence = self.validator.get_distance_confidence(far_distance)
        
        self.assertGreater(close_confidence, far_confidence)
        self.assertGreaterEqual(close_confidence, 0.0)
        self.assertLessEqual(close_confidence, 1.0)
    
    def test_occlusion_based_validation(self):
        """Test occlusion-based weight validation."""
        # Less occluded objects should have more reliable estimates
        low_occlusion = 0.1
        high_occlusion = 0.8
        
        low_confidence = self.validator.get_occlusion_confidence(low_occlusion)
        high_confidence = self.validator.get_occlusion_confidence(high_occlusion)
        
        self.assertGreater(low_confidence, high_confidence)
        self.assertGreaterEqual(low_confidence, 0.0)
        self.assertLessEqual(low_confidence, 1.0)
    
    def test_combined_validation(self):
        """Test combined validation score."""
        weight_estimate = WeightEstimate(
            weight=2.5,
            confidence=0.8,
            bounds=(2.0, 3.0),
            age_days=35,
            distance=2.0,
            occlusion_score=0.3
        )
        
        validation_score = self.validator.validate_estimate(weight_estimate)
        
        self.assertGreaterEqual(validation_score, 0.0)
        self.assertLessEqual(validation_score, 1.0)


class TestDistanceCompensationEngine(unittest.TestCase):
    """Test distance compensation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = DistanceCompensationEngine()
        self.test_bbox = BoundingBox(100, 100, 200, 200, 0.9)
    
    def test_distance_estimation(self):
        """Test distance estimation from bounding box."""
        # Larger bounding boxes should indicate closer objects
        large_bbox = BoundingBox(50, 50, 350, 350, 0.9)  # 300x300
        small_bbox = BoundingBox(100, 100, 150, 150, 0.9)  # 50x50
        
        large_distance = self.engine.estimate_distance(large_bbox)
        small_distance = self.engine.estimate_distance(small_bbox)
        
        # Larger bbox should have smaller distance
        self.assertLess(large_distance, small_distance)
        self.assertGreater(large_distance, 0)
        self.assertGreater(small_distance, 0)
    
    def test_feature_compensation(self):
        """Test distance-based feature compensation."""
        features = np.random.randn(512)
        
        # Test compensation at different distances
        close_distance = 1.5
        far_distance = 4.0
        
        close_features = self.engine.compensate_features(features, close_distance)
        far_features = self.engine.compensate_features(features, far_distance)
        
        # Compensated features should be different
        self.assertFalse(np.allclose(close_features, far_features))
        self.assertEqual(close_features.shape, features.shape)
        self.assertEqual(far_features.shape, features.shape)
    
    def test_perspective_correction(self):
        """Test perspective correction for weight estimation."""
        base_weight = 2.5
        
        # Test at different distances
        distances = [1.0, 2.0, 3.0, 4.0, 5.0]
        corrected_weights = []
        
        for distance in distances:
            corrected = self.engine.apply_perspective_correction(base_weight, distance)
            corrected_weights.append(corrected)
        
        # All corrected weights should be positive
        self.assertTrue(all(w > 0 for w in corrected_weights))
        
        # Weights should vary with distance
        self.assertFalse(all(abs(w - corrected_weights[0]) < 0.01 for w in corrected_weights))


class TestWeightEstimationIntegration(unittest.TestCase):
    """Integration tests for weight estimation pipeline."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.model = DistanceAdaptiveWeightNet()
        self.validator = WeightValidator()
        self.compensation_engine = DistanceCompensationEngine()
        
        # Create test data
        self.test_image = torch.randn(1, 3, 224, 224)
        self.test_bbox = BoundingBox(100, 100, 200, 200, 0.9)
    
    def test_full_estimation_pipeline(self):
        """Test complete weight estimation pipeline."""
        # Step 1: Estimate distance from bounding box
        distance = self.compensation_engine.estimate_distance(self.test_bbox)
        
        # Step 2: Prepare model inputs
        distance_tensor = torch.tensor([[distance]])
        occlusion_tensor = torch.tensor([[0.2]])  # Low occlusion
        age_tensor = torch.tensor([4])  # Middle age group
        
        # Step 3: Run model inference
        self.model.eval()
        with torch.no_grad():
            weight, bounds = self.model(
                self.test_image,
                distance_tensor,
                occlusion_tensor,
                age_tensor
            )
        
        # Step 4: Create weight estimate
        weight_estimate = WeightEstimate(
            weight=weight.item(),
            confidence=0.8,
            bounds=(bounds[0, 0].item(), bounds[0, 1].item()),
            age_days=28,  # 4 weeks
            distance=distance,
            occlusion_score=0.2
        )
        
        # Step 5: Validate estimate
        validation_score = self.validator.validate_estimate(weight_estimate)
        
        # Verify results
        self.assertGreater(weight.item(), 0)
        self.assertLess(bounds[0, 0].item(), bounds[0, 1].item())
        self.assertGreaterEqual(validation_score, 0.0)
        self.assertLessEqual(validation_score, 1.0)
    
    def test_batch_processing(self):
        """Test batch processing of multiple chickens."""
        batch_size = 5
        
        # Create batch inputs
        images = torch.randn(batch_size, 3, 224, 224)
        distances = torch.rand(batch_size, 1) * 4 + 1  # 1-5 meters
        occlusions = torch.rand(batch_size, 1) * 0.5   # 0-50% occlusion
        ages = torch.randint(0, 8, (batch_size,))
        
        # Process batch
        self.model.eval()
        with torch.no_grad():
            weights, bounds = self.model(images, distances, occlusions, ages)
        
        # Verify batch results
        self.assertEqual(weights.shape, (batch_size, 1))
        self.assertEqual(bounds.shape, (batch_size, 2))
        
        # All weights should be positive
        self.assertTrue(torch.all(weights > 0))
        
        # All bounds should be valid
        self.assertTrue(torch.all(bounds[:, 0] < bounds[:, 1]))
    
    def test_error_handling(self):
        """Test error handling in weight estimation."""
        # Test with invalid inputs
        invalid_image = torch.randn(1, 3, 50, 50)  # Too small
        distance = torch.tensor([[2.0]])
        occlusion = torch.tensor([[0.3]])
        age = torch.tensor([3])
        
        try:
            with torch.no_grad():
                weight, bounds = self.model(invalid_image, distance, occlusion, age)
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            # Should be a meaningful error message
            self.assertIsInstance(e, (RuntimeError, ValueError))
    
    def test_performance_requirements(self):
        """Test that weight estimation meets performance requirements."""
        import time
        
        # Prepare test data
        batch_size = 10
        images = torch.randn(batch_size, 3, 224, 224)
        distances = torch.rand(batch_size, 1) * 4 + 1
        occlusions = torch.rand(batch_size, 1) * 0.5
        ages = torch.randint(0, 8, (batch_size,))
        
        # Measure inference time
        self.model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):  # Run multiple times for average
                weights, bounds = self.model(images, distances, occlusions, ages)
        
        end_time = time.time()
        avg_time_per_batch = (end_time - start_time) / 10
        avg_time_per_sample = avg_time_per_batch / batch_size
        
        # Should process samples quickly (< 10ms per sample)
        self.assertLess(avg_time_per_sample, 0.01)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDistanceAdaptiveWeightNet))
    test_suite.addTest(unittest.makeSuite(TestFeatureExtractor))
    test_suite.addTest(unittest.makeSuite(TestAgeClassifier))
    test_suite.addTest(unittest.makeSuite(TestWeightValidator))
    test_suite.addTest(unittest.makeSuite(TestDistanceCompensationEngine))
    test_suite.addTest(unittest.makeSuite(TestWeightEstimationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("WEIGHT ESTIMATION TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")