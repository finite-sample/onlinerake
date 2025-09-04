"""Comprehensive tests for the onlinerake package."""

import pytest
import numpy as np
from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets


class TestTargets:
    """Test the Targets dataclass."""
    
    def test_default_targets(self):
        """Test default target values."""
        targets = Targets()
        assert targets.age == 0.5
        assert targets.gender == 0.5
        assert targets.education == 0.4
        assert targets.region == 0.3
    
    def test_custom_targets(self):
        """Test custom target values."""
        targets = Targets(age=0.6, gender=0.4, education=0.7, region=0.2)
        assert targets.age == 0.6
        assert targets.gender == 0.4
        assert targets.education == 0.7
        assert targets.region == 0.2
    
    def test_as_dict(self):
        """Test conversion to dictionary."""
        targets = Targets(age=0.6, gender=0.4, education=0.7, region=0.2)
        target_dict = targets.as_dict()
        expected = {"age": 0.6, "gender": 0.4, "education": 0.7, "region": 0.2}
        assert target_dict == expected


class TestOnlineRakingSGD:
    """Test the SGD-based online raking algorithm."""
    
    def test_initialization(self):
        """Test proper initialization of SGD raker."""
        targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0)
        
        assert raker.targets == targets
        assert raker.learning_rate == 1.0
        assert raker._n_obs == 0
        assert len(raker.weights) == 0
    
    def test_single_observation(self):
        """Test processing a single observation."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)
        
        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)
        
        assert raker._n_obs == 1
        assert len(raker.weights) == 1
        assert raker.weights[0] > 0
    
    def test_multiple_observations(self):
        """Test processing multiple observations."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)
        
        observations = [
            {"age": 1, "gender": 0, "education": 1, "region": 0},
            {"age": 0, "gender": 1, "education": 0, "region": 1},
            {"age": 1, "gender": 1, "education": 1, "region": 1},
        ]
        
        for obs in observations:
            raker.partial_fit(obs)
        
        assert raker._n_obs == 3
        assert len(raker.weights) == 3
        assert all(w > 0 for w in raker.weights)
    
    def test_margins_property(self):
        """Test that margins are computed correctly."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)
        
        # Add observations with known demographics
        observations = [
            {"age": 1, "gender": 1, "education": 1, "region": 1},  # all 1s
            {"age": 0, "gender": 0, "education": 0, "region": 0},  # all 0s
        ]
        
        for obs in observations:
            raker.partial_fit(obs)
        
        margins = raker.margins
        assert "age" in margins
        assert "gender" in margins
        assert "education" in margins
        assert "region" in margins
        
        # With equal weights, margins should be 0.5 for each category
        for margin in margins.values():
            assert 0 <= margin <= 1
    
    def test_effective_sample_size(self):
        """Test effective sample size calculation."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)
        
        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)
        
        ess = raker.effective_sample_size
        assert ess > 0
        assert ess <= raker._n_obs
    
    def test_loss_property(self):
        """Test that loss is computed correctly."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)
        
        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)
        
        loss = raker.loss
        assert loss >= 0


class TestOnlineRakingMWU:
    """Test the MWU-based online raking algorithm."""
    
    def test_initialization(self):
        """Test proper initialization of MWU raker."""
        targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
        raker = OnlineRakingMWU(targets, learning_rate=1.0)
        
        assert raker.targets == targets
        assert raker.learning_rate == 1.0
        assert raker._n_obs == 0
        assert len(raker.weights) == 0
    
    def test_single_observation(self):
        """Test processing a single observation with MWU."""
        targets = Targets()
        raker = OnlineRakingMWU(targets, learning_rate=1.0)
        
        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)
        
        assert raker._n_obs == 1
        assert len(raker.weights) == 1
        assert raker.weights[0] > 0
    
    def test_weight_clipping(self):
        """Test weight clipping functionality."""
        targets = Targets()
        raker = OnlineRakingMWU(targets, learning_rate=10.0, min_weight=0.1, max_weight=10.0)
        
        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)
        
        assert raker.weights[0] >= 0.1
        assert raker.weights[0] <= 10.0


class TestRealisticScenarios:
    """Test with realistic survey scenarios."""
    
    def test_gender_bias_correction(self):
        """Test correcting gender bias in a stream."""
        # US population is roughly 51% female
        targets = Targets(age=0.5, gender=0.51, education=0.4, region=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=2.0)
        
        # Simulate a biased stream (70% male respondents)
        np.random.seed(42)
        n_obs = 100
        biased_observations = []
        
        for i in range(n_obs):
            # 70% chance of male (gender=0), 30% chance of female (gender=1)
            gender = 1 if np.random.random() < 0.3 else 0
            obs = {
                "age": np.random.choice([0, 1]),
                "gender": gender,
                "education": np.random.choice([0, 1]),
                "region": np.random.choice([0, 1])
            }
            biased_observations.append(obs)
            raker.partial_fit(obs)
        
        # Check that gender margin is closer to target after raking
        final_margins = raker.margins
        raw_gender_prop = sum(obs["gender"] for obs in biased_observations) / n_obs
        
        # Raw proportion should be around 0.3 (biased)
        assert 0.25 <= raw_gender_prop <= 0.35
        
        # Weighted margin should be closer to target 0.51
        gender_error_raw = abs(raw_gender_prop - 0.51)
        gender_error_weighted = abs(final_margins["gender"] - 0.51)
        assert gender_error_weighted < gender_error_raw
    
    def test_education_bias_correction(self):
        """Test correcting education bias."""
        # Target: 40% have higher education
        targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=3.0)
        
        # Simulate over-educated sample (60% have higher education)
        np.random.seed(123)
        n_obs = 150
        
        for i in range(n_obs):
            education = 1 if np.random.random() < 0.6 else 0
            obs = {
                "age": np.random.choice([0, 1]),
                "gender": np.random.choice([0, 1]),
                "education": education,
                "region": np.random.choice([0, 1])
            }
            raker.partial_fit(obs)
        
        final_margins = raker.margins
        # Weighted education margin should be closer to 0.4
        education_error = abs(final_margins["education"] - 0.4)
        assert education_error < 0.15  # Should be reasonably close


def test_sgd_vs_mwu_comparison():
    """Compare SGD and MWU on the same stream."""
    targets = Targets(age=0.6, gender=0.5, education=0.3, region=0.4)
    
    sgd_raker = OnlineRakingSGD(targets, learning_rate=3.0)
    mwu_raker = OnlineRakingMWU(targets, learning_rate=1.0)
    
    # Generate a biased stream
    np.random.seed(456)
    observations = []
    
    for i in range(200):
        # Age bias: 80% young people
        age = 1 if np.random.random() < 0.2 else 0
        obs = {
            "age": age,
            "gender": np.random.choice([0, 1]),
            "education": np.random.choice([0, 1]),
            "region": np.random.choice([0, 1])
        }
        observations.append(obs)
        
        sgd_raker.partial_fit(obs)
        mwu_raker.partial_fit(obs)
    
    # Both should improve age margin compared to raw data
    raw_age_prop = sum(obs["age"] for obs in observations) / len(observations)
    sgd_age_margin = sgd_raker.margins["age"]
    mwu_age_margin = mwu_raker.margins["age"]
    
    raw_age_error = abs(raw_age_prop - targets.age)
    sgd_age_error = abs(sgd_age_margin - targets.age)
    mwu_age_error = abs(mwu_age_margin - targets.age)
    
    # Both algorithms should reduce the bias
    assert sgd_age_error < raw_age_error
    assert mwu_age_error < raw_age_error
    
    # Both should maintain reasonable effective sample sizes
    assert sgd_raker.effective_sample_size > 50
    assert mwu_raker.effective_sample_size > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])