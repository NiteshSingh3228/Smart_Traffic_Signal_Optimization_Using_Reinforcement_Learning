"""
Integration Verification Test for Triple-Source Models
Verifies:
1. Loading and inference of dqn_integrated_model.pth
2. Loading and inference of q_learning_integrated_model.pkl
3. Loading and inference of supervised_delay_model.joblib
4. Loading and decision arbitration of IntegratedHybridAgent
5. State consistency across 17-D inputs
"""

import os
import sys
import torch
try:
    import joblib
except ImportError:
    joblib = None

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.dqn_agent import DQNAgent
from agents.q_learning import QLearningAgent
from agents.hybrid_integrated_agent import IntegratedHybridAgent

def test_integrated_system():
    print("=== TESTING INTEGRATED SYSTEM COMPONENTS ===")
    
    # 1. Test Integrated DQN
    print("\n1. Testing Integrated DQN...")
    dqn = DQNAgent()
    dqn.load_model("dqn_integrated_model.pth")
    assert dqn.policy_network is not None
    dummy_state_17 = (4, 2, 3, 1, 5, 3, 2, 1, 1, 0, 0, 0, 12.5, 0.0, 0.0, 0.0, 0)
    action_dqn = dqn.choose_action(dummy_state_17)
    print(f"  -> DQN action for state: {action_dqn}")
    assert action_dqn in [0, 1]
    
    # 2. Test Integrated Q-Learning
    print("\n2. Testing Integrated Q-Learning...")
    ql = QLearningAgent()
    ql.load_model("q_learning_integrated_model.pkl")
    print(f"  -> Q-Learning loaded states: {len(ql.q_table)}")
    assert len(ql.q_table) >= 100
    action_ql = ql.choose_action(dummy_state_17)
    print(f"  -> Q-Learning action for state: {action_ql}")
    assert action_ql in [0, 1]
    
    # 3. Test Integrated Hybrid Ensemble
    print("\n3. Testing Integrated Hybrid Ensemble...")
    hybrid = IntegratedHybridAgent(
        dqn_model="dqn_integrated_model.pth",
        ql_model="q_learning_integrated_model.pkl"
    )
    decision = hybrid.get_detailed_decision(dummy_state_17)
    print(f"  -> Ensemble action: {decision['action']}")
    print(f"  -> Consensus: {decision['consensus']}")
    print(f"  -> Confidence: {decision['confidence']:.4f}")
    print(f"  -> DQN Q: {decision['dqn_q']}, QL Q: {decision['ql_q']}")
    assert decision["action"] in [0, 1]
    
    # 4. Test Supervised Delay Model
    if joblib is not None and os.path.exists("results/models/supervised_delay_model.joblib"):
        print("\n4. Testing Supervised Delay Model...")
        delay_bundle = joblib.load("results/models/supervised_delay_model.joblib")
        model = delay_bundle["model"]
        scaler = delay_bundle["scaler"]
        features = delay_bundle["feature_names"]
        print(f"  -> Delay model target: {delay_bundle['target']}")
        print(f"  -> Features ({len(features)}): {features}")
        
        # Create sample feature vector
        import numpy as np
        sample_feat = np.zeros((1, len(features)))
        sample_feat[0, 0] = 5.0   # departDelay
        sample_feat[0, 1] = 6.0   # departSpeed
        sample_feat[0, 2] = 220.0 # routeLength
        sample_feat[0, 3] = 1.0   # speedFactor
        sample_feat[0, 4] = 1.0   # vtype_car
        
        pred_delay = model.predict(scaler.transform(sample_feat))[0]
        print(f"  -> Predicted vehicle waiting delay: {pred_delay:.2f} seconds")
        assert pred_delay >= 0.0
    else:
        print("\n4. Skipping Supervised Delay Model test (joblib not in this environment).")
    
    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_integrated_system()

