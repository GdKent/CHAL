#!/usr/bin/env python
"""
test_config.py

Quick test script to verify configuration system works.
Run this after installing pyyaml.

Usage:
    python -m pytest tests/test_config.py
    # or
    python tests/test_config.py
"""

import sys
from chal.config import load_config

def test_default_config():
    """Test loading default configuration."""
    print("Testing default configuration...")
    config = load_config('default')

    assert config.name == "Default Debate", f"Expected 'Default Debate', got '{config.name}'"
    assert config.topic == "Does free will exist?", f"Wrong topic: {config.topic}"
    assert len(config.agents) == 2, f"Expected 2 agents, got {len(config.agents)}"
    assert config.max_rounds == 1, f"Expected 1 round, got {config.max_rounds}"
    assert config.agents[0].name == "Agent-Empiricist"
    assert config.agents[1].name == "Agent-Supernaturalist"
    assert config.adjudication.logic_weight == 1.0
    assert config.outputs.save_synthesis == True
    assert config.scribe.enabled == True

    print(f"✓ Config loaded successfully: {config.name}")
    print(f"  Topic: {config.topic}")
    print(f"  Agents: {', '.join(a.name for a in config.agents)}")
    print(f"  Max rounds: {config.max_rounds}")
    print(f"  Storage: {config.outputs.storage_dir}")

def test_quick_test_config():
    """Test loading quick_test configuration."""
    print("\nTesting quick_test configuration...")
    config = load_config('quick_test')

    assert config.name == "Quick Test"
    assert len(config.agents) == 2
    assert config.outputs.save_synthesis == False  # Disabled in quick_test
    assert config.scribe.enabled == False  # Disabled in quick_test

    print(f"✓ Config loaded successfully: {config.name}")
    print(f"  Topic: {config.topic}")
    print(f"  Scribe enabled: {config.scribe.enabled}")
    print(f"  Save synthesis: {config.outputs.save_synthesis}")

def test_storage_dir_creation():
    """Test that storage directory creation works."""
    print("\nTesting storage directory creation...")
    config = load_config('default')
    config.outputs.ensure_storage_dir()

    assert config.outputs.storage_dir.exists(), "Storage directory was not created"
    print(f"✓ Storage directory exists: {config.outputs.storage_dir}")

if __name__ == "__main__":
    try:
        test_default_config()
        test_quick_test_config()
        test_storage_dir_creation()
        print("\n" + "=" * 60)
        print("✅ All configuration tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
