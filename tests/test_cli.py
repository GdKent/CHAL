"""
Unit tests for CLI interface.

All tests use mocking to avoid actual debate execution.

Tests cover:
- Argument parsing
- Main execution flow (mocked)
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys


# ==============================================
# 1. Argument Parsing Tests
# ==============================================

@pytest.mark.skip(reason="CLI parse_args not yet implemented")
@pytest.mark.unit
@patch('sys.argv', ['debate'])
def test_cli_default_config():
    """Test that CLI uses default config if none specified."""
    from chal.utilities.cli import parse_args

    args = parse_args()

    # Should use default config
    assert args.config is None or args.config == "default"


@pytest.mark.skip(reason="CLI parse_args not yet implemented")
@pytest.mark.unit
@patch('sys.argv', ['debate', '--config', 'custom'])
def test_cli_custom_config():
    """Test loading custom config by name."""
    from chal.utilities.cli import parse_args

    args = parse_args()

    assert args.config == "custom"


@pytest.mark.skip(reason="CLI parse_args not yet implemented")
@pytest.mark.unit
@patch('sys.argv', ['debate', '--config', 'path/to/config.yaml'])
def test_cli_config_file_path():
    """Test loading config from file path."""
    from chal.utilities.cli import parse_args

    args = parse_args()

    assert "config.yaml" in args.config


# ==============================================
# 2. Main Execution Tests (Mocked)
# ==============================================

@pytest.mark.skip(reason="CLI DebateController integration not yet implemented")
@pytest.mark.unit
@patch('chal.utilities.cli.DebateController')
@patch('chal.utilities.cli.load_config')
@patch('sys.argv', ['debate', '--config', 'test'])
def test_cli_main_runs_debate(mock_load_config, mock_controller_class):
    """Test that main executes full debate workflow (mocked)."""
    # Setup mocks
    mock_config = MagicMock()
    mock_load_config.return_value = mock_config

    mock_controller = MagicMock()
    mock_controller_class.return_value = mock_controller

    from chal.utilities.cli import main

    # Should not raise exceptions
    try:
        main()
    except SystemExit:
        pass  # CLI may call sys.exit()


@pytest.mark.skip(reason="CLI DebateController integration not yet implemented")
@pytest.mark.unit
@patch('chal.utilities.cli.DebateController')
@patch('chal.utilities.cli.load_config')
@patch('sys.argv', ['debate'])
def test_cli_main_saves_outputs(mock_load_config, mock_controller_class):
    """Test that main saves all configured outputs (mocked)."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.output.storage_dir = Path("test_storage")
    mock_load_config.return_value = mock_config

    mock_controller = MagicMock()
    mock_controller.run_debate.return_value = {"agents": []}
    mock_controller_class.return_value = mock_controller

    from chal.utilities.cli import main

    try:
        main()
    except SystemExit:
        pass


@pytest.mark.skip(reason="CLI load_config not yet implemented")
@pytest.mark.unit
@patch('chal.utilities.cli.load_config')
@patch('sys.argv', ['debate'])
def test_cli_main_handles_errors(mock_load_config):
    """Test graceful error handling in main (mocked)."""
    # Setup mock to raise error
    mock_load_config.side_effect = FileNotFoundError("Config not found")

    from chal.utilities.cli import main

    # Should handle error gracefully (not crash)
    with pytest.raises((FileNotFoundError, SystemExit)):
        main()
