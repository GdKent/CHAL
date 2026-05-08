"""
Unit tests for the CHAL CLI entry point (main.py).

Tests cover:
- Argument parsing (parse_args)
- Banner display (show_banner)
- Main function routing (headless, wizard, edit modes)
"""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from rich.console import Console

from chal.cli.main import parse_args, show_banner, main, get_version


# ==============================================
# 1. Version
# ==============================================

class TestGetVersion:

    @pytest.mark.unit
    def test_returns_string(self):
        """get_version() returns a non-empty string."""
        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0


# ==============================================
# 2. Argument Parsing
# ==============================================

class TestParseArgs:

    @pytest.mark.unit
    def test_no_args(self):
        """No arguments -> config is None, edit is False."""
        args = parse_args([])
        assert args.config is None
        assert args.edit is False
        assert args.verbose is False

    @pytest.mark.unit
    def test_config_name(self):
        """--config default sets config to 'default'."""
        args = parse_args(["--config", "default"])
        assert args.config == "default"
        assert args.edit is False

    @pytest.mark.unit
    def test_config_short_flag(self):
        """-c works as short form of --config."""
        args = parse_args(["-c", "quick_test"])
        assert args.config == "quick_test"

    @pytest.mark.unit
    def test_config_file_path(self):
        """--config accepts a file path."""
        args = parse_args(["--config", "path/to/my_debate.yaml"])
        assert args.config == "path/to/my_debate.yaml"

    @pytest.mark.unit
    def test_edit_flag(self):
        """--edit flag is captured."""
        args = parse_args(["--config", "default", "--edit"])
        assert args.edit is True

    @pytest.mark.unit
    def test_edit_short_flag(self):
        """-e works as short form of --edit."""
        args = parse_args(["-c", "default", "-e"])
        assert args.edit is True

    @pytest.mark.unit
    def test_verbose_flag(self):
        """--verbose flag is captured."""
        args = parse_args(["--verbose"])
        assert args.verbose is True

    @pytest.mark.unit
    def test_verbose_short_flag(self):
        """-v works as short form of --verbose."""
        args = parse_args(["-v"])
        assert args.verbose is True

    @pytest.mark.unit
    def test_all_flags(self):
        """All flags together."""
        args = parse_args(["-c", "default", "-e", "-v"])
        assert args.config == "default"
        assert args.edit is True
        assert args.verbose is True


# ==============================================
# 3. Banner
# ==============================================

class TestShowBanner:

    @pytest.mark.unit
    def test_banner_does_not_crash(self):
        """show_banner() completes without error."""
        console = Console(file=StringIO())
        show_banner(console)

    @pytest.mark.unit
    def test_banner_contains_chal(self):
        """Banner output contains 'CHAL' or the project name."""
        buf = StringIO()
        console = Console(file=buf)
        show_banner(console)
        output = buf.getvalue()
        assert "Council" in output or "CHAL" in output


# ==============================================
# 4. Main function routing
# ==============================================

class TestMainRouting:

    @pytest.mark.unit
    @patch("chal.cli.main.run_debate")
    @patch("chal.cli.main.load_config")
    def test_headless_mode(self, mock_load_config, mock_run_debate):
        """--config without --edit triggers headless mode."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config
        mock_run_debate.return_value = 0

        result = main(["--config", "default"])

        mock_load_config.assert_called_once_with("default")
        mock_run_debate.assert_called_once()
        assert result == 0

    @pytest.mark.unit
    @patch("chal.cli.main.load_config")
    def test_headless_config_not_found(self, mock_load_config):
        """--config with missing file returns exit code 1."""
        mock_load_config.side_effect = FileNotFoundError("not found")

        result = main(["--config", "nonexistent"])

        assert result == 1

    @pytest.mark.unit
    def test_edit_without_config_returns_error(self):
        """--edit without --config returns exit code 1."""
        result = main(["--edit"])
        assert result == 1

    @pytest.mark.unit
    @patch("chal.cli.main.run_wizard")
    def test_wizard_mode_cancel(self, mock_wizard):
        """No args, wizard returns cancel -> exit 0."""
        mock_wizard.return_value = (None, "cancel")

        result = main([])

        mock_wizard.assert_called_once()
        assert result == 0

    @pytest.mark.unit
    @patch("chal.cli.main.run_debate")
    @patch("chal.cli.main.run_wizard")
    def test_wizard_mode_launch(self, mock_wizard, mock_run_debate):
        """No args, wizard returns launch -> run_debate called."""
        mock_config = MagicMock()
        mock_wizard.return_value = (mock_config, "launch")
        mock_run_debate.return_value = 0

        result = main([])

        mock_run_debate.assert_called_once()
        assert result == 0

    @pytest.mark.unit
    @patch("chal.cli.main.run_debate")
    @patch("chal.cli.main.run_wizard")
    @patch("chal.cli.main.load_config")
    def test_edit_mode(self, mock_load_config, mock_wizard, mock_run_debate):
        """--config --edit loads config, passes to wizard, then launches."""
        mock_prefill = MagicMock()
        mock_load_config.return_value = mock_prefill
        mock_config = MagicMock()
        mock_wizard.return_value = (mock_config, "launch")
        mock_run_debate.return_value = 0

        result = main(["--config", "default", "--edit"])

        mock_load_config.assert_called_once_with("default")
        mock_wizard.assert_called_once()
        # Verify prefill was passed
        _, kwargs = mock_wizard.call_args
        assert kwargs.get("prefill") is mock_prefill
        mock_run_debate.assert_called_once()
        assert result == 0

    @pytest.mark.unit
    @patch("chal.cli.main.run_wizard")
    def test_wizard_keyboard_interrupt(self, mock_wizard):
        """Ctrl+C during wizard -> exit 0."""
        mock_wizard.side_effect = KeyboardInterrupt

        result = main([])

        assert result == 0


# ==============================================
# 5. Parse Args — Phase 3 flags
# ==============================================

class TestParseArgsPhase3:

    @pytest.mark.unit
    def test_history_flag(self):
        """--history flag is captured."""
        args = parse_args(["--history"])
        assert args.history is True

    @pytest.mark.unit
    def test_replay_flag(self):
        """--replay <id> captures the debate ID."""
        args = parse_args(["--replay", "a1b2c3d4"])
        assert args.replay == "a1b2c3d4"

    @pytest.mark.unit
    def test_no_history_by_default(self):
        """--history defaults to False."""
        args = parse_args([])
        assert args.history is False

    @pytest.mark.unit
    def test_no_replay_by_default(self):
        """--replay defaults to None."""
        args = parse_args([])
        assert args.replay is None


# ==============================================
# 6. Main routing — Phase 3 (history/replay)
# ==============================================

class TestMainRoutingPhase3:

    @pytest.mark.unit
    @patch("chal.cli.history.format_history_table")
    @patch("chal.cli.history.list_debates")
    def test_history_mode(self, mock_list, mock_format):
        """--history displays history table and exits 0."""
        mock_list.return_value = [{"id": "abc123", "topic": "Test"}]

        result = main(["--history"])

        mock_list.assert_called_once()
        mock_format.assert_called_once()
        assert result == 0

    @pytest.mark.unit
    @patch("chal.cli.main.run_debate")
    @patch("chal.cli.history.load_debate_config")
    def test_replay_mode(self, mock_load, mock_run_debate):
        """--replay <id> loads config and runs debate."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_run_debate.return_value = 0

        result = main(["--replay", "abc12345"])

        mock_load.assert_called_once_with("abc12345")
        mock_run_debate.assert_called_once()
        assert result == 0

    @pytest.mark.unit
    @patch("chal.cli.history.load_debate_config")
    def test_replay_not_found(self, mock_load):
        """--replay with invalid ID returns exit code 1."""
        mock_load.side_effect = FileNotFoundError("not found")

        result = main(["--replay", "deadbeef"])

        assert result == 1
