"""
Unit tests for GoogleAgent implementation.

All tests use mocked API calls — no actual Google API calls are made.

Tests cover:
- Agent initialization
- Message generation (mocked)
- Role mapping: "assistant" → "model"
- System message filtering
- Retry logic (mocked)
- Belief state management
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from chal.agents.base import Message


# ==============================================
# 1. Initialization Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_instantiation(mock_client_class):
    """GoogleAgent sets all belief-state attributes correctly."""
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Skeptic")

    assert agent.name == "Agent-Skeptic"
    assert agent.model == "gemini-2.0-flash"
    assert agent.internal_belief == ""
    assert agent.internal_belief_obj is None
    assert agent.belief_graph is None
    assert agent.persona_label == "Skeptic"
    assert agent.all_beliefs_held == []
    assert agent.system_prompt == ""


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_instantiation_system_prompt(mock_client_class):
    """System prompt is stored correctly at construction."""
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(
        model="gemini-2.0-flash", name="Agent-Test", system_prompt="Base prompt"
    )
    assert agent.system_prompt == "Base prompt"


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_persona_label_no_prefix(mock_client_class):
    """persona_label is the full name when 'Agent-' prefix is absent."""
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(model="gemini-2.0-flash", name="Standalone")
    assert agent.persona_label == "Standalone"


# ==============================================
# 2. Message Generation Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
@patch('chal.agents.google_agent.retry_google_generate')
def test_generate_success(mock_retry, mock_client_class):
    """Mocked Gemini response is parsed into a Message."""
    from chal.agents.google_agent import GoogleAgent

    mock_response = MagicMock()
    mock_response.text = "Mocked Gemini response"
    mock_retry.return_value = mock_response

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Mocked Gemini response"


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
@patch('chal.agents.google_agent.retry_google_generate')
def test_role_mapping_assistant_to_model(mock_retry, mock_client_class):
    """'assistant' in history is mapped to 'model' before the API call."""
    from chal.agents.google_agent import GoogleAgent

    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_retry.return_value = mock_response

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    history = [
        Message(role="user", content="User question"),
        Message(role="assistant", content="Prior answer"),
    ]
    agent.generate(history)

    _args, call_kwargs = mock_retry.call_args
    contents_sent = call_kwargs["contents"]
    roles = [c.role for c in contents_sent]
    assert "model" in roles
    assert "assistant" not in roles


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
@patch('chal.agents.google_agent.retry_google_generate')
def test_system_messages_filtered(mock_retry, mock_client_class):
    """'system' messages are dropped from contents before the API call."""
    from chal.agents.google_agent import GoogleAgent

    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_retry.return_value = mock_response

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    history = [
        Message(role="system", content="System instruction"),
        Message(role="user", content="User question"),
    ]
    agent.generate(history)

    _args, call_kwargs = mock_retry.call_args
    contents_sent = call_kwargs["contents"]
    roles = [c.role for c in contents_sent]
    assert "system" not in roles
    assert len(contents_sent) == 1  # only user message


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
@patch('chal.agents.google_agent.retry_google_generate')
def test_system_prompt_prepended(mock_retry, mock_client_class):
    """Non-empty system_prompt is passed as the system_prompt kwarg to retry."""
    from chal.agents.google_agent import GoogleAgent

    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_retry.return_value = mock_response

    agent = GoogleAgent(
        model="gemini-2.0-flash", name="Agent-Test", system_prompt="Be concise."
    )
    agent.generate([Message(role="user", content="Question?")])

    call_kwargs = mock_retry.call_args.kwargs
    assert call_kwargs["system_prompt"] == "Be concise."


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
@patch('chal.agents.google_agent.retry_google_generate')
def test_generate_conversation_history(mock_retry, mock_client_class):
    """Multi-turn history is preserved in the contents kwarg (assistant -> model)."""
    from chal.agents.google_agent import GoogleAgent

    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_retry.return_value = mock_response

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    history = [
        Message(role="user", content="First message"),
        Message(role="assistant", content="First response"),
        Message(role="user", content="Second message"),
    ]
    agent.generate(history)

    contents_sent = mock_retry.call_args.kwargs["contents"]
    assert len(contents_sent) >= 3
    roles = [c.role for c in contents_sent]
    assert roles[0] == "user"
    assert roles[1] == "model"  # assistant mapped to model
    assert roles[2] == "user"


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
@patch('chal.agents.google_agent.retry_google_generate')
def test_generate_error_returns_error_message(mock_retry, mock_client_class):
    """Exceptions from retry are caught and returned as a labelled error Message."""
    from chal.agents.google_agent import GoogleAgent

    mock_retry.side_effect = RuntimeError("API failure")

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert "[Error from Agent-Test]" in result.content


# ==============================================
# 3. System Prompt / Role Card Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_receive_system_prompt(mock_client_class):
    """receive_system_prompt replaces system_prompt."""
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    agent.receive_system_prompt("New system prompt")
    assert agent.system_prompt == "New system prompt"


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_receive_role_card(mock_client_class):
    """receive_role_card appends to system_prompt."""
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(
        model="gemini-2.0-flash", name="Agent-Test", system_prompt="Base prompt"
    )
    agent.receive_role_card("Role card content")
    assert "Base prompt" in agent.system_prompt
    assert "Role card content" in agent.system_prompt


# ==============================================
# 4. Belief State Management Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_set_get_internal_belief(mock_client_class):
    """Belief text roundtrips correctly."""
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    agent.set_internal_belief("Free will is an emergent property.")
    assert agent.get_internal_belief() == "Free will is an emergent property."


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_set_internal_belief_obj_stores_dict(mock_client_class):
    """set_internal_belief_obj stores the belief dict (graph build may silently fail)."""
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    belief = {"schema_version": "CBS", "belief_id": "B1"}
    agent.set_internal_belief_obj(belief)
    assert agent.get_internal_belief_obj() == belief


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_set_internal_belief_obj_none_clears_graph(mock_client_class):
    """Setting belief obj to None clears the belief graph."""
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    agent.set_internal_belief_obj(None)
    assert agent.belief_graph is None
    assert agent.internal_belief_obj is None


# ==============================================
# 5. Retry Logic Tests (Direct on retry function)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.google_agent.time.sleep')
def test_retry_on_api_error(mock_sleep):
    """API error triggers retry; succeeds on second attempt."""
    import chal.agents.google_agent as mod

    mock_client = MagicMock()
    mock_success = MagicMock()
    mock_success.text = "Success response"

    class FakeAPIError(Exception):
        pass

    mock_client.models.generate_content.side_effect = [
        FakeAPIError("api error"),
        mock_success,
    ]

    with patch.object(mod.genai_errors, 'APIError', FakeAPIError):
        result = mod.retry_google_generate(
            client=mock_client,
            model="gemini-2.0-flash",
            contents=[],
            system_prompt="",
            temperature=0.7,
        )

    assert result is mock_success
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.unit
@patch('chal.agents.google_agent.time.sleep')
def test_retry_exhausted(mock_sleep):
    """After max_retries failures, RuntimeError is raised."""
    import chal.agents.google_agent as mod

    mock_client = MagicMock()

    class FakeAPIError(Exception):
        pass

    mock_client.models.generate_content.side_effect = FakeAPIError("api error")

    with patch.object(mod.genai_errors, 'APIError', FakeAPIError):
        with pytest.raises(RuntimeError, match="Exceeded max retries"):
            mod.retry_google_generate(
                client=mock_client,
                model="gemini-2.0-flash",
                contents=[],
                system_prompt="",
                temperature=0.7,
                max_retries=3,
            )

    assert mock_client.models.generate_content.call_count == 3


@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
@patch('chal.agents.google_agent.retry_google_generate')
def test_generate_catches_exhausted_retry(mock_retry, mock_client_class):
    """generate() wraps a RuntimeError from exhausted retries as an error Message."""
    from chal.agents.google_agent import GoogleAgent

    mock_retry.side_effect = RuntimeError("Exceeded max retries for Google Gemini API call.")

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Question")])

    assert "[Error from Agent-Test]" in result.content


# ==============================================
# 6. API Key Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
def test_api_key_from_env(mock_client_class, monkeypatch):
    """Key is read from GOOGLE_API_KEY env var when not passed explicitly."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    from chal.agents.google_agent import GoogleAgent

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    assert agent.api_key == "test-google-key"


# ==============================================
# 7. Error Handling Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.google_agent.genai.Client')
@patch('chal.agents.google_agent.retry_google_generate')
def test_generate_empty_response(mock_retry, mock_client_class):
    """Empty content from model is returned without crash."""
    from chal.agents.google_agent import GoogleAgent

    mock_response = MagicMock()
    mock_response.text = ""
    mock_retry.return_value = mock_response

    agent = GoogleAgent(model="gemini-2.0-flash", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Question")])

    assert result.content == ""
