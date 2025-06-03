"""Tests for VoltAgent types, models, and utility functions."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from voltagent.types import (
    AddEventRequest,
    AgentOptions,
    APIError,
    CreateHistoryRequest,
    Event,
    EventLevel,
    EventStatus,
    History,
    MemoryOptions,
    RetrieverOptions,
    TimelineEvent,
    TimeoutError,
    TokenUsage,
    ToolOptions,
    TraceOptions,
    TraceStatus,
    UpdateEventRequest,
    UpdateHistoryRequest,
)
from voltagent.types import ValidationError as VoltValidationError  # Enums; Models; Utilities; Exceptions
from voltagent.types import VoltAgentConfig, VoltAgentError, convert_metadata_to_camel_case, snake_to_camel


class TestEnums:
    """Test enum values and behavior."""

    def test_event_status_enum(self):
        """Test EventStatus enum values."""
        assert EventStatus.RUNNING == "running"
        assert EventStatus.COMPLETED == "completed"
        assert EventStatus.ERROR == "error"
        assert EventStatus.CANCELLED == "cancelled"

        # Test enum can be created from string
        assert EventStatus("running") == EventStatus.RUNNING

    def test_event_level_enum(self):
        """Test EventLevel enum values."""
        assert EventLevel.DEBUG == "DEBUG"
        assert EventLevel.INFO == "INFO"
        assert EventLevel.WARNING == "WARNING"
        assert EventLevel.ERROR == "ERROR"

    def test_trace_status_enum(self):
        """Test TraceStatus enum values."""
        assert TraceStatus.WORKING == "working"
        assert TraceStatus.COMPLETED == "completed"
        assert TraceStatus.ERROR == "error"


class TestUtilityFunctions:
    """Test utility functions for string conversion."""

    def test_snake_to_camel_conversion(self):
        """Test snake_case to camelCase conversion."""
        # Basic cases
        assert snake_to_camel("model_parameters") == "modelParameters"
        assert snake_to_camel("parent_agent") == "parentAgent"
        assert snake_to_camel("max_retries") == "maxRetries"
        assert snake_to_camel("api_version") == "apiVersion"

        # Edge cases
        assert snake_to_camel("single") == "single"  # No underscore
        assert snake_to_camel("already_camel") == "alreadyCamel"
        assert snake_to_camel("multiple_word_test") == "multipleWordTest"
        assert snake_to_camel("") == ""  # Empty string
        assert snake_to_camel("_leading") == "Leading"  # Leading underscore

    def test_convert_metadata_to_camel_case(self, metadata_test_cases):
        """Test metadata conversion from snake_case to camelCase."""
        for case in metadata_test_cases:
            result = convert_metadata_to_camel_case(case["input"])
            assert result == case["expected"], f"Failed for input: {case['input']}"

    def test_convert_metadata_none(self):
        """Test metadata conversion with None input."""
        assert convert_metadata_to_camel_case(None) is None
        assert convert_metadata_to_camel_case({}) == {}

    def test_convert_metadata_nested_conversion(self):
        """Test deep nested metadata conversion."""
        input_data = {
            "model_parameters": {"temperature": 0.7, "max_tokens": 150, "response_format": {"type": "json"}},
            "retry_config": {"max_attempts": 3, "backoff_factor": 2.0},
        }

        expected = {
            "modelParameters": {"temperature": 0.7, "maxTokens": 150, "responseFormat": {"type": "json"}},
            "retryConfig": {"maxAttempts": 3, "backoffFactor": 2.0},
        }

        result = convert_metadata_to_camel_case(input_data)
        assert result == expected


class TestCoreModels:
    """Test core Pydantic models."""

    def test_token_usage_model(self):
        """Test TokenUsage model validation."""
        # Valid data
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.total_tokens == 150

        # Partial data (optional fields)
        usage_partial = TokenUsage(total_tokens=100)
        assert usage_partial.total_tokens == 100
        assert usage_partial.prompt_tokens is None

    def test_timeline_event_model(self):
        """Test TimelineEvent model validation."""
        now = datetime.now(timezone.utc).isoformat()

        event = TimelineEvent(
            id="test-event",
            name="agent:start",
            type="agent",
            startTime=now,
            status=EventStatus.RUNNING,
            level=EventLevel.INFO,
            input={"task": "test"},
            metadata={"agentId": "test-agent"},
        )

        assert event.id == "test-event"
        assert event.name == "agent:start"
        assert event.type == "agent"
        assert event.status == EventStatus.RUNNING

        # Test with minimal required fields
        minimal_event = TimelineEvent(name="tool:start", type="tool")
        assert minimal_event.name == "tool:start"
        assert minimal_event.status == EventStatus.RUNNING  # Default
        assert minimal_event.level == EventLevel.INFO  # Default

    def test_event_model(self):
        """Test Event API response model."""
        now = datetime.now(timezone.utc).isoformat()

        event = Event(
            id="test-event",
            history_id="test-trace",
            event_type="agent",
            event_name="agent:start",
            status=EventStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )

        assert event.id == "test-event"
        assert event.history_id == "test-trace"
        assert event.event_type == "agent"
        assert event.status == EventStatus.RUNNING

    def test_history_model(self):
        """Test History model validation."""
        now = datetime.now(timezone.utc).isoformat()

        history = History(
            id="test-trace", agent_id="test-agent", status=TraceStatus.WORKING, created_at=now, updated_at=now
        )

        assert history.id == "test-trace"
        assert history.agent_id == "test-agent"
        assert history.status == TraceStatus.WORKING

        # Test with full data
        full_history = History(
            id="full-trace",
            agent_id="full-agent",
            user_id="test-user",
            conversation_id="test-conv",
            start_time=now,
            end_time=now,
            status=TraceStatus.COMPLETED,
            input={"query": "test"},
            output={"response": "test response"},
            tags=["test"],
            metadata={"source": "test"},
            usage=TokenUsage(total_tokens=100),
            created_at=now,
            updated_at=now,
        )

        assert full_history.user_id == "test-user"
        assert full_history.input == {"query": "test"}
        assert full_history.usage.total_tokens == 100


class TestConfigurationModels:
    """Test configuration and options models."""

    def test_volt_agent_config(self):
        """Test VoltAgentConfig model."""
        config = VoltAgentConfig(base_url="https://api.test.com", public_key="pub-key", secret_key="secret-key")

        assert config.base_url == "https://api.test.com"
        assert config.timeout == 30  # Default value
        assert config.auto_flush is True  # Default value

        # Test with custom values
        custom_config = VoltAgentConfig(
            base_url="https://custom.api.com",
            public_key="custom-pub",
            secret_key="custom-secret",
            timeout=60,
            auto_flush=False,
            flush_interval=10,
            max_retries=5,
            headers={"Custom-Header": "test"},
        )

        assert custom_config.timeout == 60
        assert custom_config.auto_flush is False
        assert custom_config.flush_interval == 10
        assert custom_config.max_retries == 5
        assert custom_config.headers == {"Custom-Header": "test"}

    def test_trace_options(self):
        """Test TraceOptions model with camelCase support."""
        # Test with camelCase (primary format)
        options = TraceOptions(
            agentId="test-agent",
            userId="test-user",
            conversationId="test-conv",
            input={"query": "test"},
            startTime=datetime.now(timezone.utc).isoformat(),
        )

        assert options.agentId == "test-agent"
        assert options.userId == "test-user"
        assert options.conversationId == "test-conv"

        # Test with snake_case (alias support)
        options_snake = TraceOptions(agent_id="test-agent-2", user_id="test-user-2", conversation_id="test-conv-2")

        assert options_snake.agentId == "test-agent-2"
        assert options_snake.userId == "test-user-2"
        assert options_snake.conversationId == "test-conv-2"

    def test_agent_options(self):
        """Test AgentOptions model."""
        options = AgentOptions(
            name="Test Agent", input={"task": "test task"}, instructions="Do something", metadata={"model": "gpt-4"}
        )

        assert options.name == "Test Agent"
        assert options.input == {"task": "test task"}
        assert options.instructions == "Do something"
        assert options.metadata == {"model": "gpt-4"}

        # Test minimal options
        minimal = AgentOptions(name="Minimal Agent")
        assert minimal.name == "Minimal Agent"
        assert minimal.input is None

    def test_tool_options(self):
        """Test ToolOptions model."""
        options = ToolOptions(name="Test Tool", input={"param": "value"}, metadata={"version": "1.0"})

        assert options.name == "Test Tool"
        assert options.input == {"param": "value"}
        assert options.metadata == {"version": "1.0"}

    def test_memory_options(self):
        """Test MemoryOptions model."""
        options = MemoryOptions(name="Test Memory", input={"key": "value"}, metadata={"type": "redis"})

        assert options.name == "Test Memory"
        assert options.input == {"key": "value"}
        assert options.metadata == {"type": "redis"}

    def test_retriever_options(self):
        """Test RetrieverOptions model."""
        options = RetrieverOptions(
            name="Test Retriever", input={"query": "search query"}, metadata={"index": "documents"}
        )

        assert options.name == "Test Retriever"
        assert options.input == {"query": "search query"}
        assert options.metadata == {"index": "documents"}


class TestRequestModels:
    """Test API request models."""

    def test_create_history_request(self):
        """Test CreateHistoryRequest model."""
        request = CreateHistoryRequest(
            agent_id="test-agent",
            userId="test-user",
            conversationId="test-conv",
            input={"query": "test"},
            status=TraceStatus.WORKING,
        )

        assert request.agent_id == "test-agent"
        assert request.userId == "test-user"
        assert request.conversationId == "test-conv"
        assert request.status == TraceStatus.WORKING

    def test_update_history_request(self):
        """Test UpdateHistoryRequest model."""
        now = datetime.now(timezone.utc).isoformat()

        request = UpdateHistoryRequest(
            id="test-trace", status=TraceStatus.COMPLETED, end_time=now, output={"result": "success"}
        )

        assert request.id == "test-trace"
        assert request.status == TraceStatus.COMPLETED
        assert request.end_time == now
        assert request.output == {"result": "success"}

    def test_add_event_request(self, sample_timeline_event):
        """Test AddEventRequest model."""
        request = AddEventRequest(history_id="test-trace", event=sample_timeline_event)

        assert request.history_id == "test-trace"
        assert request.event == sample_timeline_event

    def test_update_event_request(self):
        """Test UpdateEventRequest model."""
        now = datetime.now(timezone.utc).isoformat()

        request = UpdateEventRequest(
            id="test-event", status=EventStatus.COMPLETED, end_time=now, output={"result": "success"}
        )

        assert request.id == "test-event"
        assert request.status == EventStatus.COMPLETED
        assert request.end_time == now
        assert request.output == {"result": "success"}


class TestExceptions:
    """Test custom exceptions."""

    def test_volt_agent_error(self):
        """Test base VoltAgentError exception."""
        error = VoltAgentError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_api_error(self):
        """Test APIError exception."""
        error = APIError("API failed", status_code=400, response_data={"error": "Invalid request"})

        assert str(error) == "API failed"
        assert error.status_code == 400
        assert error.response_data == {"error": "Invalid request"}
        assert isinstance(error, VoltAgentError)

    def test_timeout_error(self):
        """Test TimeoutError exception."""
        error = TimeoutError("Request timed out")
        assert str(error) == "Request timed out"
        assert isinstance(error, VoltAgentError)

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = VoltValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, VoltAgentError)


class TestModelValidation:
    """Test Pydantic model validation behavior."""

    def test_required_field_validation(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            # Missing required agent_id
            CreateHistoryRequest()

        with pytest.raises(ValidationError):
            # Missing required name
            AgentOptions()

        with pytest.raises(ValidationError):
            # Missing required agentId
            TraceOptions()

    def test_enum_validation(self):
        """Test enum field validation."""
        # Valid enum value
        event = TimelineEvent(name="test", type="agent", status=EventStatus.RUNNING)
        assert event.status == EventStatus.RUNNING

        # Invalid enum value should raise ValidationError
        with pytest.raises(ValidationError):
            TimelineEvent(name="test", type="agent", status="invalid_status")  # Not a valid EventStatus

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in models with extra='allow'."""
        # Most models should allow extra fields
        options = AgentOptions(
            name="Test Agent", custom_field="custom_value", another_extra={"nested": "data"}  # Extra field
        )

        # Extra fields should be preserved
        assert hasattr(options, "custom_field")
        assert options.custom_field == "custom_value"
