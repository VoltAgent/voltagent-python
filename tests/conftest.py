"""Pytest configuration and fixtures for VoltAgent SDK tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import pytest_asyncio
import respx

from voltagent import VoltAgentSDK
from voltagent.client import VoltAgentCoreAPI
from voltagent.types import (
    Event,
    EventLevel,
    EventStatus,
    History,
    TimelineEvent,
    TokenUsage,
    TraceStatus,
    VoltAgentConfig,
)


@pytest.fixture
def sample_config():
    """Sample VoltAgent configuration for testing."""
    return VoltAgentConfig(
        base_url="https://api.test.voltagent.dev",
        public_key="test-public-key",
        secret_key="test-secret-key",
        timeout=30,
        auto_flush=False,  # Disable auto-flush for tests
    )


@pytest.fixture
def sample_history():
    """Sample history/trace data for testing."""
    return History(
        id="test-trace-123",
        agent_id="test-agent",
        user_id="test-user",
        conversation_id="test-conv",
        start_time=datetime.now(timezone.utc).isoformat(),
        status=TraceStatus.WORKING,
        input={"query": "test query"},
        metadata={"source": "test"},
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def sample_event():
    """Sample event data for testing."""
    return Event(
        id="test-event-456",
        history_id="test-trace-123",
        event_type="agent",
        event_name="agent:start",
        start_time=datetime.now(timezone.utc).isoformat(),
        status=EventStatus.RUNNING,
        level=EventLevel.INFO,
        input={"task": "test task"},
        metadata={"agentId": "test-agent", "displayName": "Test Agent"},
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def sample_timeline_event():
    """Sample timeline event for testing."""
    return TimelineEvent(
        id="test-timeline-789",
        name="agent:start",
        type="agent",
        startTime=datetime.now(timezone.utc).isoformat(),
        status=EventStatus.RUNNING,
        level=EventLevel.INFO,
        input={"task": "test task"},
        metadata={"agentId": "test-agent"},
    )


@pytest.fixture
def sample_token_usage():
    """Sample token usage data for testing."""
    return TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )


@pytest.fixture
def mock_client(sample_config):
    """Mock VoltAgentCoreAPI client for testing."""
    client = Mock(spec=VoltAgentCoreAPI)
    client.config = sample_config

    # Mock async methods
    client.add_history_async = AsyncMock()
    client.update_history_async = AsyncMock()
    client.add_event_async = AsyncMock()
    client.update_event_async = AsyncMock()
    client.aclose = AsyncMock()

    # Mock sync methods
    client.add_history = Mock()
    client.update_history = Mock()
    client.add_event = Mock()
    client.update_event = Mock()
    client.close = Mock()

    return client


@pytest.fixture
def sdk_with_mock_client(sample_config, mock_client):
    """VoltAgent SDK with mocked client for testing."""
    sdk = VoltAgentSDK(**sample_config.model_dump())
    sdk._client = mock_client
    return sdk


@pytest.fixture
def mock_http_responses():
    """Mock HTTP responses for testing."""
    return {
        "history_created": {
            "id": "test-trace-123",
            "agent_id": "test-agent",
            "user_id": "test-user",
            "status": "working",
            "input": {"query": "test"},
            "metadata": {"source": "test"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        "event_created": {
            "id": "test-event-456",
            "history_id": "test-trace-123",
            "event_type": "agent",
            "event_name": "agent:start",
            "status": "running",
            "level": "INFO",
            "input": {"task": "test"},
            "metadata": {"agentId": "test-agent"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


@pytest.fixture
def respx_mock():
    """Respx mock for HTTP testing."""
    with respx.mock as mock:
        yield mock


@pytest.fixture
def error_test_data():
    """Test data for error scenarios."""
    return {
        "api_error": {
            "status_code": 400,
            "response": {"message": "Invalid request", "errors": ["Field required"]},
        },
        "timeout_error": {
            "status_code": 408,
            "response": {"message": "Request timeout"},
        },
        "network_error": {
            "status_code": 0,
            "response": {"message": "Network error"},
        },
    }


@pytest_asyncio.fixture
async def sample_trace_context(sdk_with_mock_client, sample_history):
    """Sample trace context for testing."""
    # Mock the client to return our sample history
    sdk_with_mock_client._client.add_history_async.return_value = sample_history

    trace_context = await sdk_with_mock_client.create_trace(
        agentId="test-agent",
        input={"query": "test"},
    )
    return trace_context


@pytest.fixture
def metadata_test_cases():
    """Test cases for metadata conversion."""
    return [
        # snake_case to camelCase conversion
        {
            "input": {"model_parameters": {"model": "gpt-4"}},
            "expected": {"modelParameters": {"model": "gpt-4"}},
        },
        {
            "input": {"parent_agent": "agent-123", "max_retries": 3},
            "expected": {"parentAgent": "agent-123", "maxRetries": 3},
        },
        # Nested objects
        {
            "input": {
                "model_parameters": {"model": "gpt-4", "temperature": 0.7},
                "api_config": {"base_url": "test", "timeout_seconds": 30},
            },
            "expected": {
                "modelParameters": {"model": "gpt-4", "temperature": 0.7},
                "apiConfig": {"baseUrl": "test", "timeoutSeconds": 30},
            },
        },
        # Already camelCase (should remain unchanged)
        {
            "input": {"modelParameters": {"model": "gpt-4"}},
            "expected": {"modelParameters": {"model": "gpt-4"}},
        },
        # Mixed case
        {
            "input": {"model_parameters": {"model": "gpt-4"}, "agentId": "test"},
            "expected": {"modelParameters": {"model": "gpt-4"}, "agentId": "test"},
        },
    ]


# Pytest configuration
@pytest.fixture(autouse=True)
def reset_environment() -> None:
    """Reset environment variables for each test."""
    import os

    original_env = os.environ.copy()

    # Set test environment variables
    os.environ.update(
        {
            "VOLTAGENT_BASE_URL": "https://api.test.voltagent.dev",
            "VOLTAGENT_PUBLIC_KEY": "test-public-key",
            "VOLTAGENT_SECRET_KEY": "test-secret-key",
        }
    )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Async test configuration
pytest_plugins = ["pytest_asyncio"]
