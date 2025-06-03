"""Tests for VoltAgent HTTP client (VoltAgentCoreAPI)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import respx

from voltagent.client import VoltAgentCoreAPI
from voltagent.types import (
    AddEventRequest,
    APIError,
    CreateHistoryRequest,
    EventLevel,
    EventStatus,
    TimelineEvent,
    TimeoutError,
    TraceStatus,
    UpdateEventRequest,
    UpdateHistoryRequest,
    VoltAgentConfig,
)


class TestVoltAgentCoreAPI:
    """Test VoltAgentCoreAPI initialization and configuration."""

    def test_client_initialization(self, sample_config):
        """Test client initialization with configuration."""
        client = VoltAgentCoreAPI(sample_config)

        assert client.base_url == "https://api.test.voltagent.dev"
        assert client.timeout == 30
        assert "x-public-key" in client.headers
        assert "x-secret-key" in client.headers
        assert client.headers["x-public-key"] == "test-public-key"
        assert client.headers["x-secret-key"] == "test-secret-key"

    def test_client_base_url_normalization(self):
        """Test base URL trailing slash normalization."""
        config_with_slash = VoltAgentConfig(base_url="https://api.test.com/", public_key="key", secret_key="secret")
        client = VoltAgentCoreAPI(config_with_slash)
        assert client.base_url == "https://api.test.com"

        config_without_slash = VoltAgentConfig(base_url="https://api.test.com", public_key="key", secret_key="secret")
        client2 = VoltAgentCoreAPI(config_without_slash)
        assert client2.base_url == "https://api.test.com"

    def test_client_custom_headers(self):
        """Test client with custom headers."""
        config = VoltAgentConfig(
            base_url="https://api.test.com",
            public_key="key",
            secret_key="secret",
            headers={"Custom-Header": "test-value"},
        )
        client = VoltAgentCoreAPI(config)

        assert client.headers["Custom-Header"] == "test-value"
        assert client.headers["x-public-key"] == "key"


class TestHistoryOperations:
    """Test history/trace CRUD operations."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_add_history_success(self, sample_config, mock_http_responses):
        """Test successful history creation."""
        # Mock the API response
        respx.post("https://api.test.voltagent.dev/history").mock(
            return_value=httpx.Response(200, json=mock_http_responses["history_created"])
        )

        client = VoltAgentCoreAPI(sample_config)

        request = CreateHistoryRequest(agent_id="test-agent", userId="test-user", input={"query": "test"})

        result = await client.add_history_async(request)

        assert result.id == "test-trace-123"
        assert result.agent_id == "test-agent"
        assert result.user_id == "test-user"

    @pytest.mark.asyncio
    @respx.mock
    async def test_add_history_api_error(self, sample_config, error_test_data):
        """Test history creation with API error."""
        # Mock API error response
        respx.post("https://api.test.voltagent.dev/history").mock(
            return_value=httpx.Response(400, json=error_test_data["api_error"]["response"])
        )

        client = VoltAgentCoreAPI(sample_config)

        request = CreateHistoryRequest(agent_id="test-agent", input={"query": "test"})

        with pytest.raises(APIError) as exc_info:
            await client.add_history_async(request)

        assert exc_info.value.status_code == 400
        assert "Invalid request" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_update_history_success(self, sample_config, mock_http_responses):
        """Test successful history update."""
        updated_history = {
            **mock_http_responses["history_created"],
            "status": "completed",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "output": {"result": "success"},
        }

        respx.patch("https://api.test.voltagent.dev/history/test-trace-123").mock(
            return_value=httpx.Response(200, json=updated_history)
        )

        client = VoltAgentCoreAPI(sample_config)

        request = UpdateHistoryRequest(id="test-trace-123", status=TraceStatus.COMPLETED, output={"result": "success"})

        result = await client.update_history_async(request)

        assert result.id == "test-trace-123"
        assert result.status == "completed"
        assert result.output == {"result": "success"}

    def test_add_history_sync(self, sample_config):
        """Test synchronous history creation."""
        client = VoltAgentCoreAPI(sample_config)

        # Mock the private _sync_client attribute instead of the property
        mock_sync = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "id": "sync-trace-123",
            "agent_id": "sync-agent",
            "status": "working",
            # Add required fields that were missing
            "user_id": None,
            "conversation_id": None,
            "start_time": None,
            "end_time": None,
            "completion_start_time": None,
            "input": {"query": "sync test"},
            "output": None,
            "tags": None,
            "metadata": None,
            "version": None,
            "level": None,
            "usage": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_sync.request.return_value = mock_response
        client._sync_client = mock_sync

        request = CreateHistoryRequest(agent_id="sync-agent", input={"query": "sync test"})

        result = client.add_history(request)

        assert result.id == "sync-trace-123"
        assert result.agent_id == "sync-agent"


class TestEventOperations:
    """Test event CRUD operations."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_add_event_success(self, sample_config, mock_http_responses):
        """Test successful event creation."""
        respx.post("https://api.test.voltagent.dev/history-events").mock(
            return_value=httpx.Response(200, json=mock_http_responses["event_created"])
        )

        client = VoltAgentCoreAPI(sample_config)

        timeline_event = TimelineEvent(
            id="test-event",
            name="agent:start",
            type="agent",
            startTime=datetime.now(timezone.utc).isoformat(),
            status=EventStatus.RUNNING,
            input={"task": "test"},
        )

        request = AddEventRequest(history_id="test-trace-123", event=timeline_event)

        result = await client.add_event_async(request)

        assert result.id == "test-event-456"
        assert result.history_id == "test-trace-123"
        assert result.event_type == "agent"

    @pytest.mark.asyncio
    @respx.mock
    async def test_add_event_dto_transformation(self, sample_config):
        """Test that event data is properly transformed to DTO format."""
        # Capture the request that gets sent
        requests = []

        def capture_request(request):
            requests.append(request)
            # Return a complete event response instead of just {"id": "test"}
            return httpx.Response(
                200,
                json={
                    "id": "test-event-123",
                    "history_id": "test-trace-123",
                    "event_type": "agent",
                    "event_name": "agent:start",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "end_time": None,
                    "status": "running",
                    "status_message": None,
                    "level": "INFO",
                    "parent_event_id": "parent-123",
                    "input": {"task": "test"},
                    "output": None,
                    "tags": None,
                    "metadata": {"agentId": "test-agent"},
                    "version": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        respx.post("https://api.test.voltagent.dev/history-events").mock(side_effect=capture_request)

        client = VoltAgentCoreAPI(sample_config)

        timeline_event = TimelineEvent(
            id="test-event",
            name="agent:start",
            type="agent",
            startTime=datetime.now(timezone.utc).isoformat(),
            status=EventStatus.RUNNING,
            level=EventLevel.INFO,
            parentEventId="parent-123",
            input={"task": "test"},
            metadata={"agentId": "test-agent"},
        )

        request = AddEventRequest(history_id="test-trace-123", event=timeline_event)

        result = await client.add_event_async(request)

        # Verify we got a valid result back
        assert result.id == "test-event-123"
        assert result.history_id == "test-trace-123"

        # Verify the request was properly transformed to DTO format
        sent_request = requests[0]
        sent_data = sent_request.content.decode()

        # Should contain snake_case field names for the API
        assert "history_id" in sent_data
        assert "event_type" in sent_data
        assert "event_name" in sent_data
        assert "start_time" in sent_data
        assert "parent_event_id" in sent_data

    @pytest.mark.asyncio
    @respx.mock
    async def test_update_event_success(self, sample_config, mock_http_responses):
        """Test successful event update."""
        updated_event = {
            **mock_http_responses["event_created"],
            "status": "completed",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "output": {"result": "success"},
        }

        respx.patch("https://api.test.voltagent.dev/history-events/test-event-456").mock(
            return_value=httpx.Response(200, json=updated_event)
        )

        client = VoltAgentCoreAPI(sample_config)

        request = UpdateEventRequest(id="test-event-456", status=EventStatus.COMPLETED, output={"result": "success"})

        result = await client.update_event_async(request)

        assert result.id == "test-event-456"
        assert result.status == "completed"
        assert result.output == {"result": "success"}


class TestErrorHandling:
    """Test various error scenarios and error handling."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_400_error(self, sample_config):
        """Test HTTP 400 Bad Request error."""
        respx.post("https://api.test.voltagent.dev/history").mock(
            return_value=httpx.Response(400, json={"message": "Bad Request", "errors": ["Invalid agent_id"]})
        )

        client = VoltAgentCoreAPI(sample_config)

        request = CreateHistoryRequest(agent_id="")  # Invalid empty agent_id

        with pytest.raises(APIError) as exc_info:
            await client.add_history_async(request)

        error = exc_info.value
        assert error.status_code == 400
        assert "Bad Request" in str(error)

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_500_error(self, sample_config):
        """Test HTTP 500 Internal Server Error."""
        respx.post("https://api.test.voltagent.dev/history").mock(
            return_value=httpx.Response(500, json={"message": "Internal Server Error"})
        )

        client = VoltAgentCoreAPI(sample_config)

        request = CreateHistoryRequest(agent_id="test-agent")

        with pytest.raises(APIError) as exc_info:
            await client.add_history_async(request)

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_timeout_error(self, sample_config):
        """Test request timeout handling."""
        # Create client with very short timeout
        config = VoltAgentConfig(
            base_url="https://api.test.voltagent.dev",
            public_key="key",
            secret_key="secret",
            timeout=0.001,  # 1ms timeout
        )
        client = VoltAgentCoreAPI(config)

        # Mock the async client to simulate timeout
        mock_async = AsyncMock()
        mock_async.request.side_effect = httpx.TimeoutException("Request timeout")
        client._async_client = mock_async

        request = CreateHistoryRequest(agent_id="test-agent")

        with pytest.raises(TimeoutError) as exc_info:
            await client.add_history_async(request)

        # Should be timeout error
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error(self, sample_config):
        """Test network connection error."""
        # Mock network error
        respx.post("https://api.test.voltagent.dev/history").mock(side_effect=httpx.ConnectError("Connection failed"))

        client = VoltAgentCoreAPI(sample_config)

        request = CreateHistoryRequest(agent_id="test-agent")

        with pytest.raises(APIError) as exc_info:
            await client.add_history_async(request)

        assert exc_info.value.status_code == 0  # Network errors have status 0
        assert "network error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_invalid_json_response(self, sample_config):
        """Test handling of invalid JSON response."""
        respx.post("https://api.test.voltagent.dev/history").mock(
            return_value=httpx.Response(200, content="Invalid JSON")
        )

        client = VoltAgentCoreAPI(sample_config)

        request = CreateHistoryRequest(agent_id="test-agent")

        with pytest.raises(Exception):  # Should raise JSON decode error
            await client.add_history_async(request)


class TestClientConfiguration:
    """Test client configuration and options."""

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        config = VoltAgentConfig(base_url="https://api.test.com", public_key="key", secret_key="secret", timeout=60)
        client = VoltAgentCoreAPI(config)

        assert client.timeout == 60

    def test_headers_configuration(self):
        """Test custom headers configuration."""
        config = VoltAgentConfig(
            base_url="https://api.test.com",
            public_key="key",
            secret_key="secret",
            headers={"User-Agent": "VoltAgent-Python/1.0", "Custom-Header": "test-value"},
        )
        client = VoltAgentCoreAPI(config)

        assert client.headers["User-Agent"] == "VoltAgent-Python/1.0"
        assert client.headers["Custom-Header"] == "test-value"
        assert client.headers["Content-Type"] == "application/json"
        assert client.headers["x-public-key"] == "key"


class TestAsyncContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_aclose_method(self, sample_config):
        """Test client cleanup with aclose method."""
        client = VoltAgentCoreAPI(sample_config)

        # Mock the httpx async client
        with patch.object(client, "_async_client") as mock_async:
            mock_async.aclose = AsyncMock()  # Use AsyncMock for async method
            await client.aclose()
            mock_async.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, sample_config):
        """Test using client as async context manager."""
        async with VoltAgentCoreAPI(sample_config) as client:
            assert isinstance(client, VoltAgentCoreAPI)

        # Client should be closed after context exit


class TestSyncMethods:
    """Test synchronous method implementations."""

    def test_sync_method_delegation(self, sample_config):
        """Test that sync methods properly delegate to async ones."""
        client = VoltAgentCoreAPI(sample_config)

        # Mock the private _sync_client attribute instead of the property
        mock_sync = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "id": "test-trace-123",
            "agent_id": "test-agent",
            "status": "working",
            "user_id": None,
            "conversation_id": None,
            "start_time": None,
            "end_time": None,
            "completion_start_time": None,
            "input": None,
            "output": None,
            "tags": None,
            "metadata": None,
            "version": None,
            "level": None,
            "usage": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_sync.request.return_value = mock_response
        client._sync_client = mock_sync

        request = CreateHistoryRequest(agent_id="test-agent")

        result = client.add_history(request)

        # Verify the sync method was called and returned expected result
        assert result.id == "test-trace-123"
        assert result.agent_id == "test-agent"
        mock_sync.request.assert_called_once()


class TestRequestValidation:
    """Test request validation and data transformation."""

    def test_history_request_validation(self):
        """Test history request validation."""
        # Valid request
        request = CreateHistoryRequest(agent_id="test-agent", userId="test-user", input={"query": "test"})
        assert request.agent_id == "test-agent"
        assert request.userId == "test-user"

        # Test with missing required field
        with pytest.raises(ValueError):
            CreateHistoryRequest()  # Missing agent_id

    def test_event_request_validation(self):
        """Test event request validation."""
        timeline_event = TimelineEvent(name="test:event", type="test")

        request = AddEventRequest(history_id="test-trace", event=timeline_event)

        assert request.history_id == "test-trace"
        assert request.event.name == "test:event"

        # Test with missing required field
        with pytest.raises(ValueError):
            AddEventRequest(event=timeline_event)  # Missing history_id
