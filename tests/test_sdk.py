"""Tests for VoltAgent SDK including context managers and high-level operations."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic_core import ValidationError as PydanticValidationError

from voltagent import VoltAgentSDK
from voltagent.sdk import (
    AgentContext,
    EventContext,
    MemoryContext,
    RetrieverContext,
    ToolContext,
    TraceContext,
    TraceContextManager,
)
from voltagent.types import (
    AgentOptions,
    APIError,
    EventLevel,
    EventStatus,
    MemoryOptions,
    RetrieverOptions,
    TokenUsage,
    ToolOptions,
    TraceStatus,
    ValidationError,
    VoltAgentConfig,
)


class TestVoltAgentSDK:
    """Test VoltAgentSDK initialization and basic functionality."""

    def test_sdk_initialization(self, sample_config):
        """Test SDK initialization with configuration."""
        sdk = VoltAgentSDK(**sample_config.model_dump())

        assert isinstance(sdk.config, VoltAgentConfig)
        assert sdk.config.base_url == sample_config.base_url
        assert sdk.config.public_key == sample_config.public_key
        assert sdk._traces == {}
        assert sdk._event_queue == []

    def test_sdk_initialization_from_env(self):
        """Test SDK initialization from environment variables."""
        sdk = VoltAgentSDK(
            base_url="https://api.test.voltagent.dev", public_key="test-public-key", secret_key="test-secret-key"
        )

        assert sdk.config.base_url == "https://api.test.voltagent.dev"
        assert sdk.config.public_key == "test-public-key"
        assert sdk.config.secret_key == "test-secret-key"

    def test_sdk_auto_flush_disabled(self):
        """Test SDK with auto-flush disabled."""
        sdk = VoltAgentSDK(base_url="https://api.test.com", public_key="key", secret_key="secret", auto_flush=False)

        assert sdk.config.auto_flush is False
        assert sdk._auto_flush_task is None


class TestTraceManagement:
    """Test trace creation and management."""

    @pytest.mark.asyncio
    async def test_create_trace_success(self, sdk_with_mock_client, sample_history):
        """Test successful trace creation."""
        # Setup mock
        sdk_with_mock_client._client.add_history_async.return_value = sample_history

        trace_context = await sdk_with_mock_client.create_trace(
            agentId="test-agent", input={"query": "test"}, userId="test-user"
        )

        assert isinstance(trace_context, TraceContext)
        assert trace_context.id == "test-trace-123"
        assert trace_context.agent_id == "test-agent"
        assert sample_history.id in sdk_with_mock_client._traces

    @pytest.mark.asyncio
    async def test_trace_context_manager(self, sdk_with_mock_client, sample_history):
        """Test trace context manager."""
        # Setup mock
        sdk_with_mock_client._client.add_history_async.return_value = sample_history
        sdk_with_mock_client._client.update_history_async.return_value = sample_history

        async with sdk_with_mock_client.trace(agentId="test-agent", input={"query": "test"}) as trace:
            assert isinstance(trace, TraceContext)
            assert trace.id == "test-trace-123"

        # Verify trace was ended
        sdk_with_mock_client._client.update_history_async.assert_called()

    @pytest.mark.asyncio
    async def test_trace_context_manager_with_error(self, sdk_with_mock_client, sample_history):
        """Test trace context manager handles errors."""
        sdk_with_mock_client._client.add_history_async.return_value = sample_history
        sdk_with_mock_client._client.update_history_async.return_value = sample_history

        with pytest.raises(ValueError):
            async with sdk_with_mock_client.trace(agentId="test-agent") as trace:
                # Simulate error
                raise ValueError("Test error")

        # Verify trace was marked as error
        update_call = sdk_with_mock_client._client.update_history_async.call_args
        assert update_call[0][0].status == TraceStatus.ERROR

    @pytest.mark.asyncio
    async def test_update_trace(self, sdk_with_mock_client, sample_history):
        """Test trace update functionality."""
        sdk_with_mock_client._client.update_history_async.return_value = sample_history

        result = await sdk_with_mock_client.update_trace(
            "test-trace-123", status=TraceStatus.COMPLETED, output={"result": "success"}
        )

        assert result == sample_history
        sdk_with_mock_client._client.update_history_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_trace(self, sdk_with_mock_client, sample_history):
        """Test trace retrieval."""
        # Add trace to internal storage
        sdk_with_mock_client._traces["test-trace-123"] = sample_history

        result = await sdk_with_mock_client.get_trace("test-trace-123")
        assert result == sample_history

        # Test non-existent trace
        result = await sdk_with_mock_client.get_trace("non-existent")
        assert result is None


class TestTraceContext:
    """Test TraceContext functionality."""

    @pytest.mark.asyncio
    async def test_trace_context_update(self, sample_trace_context):
        """Test trace context update."""
        await sample_trace_context.update(status=TraceStatus.COMPLETED, output={"result": "success"})

        # Verify update was called on SDK
        sample_trace_context._sdk._client.update_history_async.assert_called()

    @pytest.mark.asyncio
    async def test_trace_context_end(self, sample_trace_context, sample_token_usage):
        """Test trace context end."""
        await sample_trace_context.end(
            output={"result": "success"},
            status=TraceStatus.COMPLETED,
            usage=sample_token_usage,
            metadata={"final": True},
        )

        # Verify end was called with correct parameters
        update_call = sample_trace_context._sdk._client.update_history_async.call_args
        # call_args is a tuple: (args, kwargs)
        # For update_history_async(request), the request is the first argument
        update_request = update_call[0][0]  # First positional argument

        assert update_request.status == TraceStatus.COMPLETED
        assert update_request.output == {"result": "success"}
        assert update_request.usage == sample_token_usage

    @pytest.mark.asyncio
    async def test_trace_add_agent(self, sample_trace_context, sample_event):
        """Test adding agent to trace."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event

        agent_context = await sample_trace_context.add_agent(
            {"name": "Test Agent", "input": {"task": "test"}, "instructions": "Do something"}
        )

        assert isinstance(agent_context, AgentContext)
        assert agent_context._event == sample_event

        # Verify event was created with correct data
        event_call = sample_trace_context._sdk._client.add_event_async.call_args
        event_data = event_call[0][0].event

        assert event_data.name == "agent:start"
        assert event_data.type == "agent"
        assert event_data.metadata["displayName"] == "Test Agent"

    @pytest.mark.asyncio
    async def test_trace_add_custom_event(self, sample_trace_context, sample_event):
        """Test adding custom event to trace."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event

        event_context = await sample_trace_context.add_event(
            {"name": "custom:event", "type": "custom", "input": {"data": "test"}}
        )

        assert isinstance(event_context, EventContext)
        assert event_context._event == sample_event


class TestAgentContext:
    """Test AgentContext functionality."""

    @pytest.mark.asyncio
    async def test_agent_add_tool(self, sample_trace_context, sample_event):
        """Test adding tool to agent."""
        # Create agent context
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})

        # Add tool to agent
        tool_context = await agent.add_tool(
            {"name": "Test Tool", "input": {"param": "value"}, "metadata": {"version": "1.0"}}
        )

        assert isinstance(tool_context, ToolContext)

        # Verify tool event was created correctly
        tool_call = sample_trace_context._sdk._client.add_event_async.call_args
        tool_event = tool_call[0][0].event

        assert tool_event.name == "tool:start"
        assert tool_event.type == "tool"
        assert tool_event.metadata["displayName"] == "Test Tool"
        assert tool_event.parentEventId == agent.id

    @pytest.mark.asyncio
    async def test_agent_add_memory(self, sample_trace_context, sample_event):
        """Test adding memory operation to agent."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})

        memory_context = await agent.add_memory(
            {"name": "Test Memory", "input": {"key": "value"}, "metadata": {"type": "redis"}}
        )

        assert isinstance(memory_context, MemoryContext)

        # Verify memory event
        memory_call = sample_trace_context._sdk._client.add_event_async.call_args
        memory_event = memory_call[0][0].event

        assert memory_event.name == "memory:write_start"
        assert memory_event.type == "memory"
        assert memory_event.parentEventId == agent.id

    @pytest.mark.asyncio
    async def test_agent_add_retriever(self, sample_trace_context, sample_event):
        """Test adding retriever to agent."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})

        retriever_context = await agent.add_retriever(
            {"name": "Test Retriever", "input": {"query": "search"}, "metadata": {"index": "docs"}}
        )

        assert isinstance(retriever_context, RetrieverContext)

        # Verify retriever event
        retriever_call = sample_trace_context._sdk._client.add_event_async.call_args
        retriever_event = retriever_call[0][0].event

        assert retriever_event.name == "retriever:start"
        assert retriever_event.type == "retriever"
        assert retriever_event.parentEventId == agent.id

    @pytest.mark.asyncio
    async def test_agent_add_sub_agent(self, sample_trace_context, sample_event):
        """Test adding sub-agent to agent."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event

        parent_agent = await sample_trace_context.add_agent({"name": "Parent Agent"})
        sub_agent = await parent_agent.add_agent({"name": "Sub Agent", "instructions": "Sub task"})

        assert isinstance(sub_agent, AgentContext)
        assert sub_agent.parent_id == parent_agent.id

        # Verify sub-agent event
        sub_agent_call = sample_trace_context._sdk._client.add_event_async.call_args
        sub_agent_event = sub_agent_call[0][0].event

        assert sub_agent_event.parentEventId == parent_agent.id

    @pytest.mark.asyncio
    async def test_agent_success(self, sample_trace_context, sample_event, sample_token_usage):
        """Test agent success completion."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})

        await agent.success(output={"result": "completed"}, usage=sample_token_usage, metadata={"final": True})

        # Verify success event was created
        success_call = sample_trace_context._sdk._client.add_event_async.call_args
        success_event = success_call[0][0].event

        assert success_event.name == "agent:success"
        assert success_event.status == EventStatus.COMPLETED
        assert success_event.output == {"result": "completed"}
        assert success_event.parentEventId == agent.id


class TestEventContext:
    """Test event context operations."""

    @pytest.mark.asyncio
    async def test_event_update(self, sample_trace_context, sample_event):
        """Test event context update."""
        # Create an actual EventContext
        event_context = EventContext(sample_event, "test-trace-123", sample_trace_context._sdk)

        await event_context.update(status=EventStatus.COMPLETED, output={"result": "done"})

        # Verify update was called
        sample_trace_context._sdk._client.update_event_async.assert_called()

    @pytest.mark.asyncio
    async def test_event_success(self, sample_trace_context, sample_event):
        """Test event success completion."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event

        event_context = EventContext(sample_event, "test-trace-123", sample_trace_context._sdk)

        await event_context.success(output={"result": "success"}, metadata={"extra": "data"})

        # Verify success event was created
        success_call = sample_trace_context._sdk._client.add_event_async.call_args
        success_event = success_call[0][0].event

        assert success_event.name == f"{sample_event.event_type}:success"
        assert success_event.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_event_error_with_exception(self, sample_trace_context, sample_event):
        """Test event error with Python exception."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event

        event_context = EventContext(sample_event, "test-trace-123", sample_trace_context._sdk)

        test_error = ValueError("Test error")
        await event_context.error(status_message=test_error, metadata={"error_type": "validation"})

        # Verify error event was created
        error_call = sample_trace_context._sdk._client.add_event_async.call_args
        error_event = error_call[0][0].event

        assert error_event.name == f"{sample_event.event_type}:error"
        assert error_event.status == EventStatus.ERROR
        assert error_event.level == EventLevel.ERROR
        assert "message" in error_event.statusMessage

    @pytest.mark.asyncio
    async def test_event_error_with_string(self, sample_trace_context, sample_event):
        """Test event error with string message."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event

        event_context = EventContext(sample_event, "test-trace-123", sample_trace_context._sdk)

        await event_context.error(status_message="Simple error message", metadata={"category": "user_error"})

        # Verify error event was created
        error_call = sample_trace_context._sdk._client.add_event_async.call_args
        error_event = error_call[0][0].event

        assert error_event.name == f"{sample_event.event_type}:error"
        assert error_event.statusMessage == "Simple error message"


class TestSpecializedContexts:
    """Test specialized context operations."""

    @pytest.mark.asyncio
    async def test_tool_context_success(self, sample_trace_context, sample_event):
        """Test tool context success."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})
        tool = await agent.add_tool({"name": "Test Tool"})

        await tool.success(output={"data": "processed"}, metadata={"duration": "0.5s"})

        # Verify tool success event
        success_call = sample_trace_context._sdk._client.add_event_async.call_args
        success_event = success_call[0][0].event

        assert success_event.name == "tool:success"
        assert success_event.type == "tool"
        assert success_event.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_memory_context_success(self, sample_trace_context, sample_event):
        """Test memory context success."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})
        memory = await agent.add_memory({"name": "Test Memory"})

        await memory.success(output={"stored": True, "key": "data_key"}, metadata={"ttl": 3600})

        # Verify memory success event
        success_call = sample_trace_context._sdk._client.add_event_async.call_args
        success_event = success_call[0][0].event

        assert success_event.name == "memory:write_success"
        assert success_event.type == "memory"
        assert success_event.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_retriever_context_success(self, sample_trace_context, sample_event):
        """Test retriever context success."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})
        retriever = await agent.add_retriever({"name": "Test Retriever"})

        await retriever.success(output={"documents": ["doc1", "doc2"], "count": 2}, metadata={"search_time": "0.2s"})

        # Verify retriever success event
        success_call = sample_trace_context._sdk._client.add_event_async.call_args
        success_event = success_call[0][0].event

        assert success_event.name == "retriever:success"
        assert success_event.type == "retriever"
        assert success_event.status == EventStatus.COMPLETED


class TestMetadataConversion:
    """Test metadata conversion in context operations."""

    @pytest.mark.asyncio
    async def test_agent_metadata_conversion(self, sample_trace_context, sample_event):
        """Test that agent metadata is converted to camelCase."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event

        await sample_trace_context.add_agent(
            {
                "name": "Test Agent",
                "metadata": {
                    "model_parameters": {"model": "gpt-4"},
                    "max_retries": 3,
                    "api_config": {"base_url": "test"},
                },
            }
        )

        # Verify metadata was converted
        event_call = sample_trace_context._sdk._client.add_event_async.call_args
        event_data = event_call[0][0].event

        assert "modelParameters" in event_data.metadata
        assert "maxRetries" in event_data.metadata
        assert "apiConfig" in event_data.metadata
        assert event_data.metadata["modelParameters"]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_tool_metadata_conversion(self, sample_trace_context, sample_event):
        """Test that tool metadata is converted to camelCase."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})

        await agent.add_tool(
            {"name": "Test Tool", "metadata": {"execution_timeout": 30, "retry_policy": {"max_attempts": 3}}}
        )

        # Verify metadata was converted
        tool_call = sample_trace_context._sdk._client.add_event_async.call_args
        tool_event = tool_call[0][0].event

        assert "executionTimeout" in tool_event.metadata
        assert "retryPolicy" in tool_event.metadata
        assert tool_event.metadata["retryPolicy"]["maxAttempts"] == 3


class TestDictToObjectConversion:
    """Test automatic conversion from dict to Pydantic objects."""

    @pytest.mark.asyncio
    async def test_agent_options_dict_conversion(self, sample_trace_context, sample_event):
        """Test that dict is automatically converted to AgentOptions."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event

        agent_dict = {
            "name": "Dict Agent",
            "input": {"task": "test"},
            "instructions": "Test instructions",
            "metadata": {"type": "test"},
        }

        agent_context = await sample_trace_context.add_agent(agent_dict)
        assert isinstance(agent_context, AgentContext)

        # Verify the dict was properly converted
        event_call = sample_trace_context._sdk._client.add_event_async.call_args
        event_data = event_call[0][0].event

        assert event_data.metadata["displayName"] == "Dict Agent"
        assert event_data.input == {"task": "test"}

    @pytest.mark.asyncio
    async def test_tool_options_dict_conversion(self, sample_trace_context, sample_event):
        """Test that dict is automatically converted to ToolOptions."""
        sample_trace_context._sdk._client.add_event_async.return_value = sample_event
        agent = await sample_trace_context.add_agent({"name": "Test Agent"})

        tool_dict = {"name": "Dict Tool", "input": {"param": "value"}, "metadata": {"version": "1.0"}}

        tool_context = await agent.add_tool(tool_dict)
        assert isinstance(tool_context, ToolContext)


class TestSDKLifecycle:
    """Test SDK lifecycle management."""

    @pytest.mark.asyncio
    async def test_sdk_context_manager(self, sample_config):
        """Test SDK as async context manager."""
        async with VoltAgentSDK(**sample_config.model_dump()) as sdk:
            assert isinstance(sdk, VoltAgentSDK)

        # SDK should be shut down after context exit

    @pytest.mark.asyncio
    async def test_sdk_flush(self, sdk_with_mock_client):
        """Test SDK flush method."""
        await sdk_with_mock_client.flush()
        # Currently a no-op, but should not raise errors

    @pytest.mark.asyncio
    async def test_sdk_shutdown(self, sdk_with_mock_client):
        """Test SDK shutdown."""
        # Create a mock auto-flush task that is already done
        mock_task = Mock()
        mock_task.done.return_value = True  # Task is already done, so cancel won't be called
        mock_task.cancel = Mock()  # Mock cancel method
        sdk_with_mock_client._auto_flush_task = mock_task

        await sdk_with_mock_client.shutdown()

        # Verify cleanup was called
        sdk_with_mock_client._client.aclose.assert_called_once()
        # Cancel should NOT be called since task is already done
        mock_task.cancel.assert_not_called()


class TestEventAddition:
    """Test event addition to traces."""

    @pytest.mark.asyncio
    async def test_add_event_to_trace(self, sdk_with_mock_client, sample_event):
        """Test adding event to trace."""
        sdk_with_mock_client._client.add_event_async.return_value = sample_event

        event_data = {"name": "custom:event", "type": "custom", "input": {"data": "test"}}

        result = await sdk_with_mock_client.add_event_to_trace("test-trace-123", event_data)

        assert result == sample_event

        # Verify event was created with UUID and timestamp
        event_call = sdk_with_mock_client._client.add_event_async.call_args
        timeline_event = event_call[0][0].event

        assert timeline_event.name == "custom:event"
        assert timeline_event.type == "custom"
        assert timeline_event.id is not None  # UUID should be generated
        assert timeline_event.startTime is not None  # Timestamp should be set


class TestErrorScenarios:
    """Test various error scenarios in SDK operations."""

    @pytest.mark.asyncio
    async def test_trace_creation_error(self, sdk_with_mock_client):
        """Test trace creation API error."""
        # Mock API error
        sdk_with_mock_client._client.add_history_async.side_effect = APIError("API Error", status_code=400)

        with pytest.raises(APIError):
            await sdk_with_mock_client.create_trace(agentId="test-agent")

    @pytest.mark.asyncio
    async def test_event_creation_error(self, sample_trace_context):
        """Test event creation error."""
        # Mock API error for event creation
        sample_trace_context._sdk._client.add_event_async.side_effect = APIError(
            "Event creation failed", status_code=400
        )

        with pytest.raises(APIError):
            await sample_trace_context.add_agent({"name": "Test Agent"})

    @pytest.mark.asyncio
    async def test_invalid_agent_options(self, sample_trace_context):
        """Test invalid agent options validation."""
        with pytest.raises(PydanticValidationError):
            await sample_trace_context.add_agent({})  # Missing required 'name' field

    @pytest.mark.asyncio
    async def test_invalid_trace_options(self, sdk_with_mock_client):
        """Test invalid trace options validation."""
        with pytest.raises(PydanticValidationError):
            await sdk_with_mock_client.create_trace()  # Missing required 'agentId' field


class TestPropertyAccess:
    """Test property access on SDK and contexts."""

    def test_sdk_client_property(self, sdk_with_mock_client):
        """Test SDK client property access."""
        client = sdk_with_mock_client.client
        assert client == sdk_with_mock_client._client

    def test_trace_context_properties(self, sample_trace_context):
        """Test trace context property access."""
        assert sample_trace_context.id == "test-trace-123"
        assert sample_trace_context.agent_id == "test-agent"
