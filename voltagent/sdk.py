"""Main VoltAgent SDK with high-level async/sync API and context managers."""

import asyncio
import uuid
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, AsyncContextManager, ContextManager, Dict, List, Optional, Union

from .client import VoltAgentCoreAPI
from .types import (
    AddEventRequest,
    AgentOptions,
    AgentSuccessOptions,
    CreateHistoryRequest,
    ErrorOptions,
    Event,
    EventLevel,
    EventStatus,
    History,
    MemoryOptions,
    RetrieverOptions,
    SuccessOptions,
    TimelineEvent,
    TokenUsage,
    ToolOptions,
    TraceOptions,
    TraceStatus,
    UpdateEventRequest,
    UpdateHistoryRequest,
    VoltAgentConfig,
    convert_metadata_to_camel_case,
)

# ===== CONTEXT CLASSES =====


class TraceContextManager:
    """Async context manager for trace creation."""

    def __init__(self, sdk: "VoltAgentSDK", options: Dict[str, Any]):
        self._sdk = sdk
        self._options = options
        self._trace_context: Optional["TraceContext"] = None

    async def __aenter__(self) -> "TraceContext":
        """Create and return trace context."""
        self._trace_context = await self._sdk.create_trace(**self._options)
        return self._trace_context

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]
    ) -> None:
        """Handle trace completion."""
        if self._trace_context:
            await self._trace_context.__aexit__(exc_type, exc_val, exc_tb)


class EventContext:
    """Context for managing individual events."""

    def __init__(self, event: Event, trace_id: str, sdk: "VoltAgentSDK", parent_id: Optional[str] = None):
        self.id = event.id
        self.trace_id = trace_id
        self.parent_id = parent_id
        self._event = event
        self._sdk = sdk

    async def update(self, **kwargs) -> None:
        """Update the event."""
        await self._sdk.update_event(self.id, **kwargs)

    async def success(self, output: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark event as successful."""
        event_name = f"{self._event.event_type}:success"
        await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": event_name,
                "type": self._event.event_type,
                "status": EventStatus.COMPLETED,
                "output": output or {},
                "parentEventId": self.id,
                "metadata": {**(self._event.metadata or {}), **(metadata or {})},
            },
        )

    async def error(
        self, status_message: Union[str, Exception, Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark event as failed."""
        # Handle different error types
        if isinstance(status_message, Exception):
            error_data = {
                "message": str(status_message),
                "type": type(status_message).__name__,
                "stack": getattr(status_message, "__traceback__", None),
            }
        else:
            error_data = status_message

        event_name = f"{self._event.event_type}:error"
        await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": event_name,
                "type": self._event.event_type,
                "status": EventStatus.ERROR,
                "level": EventLevel.ERROR,
                "statusMessage": error_data,
                "parentEventId": self.id,
                "metadata": {**(self._event.metadata or {}), **(metadata or {})},
            },
        )


class AgentContext(EventContext):
    """Context for managing agent events with nested capabilities."""

    async def add_agent(self, options: Union[AgentOptions, Dict[str, Any]]) -> "AgentContext":
        """Add a sub-agent to this agent."""
        # Convert dict to AgentOptions if needed
        if isinstance(options, dict):
            agent_options = AgentOptions(**options)
        else:
            agent_options = options

        # Convert metadata to camelCase for API compatibility
        converted_metadata = convert_metadata_to_camel_case(agent_options.metadata)

        sub_agent_event = await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": "agent:start",
                "type": "agent",
                "status": EventStatus.RUNNING,
                "input": agent_options.input or {},
                "metadata": {
                    "displayName": agent_options.name,
                    "id": agent_options.name,
                    "agentId": self._event.metadata.get("id") if self._event.metadata else self.id,
                    "instructions": agent_options.instructions,
                    **(converted_metadata or {}),
                },
                "parentEventId": self.id,
            },
        )

        return AgentContext(sub_agent_event, self.trace_id, self._sdk, self.id)

    async def add_tool(self, options: Union[ToolOptions, Dict[str, Any]]) -> "ToolContext":
        """Add a tool to this agent."""
        # Convert dict to ToolOptions if needed
        if isinstance(options, dict):
            tool_options = ToolOptions(**options)
        else:
            tool_options = options

        # Convert metadata to camelCase for API compatibility
        converted_metadata = convert_metadata_to_camel_case(tool_options.metadata)

        tool_event = await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": "tool:start",
                "type": "tool",
                "input": tool_options.input or {},
                "status": EventStatus.RUNNING,
                "metadata": {
                    "displayName": tool_options.name,
                    "id": tool_options.name,
                    "agentId": self._event.metadata.get("id") if self._event.metadata else self.id,
                    **(converted_metadata or {}),
                },
                "parentEventId": self.id,
            },
        )

        return ToolContext(tool_event, self.trace_id, self._sdk, self.id)

    async def add_memory(self, options: Union[MemoryOptions, Dict[str, Any]]) -> "MemoryContext":
        """Add a memory operation to this agent."""
        # Convert dict to MemoryOptions if needed
        if isinstance(options, dict):
            memory_options = MemoryOptions(**options)
        else:
            memory_options = options

        # Convert metadata to camelCase for API compatibility
        converted_metadata = convert_metadata_to_camel_case(memory_options.metadata)

        memory_event = await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": "memory:write_start",
                "type": "memory",
                "input": memory_options.input or {},
                "status": EventStatus.RUNNING,
                "metadata": {
                    "displayName": memory_options.name,
                    "id": memory_options.name,
                    "agentId": self._event.metadata.get("id") if self._event.metadata else self.id,
                    **(converted_metadata or {}),
                },
                "parentEventId": self.id,
            },
        )

        return MemoryContext(memory_event, self.trace_id, self._sdk, self.id)

    async def add_retriever(self, options: Union[RetrieverOptions, Dict[str, Any]]) -> "RetrieverContext":
        """Add a retriever to this agent."""
        # Convert dict to RetrieverOptions if needed
        if isinstance(options, dict):
            retriever_options = RetrieverOptions(**options)
        else:
            retriever_options = options

        # Convert metadata to camelCase for API compatibility
        converted_metadata = convert_metadata_to_camel_case(retriever_options.metadata)

        retriever_event = await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": "retriever:start",
                "type": "retriever",
                "input": retriever_options.input or {},
                "status": EventStatus.RUNNING,
                "metadata": {
                    "displayName": retriever_options.name,
                    "id": retriever_options.name,
                    "agentId": self._event.metadata.get("id") if self._event.metadata else self.id,
                    **(converted_metadata or {}),
                },
                "parentEventId": self.id,
            },
        )

        return RetrieverContext(retriever_event, self.trace_id, self._sdk, self.id)

    async def success(
        self,
        output: Optional[Dict[str, Any]] = None,
        usage: Optional[TokenUsage] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark agent as successful."""
        final_metadata = {**(self._event.metadata or {}), **(metadata or {})}
        if usage:
            final_metadata["usage"] = usage.model_dump()

        await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": "agent:success",
                "type": "agent",
                "status": EventStatus.COMPLETED,
                "output": output or {},
                "parentEventId": self.id,
                "metadata": final_metadata,
            },
        )


class ToolContext(EventContext):
    """Context for managing tool events."""

    async def success(self, output: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark tool as successful."""
        await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": "tool:success",
                "type": "tool",
                "status": EventStatus.COMPLETED,
                "output": output or {},
                "parentEventId": self.id,
                "metadata": {**(self._event.metadata or {}), **(metadata or {})},
            },
        )


class MemoryContext(EventContext):
    """Context for managing memory events."""

    async def success(self, output: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark memory operation as successful."""
        await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": "memory:write_success",
                "type": "memory",
                "status": EventStatus.COMPLETED,
                "output": output or {},
                "parentEventId": self.id,
                "metadata": {**(self._event.metadata or {}), **(metadata or {})},
            },
        )


class RetrieverContext(EventContext):
    """Context for managing retriever events."""

    async def success(self, output: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark retriever as successful."""
        await self._sdk.add_event_to_trace(
            self.trace_id,
            {
                "name": "retriever:success",
                "type": "retriever",
                "status": EventStatus.COMPLETED,
                "output": output or {},
                "parentEventId": self.id,
                "metadata": {**(self._event.metadata or {}), **(metadata or {})},
            },
        )


class TraceContext:
    """Context for managing traces with nested operations."""

    def __init__(self, history: History, sdk: "VoltAgentSDK"):
        self.id = history.id
        self.agent_id = history.agent_id
        self._history = history
        self._sdk = sdk

    async def update(self, **kwargs) -> None:
        """Update the trace."""
        self._history = await self._sdk.update_trace(self.id, **kwargs)

    async def end(
        self,
        output: Optional[Dict[str, Any]] = None,
        status: TraceStatus = TraceStatus.COMPLETED,
        usage: Optional[TokenUsage] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End the trace."""
        update_data: Dict[str, Any] = {
            "status": status,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }

        if output:
            update_data["output"] = output
        if usage:
            update_data["usage"] = usage
        if metadata:
            update_data["metadata"] = {**(self._history.metadata or {}), **metadata}

        await self._sdk.update_trace(self.id, **update_data)

    async def add_agent(self, options: Union[AgentOptions, Dict[str, Any]]) -> AgentContext:
        """Add an agent to this trace."""
        # Convert dict to AgentOptions if needed
        if isinstance(options, dict):
            agent_options = AgentOptions(**options)
        else:
            agent_options = options

        # Convert metadata to camelCase for API compatibility
        converted_metadata = convert_metadata_to_camel_case(agent_options.metadata)

        agent_event = await self._sdk.add_event_to_trace(
            self.id,
            {
                "name": "agent:start",
                "type": "agent",
                "input": agent_options.input or {},
                "status": EventStatus.RUNNING,
                "metadata": {
                    "displayName": agent_options.name,
                    "id": agent_options.name,
                    "agentId": self.agent_id,
                    "instructions": agent_options.instructions,
                    **(converted_metadata or {}),
                },
            },
        )

        return AgentContext(agent_event, self.id, self._sdk)

    async def add_event(self, event_data: Dict[str, Any]) -> EventContext:
        """Add a custom event to this trace."""
        created_event = await self._sdk.add_event_to_trace(self.id, event_data)
        return EventContext(created_event, self.id, self._sdk)

    # Context manager support
    async def __aenter__(self) -> "TraceContext":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]
    ) -> None:
        """Async context manager exit."""
        if exc_type:
            # Error occurred, mark trace as failed
            await self.end(status=TraceStatus.ERROR, metadata={"error": str(exc_val) if exc_val else "Unknown error"})
        else:
            # Normal completion
            await self.end()


# ===== MAIN SDK CLASS =====


class VoltAgentSDK:
    """Main VoltAgent SDK with async support and context managers.

    This SDK provides high-level methods for tracing AI agent operations
    with support for both synchronous and asynchronous usage patterns.
    """

    def __init__(self, **config_kwargs):
        """Initialize the SDK.

        Args:
            **config_kwargs: Configuration parameters (base_url, public_key, secret_key, etc.)
        """
        self.config = VoltAgentConfig(**config_kwargs)
        self._client = VoltAgentCoreAPI(self.config)
        self._traces: Dict[str, History] = {}
        self._auto_flush_task: Optional[asyncio.Task] = None
        self._event_queue: List[Dict[str, Any]] = []

        # Start auto-flush if enabled
        if self.config.auto_flush:
            self._start_auto_flush()

    def _start_auto_flush(self) -> None:
        """Start the auto-flush background task."""

        async def auto_flush_loop():
            while True:
                await asyncio.sleep(self.config.flush_interval)
                try:
                    await self.flush()
                except Exception as e:
                    # Log error but continue
                    print(f"Auto-flush error: {e}")

        try:
            # Get current event loop or create one
            loop = asyncio.get_event_loop()
            self._auto_flush_task = loop.create_task(auto_flush_loop())
        except RuntimeError:
            # No event loop running, auto-flush will be disabled
            pass

    # ===== TRACE MANAGEMENT =====

    def trace(self, **options):
        """Create a new trace context manager.

        Args:
            **options: Trace options (agent_id, input, user_id, etc.)

        Returns:
            TraceContextManager for use with async with
        """
        return TraceContextManager(self, options)

    async def create_trace(self, **options) -> TraceContext:
        """Create a new trace (direct method).

        Args:
            **options: Trace options (agent_id, input, user_id, etc.)

        Returns:
            TraceContext for managing the trace
        """
        trace_options = TraceOptions(**options)

        # Prepare request data
        request_data = CreateHistoryRequest(
            id=trace_options.id or str(uuid.uuid4()),
            agent_id=trace_options.agentId,
            userId=trace_options.userId,
            conversationId=trace_options.conversationId,
            input=trace_options.input,
            tags=trace_options.tags,
            metadata={
                "agentId": trace_options.agentId,
                **(trace_options.metadata or {}),
            },
            version=trace_options.version,
            level=trace_options.level,
            startTime=trace_options.startTime or datetime.now(timezone.utc).isoformat(),
            completionStartTime=trace_options.completionStartTime,
            status=TraceStatus.WORKING,
        )

        # Create trace
        history = await self._client.add_history_async(request_data)
        self._traces[history.id] = history

        return TraceContext(history, self)

    async def get_trace(self, trace_id: str) -> Optional[History]:
        """Get trace data by ID."""
        return self._traces.get(trace_id)

    async def update_trace(self, trace_id: str, **kwargs) -> History:
        """Update a trace."""
        update_request = UpdateHistoryRequest(id=trace_id, **kwargs)
        updated_history = await self._client.update_history_async(update_request)
        self._traces[trace_id] = updated_history
        return updated_history

    # ===== EVENT MANAGEMENT =====

    async def add_event_to_trace(self, trace_id: str, event_data: Dict[str, Any]) -> Event:
        """Add an event to a trace."""
        # Create timeline event
        timeline_event = TimelineEvent(
            id=str(uuid.uuid4()), startTime=datetime.now(timezone.utc).isoformat(), traceId=trace_id, **event_data
        )

        # Create request
        request = AddEventRequest(history_id=trace_id, event=timeline_event)

        # Add event
        return await self._client.add_event_async(request)

    async def update_event(self, event_id: str, **kwargs) -> Event:
        """Update an event."""
        update_request = UpdateEventRequest(id=event_id, **kwargs)
        return await self._client.update_event_async(update_request)

    # ===== FLUSH AND CLEANUP =====

    async def flush(self) -> None:
        """Flush any pending events."""
        # For now, just a placeholder as events are sent immediately
        # In future versions, this could batch events for efficiency
        pass

    async def shutdown(self) -> None:
        """Shutdown the SDK and clean up resources."""
        # Cancel auto-flush task if running
        if self._auto_flush_task and not self._auto_flush_task.done():
            self._auto_flush_task.cancel()
            try:
                await self._auto_flush_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling

        # Final flush
        await self.flush()

        # Close HTTP client
        await self._client.aclose()

    # ===== CONTEXT MANAGER SUPPORT =====

    async def __aenter__(self) -> "VoltAgentSDK":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]
    ) -> None:
        """Async context manager exit."""
        await self.shutdown()

    # ===== DIRECT ACCESS =====

    @property
    def client(self) -> VoltAgentCoreAPI:
        """Get direct access to the core API client."""
        return self._client
