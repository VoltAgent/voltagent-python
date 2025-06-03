"""
VoltAgent Python SDK - Observability and Tracing for AI Agents

This SDK provides comprehensive observability for AI agent applications,
allowing you to trace agent interactions, tool usage, and system events.

Example usage:
    ```python
    import asyncio
    from voltagent import VoltAgentSDK

    async def main():
        sdk = VoltAgentSDK(
            base_url="https://api.voltagent.dev",
            public_key="your-public-key",
            secret_key="your-secret-key"
        )

        async with sdk.trace(agent_id="my-agent") as trace:
            async with trace.add_agent(name="Assistant") as agent:
                await agent.success(output="Task completed")

    asyncio.run(main())
    ```
"""

from .client import VoltAgentCoreAPI
from .sdk import VoltAgentSDK
from .types import (  # Configuration; Request/Response Types; Core Data Types; Context Options; Status and Level Enums; Usage Tracking; Exception Types
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
    ValidationError,
    VoltAgentConfig,
    VoltAgentError,
)

# Version info
__version__ = "0.1.1"
__author__ = "VoltAgent Team"

# Public API
__all__ = [
    # Main SDK class
    "VoltAgentSDK",
    "VoltAgentCoreAPI",
    # Configuration
    "VoltAgentConfig",
    # Request/Response Types
    "CreateHistoryRequest",
    "UpdateHistoryRequest",
    "AddEventRequest",
    "UpdateEventRequest",
    # Core Data Types
    "History",
    "Event",
    "TimelineEvent",
    # Context Options
    "TraceOptions",
    "AgentOptions",
    "ToolOptions",
    "MemoryOptions",
    "RetrieverOptions",
    # Enums
    "EventStatus",
    "EventLevel",
    "TraceStatus",
    # Usage Tracking
    "TokenUsage",
    # Exceptions
    "VoltAgentError",
    "APIError",
    "TimeoutError",
    "ValidationError",
    # Version
    "__version__",
]
