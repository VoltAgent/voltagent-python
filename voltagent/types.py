"""Type definitions for VoltAgent Python SDK using Pydantic for validation."""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ===== UTILITY FUNCTIONS =====


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase.

    Args:
        snake_str: String in snake_case format

    Returns:
        String in camelCase format
    """
    components = snake_str.split("_")
    return components[0] + "".join(word.capitalize() for word in components[1:])


def convert_metadata_to_camel_case(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert metadata dictionary keys from snake_case to camelCase.

    This allows Python developers to use snake_case (Python convention)
    while sending camelCase to the API (API convention).

    Args:
        metadata: Dictionary with potentially snake_case keys

    Returns:
        Dictionary with camelCase keys
    """
    if not metadata:
        return metadata

    converted = {}
    for key, value in metadata.items():
        # Convert key to camelCase
        camel_key = snake_to_camel(key)

        # Recursively convert nested dictionaries
        if isinstance(value, dict):
            converted[camel_key] = convert_metadata_to_camel_case(value)
        else:
            converted[camel_key] = value

    return converted


# ===== ENUMS =====


class EventStatus(str, Enum):
    """Status of an event or trace."""

    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class EventLevel(str, Enum):
    """Logging level for events."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class TraceStatus(str, Enum):
    """Status of a trace/history."""

    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"


# ===== CORE DATA MODELS =====


class TokenUsage(BaseModel):
    """Token usage tracking for LLM operations."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the prompt")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in the completion")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens used")


class TimelineEvent(BaseModel):
    """Core timeline event structure."""

    model_config = ConfigDict(extra="allow")

    # Core identification
    id: Optional[str] = Field(None, description="Unique event identifier")
    name: str = Field(..., description="Event name")
    type: str = Field(..., description="Event type (agent, tool, memory, retriever)")

    # Timing
    startTime: Optional[str] = Field(None, description="Event start time (ISO 8601)")
    endTime: Optional[str] = Field(None, description="Event end time (ISO 8601)")

    # Status and level
    status: EventStatus = Field(EventStatus.RUNNING, description="Current event status")
    level: EventLevel = Field(EventLevel.INFO, description="Event logging level")
    statusMessage: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Status message or error details")

    # Hierarchy
    parentEventId: Optional[str] = Field(None, description="Parent event ID for nested events")
    traceId: Optional[str] = Field(None, description="Associated trace ID")

    # Data
    input: Optional[Dict[str, Any]] = Field(None, description="Event input data")
    output: Optional[Dict[str, Any]] = Field(None, description="Event output data")

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Event tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    version: Optional[str] = Field(None, description="Event version")


class Event(BaseModel):
    """API response event model."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Unique event identifier")
    history_id: str = Field(..., description="Associated history/trace ID")
    event_type: str = Field(..., description="Event type")
    event_name: str = Field(..., description="Event name")

    # Timing
    start_time: Optional[str] = Field(None, description="Event start time")
    end_time: Optional[str] = Field(None, description="Event end time")

    # Status
    status: EventStatus = Field(..., description="Event status")
    status_message: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Status message")
    level: EventLevel = Field(EventLevel.INFO, description="Event level")

    # Hierarchy
    parent_event_id: Optional[str] = Field(None, description="Parent event ID")

    # Data
    input: Optional[Dict[str, Any]] = Field(None, description="Event input")
    output: Optional[Dict[str, Any]] = Field(None, description="Event output")

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Event tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Event metadata")
    version: Optional[str] = Field(None, description="Event version")

    # Timestamps
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class History(BaseModel):
    """History/Trace model."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Unique history identifier")
    agent_id: str = Field(..., description="Associated agent identifier")

    # User context
    user_id: Optional[str] = Field(None, description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")

    # Timing
    start_time: Optional[str] = Field(None, description="Trace start time")
    end_time: Optional[str] = Field(None, description="Trace end time")
    completion_start_time: Optional[str] = Field(None, description="Completion start time")

    # Status
    status: TraceStatus = Field(TraceStatus.WORKING, description="Trace status")

    # Data
    input: Optional[Dict[str, Any]] = Field(None, description="Trace input")
    output: Optional[Dict[str, Any]] = Field(None, description="Trace output")

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Trace tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Trace metadata")
    version: Optional[str] = Field(None, description="Trace version")
    level: Optional[EventLevel] = Field(None, description="Trace level")

    # Usage tracking
    usage: Optional[TokenUsage] = Field(None, description="Token usage for this trace")

    # Timestamps
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


# ===== REQUEST MODELS =====


class CreateHistoryRequest(BaseModel):
    """Request model for creating a new history/trace."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = Field(None, description="Custom trace ID")
    agent_id: str = Field(..., description="Agent identifier")

    # User context
    userId: Optional[str] = Field(None, description="User identifier")
    conversationId: Optional[str] = Field(None, description="Conversation identifier")

    # Timing
    startTime: Optional[str] = Field(None, description="Trace start time")
    completionStartTime: Optional[str] = Field(None, description="Completion start time")

    # Status
    status: TraceStatus = Field(TraceStatus.WORKING, description="Initial trace status")

    # Data
    input: Optional[Dict[str, Any]] = Field(None, description="Trace input")

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Trace tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Trace metadata")
    version: Optional[str] = Field(None, description="Trace version")
    level: Optional[EventLevel] = Field(None, description="Trace level")


class UpdateHistoryRequest(BaseModel):
    """Request model for updating a history/trace."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="History ID to update")

    # Timing
    end_time: Optional[str] = Field(None, description="Trace end time")
    completion_start_time: Optional[str] = Field(None, description="Completion start time")

    # Status
    status: Optional[TraceStatus] = Field(None, description="Updated trace status")

    # Data
    input: Optional[Dict[str, Any]] = Field(None, description="Updated trace input")
    output: Optional[Dict[str, Any]] = Field(None, description="Trace output")

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Updated trace tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated trace metadata")
    version: Optional[str] = Field(None, description="Updated trace version")
    level: Optional[EventLevel] = Field(None, description="Updated trace level")

    # Usage
    usage: Optional[TokenUsage] = Field(None, description="Token usage for this trace")


class AddEventRequest(BaseModel):
    """Request model for adding an event to a trace."""

    model_config = ConfigDict(extra="allow")

    history_id: str = Field(..., description="History/trace ID")
    event: TimelineEvent = Field(..., description="Event to add")


class UpdateEventRequest(BaseModel):
    """Request model for updating an event."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Event ID to update")

    # Timing
    end_time: Optional[str] = Field(None, description="Event end time")

    # Status
    status: Optional[EventStatus] = Field(None, description="Updated event status")
    status_message: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Updated status message")
    level: Optional[EventLevel] = Field(None, description="Updated event level")

    # Data
    input: Optional[Dict[str, Any]] = Field(None, description="Updated event input")
    output: Optional[Dict[str, Any]] = Field(None, description="Event output")

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Updated event tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated event metadata")
    version: Optional[str] = Field(None, description="Updated event version")


# ===== CONFIGURATION =====


class VoltAgentConfig(BaseModel):
    """SDK configuration model."""

    model_config = ConfigDict(extra="allow")

    base_url: str = Field(..., description="API base URL")
    public_key: str = Field(..., description="Public API key")
    secret_key: str = Field(..., description="Secret API key")

    # HTTP settings
    timeout: Union[int, float] = Field(30, description="Request timeout in seconds")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional HTTP headers")

    # Flush settings
    auto_flush: bool = Field(True, description="Enable automatic event flushing")
    flush_interval: int = Field(5, description="Auto-flush interval in seconds")

    # Retry settings
    max_retries: int = Field(3, description="Maximum number of retries")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")


# ===== CONTEXT OPTIONS =====


class TraceOptions(BaseModel):
    """Options for creating a new trace."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: Optional[str] = Field(None, description="Custom trace ID")
    agentId: str = Field(..., alias="agent_id", description="Agent identifier")

    # User context
    userId: Optional[str] = Field(None, alias="user_id", description="User identifier")
    conversationId: Optional[str] = Field(None, alias="conversation_id", description="Conversation identifier")

    # Data
    input: Optional[Dict[str, Any]] = Field(None, description="Trace input data")

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Trace tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Trace metadata")
    version: Optional[str] = Field(None, description="Trace version")
    level: Optional[EventLevel] = Field(None, description="Trace level")

    # Timing
    startTime: Optional[str] = Field(None, alias="start_time", description="Custom start time")
    completionStartTime: Optional[str] = Field(None, alias="completion_start_time", description="Completion start time")


class AgentOptions(BaseModel):
    """Options for creating an agent event."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Agent name")
    input: Optional[Dict[str, Any]] = Field(None, description="Agent input data")
    instructions: Optional[str] = Field(None, description="Agent instructions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Agent metadata")


class ToolOptions(BaseModel):
    """Options for creating a tool event."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Tool name")
    input: Optional[Dict[str, Any]] = Field(None, description="Tool input data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Tool metadata")


class MemoryOptions(BaseModel):
    """Options for creating a memory event."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Memory operation name")
    input: Optional[Dict[str, Any]] = Field(None, description="Memory input data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Memory metadata")


class RetrieverOptions(BaseModel):
    """Options for creating a retriever event."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Retriever name")
    input: Optional[Dict[str, Any]] = Field(None, description="Retriever input data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Retriever metadata")


# ===== SUCCESS/ERROR OPTIONS =====


class SuccessOptions(BaseModel):
    """Base success options."""

    model_config = ConfigDict(extra="allow")

    output: Optional[Dict[str, Any]] = Field(None, description="Success output data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AgentSuccessOptions(SuccessOptions):
    """Success options for agents."""

    usage: Optional[TokenUsage] = Field(None, description="Token usage")


class ErrorOptions(BaseModel):
    """Base error options."""

    model_config = ConfigDict(extra="allow")

    status_message: Union[str, Dict[str, Any], Any] = Field(..., description="Error message or object")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional error metadata")


# Type aliases for convenience
ToolSuccessOptions = SuccessOptions
MemorySuccessOptions = SuccessOptions
RetrieverSuccessOptions = SuccessOptions

ToolErrorOptions = ErrorOptions
MemoryErrorOptions = ErrorOptions
RetrieverErrorOptions = ErrorOptions
AgentErrorOptions = ErrorOptions


# ===== EXCEPTIONS =====


class VoltAgentError(Exception):
    """Base exception for VoltAgent SDK."""

    pass


class APIError(VoltAgentError):
    """API-related errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class TimeoutError(VoltAgentError):
    """Request timeout errors."""

    pass


class ValidationError(VoltAgentError):
    """Data validation errors."""

    pass
