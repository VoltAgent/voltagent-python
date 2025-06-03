"""Core HTTP client for VoltAgent API communication."""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional, Union

import httpx
from pydantic import ValidationError as PydanticValidationError

from .types import (
    AddEventRequest,
    APIError,
    CreateHistoryRequest,
    Event,
    History,
    TimelineEvent,
    TimeoutError,
    UpdateEventRequest,
    UpdateHistoryRequest,
    ValidationError,
    VoltAgentConfig,
)


class VoltAgentCoreAPI:
    """Core HTTP client for VoltAgent API communication.

    This class provides low-level HTTP methods for communicating with the VoltAgent API.
    It supports both synchronous and asynchronous operations.
    """

    def __init__(self, config: Union[VoltAgentConfig, Dict[str, Any]]):
        """Initialize the API client.

        Args:
            config: Configuration object or dictionary with API settings
        """
        if isinstance(config, dict):
            self.config = VoltAgentConfig(**config)
        else:
            self.config = config

        # Prepare base URL (remove trailing slash)
        self.base_url = self.config.base_url.rstrip("/")

        # Set timeout directly for easy access (matching TypeScript client)
        self.timeout = self.config.timeout

        # Prepare headers
        self.headers = {
            "Content-Type": "application/json",
            "x-public-key": self.config.public_key,
            "x-secret-key": self.config.secret_key,
        }

        if self.config.headers:
            self.headers.update(self.config.headers)

        # HTTP clients (created on demand)
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    @property
    def sync_client(self) -> httpx.Client:
        """Get or create synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self.timeout,
                headers=self.headers,
            )
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create asynchronous HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.headers,
            )
        return self._async_client

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and extract data.

        Args:
            response: HTTP response object

        Returns:
            Parsed response data as expected by calling methods

        Raises:
            APIError: If the request failed
        """
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise APIError(f"Invalid JSON response: {response.text}", status_code=response.status_code)

        if not response.is_success:
            error_message = data.get("message", "An error occurred")
            raise APIError(error_message, status_code=response.status_code, response_data=data)

        return data

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, is_async: bool = False
    ) -> Union[Dict[str, Any], Any]:  # Any for coroutine
        """Make HTTP request (sync or async).

        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            endpoint: API endpoint (without base URL)
            data: Request data
            is_async: Whether to make async request

        Returns:
            Response data or coroutine
        """
        url = f"{self.base_url}{endpoint}"

        request_kwargs = {}
        if data:
            request_kwargs["json"] = data

        if is_async:
            return self._make_async_request(method, url, **request_kwargs)
        else:
            return self._make_sync_request(method, url, **request_kwargs)

    def _make_sync_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make synchronous HTTP request."""
        try:
            response = self.sync_client.request(method, url, **kwargs)
            return self._handle_response(response)
        except httpx.TimeoutException:
            raise TimeoutError("Request timeout")
        except httpx.RequestError as e:
            # Handle network errors with status_code 0 as expected by tests
            raise APIError(f"Network error: {e}", status_code=0)

    async def _make_async_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make asynchronous HTTP request."""
        try:
            response = await self.async_client.request(method, url, **kwargs)
            return self._handle_response(response)
        except httpx.TimeoutException:
            raise TimeoutError("Request timeout")
        except httpx.RequestError as e:
            # Handle network errors with status_code 0 as expected by tests
            raise APIError(f"Network error: {e}", status_code=0)

    # ===== HISTORY/TRACE METHODS =====

    def add_history(self, data: CreateHistoryRequest) -> History:
        """Create a new history/trace (sync).

        Args:
            data: History creation request

        Returns:
            Created history object
        """
        try:
            response_data = self._make_request("POST", "/history", data.model_dump(exclude_none=True))
            # API directly returns History object format
            return History(**response_data)
        except PydanticValidationError as e:
            raise ValidationError(f"Invalid response data: {e}")

    async def add_history_async(self, data: CreateHistoryRequest) -> History:
        """Create a new history/trace (async).

        Args:
            data: History creation request

        Returns:
            Created history object
        """
        try:
            response_data = await self._make_request(
                "POST", "/history", data.model_dump(exclude_none=True), is_async=True
            )
            # API directly returns History object format
            return History(**response_data)
        except PydanticValidationError as e:
            raise ValidationError(f"Invalid response data: {e}")

    def update_history(self, data: UpdateHistoryRequest) -> History:
        """Update an existing history/trace (sync).

        Args:
            data: History update request

        Returns:
            Updated history object
        """
        try:
            update_data = data.model_dump(exclude={"id"}, exclude_none=True)
            response_data = self._make_request("PATCH", f"/history/{data.id}", update_data)
            # API directly returns History object format
            return History(**response_data)
        except PydanticValidationError as e:
            raise ValidationError(f"Invalid response data: {e}")

    async def update_history_async(self, data: UpdateHistoryRequest) -> History:
        """Update an existing history/trace (async).

        Args:
            data: History update request

        Returns:
            Updated history object
        """
        try:
            update_data = data.model_dump(exclude={"id"}, exclude_none=True)
            response_data = await self._make_request("PATCH", f"/history/{data.id}", update_data, is_async=True)
            # API directly returns History object format
            return History(**response_data)
        except PydanticValidationError as e:
            raise ValidationError(f"Invalid response data: {e}")

    # ===== EVENT METHODS =====

    def add_event(self, data: AddEventRequest) -> Event:
        """Add an event to a history/trace (sync).

        Args:
            data: Event creation request

        Returns:
            Created event object
        """
        try:
            # Convert TimelineEvent to API format
            event_dto = self._timeline_event_to_dto(data.event, data.history_id)
            response_data = self._make_request("POST", "/history-events", event_dto)
            # API directly returns Event object format
            return Event(**response_data)
        except PydanticValidationError as e:
            raise ValidationError(f"Invalid response data: {e}")

    async def add_event_async(self, data: AddEventRequest) -> Event:
        """Add an event to a history/trace (async).

        Args:
            data: Event creation request

        Returns:
            Created event object
        """
        try:
            # Convert TimelineEvent to API format
            event_dto = self._timeline_event_to_dto(data.event, data.history_id)
            response_data = await self._make_request("POST", "/history-events", event_dto, is_async=True)
            # API directly returns Event object format
            return Event(**response_data)
        except PydanticValidationError as e:
            raise ValidationError(f"Invalid response data: {e}")

    def update_event(self, data: UpdateEventRequest) -> Event:
        """Update an existing event (sync).

        Args:
            data: Event update request

        Returns:
            Updated event object
        """
        try:
            update_data = data.model_dump(exclude={"id"}, exclude_none=True)
            response_data = self._make_request("PATCH", f"/history-events/{data.id}", update_data)
            # API directly returns Event object format
            return Event(**response_data)
        except PydanticValidationError as e:
            raise ValidationError(f"Invalid response data: {e}")

    async def update_event_async(self, data: UpdateEventRequest) -> Event:
        """Update an existing event (async).

        Args:
            data: Event update request

        Returns:
            Updated event object
        """
        try:
            update_data = data.model_dump(exclude={"id"}, exclude_none=True)
            response_data = await self._make_request("PATCH", f"/history-events/{data.id}", update_data, is_async=True)
            # API directly returns Event object format
            return Event(**response_data)
        except PydanticValidationError as e:
            raise ValidationError(f"Invalid response data: {e}")

    def _timeline_event_to_dto(self, event: TimelineEvent, history_id: str) -> Dict[str, Any]:
        """Convert TimelineEvent to API DTO format.

        Args:
            event: Timeline event object
            history_id: Associated history ID

        Returns:
            API DTO dictionary
        """
        return {
            "history_id": history_id,
            "event_type": event.type,
            "event_name": event.name,
            "start_time": event.startTime,
            "end_time": event.endTime,
            "status": event.status.value,
            "status_message": event.statusMessage,
            "level": event.level.value,
            "version": event.version,
            "parent_event_id": event.parentEventId,
            "tags": event.tags,
            "metadata": event.metadata,
            "input": event.input,
            "output": event.output,
        }

    # ===== CLEANUP METHODS =====

    def close(self):
        """Close synchronous HTTP client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self):
        """Close asynchronous HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()
