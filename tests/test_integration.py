"""Integration tests for VoltAgent SDK - End-to-end scenarios."""

from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
import respx

from voltagent import EventLevel, TokenUsage, VoltAgentSDK
from voltagent.types import APIError, EventStatus, TimeoutError, TraceStatus, ValidationError


class TestBasicIntegration:
    """Test basic end-to-end integration scenarios."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_basic_trace_and_agent_workflow(self, sample_config):
        """Test basic trace creation with agent and tool operations."""
        # Mock API responses
        trace_response = {
            "id": "integration-trace-001",
            "agent_id": "weather-agent",
            "user_id": "user-123",
            "status": "working",
            "input": {"query": "weather in Tokyo"},
            "metadata": {"agentId": "weather-agent"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        event_response = {
            "id": "event-001",
            "history_id": "integration-trace-001",
            "event_type": "agent",
            "event_name": "agent:start",
            "status": "running",
            "level": "INFO",
            "input": {"task": "get weather"},
            "metadata": {"agentId": "weather-agent", "displayName": "Weather Agent"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Setup API mocks
        respx.post(f"{sample_config.base_url}/history").mock(return_value=httpx.Response(200, json=trace_response))

        respx.post(f"{sample_config.base_url}/history-events").mock(
            return_value=httpx.Response(200, json=event_response)
        )

        respx.patch(f"{sample_config.base_url}/history/integration-trace-001").mock(
            return_value=httpx.Response(200, json={**trace_response, "status": "completed"})
        )

        # Execute workflow
        sdk = VoltAgentSDK(**sample_config.model_dump())

        async with sdk.trace(
            agentId="weather-agent",
            input={"query": "weather in Tokyo"},
            userId="user-123",
            tags=["weather", "integration-test"],
        ) as trace:
            assert trace.id == "integration-trace-001"

            # Add agent
            agent = await trace.add_agent(
                {
                    "name": "Weather Agent",
                    "input": {"city": "Tokyo"},
                    "instructions": "Get weather information",
                    "metadata": {"model_parameters": {"model": "gpt-4"}, "api_version": "v2"},
                }
            )

            # Add tool to agent
            weather_tool = await agent.add_tool(
                {"name": "weather-api", "input": {"city": "Tokyo", "units": "celsius"}, "metadata": {"timeout": 5000}}
            )

            # Complete tool successfully
            await weather_tool.success(
                output={"temperature": 24, "condition": "rainy"}, metadata={"response_time": "500ms"}
            )

            # Complete agent successfully
            await agent.success(
                output={"response": "Weather in Tokyo is 24°C and rainy"},
                usage=TokenUsage(prompt_tokens=45, completion_tokens=25, total_tokens=70),
                metadata={"confidence": 0.95},
            )

        await sdk.shutdown()

    @pytest.mark.asyncio
    @respx.mock
    async def test_error_handling_integration(self, sample_config):
        """Test error scenarios in integration workflow."""
        # Mock error responses
        respx.post(f"{sample_config.base_url}/history").mock(
            return_value=httpx.Response(400, json={"message": "Invalid agent_id", "errors": ["agent_id is required"]})
        )

        sdk = VoltAgentSDK(**sample_config.model_dump())

        # Test API error propagation
        with pytest.raises(APIError) as exc_info:
            async with sdk.trace(agentId="") as trace:  # Invalid empty agent_id
                pass

        assert exc_info.value.status_code == 400
        assert "Invalid agent_id" in str(exc_info.value)

        await sdk.shutdown()


class TestComplexHierarchyIntegration:
    """Test complex multi-agent hierarchy scenarios."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_multi_agent_hierarchy(self, sample_config):
        """Test complex multi-agent hierarchy with sub-agents."""
        # Mock responses for main trace
        main_trace_response = {
            "id": "research-trace-001",
            "agent_id": "research-coordinator",
            "status": "working",
            "input": {"topic": "AI trends"},
            "metadata": {"agentId": "research-coordinator"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Generic event response template
        def create_event_response(event_id, event_type, event_name):
            return {
                "id": event_id,
                "history_id": "research-trace-001",
                "event_type": event_type,
                "event_name": event_name,
                "status": "running",
                "level": "INFO",
                "metadata": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        # Setup mocks
        respx.post(f"{sample_config.base_url}/history").mock(return_value=httpx.Response(200, json=main_trace_response))

        respx.post(f"{sample_config.base_url}/history-events").mock(
            return_value=httpx.Response(200, json=create_event_response("event-001", "agent", "agent:start"))
        )

        respx.patch(f"{sample_config.base_url}/history/research-trace-001").mock(
            return_value=httpx.Response(200, json={**main_trace_response, "status": "completed"})
        )

        sdk = VoltAgentSDK(**sample_config.model_dump())

        async with sdk.trace(
            agentId="research-coordinator",
            input={"topic": "AI trends", "depth": "comprehensive"},
            userId="researcher-789",
            tags=["research", "multi-agent", "ai-trends"],
        ) as trace:
            # Main coordinator agent
            coordinator = await trace.add_agent(
                {
                    "name": "Research Coordinator",
                    "input": {"task": "coordinate global AI research"},
                    "metadata": {
                        "role": "coordinator",
                        "experience_level": "senior",
                        "model_parameters": {"model": "gpt-4"},
                    },
                }
            )

            # Add retriever to coordinator
            planning_retriever = await coordinator.add_retriever(
                {
                    "name": "research-planning-retriever",
                    "input": {"query": "AI research methodology"},
                    "metadata": {"vector_store": "pinecone", "embedding_model": "text-embedding-ada-002"},
                }
            )

            await planning_retriever.success(
                output={"documents": ["Research methodology guide", "Best practices"], "relevance_scores": [0.95, 0.88]}
            )

            # Sub-agent 1: Data collector
            data_collector = await coordinator.add_agent(
                {
                    "name": "Data Collection Agent",
                    "input": {"task": "collect AI development data"},
                    "metadata": {"role": "data-collector", "specialization": "global-ai-landscape"},
                }
            )

            # Add tool to data collector
            web_search_tool = await data_collector.add_tool(
                {
                    "name": "web-search-tool",
                    "input": {"query": "global AI developments 2024"},
                    "metadata": {"search_engine": "google", "max_results": 20},
                }
            )

            await web_search_tool.success(output={"results": ["Article 1", "Article 2", "Article 3"], "total_found": 3})

            # Sub-sub-agent: Paper analyzer (under data collector)
            paper_analyzer = await data_collector.add_agent(
                {
                    "name": "Academic Paper Analyzer",
                    "input": {"task": "analyze academic papers"},
                    "metadata": {"role": "academic-analyzer", "parent_agent": data_collector.id},
                }
            )

            # Add tool to paper analyzer
            analysis_tool = await paper_analyzer.add_tool(
                {
                    "name": "paper-analysis-tool",
                    "input": {"papers": ["arxiv_paper_1.pdf", "nature_ai_2024.pdf"]},
                    "metadata": {"pdf_parser": "advanced"},
                }
            )

            await analysis_tool.success(
                output={"analyzed_papers": 2, "key_findings": ["Multimodal AI advances", "Enterprise adoption"]}
            )

            # Complete paper analyzer
            await paper_analyzer.success(
                output={"summary": "2 papers analyzed successfully"},
                usage=TokenUsage(prompt_tokens=200, completion_tokens=150, total_tokens=350),
            )

            # Sub-agent 2: Translation agent
            translator = await coordinator.add_agent(
                {
                    "name": "Translation Agent",
                    "input": {"task": "translate research data"},
                    "metadata": {"role": "translator", "languages": ["en", "es", "zh", "fr"]},
                }
            )

            # Add translation tool
            translation_tool = await translator.add_tool(
                {
                    "name": "ai-translation-tool",
                    "input": {"text": "AI research findings", "from_lang": "en", "to_lang": "es"},
                }
            )

            await translation_tool.success(output={"translated_text": "[ES] AI research findings"})

            # Complete translator
            await translator.success(output={"translation_completed": True, "total_words": 250})

            # Complete data collector
            await data_collector.success(
                output={"data_collected": True, "total_sources": 25},
                usage=TokenUsage(prompt_tokens=450, completion_tokens=280, total_tokens=730),
            )

            # Add final memory operation to coordinator
            final_results = await coordinator.add_memory(
                {
                    "name": "final-research-results",
                    "input": {"key": "global_ai_research_final", "value": {"status": "completed"}},
                    "metadata": {"storage_type": "permanent"},
                }
            )

            await final_results.success(output={"stored": True, "archived": True})

            # Complete coordinator
            await coordinator.success(
                output={"project_completed": True, "sub_agents_managed": 2, "total_operations": 8},
                usage=TokenUsage(prompt_tokens=1200, completion_tokens=850, total_tokens=2050),
            )

        await sdk.shutdown()


class TestErrorRecoveryIntegration:
    """Test error recovery and handling in integration scenarios."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_agent_failure_recovery(self, sample_config):
        """Test agent failure and recovery patterns."""
        # Mock successful trace creation
        trace_response = {
            "id": "error-recovery-trace",
            "agent_id": "test-agent",
            "status": "working",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        event_response = {
            "id": "event-001",
            "history_id": "error-recovery-trace",
            "event_type": "agent",
            "event_name": "agent:start",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        respx.post(f"{sample_config.base_url}/history").mock(return_value=httpx.Response(200, json=trace_response))

        respx.post(f"{sample_config.base_url}/history-events").mock(
            return_value=httpx.Response(200, json=event_response)
        )

        respx.patch(f"{sample_config.base_url}/history/error-recovery-trace").mock(
            return_value=httpx.Response(200, json={**trace_response, "status": "error"})
        )

        sdk = VoltAgentSDK(**sample_config.model_dump())

        # Test error handling with context manager
        with pytest.raises(ValueError):
            async with sdk.trace(agentId="test-agent") as trace:
                agent = await trace.add_agent({"name": "Failing Agent"})

                # Simulate tool failure
                failing_tool = await agent.add_tool(
                    {"name": "failing-api-tool", "input": {"endpoint": "https://nonexistent.api"}}
                )

                # Record tool error
                api_error = Exception("API endpoint not reachable")
                await failing_tool.error(
                    status_message=api_error, metadata={"http_status": 404, "error_category": "network_error"}
                )

                # Record agent error
                await agent.error(
                    status_message="Agent failed due to tool failure", metadata={"failed_tool": failing_tool.id}
                )

                # Trigger exception to test context manager error handling
                raise ValueError("Simulated critical error")

        await sdk.shutdown()

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_failure_scenarios(self, sample_config):
        """Test various API failure scenarios."""
        # Test timeout scenario
        respx.post(f"{sample_config.base_url}/history").mock(side_effect=httpx.TimeoutException("Request timeout"))

        sdk = VoltAgentSDK(**sample_config.model_dump())

        with pytest.raises(TimeoutError) as exc_info:
            await sdk.create_trace(agentId="timeout-test")

        assert "Request timeout" in str(exc_info.value)

        await sdk.shutdown()


class TestMetadataConversionIntegration:
    """Test metadata conversion in real-world scenarios."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_snake_case_to_camel_case_workflow(self, sample_config):
        """Test that snake_case metadata is properly converted to camelCase."""
        # Capture requests to verify conversion
        captured_requests = []

        def capture_request(request):
            captured_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "metadata-test-trace",
                    "agent_id": "metadata-agent",
                    "status": "working",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        def capture_event_request(request):
            captured_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "metadata-event",
                    "history_id": "metadata-test-trace",
                    "event_type": "agent",
                    "event_name": "agent:start",
                    "status": "running",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        respx.post(f"{sample_config.base_url}/history").mock(side_effect=capture_request)
        respx.post(f"{sample_config.base_url}/history-events").mock(side_effect=capture_event_request)
        respx.patch(f"{sample_config.base_url}/history/metadata-test-trace").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "metadata-test-trace",
                    "agent_id": "metadata-agent",
                    "status": "completed",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

        sdk = VoltAgentSDK(**sample_config.model_dump())

        async with sdk.trace(agentId="metadata-agent") as trace:
            # Add agent with snake_case metadata
            agent = await trace.add_agent(
                {
                    "name": "Metadata Test Agent",
                    "metadata": {
                        "model_parameters": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 150},
                        "retry_config": {"max_attempts": 3, "backoff_factor": 2.0},
                        "api_settings": {"base_url": "https://api.openai.com", "timeout_seconds": 30},
                    },
                }
            )

            # Add tool with snake_case metadata
            await agent.add_tool(
                {
                    "name": "Test Tool",
                    "metadata": {"execution_timeout": 60, "memory_limit": "512MB", "output_format": "json"},
                }
            )

        # Verify requests were captured
        assert len(captured_requests) >= 2

        # Find the agent creation request
        agent_request = None
        for req in captured_requests:
            if b"agent:start" in req.content:
                agent_request = req
                break

        assert agent_request is not None

        # Parse request content to verify camelCase conversion
        import json

        request_data = json.loads(agent_request.content.decode())
        metadata = request_data.get("metadata", {})

        # Verify camelCase conversion occurred
        assert "modelParameters" in metadata
        assert "retryConfig" in metadata
        assert "apiSettings" in metadata
        assert metadata["modelParameters"]["maxTokens"] == 150
        assert metadata["retryConfig"]["maxAttempts"] == 3
        assert metadata["apiSettings"]["timeoutSeconds"] == 30

        await sdk.shutdown()


class TestConcurrentOperations:
    """Test concurrent operations and race conditions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_concurrent_agent_operations(self, sample_config):
        """Test concurrent agent and tool operations."""
        # Mock responses
        trace_response = {
            "id": "concurrent-trace",
            "agent_id": "concurrent-agent",
            "status": "working",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        event_response = {
            "id": "concurrent-event",
            "history_id": "concurrent-trace",
            "event_type": "agent",
            "event_name": "agent:start",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        respx.post(f"{sample_config.base_url}/history").mock(return_value=httpx.Response(200, json=trace_response))

        respx.post(f"{sample_config.base_url}/history-events").mock(
            return_value=httpx.Response(200, json=event_response)
        )

        respx.patch(f"{sample_config.base_url}/history/concurrent-trace").mock(
            return_value=httpx.Response(200, json={**trace_response, "status": "completed"})
        )

        sdk = VoltAgentSDK(**sample_config.model_dump())

        async with sdk.trace(agentId="concurrent-agent") as trace:
            agent = await trace.add_agent({"name": "Concurrent Agent"})

            # Create multiple tools concurrently
            import asyncio

            async def create_tool(name):
                tool = await agent.add_tool({"name": f"Tool-{name}"})
                await tool.success(output={"tool": name, "result": "completed"})
                return tool

            # Run multiple tool operations concurrently
            tools = await asyncio.gather(create_tool("A"), create_tool("B"), create_tool("C"), return_exceptions=True)

            # Verify all tools were created successfully
            assert len(tools) == 3
            for tool in tools:
                assert not isinstance(tool, Exception)

            await agent.success(output={"concurrent_tools": 3})

        await sdk.shutdown()


class TestLongRunningOperations:
    """Test long-running operations and timeouts."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_long_running_agent_workflow(self, sample_config):
        """Test long-running agent workflow with multiple steps."""
        # Mock responses
        trace_response = {
            "id": "long-running-trace",
            "agent_id": "long-running-agent",
            "status": "working",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        event_response = {
            "id": "long-running-event",
            "history_id": "long-running-trace",
            "event_type": "agent",
            "event_name": "agent:start",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        respx.post(f"{sample_config.base_url}/history").mock(return_value=httpx.Response(200, json=trace_response))

        respx.post(f"{sample_config.base_url}/history-events").mock(
            return_value=httpx.Response(200, json=event_response)
        )

        respx.patch(f"{sample_config.base_url}/history/long-running-trace").mock(
            return_value=httpx.Response(200, json={**trace_response, "status": "completed"})
        )

        sdk = VoltAgentSDK(**sample_config.model_dump())

        async with sdk.trace(agentId="long-running-agent") as trace:
            agent = await trace.add_agent(
                {"name": "Long Running Agent", "metadata": {"estimated_duration": "5_minutes"}}
            )

            # Simulate multiple processing steps
            for step in range(1, 4):  # 3 steps
                step_tool = await agent.add_tool(
                    {
                        "name": f"processing-step-{step}",
                        "input": {"step": step, "data": f"step_{step}_data"},
                        "metadata": {"step_number": step},
                    }
                )

                # Simulate some processing time
                import asyncio

                await asyncio.sleep(0.01)  # Small delay to simulate processing

                await step_tool.success(
                    output={"step": step, "processed": True, "data_size": f"{step * 100}MB"},
                    metadata={"processing_time": f"{step * 0.5}s"},
                )

            # Final memory storage
            final_storage = await agent.add_memory(
                {
                    "name": "final-results-storage",
                    "input": {"results": "processed_data", "steps_completed": 3},
                    "metadata": {"storage_type": "persistent"},
                }
            )

            await final_storage.success(output={"stored": True, "size": "300MB"})

            await agent.success(
                output={"workflow_completed": True, "steps_processed": 3, "total_data": "300MB"},
                usage=TokenUsage(prompt_tokens=800, completion_tokens=600, total_tokens=1400),
            )

        await sdk.shutdown()


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_customer_service_bot_scenario(self, sample_config):
        """Test customer service bot scenario with user interaction."""
        # Mock trace and event responses
        trace_response = {
            "id": "customer-service-trace",
            "agent_id": "customer-service-bot",
            "user_id": "customer-123",
            "conversation_id": "support-conv-456",
            "status": "working",
            "input": {"query": "Help with order #12345"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        event_response = {
            "id": "cs-event",
            "history_id": "customer-service-trace",
            "event_type": "agent",
            "event_name": "agent:start",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        respx.post(f"{sample_config.base_url}/history").mock(return_value=httpx.Response(200, json=trace_response))

        respx.post(f"{sample_config.base_url}/history-events").mock(
            return_value=httpx.Response(200, json=event_response)
        )

        respx.patch(f"{sample_config.base_url}/history/customer-service-trace").mock(
            return_value=httpx.Response(200, json={**trace_response, "status": "completed"})
        )

        sdk = VoltAgentSDK(**sample_config.model_dump())

        async with sdk.trace(
            agentId="customer-service-bot",
            input={"query": "Help with order #12345", "priority": "medium"},
            userId="customer-123",
            conversationId="support-conv-456",
            tags=["customer-service", "order-support"],
        ) as trace:
            # Main customer service agent
            cs_agent = await trace.add_agent(
                {
                    "name": "Customer Service Agent",
                    "input": {"customer_query": "Help with order #12345"},
                    "instructions": "Help customer with their order inquiry",
                    "metadata": {"agent_type": "customer_service", "skill_level": "expert", "languages": ["en", "es"]},
                }
            )

            # Order lookup tool
            order_lookup = await cs_agent.add_tool(
                {
                    "name": "order-lookup-system",
                    "input": {"order_id": "12345"},
                    "metadata": {"system": "order_management", "version": "v2.1"},
                }
            )

            await order_lookup.success(
                output={
                    "order": {"id": "12345", "status": "shipped", "tracking": "TRACK123", "delivery_date": "2024-01-15"}
                }
            )

            # Memory: Store customer interaction
            customer_memory = await cs_agent.add_memory(
                {
                    "name": "customer-interaction-log",
                    "input": {
                        "customer_id": "customer-123",
                        "interaction_type": "order_inquiry",
                        "order_id": "12345",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    "metadata": {"retention_period": "1_year"},
                }
            )

            await customer_memory.success(output={"logged": True, "interaction_id": "int-789"})

            # Knowledge retrieval for similar issues
            knowledge_retriever = await cs_agent.add_retriever(
                {
                    "name": "knowledge-base-search",
                    "input": {"query": "order tracking shipping status"},
                    "metadata": {"kb_version": "2024.1", "search_type": "semantic"},
                }
            )

            await knowledge_retriever.success(
                output={
                    "articles": ["How to track your order", "Shipping status meanings", "Delivery troubleshooting"],
                    "relevance_scores": [0.95, 0.87, 0.82],
                }
            )

            # Complete customer service interaction
            await cs_agent.success(
                output={
                    "response": "Your order #12345 has been shipped with tracking TRACK123. Expected delivery: 2024-01-15.",
                    "satisfaction_score": 0.92,
                    "resolution_status": "resolved",
                    "follow_up_needed": False,
                },
                usage=TokenUsage(prompt_tokens=150, completion_tokens=80, total_tokens=230),
                metadata={"interaction_duration": "3_minutes", "customer_satisfaction": "high"},
            )

        await sdk.shutdown()


# Add __init__.py file for tests package
@pytest.mark.asyncio
async def test_imports():
    """Test that all imports work correctly."""
    from voltagent import VoltAgentSDK
    from voltagent.client import VoltAgentCoreAPI
    from voltagent.types import EventStatus, TokenUsage, TraceStatus

    assert VoltAgentSDK is not None
    assert EventStatus is not None
    assert TraceStatus is not None
    assert TokenUsage is not None
    assert VoltAgentCoreAPI is not None
