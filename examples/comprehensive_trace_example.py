"""
VoltAgent Python SDK - Comprehensive Trace and Agent Hierarchy Examples

This example demonstrates:
- Basic trace and agent creation
- Multi-level agent hierarchies
- Tool, memory, and retriever operations
- Error handling patterns
- Context manager usage (Python-specific)

Run with:
    python examples/comprehensive_trace_example.py
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

# Import VoltAgent SDK
from voltagent import EventLevel, TokenUsage, VoltAgentSDK

# ===== SIMULATED EXTERNAL SERVICES =====


async def sleep(seconds: float) -> None:
    """Simulated delay for demo purposes."""
    await asyncio.sleep(seconds)


async def call_weather_api(city: str) -> Dict[str, Any]:
    """Simulated weather API call."""
    await sleep(0.5)  # API delay simulation

    # Simulated weather data
    return {
        "temperature": 24,
        "condition": "rainy",
        "humidity": 65,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def search_web(query: str) -> List[str]:
    """Simulated web search API."""
    await sleep(0.3)
    return [
        f"Search result 1 for: {query}",
        f"Search result 2 for: {query}",
        f"Search result 3 for: {query}",
    ]


async def translate_text(text: str, target_lang: str) -> str:
    """Simulated translation service."""
    await sleep(0.4)
    return f"[{target_lang.upper()}] {text}"


# ===== EXAMPLE FUNCTIONS =====


async def basic_trace_example(sdk: VoltAgentSDK) -> None:
    """Basic Trace and Agent Example - Python Style with Context Managers."""
    print("\n🚀 Basic Trace and Agent Example Starting...")

    # Using async context manager - Python best practice!
    async with sdk.trace(
        agentId="weather-agent-v1",
        input={"query": "What's the weather in Tokyo?"},
        userId="user-123",
        conversationId="conv-456",
        tags=["weather", "basic-example"],
        metadata={"source": "python-sdk-example", "version": "1.0", "language": "python"},
    ) as trace:
        print(f"✅ Trace created: {trace.id}")

        # Add main agent
        agent = await trace.add_agent(
            {
                "name": "Weather Agent",
                "input": {"city": "Tokyo"},
                "instructions": "You are a weather agent responsible for providing weather information.",
                "metadata": {"model_parameters": {"model": "gpt-4", "temperature": 0.7}},
            }
        )
        print(f"✅ Main agent added: {agent.id}")

        # Add tool to agent
        weather_tool = await agent.add_tool(
            {
                "name": "weather-api",
                "input": {"city": "Tokyo", "units": "celsius"},
                "metadata": {"api_version": "v2", "timeout": 5000},
            }
        )
        print(f"🔧 Weather tool started: {weather_tool.id}")

        # Simulate weather API call with error handling
        try:
            weather_data = await call_weather_api("Tokyo")
            await weather_tool.success(
                output={
                    "temperature": weather_data["temperature"],
                    "condition": weather_data["condition"],
                    "humidity": weather_data["humidity"],
                    "timestamp": weather_data["timestamp"],
                },
                metadata={"data_source": "weather-api-v2", "response_time_ms": 500},
            )
            print(f"✅ Weather tool successful: {weather_data['temperature']}°C, {weather_data['condition']}")
        except Exception as error:
            await weather_tool.error(
                status_message=error, metadata={"error_type": "api_failure", "retry_attempted": False}
            )
            print(f"❌ Weather tool error: {error}")
            return

        # Add memory operation
        memory_op = await agent.add_memory(
            {
                "name": "cache-weather-data",
                "input": {
                    "key": "weather_tokyo",
                    "value": {
                        "temp": weather_data["temperature"],
                        "condition": weather_data["condition"],
                        "cached_at": datetime.now(timezone.utc).timestamp(),
                    },
                    "ttl": 3600,  # 1 hour cache
                },
                "metadata": {"type": "redis", "region": "ap-northeast-1"},
            }
        )
        print(f"💾 Memory operation started: {memory_op.id}")

        await memory_op.success(
            output={"cached": True, "key": "weather_tokyo", "data_size": "124 bytes"},
            metadata={"cache_hit": False, "ttl": 3600},
        )
        print("✅ Memory operation successful")

        # Complete agent successfully
        await agent.success(
            output={
                "response": f"Weather in Tokyo is {weather_data['temperature']}°C and {weather_data['condition']}.",
                "confidence": 0.95,
                "sources": ["weather-api"],
            },
            usage=TokenUsage(prompt_tokens=45, completion_tokens=25, total_tokens=70),
            metadata={"completion_time": datetime.now(timezone.utc).isoformat()},
        )
        print("✅ Main agent completed")

        # Trace will be automatically ended by context manager
        print(f"🎉 Trace completed: {trace.id}")


async def complex_hierarchy_example(sdk: VoltAgentSDK) -> None:
    """Complex Multi-Agent Hierarchy Example."""
    print("\n🌟 Complex Multi-Agent Hierarchy Example Starting...")

    async with sdk.trace(
        agentId="research-coordinator",
        input={
            "topic": "Global AI developments and emerging trends",
            "depth": "comprehensive",
            "languages": ["en", "zh", "es"],
        },
        userId="researcher-789",
        conversationId="research-session-001",
        tags=["research", "multi-agent", "ai-trends", "global"],
        metadata={
            "priority": "high",
            "deadline": "2024-06-01",
            "requester": "research-team",
            "example_type": "complex_hierarchy",
        },
    ) as trace:
        print(f"✅ Research trace created: {trace.id}")

        # Main Coordinator Agent
        coordinator = await trace.add_agent(
            {
                "name": "Research Coordinator",
                "input": {
                    "task": "Coordinate global AI research project and manage sub-agents",
                    "strategy": "divide-and-conquer",
                },
                "metadata": {
                    "role": "coordinator",
                    "experience_level": "senior",
                    "specialization": "research-management",
                    "model_parameters": {"model": "gpt-4"},
                },
            }
        )
        print(f"👑 Coordinator agent created: {coordinator.id}")

        # Add retriever to coordinator (research planning)
        planning_retriever = await coordinator.add_retriever(
            {
                "name": "research-planning-retriever",
                "input": {
                    "query": "AI research methodology best practices",
                    "sources": ["academic-db", "research-guidelines"],
                    "max_results": 10,
                },
                "metadata": {"vector_store": "pinecone", "embedding_model": "text-embedding-ada-002"},
            }
        )
        print(f"🔍 Planning retriever started: {planning_retriever.id}")

        await planning_retriever.success(
            output={
                "documents": [
                    "Research methodology guide for AI topics",
                    "Best practices for multi-agent coordination",
                    "Academic research standards for AI studies",
                ],
                "relevance_scores": [0.95, 0.88, 0.82],
            },
            metadata={"search_time": "0.3s", "vector_space": "1536-dimensions"},
        )

        # SUB-AGENT 1: Data Collection Agent
        data_collector = await coordinator.add_agent(
            {
                "name": "Data Collection Agent",
                "input": {
                    "task": "Collect data about global AI developments and trends",
                    "sources": ["news", "academic-papers", "tech-reports", "industry-analysis"],
                    "timeframe": "last-2-years",
                },
                "metadata": {
                    "role": "data-collector",
                    "specialization": "global-ai-landscape",
                    "model_parameters": {"model": "gpt-4"},
                },
            }
        )
        print(f"📊 Data collector sub-agent created: {data_collector.id}")

        # Add web search tool to data collector
        web_search_tool = await data_collector.add_tool(
            {
                "name": "web-search-tool",
                "input": {
                    "query": "global artificial intelligence developments trends 2023 2024",
                    "search_engine": "google",
                    "max_results": 20,
                },
                "metadata": {"search_type": "comprehensive", "language": "en"},
            }
        )
        print(f"🔍 Web search tool started: {web_search_tool.id}")

        try:
            search_results = await search_web("global artificial intelligence developments trends")
            await web_search_tool.success(
                output={"results": search_results, "total_found": len(search_results), "search_time": "0.8s"},
                metadata={"search_engine": "google", "results_filtered": True},
            )
            print(f"✅ Web search successful: {len(search_results)} results found")
        except Exception as error:
            await web_search_tool.error(
                status_message=error, metadata={"search_engine": "google", "query_type": "comprehensive"}
            )

        # Add memory operation to data collector
        data_memory = await data_collector.add_memory(
            {
                "name": "collected-data-storage",
                "input": {
                    "key": "global_ai_data_2024",
                    "value": {
                        "sources": ["news-articles", "academic-papers", "tech-reports"],
                        "data_points": 85,
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                    },
                    "category": "research-data",
                },
                "metadata": {"storage_type": "long-term", "encryption": True},
            }
        )
        print(f"💾 Data memory operation started: {data_memory.id}")

        await data_memory.success(output={"stored": True, "data_size": "4.7MB", "compression_ratio": 0.65})

        # SUB-SUB-AGENT: Academic Paper Analyzer (under Data Collector)
        paper_analyzer = await data_collector.add_agent(
            {
                "name": "Academic Paper Analyzer",
                "input": {
                    "task": "Analyze academic papers and extract key findings from global AI research",
                    "focus": "global-ai-research-trends",
                    "analysis_depth": "detailed",
                },
                "metadata": {
                    "role": "academic-analyzer",
                    "specialization": "paper-analysis",
                    "parent_agent": data_collector.id,
                    "model_parameters": {"model": "gpt-4"},
                },
            }
        )
        print(f"📚 Academic paper analyzer (sub-sub-agent) created: {paper_analyzer.id}")

        # Add tool to paper analyzer
        paper_analysis_tool = await paper_analyzer.add_tool(
            {
                "name": "paper-analysis-tool",
                "input": {
                    "papers": ["arxiv_paper_1.pdf", "nature_ai_2024.pdf", "ieee_ml_trends.pdf"],
                    "analysis_type": "content-extraction",
                    "language": "mixed",
                },
                "metadata": {"pdf_parser": "advanced", "nlp_model": "bert-multilingual"},
            }
        )
        print(f"🔬 Paper analysis tool started: {paper_analysis_tool.id}")

        await paper_analysis_tool.success(
            output={
                "analyzed_papers": 3,
                "key_findings": [
                    "Multimodal AI systems showing 60% improvement in 2024",
                    "Enterprise AI adoption reached 70% globally",
                    "Significant breakthroughs in AI safety and alignment",
                ],
                "confidence": 0.89,
            },
            metadata={"processing_time": "12.3s", "nlp_model": "bert-multilingual-v2"},
        )

        # Complete paper analyzer
        await paper_analyzer.success(
            output={
                "summary": "3 academic papers analyzed successfully",
                "key_insights": ["Multimodal AI advances", "Enterprise adoption growth", "AI safety progress"],
                "next_steps": ["Deep dive into multimodal research", "Enterprise case studies analysis"],
            },
            metadata={"total_papers_processed": 3, "analysis_accuracy": 0.94},
        )

        # SUB-AGENT 2: Translation Agent
        translator = await coordinator.add_agent(
            {
                "name": "Translation Agent",
                "input": {
                    "task": "Translate collected data to multiple languages",
                    "source_language": "english",
                    "target_languages": ["spanish", "chinese", "french"],
                    "preserve_terminology": True,
                },
                "metadata": {
                    "role": "translator",
                    "specialization": "technical-translation",
                    "languages": ["en", "es", "zh", "fr"],
                    "model_parameters": {"model": "gpt-4"},
                },
            }
        )
        print(f"🌍 Translation sub-agent created: {translator.id}")

        # Add translation tool
        translation_tool = await translator.add_tool(
            {
                "name": "ai-translation-tool",
                "input": {
                    "text": "Multimodal AI systems are showing significant improvements in 2024",
                    "from_lang": "en",
                    "to_lang": "es",
                    "domain": "technology",
                },
                "metadata": {"model": "neural-translation-v3", "quality_check": True},
            }
        )
        print(f"🔤 Translation tool started: {translation_tool.id}")

        try:
            translated_text = await translate_text(
                "Multimodal AI systems are showing significant improvements in 2024", "es"
            )
            await translation_tool.success(
                output={"translated_text": translated_text, "confidence": 0.96, "word_count": 10},
                metadata={"model": "neural-translation-v3"},
            )
            print(f"✅ Translation successful: {translated_text}")
        except Exception as error:
            await translation_tool.error(
                status_message=error, metadata={"translation_pair": "en-es", "model": "neural-translation-v3"}
            )

        # SUB-SUB-AGENT: Quality Checker (under Translator)
        quality_checker = await translator.add_agent(
            {
                "name": "Translation Quality Control Agent",
                "input": {
                    "task": "Check translation quality and suggest improvements",
                    "criteria": ["accuracy", "fluency", "terminology"],
                    "threshold": 0.9,
                },
                "metadata": {
                    "role": "quality-checker",
                    "specialization": "translation-qa",
                    "parent_agent": translator.id,
                    "model_parameters": {"model": "gpt-4"},
                },
            }
        )
        print(f"✅ Quality checker (sub-sub-agent) created: {quality_checker.id}")

        # Add retriever to quality checker
        terminology_retriever = await quality_checker.add_retriever(
            {
                "name": "ai-terminology-retriever",
                "input": {
                    "query": "AI technical terms multilingual translation verification",
                    "domain": "artificial-intelligence",
                    "verification_mode": True,
                },
                "metadata": {"terminology_db": "global-tech-terms-v3", "languages": ["en", "es", "zh", "fr"]},
            }
        )
        print(f"📖 Terminology retriever started: {terminology_retriever.id}")

        await terminology_retriever.success(
            output={
                "verified_terms": [
                    "multimodal AI -> IA multimodal (es)",
                    "artificial intelligence -> inteligencia artificial (es)",
                    "machine learning -> aprendizaje automático (es)",
                ],
                "accuracy": 0.98,
            },
            metadata={"database_version": "global-tech-terms-v3"},
        )

        # Complete quality checker
        await quality_checker.success(
            output={
                "quality_score": 0.94,
                "issues": [],
                "recommendations": ["Excellent translation quality", "Terminology consistency maintained"],
            },
            usage=TokenUsage(prompt_tokens=120, completion_tokens=80, total_tokens=200),
            metadata={"criteria_checked": 3},
        )

        # Complete translator
        await translator.success(
            output={"translation_completed": True, "total_words": 250, "average_quality": 0.94},
            usage=TokenUsage(prompt_tokens=350, completion_tokens=180, total_tokens=530),
            metadata={"language_pairs": ["en-es", "en-zh", "en-fr"], "quality_threshold": 0.9},
        )

        # Complete data collector
        await data_collector.success(
            output={"data_collected": True, "total_sources": 25, "key_data_points": 45},
            usage=TokenUsage(prompt_tokens=450, completion_tokens=280, total_tokens=730),
            metadata={"sub_agents_used": 1, "analysis_accuracy": 0.91},
        )

        # Add final memory operation to coordinator
        final_results = await coordinator.add_memory(
            {
                "name": "final-research-results",
                "input": {
                    "key": "global_ai_research_final",
                    "value": {
                        "data_points": 85,
                        "translations": 250,
                        "quality_score": 0.94,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "category": "final-results",
                },
                "metadata": {"storage_type": "permanent", "backup": True},
            }
        )
        print(f"💾 Final results memory started: {final_results.id}")

        await final_results.success(
            output={"stored": True, "archived": True, "backup_location": "s3://research-backups/"},
            metadata={"storage_provider": "aws-s3"},
        )

        # Complete coordinator
        await coordinator.success(
            output={
                "project_completed": True,
                "sub_agents_managed": 2,
                "total_operations": 8,
                "overall_success": True,
                "summary": "Global AI research completed successfully",
                "recommendations": [
                    "Continue monitoring global AI development trends",
                    "Schedule follow-up research in 6 months",
                    "Share findings with international research community",
                ],
            },
            usage=TokenUsage(prompt_tokens=1200, completion_tokens=850, total_tokens=2050),
            metadata={"success_rate": 1.0},
        )
        print("👑 Coordinator agent completed")

        # Trace will be automatically ended with success by context manager
        print(f"🎉 Complex hierarchy trace completed: {trace.id}")


async def error_handling_example(sdk: VoltAgentSDK) -> None:
    """Error Handling Example - Demonstrating Python exception handling."""
    print("\n⚠️ Error Handling Example Starting...")

    try:
        async with sdk.trace(agentId="error-test-agent", input={"test_type": "error-scenarios"}) as trace:
            agent = await trace.add_agent({"name": "Error Test Agent", "input": {"scenario": "api-failure"}})

            # Failed tool example
            failing_tool = await agent.add_tool(
                {"name": "failing-api-tool", "input": {"endpoint": "https://nonexistent-api.com/data"}}
            )

            # Simulated API failure
            api_error = Exception("API endpoint not reachable")
            await failing_tool.error(
                status_message=api_error,
                metadata={
                    "endpoint": "https://nonexistent-api.com/data",
                    "http_status": 404,
                    "error_category": "network_error",
                },
            )
            print("❌ Tool error recorded")

            # Mark agent as failed as well
            agent_error = Exception("Agent failed due to tool failure")
            await agent.error(
                status_message=agent_error,
                metadata={"failed_tool": failing_tool.id, "error_category": "api_failure", "recovery_attempted": False},
            )
            print("❌ Agent error recorded")

            # Manually trigger an exception to test context manager error handling
            raise Exception("Simulated critical error")

    except Exception as e:
        print(f"❌ Error handled by context manager: {e}")
        print("❌ Trace automatically marked as failed")


# ===== MAIN FUNCTION =====


async def main() -> None:
    """Main function demonstrating all SDK features."""
    print("🌟 VoltAgent Python SDK - Comprehensive Examples")
    print("=" * 70)

    # Initialize SDK
    sdk = VoltAgentSDK(
        base_url=os.getenv("VOLTAGENT_BASE_URL", "https://api.voltagent.dev"),
        public_key=os.getenv("VOLTAGENT_PUBLIC_KEY", "demo-public-key"),
        secret_key=os.getenv("VOLTAGENT_SECRET_KEY", "demo-secret-key"),
        auto_flush=True,
        flush_interval=3,  # Send every 3 seconds
        timeout=30,
    )

    try:
        # Run examples
        await basic_trace_example(sdk)
        await complex_hierarchy_example(sdk)
        await error_handling_example(sdk)

        print("\n✅ All examples completed!")

        # Manual flush (though auto-flush is enabled)
        await sdk.flush()
        print("📤 All events sent")

    except Exception as e:
        print(f"❌ Main function error: {e}")
    finally:
        # Cleanup SDK resources
        await sdk.shutdown()
        print("🔒 SDK shutdown complete")


# ===== ENTRY POINT =====

if __name__ == "__main__":
    """
    Run the examples.

    Set environment variables:
    export VOLTAGENT_BASE_URL="https://api.voltagent.dev"
    export VOLTAGENT_PUBLIC_KEY="your-public-key"
    export VOLTAGENT_SECRET_KEY="your-secret-key"

    Then run:
    python examples/comprehensive_trace_example.py
    """
    asyncio.run(main())
