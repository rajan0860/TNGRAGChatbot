"""
Unit tests for the chatbot module.
Tests agent construction, tool functions, and query handling.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestDuckDuckGoSearchFunc:
    """Tests for the duckduckgo_search_func function."""
    
    @patch('src.chatbot.DDGS')
    def test_returns_formatted_results(self, mock_ddgs):
        """Should return formatted search results."""
        from src.chatbot import duckduckgo_search_func
        
        # Setup mock
        mock_results = [
            {
                'title': 'Star Trek TNG',
                'href': 'https://example.com/tng',
                'body': 'Star Trek The Next Generation info'
            },
            {
                'title': 'Data Android',
                'href': 'https://example.com/data',
                'body': 'Information about Data'
            }
        ]
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Execute
        result = duckduckgo_search_func("Star Trek Data")
        
        # Verify
        assert "Star Trek TNG" in result
        assert "https://example.com/tng" in result
        assert "Data Android" in result
        mock_ddgs_instance.text.assert_called_once_with("Star Trek Data", max_results=5)
    
    @patch('src.chatbot.DDGS')
    def test_returns_no_results_message(self, mock_ddgs):
        """Should return 'No results found.' when search returns empty."""
        from src.chatbot import duckduckgo_search_func
        
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs.return_value = mock_ddgs_instance
        
        result = duckduckgo_search_func("nonexistent query xyz123")
        
        assert result == "No results found."
    
    @patch('src.chatbot.DDGS')
    def test_handles_search_error(self, mock_ddgs):
        """Should handle search errors gracefully."""
        from src.chatbot import duckduckgo_search_func
        
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.side_effect = Exception("Network error")
        mock_ddgs.return_value = mock_ddgs_instance
        
        result = duckduckgo_search_func("test query")
        
        assert "Search error" in result
        assert "Network error" in result


class TestBuildRagChain:
    """Tests for the build_rag_chain function."""
    
    @patch('src.chatbot.AgentExecutor')
    @patch('src.chatbot.create_react_agent')
    @patch('src.chatbot.PromptTemplate')
    @patch('src.chatbot.ChatOllama')
    @patch('src.chatbot.create_retriever_tool')
    @patch('src.chatbot.Tool')
    def test_builds_agent_with_tools(
        self, mock_tool, mock_create_retriever_tool, 
        mock_chat_ollama, mock_prompt_template, 
        mock_create_react_agent, mock_agent_executor
    ):
        """Should build agent with search and retriever tools."""
        from src.chatbot import build_rag_chain
        
        # Setup mocks
        mock_retriever = MagicMock()
        mock_search_tool = MagicMock()
        mock_tool.return_value = mock_search_tool
        
        mock_retriever_tool = MagicMock()
        mock_create_retriever_tool.return_value = mock_retriever_tool
        
        mock_llm = MagicMock()
        mock_chat_ollama.return_value = mock_llm
        
        mock_prompt = MagicMock()
        mock_prompt_template.from_template.return_value = mock_prompt
        
        mock_agent = MagicMock()
        mock_create_react_agent.return_value = mock_agent
        
        mock_executor = MagicMock()
        mock_agent_executor.return_value = mock_executor
        
        # Execute
        result = build_rag_chain(mock_retriever)
        
        # Verify tools were created
        mock_tool.assert_called_once()
        mock_create_retriever_tool.assert_called_once_with(
            mock_retriever,
            "tng_knowledge_base",
            "Search for lines and exact dialogues from Star Trek: The Next Generation scripts. Use this when asked about specific quotes or plot points from the show."
        )
        
        # Verify LLM was created
        mock_chat_ollama.assert_called_once()
        
        # Verify agent was created
        mock_create_react_agent.assert_called_once()
        
        # Verify executor was created
        mock_agent_executor.assert_called_once()
        
        assert result == mock_executor
    
    @patch('src.chatbot.AgentExecutor')
    @patch('src.chatbot.create_react_agent')
    @patch('src.chatbot.PromptTemplate')
    @patch('src.chatbot.ChatOllama')
    @patch('src.chatbot.create_retriever_tool')
    @patch('src.chatbot.Tool')
    def test_agent_executor_has_verbose_enabled(
        self, mock_tool, mock_create_retriever_tool, 
        mock_chat_ollama, mock_prompt_template, 
        mock_create_react_agent, mock_agent_executor
    ):
        """Should create AgentExecutor with verbose=True."""
        from src.chatbot import build_rag_chain
        
        mock_retriever = MagicMock()
        mock_executor = MagicMock()
        mock_agent_executor.return_value = mock_executor
        
        build_rag_chain(mock_retriever)
        
        # Verify verbose is True in the call
        call_kwargs = mock_agent_executor.call_args[1]
        assert call_kwargs.get('verbose') is True
        assert call_kwargs.get('handle_parsing_errors') is True


class TestQueryChain:
    """Tests for the query_chain function."""
    
    def test_invokes_chain_with_question(self):
        """Should invoke chain and return output."""
        from src.chatbot import query_chain
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"output": "I am Data, an android."}
        
        result = query_chain(mock_chain, "Who are you?")
        
        mock_chain.invoke.assert_called_once_with({"input": "Who are you?"})
        assert result == "I am Data, an android."
    
    def test_extracts_output_from_result(self):
        """Should extract the 'output' key from result dictionary."""
        from src.chatbot import query_chain
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "input": "Test question",
            "output": "Test answer",
            "intermediate_steps": []
        }
        
        result = query_chain(mock_chain, "Test question")
        
        assert result == "Test answer"
