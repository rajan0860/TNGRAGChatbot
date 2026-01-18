"""
Integration tests for the TNGRAGChatbot project.
These tests verify that components work together correctly.
"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock


class TestIngestPipeline:
    """Integration tests for the data ingestion pipeline."""
    
    @pytest.fixture
    def temp_script_dir(self, tmp_path):
        """Create a temporary directory with test scripts."""
        # Create test script files
        script1 = tmp_path / "scripts" / "episode1.txt"
        script1.parent.mkdir(parents=True, exist_ok=True)
        script1.write_text("""
DATA
Greetings. I am Lieutenant Commander Data.

PICARD
Engage!

DATA
Acknowledged, Captain.

""")
        
        script2 = tmp_path / "scripts" / "episode2.txt"
        script2.write_text("""
DATA
Fascinating observation.

WORF
Today is a good day to die.

DATA
I do not understand that expression.

""")
        return str(tmp_path / "scripts")
    
    def test_full_extraction_pipeline(self, temp_script_dir, tmp_path):
        """Test complete dialogue extraction from scripts to file."""
        from src.processor import process_directory, save_dialogues
        
        # Extract dialogues
        dialogues = process_directory(temp_script_dir, "DATA")
        
        # Verify extraction
        assert len(dialogues) == 4  # 4 DATA lines across 2 scripts
        
        # Save to file
        output_path = str(tmp_path / "output" / "data_lines.txt")
        save_dialogues(dialogues, output_path)
        
        # Verify file was created and contains content
        assert os.path.exists(output_path)
        with open(output_path) as f:
            content = f.read()
            assert "Greetings" in content or "Lieutenant Commander Data" in content


class TestVectorStoreIntegration:
    """Integration tests for vector store operations."""
    
    @patch('src.vector_store.FAISS')
    @patch('src.vector_store.RecursiveCharacterTextSplitter')
    @patch('src.vector_store.get_embeddings')
    def test_create_and_save_index(self, mock_get_embeddings, mock_splitter_class, mock_faiss, tmp_path):
        """Test creating and saving a vector index."""
        from src.vector_store import create_index_recursive, save_index
        
        # Setup mocks
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_splitter = MagicMock()
        mock_docs = [MagicMock()]
        mock_splitter.create_documents.return_value = mock_docs
        mock_splitter_class.return_value = mock_splitter
        
        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store
        
        # Create index
        test_text = "I am Data, an android aboard the USS Enterprise."
        vector_store = create_index_recursive(test_text)
        
        # Save index
        index_path = str(tmp_path / "test_index")
        save_index(vector_store, index_path)
        
        # Verify save was called
        mock_vector_store.save_local.assert_called_once_with(index_path)


class TestChatbotChainIntegration:
    """Integration tests for chatbot chain construction."""
    
    @patch('src.chatbot.AgentExecutor')
    @patch('src.chatbot.create_react_agent')
    @patch('src.chatbot.PromptTemplate')
    @patch('src.chatbot.ChatOllama')
    @patch('src.chatbot.create_retriever_tool')
    @patch('src.chatbot.Tool')
    def test_build_and_query_chain(
        self, mock_tool, mock_create_retriever_tool,
        mock_chat_ollama, mock_prompt_template,
        mock_create_react_agent, mock_agent_executor
    ):
        """Test building a chain and querying it."""
        from src.chatbot import build_rag_chain, query_chain
        
        # Setup mocks
        mock_retriever = MagicMock()
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "I am an android."}
        mock_agent_executor.return_value = mock_executor
        
        # Build chain
        chain = build_rag_chain(mock_retriever)
        
        # Query chain
        result = query_chain(chain, "What are you?")
        
        assert result == "I am an android."
        mock_executor.invoke.assert_called_once_with({"input": "What are you?"})


class TestEndToEndMocked:
    """End-to-end tests with mocked external dependencies."""
    
    @pytest.fixture
    def mock_script_environment(self, tmp_path):
        """Create a complete mock environment for testing."""
        # Create scripts directory
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        
        # Create a sample script
        script = scripts_dir / "test_episode.txt"
        script.write_text("""
DATA
Captain, I have analyzed the sensor readings.

PICARD
What do you make of them, Data?

DATA
They appear to indicate a temporal anomaly.

""")
        
        # Create data output directory
        data_dir = tmp_path / "data" / "processed"
        data_dir.mkdir(parents=True)
        
        return {
            'scripts_dir': str(scripts_dir),
            'data_dir': str(data_dir),
            'output_file': str(data_dir / "data_lines.txt")
        }
    
    def test_complete_ingest_flow(self, mock_script_environment):
        """Test the complete data ingestion flow."""
        from src.processor import process_directory, save_dialogues
        
        # Process scripts
        dialogues = process_directory(
            mock_script_environment['scripts_dir'],
            "DATA"
        )
        
        # Verify expected dialogues were extracted
        assert len(dialogues) == 2
        assert any("sensor readings" in d for d in dialogues)
        assert any("temporal anomaly" in d for d in dialogues)
        
        # Save dialogues
        save_dialogues(dialogues, mock_script_environment['output_file'])
        
        # Verify file exists and has content
        with open(mock_script_environment['output_file']) as f:
            lines = f.readlines()
            assert len(lines) == 2
