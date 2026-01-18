"""
Unit tests for the vector_store module.
Tests embedding generation, index creation, and persistence.
"""
import pytest
import os
from unittest.mock import patch, MagicMock


class TestGetEmbeddings:
    """Tests for the get_embeddings function."""
    
    @patch('src.vector_store.OllamaEmbeddings')
    def test_returns_ollama_embeddings(self, mock_ollama):
        """Should return OllamaEmbeddings instance with correct model."""
        from src.vector_store import get_embeddings
        from src.config import EMBEDDING_MODEL_NAME
        
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance
        
        result = get_embeddings()
        
        mock_ollama.assert_called_once_with(model=EMBEDDING_MODEL_NAME)
        assert result == mock_instance


class TestCreateIndexSemantic:
    """Tests for the create_index_semantic function."""
    
    @patch('src.vector_store.FAISS')
    @patch('src.vector_store.SemanticChunker')
    @patch('src.vector_store.get_embeddings')
    def test_creates_semantic_index(self, mock_get_embeddings, mock_chunker, mock_faiss):
        """Should create FAISS index using semantic chunking."""
        from src.vector_store import create_index_semantic
        
        # Setup mocks
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_splitter = MagicMock()
        mock_docs = [MagicMock()]
        mock_splitter.create_documents.return_value = mock_docs
        mock_chunker.return_value = mock_splitter
        
        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store
        
        # Execute
        result = create_index_semantic("Test text content")
        
        # Verify
        mock_get_embeddings.assert_called_once()
        mock_chunker.assert_called_once_with(
            mock_embeddings,
            breakpoint_threshold_type="percentile"
        )
        mock_splitter.create_documents.assert_called_once_with(["Test text content"])
        mock_faiss.from_documents.assert_called_once_with(mock_docs, mock_embeddings)
        assert result == mock_vector_store


class TestCreateIndexRecursive:
    """Tests for the create_index_recursive function."""
    
    @patch('src.vector_store.FAISS')
    @patch('src.vector_store.RecursiveCharacterTextSplitter')
    @patch('src.vector_store.get_embeddings')
    def test_creates_recursive_index(self, mock_get_embeddings, mock_splitter_class, mock_faiss):
        """Should create FAISS index using recursive character splitting."""
        from src.vector_store import create_index_recursive
        
        # Setup mocks
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_splitter = MagicMock()
        mock_docs = [MagicMock()]
        mock_splitter.create_documents.return_value = mock_docs
        mock_splitter_class.return_value = mock_splitter
        
        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store
        
        # Execute
        result = create_index_recursive("Test text content")
        
        # Verify
        mock_get_embeddings.assert_called_once()
        mock_splitter_class.assert_called_once_with(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        mock_splitter.create_documents.assert_called_once_with(["Test text content"])
        mock_faiss.from_documents.assert_called_once_with(mock_docs, mock_embeddings)
        assert result == mock_vector_store


class TestSaveIndex:
    """Tests for the save_index function."""
    
    def test_calls_save_local(self):
        """Should call save_local on the vector store."""
        from src.vector_store import save_index
        
        mock_vector_store = MagicMock()
        save_index(mock_vector_store, "/test/path")
        
        mock_vector_store.save_local.assert_called_once_with("/test/path")


class TestLoadIndex:
    """Tests for the load_index function."""
    
    @patch('src.vector_store.FAISS')
    @patch('src.vector_store.get_embeddings')
    def test_loads_index_from_path(self, mock_get_embeddings, mock_faiss):
        """Should load FAISS index from local path."""
        from src.vector_store import load_index
        
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_faiss.load_local.return_value = mock_vector_store
        
        result = load_index("/test/path")
        
        mock_get_embeddings.assert_called_once()
        mock_faiss.load_local.assert_called_once_with(
            "/test/path", 
            mock_embeddings, 
            allow_dangerous_deserialization=True
        )
        assert result == mock_vector_store
