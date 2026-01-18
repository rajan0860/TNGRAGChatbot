"""
Unit tests for the config module.
Tests configuration values and environment settings.
"""
import pytest
import os


class TestConfigValues:
    """Tests for configuration values."""
    
    def test_llm_model_name_is_set(self):
        """Should have LLM model name configured."""
        from src.config import LLM_MODEL_NAME
        
        assert LLM_MODEL_NAME is not None
        assert isinstance(LLM_MODEL_NAME, str)
        assert len(LLM_MODEL_NAME) > 0
    
    def test_embedding_model_name_is_set(self):
        """Should have embedding model name configured."""
        from src.config import EMBEDDING_MODEL_NAME
        
        assert EMBEDDING_MODEL_NAME is not None
        assert isinstance(EMBEDDING_MODEL_NAME, str)
        assert len(EMBEDDING_MODEL_NAME) > 0
    
    def test_data_output_path_is_valid(self):
        """Should have valid data output path."""
        from src.config import DATA_OUTPUT_PATH
        
        assert DATA_OUTPUT_PATH is not None
        assert isinstance(DATA_OUTPUT_PATH, str)
        assert "data" in DATA_OUTPUT_PATH
        assert DATA_OUTPUT_PATH.endswith(".txt")
    
    def test_scripts_dir_is_set(self):
        """Should have scripts directory configured."""
        from src.config import SCRIPTS_DIR
        
        assert SCRIPTS_DIR is not None
        assert isinstance(SCRIPTS_DIR, str)
    
    def test_kmp_duplicate_lib_env_var(self):
        """Should set KMP_DUPLICATE_LIB_OK environment variable."""
        # Import config to trigger the os.environ setting
        from src import config
        
        assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"


class TestConfigPaths:
    """Tests for path configuration."""
    
    def test_project_root_is_absolute(self):
        """PROJECT_ROOT should be an absolute path."""
        from src.config import PROJECT_ROOT
        
        # PROJECT_ROOT is set using os.getcwd(), which returns absolute path
        assert os.path.isabs(PROJECT_ROOT) or PROJECT_ROOT == os.getcwd()
    
    def test_data_output_path_contains_processed(self):
        """DATA_OUTPUT_PATH should point to processed data."""
        from src.config import DATA_OUTPUT_PATH
        
        assert "processed" in DATA_OUTPUT_PATH
        assert "data_lines.txt" in DATA_OUTPUT_PATH
