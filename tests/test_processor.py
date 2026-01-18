"""
Unit tests for the processor module.
Tests dialogue extraction and text processing functions.
"""
import pytest
import os
import tempfile
from src.processor import (
    strip_parentheses,
    is_single_word_all_caps,
    extract_character_lines,
    process_directory,
    save_dialogues
)


class TestStripParentheses:
    """Tests for the strip_parentheses function."""
    
    def test_removes_single_parentheses(self):
        """Should remove content within single parentheses."""
        result = strip_parentheses("Hello (world) there")
        assert result == "Hello  there"
    
    def test_removes_multiple_parentheses(self):
        """Should remove content within multiple parentheses."""
        result = strip_parentheses("Hello (world) and (goodbye) there")
        assert result == "Hello  and  there"
    
    def test_no_parentheses(self):
        """Should return the string unchanged if no parentheses."""
        result = strip_parentheses("Hello world")
        assert result == "Hello world"
    
    def test_empty_string(self):
        """Should handle empty string."""
        result = strip_parentheses("")
        assert result == ""
    
    def test_empty_parentheses(self):
        """Should handle empty parentheses."""
        result = strip_parentheses("Hello () world")
        assert result == "Hello  world"
    
    def test_nested_text_in_parentheses(self):
        """Should remove text within parentheses including special characters."""
        result = strip_parentheses("Data (android, second officer) speaks")
        assert result == "Data  speaks"


class TestIsSingleWordAllCaps:
    """Tests for the is_single_word_all_caps function."""
    
    def test_single_caps_word(self):
        """Should return True for a single all-caps word."""
        assert is_single_word_all_caps("DATA") is True
    
    def test_single_caps_word_picard(self):
        """Should return True for character name PICARD."""
        assert is_single_word_all_caps("PICARD") is True
    
    def test_single_lowercase_word(self):
        """Should return False for a lowercase word."""
        assert is_single_word_all_caps("data") is False
    
    def test_mixed_case_word(self):
        """Should return False for mixed case word."""
        assert is_single_word_all_caps("Data") is False
    
    def test_multiple_words(self):
        """Should return False for multiple words."""
        assert is_single_word_all_caps("DATA SPEAKS") is False
    
    def test_empty_string(self):
        """Should return False for empty string."""
        assert is_single_word_all_caps("") is False
    
    def test_word_with_number(self):
        """Should return False for words containing numbers."""
        assert is_single_word_all_caps("DATA1") is False
    
    def test_line_number(self):
        """Should return False for line numbers."""
        assert is_single_word_all_caps("123") is False
    
    def test_word_with_whitespace(self):
        """Should handle leading/trailing whitespace correctly."""
        # Note: The function splits on whitespace, so this should work
        assert is_single_word_all_caps("  DATA  ") is True


class TestExtractCharacterLines:
    """Tests for the extract_character_lines function."""
    
    @pytest.fixture
    def sample_script(self, tmp_path):
        """Create a sample script file for testing."""
        script_content = """
PICARD
Captain's log, stardate 12345.

DATA
I am an android. I do not require sleep.

PICARD
Interesting observation, Mr. Data.

DATA
Thank you, Captain. I was merely stating a fact.

"""
        script_file = tmp_path / "test_script.txt"
        script_file.write_text(script_content)
        return str(script_file)
    
    def test_extracts_data_lines(self, sample_script):
        """Should extract only DATA's dialogues."""
        dialogues = []
        extract_character_lines(sample_script, "DATA", dialogues)
        assert len(dialogues) == 2
        assert "I am an android. I do not require sleep." in dialogues
        assert "Thank you, Captain. I was merely stating a fact." in dialogues
    
    def test_extracts_picard_lines(self, sample_script):
        """Should extract only PICARD's dialogues."""
        dialogues = []
        extract_character_lines(sample_script, "PICARD", dialogues)
        assert len(dialogues) == 2
        assert "Captain's log, stardate 12345." in dialogues
    
    def test_no_matching_character(self, sample_script):
        """Should return empty list for non-existent character."""
        dialogues = []
        extract_character_lines(sample_script, "WORF", dialogues)
        assert len(dialogues) == 0
    
    @pytest.fixture
    def script_with_parentheses(self, tmp_path):
        """Create a script with parenthetical directions."""
        script_content = """
DATA
(concerned) Captain, I believe we have a problem.

"""
        script_file = tmp_path / "test_script_paren.txt"
        script_file.write_text(script_content)
        return str(script_file)
    
    def test_strips_parentheses_from_dialogue(self, script_with_parentheses):
        """Should strip parenthetical stage directions from dialogues."""
        dialogues = []
        extract_character_lines(script_with_parentheses, "DATA", dialogues)
        assert len(dialogues) == 1
        assert "concerned" not in dialogues[0]
        assert "Captain, I believe we have a problem." in dialogues[0]


class TestProcessDirectory:
    """Tests for the process_directory function."""
    
    @pytest.fixture
    def script_directory(self, tmp_path):
        """Create a directory with multiple script files."""
        # Script 1
        script1 = tmp_path / "episode1.txt"
        script1.write_text("""
DATA
Greetings from episode one.

""")
        # Script 2
        script2 = tmp_path / "episode2.txt"
        script2.write_text("""
DATA
Hello from episode two.

""")
        return str(tmp_path)
    
    def test_processes_all_files(self, script_directory):
        """Should process all files in directory."""
        dialogues = process_directory(script_directory, "DATA")
        assert len(dialogues) == 2
    
    def test_nonexistent_directory(self):
        """Should handle non-existent directory gracefully."""
        dialogues = process_directory("/nonexistent/path", "DATA")
        assert dialogues == []
    
    def test_empty_directory(self, tmp_path):
        """Should handle empty directory."""
        dialogues = process_directory(str(tmp_path), "DATA")
        assert dialogues == []


class TestSaveDialogues:
    """Tests for the save_dialogues function."""
    
    def test_saves_dialogues_to_file(self, tmp_path):
        """Should save dialogues list to a file."""
        output_path = tmp_path / "output" / "dialogues.txt"
        dialogues = ["Line one", "Line two", "Line three"]
        save_dialogues(dialogues, str(output_path))
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "Line one" in content
        assert "Line two" in content
        assert "Line three" in content
    
    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if they don't exist."""
        output_path = tmp_path / "nested" / "deep" / "dialogues.txt"
        save_dialogues(["Test line"], str(output_path))
        assert output_path.exists()
    
    def test_handles_empty_list(self, tmp_path):
        """Should handle empty dialogues list."""
        output_path = tmp_path / "empty.txt"
        save_dialogues([], str(output_path))
        assert output_path.exists()
        assert output_path.read_text() == ""
