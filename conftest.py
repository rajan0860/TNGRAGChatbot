"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sample_dialogue_text():
    """Sample dialogue text for testing."""
    return """I am an android. I do not require sleep.
Thank you, Captain. I was merely stating a fact.
Fascinating. This data is most intriguing."""


@pytest.fixture
def sample_script_content():
    """Sample script content in the expected format."""
    return """
PICARD
Make it so.

DATA
Acknowledged, Captain. I am initiating the sequence now.

RIKER
Number One, reporting.

DATA
Commander, I have completed the analysis.

WORF
Shields are at maximum.

"""
