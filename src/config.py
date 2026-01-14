import os

# Paths
# Assuming the script runs from the project root
PROJECT_ROOT = os.getcwd()
# You might want to make this configurable via env var
SCRIPTS_DIR = "/Users/rajanmehta/Documents/MLProjects/scripts_tng" 
DATA_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "data_lines.txt")

# Models
LLM_MODEL_NAME = "qwen2.5:7b-instruct"
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"

# Environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
