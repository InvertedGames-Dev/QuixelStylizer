import os

TOOL_VERSION = "2.0.0"
TOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW_TITLE = f"Quixel Stylizer v{TOOL_VERSION}"
LOG_PATH = os.path.join(TOOL_DIR, "stylizer.log")
