import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parents[1]
FOLDER_SRC = AKAR / "src"

if str(FOLDER_SRC) not in sys.path:
    sys.path.insert(0, str(FOLDER_SRC))