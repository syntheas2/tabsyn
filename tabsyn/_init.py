# Add 'root dir' to the path so Python can find 'tabsyn'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))