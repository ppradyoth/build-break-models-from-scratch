import sys
from pathlib import Path

# Make `import ctf` work when running pytest without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent))
