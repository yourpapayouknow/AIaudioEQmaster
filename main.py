from __future__ import annotations
import sys
from pathlib import Path

def _ensure_root_on_path():
    root=Path(__file__).resolve().parent
    if str(root) not in sys.path: sys.path.insert(0,str(root))

def main()->int:
    _ensure_root_on_path()
    from src.cli import main as cli_main
    return int(cli_main())

if __name__=='__main__':
    raise SystemExit(main())
