import os
import sys
from pathlib import Path

from popper.loop import Popper, learn_solution
from popper.util import Settings

CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())

from src.reasoning.to_popper import representation_to_popper  # noqa: E402


