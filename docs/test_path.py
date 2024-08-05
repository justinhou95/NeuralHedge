from pathlib import Path
import sys

x = Path(__file__).parents[1].joinpath("src").resolve().as_posix()
print(x)
