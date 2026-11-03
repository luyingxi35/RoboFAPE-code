"""Check the independent hand-calculated expected labeling result."""
import json
from pathlib import Path
import sys

data=json.loads(Path(sys.argv[1]).read_text())
assert data["progress"] == [0.5,1.0,0.25], data["progress"]
assert [x["reference_frame"] for x in data["alignment"]] == [1,3,0]
print("Analytical example: PASS")
