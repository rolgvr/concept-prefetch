"""Parser test on a synthetic log (no real logs). Run: .venv/Scripts/python replay/test_parse.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_claude_logs import parse_session  # noqa: E402

RECS = [
    {"type": "user", "timestamp": "2026-01-01", "message": {"content": "fix the bug in engine.py please"}},
    {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "Looking at toy/engine.py"},
        {"type": "tool_use", "name": "Read", "input": {"file_path": r"C:\proj\toy\engine.py"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "content": r'Traceback: File "C:\proj\toy\data.py", line 3'}]}},
    {"type": "user", "message": {"content": "now what about data.py"}},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "C:/proj/toy/data.py"}}]}},
    {"type": "user", "message": {"content": "<command-name>/clear</command-name>"}},
    {"type": "user", "isSidechain": True, "message": {"content": "subagent noise run.py"}},
    {"type": "user", "message": {"content": "and run.py"}},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/c/proj/toy/run.py"}}]}},
]

with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "s.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in RECS))
    s = parse_session(f)

t = s["turns"]
assert len(t) == 3, t
assert t[0]["needed"] == ["c:/proj/toy/engine.py"], t[0]
assert "c:/proj/toy/engine.py" in t[0]["conv"], t[0]          # basename mention resolved at turn close
assert t[0]["env"] == ["c:/proj/toy/data.py"], t[0]           # traceback path = environment cue
assert t[1]["needed"] == ["c:/proj/toy/data.py"]
assert t[2]["needed"] == ["c:/proj/toy/run.py"]               # /c/ -> c:/ normalization
print("ok parse:", json.dumps(t))
