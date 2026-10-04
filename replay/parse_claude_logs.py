"""Turn local Claude Code session logs into the normalized replay format (dataset A in DESIGN.md).

PRIVACY: the output goes to data/private/ (gitignored). This script prints aggregate counts only, never content or
paths. Only the aggregate metrics produced by replay.py are meant to be committed.

Usage: .venv/Scripts/python replay/parse_claude_logs.py [--root ~/.claude/projects] [--exclude-session <id> ...]
"""
import argparse
import json
import re
from pathlib import Path

PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/[a-z]/|\./|\.\./)?(?:[\w.\-]+[\\/])*[\w\-]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml|toml|"
    r"html|css|cpp|h|hpp|cs|go|rs|java|sql|sh|ps1|txt|csv|ipynb|ini|cfg)\b"
)
FILE_TOOLS = {"Read", "Edit", "Write", "NotebookEdit", "MultiEdit"}


def norm(p):
    p = p.replace("\\", "/").lower()
    p = re.sub(r"^/([a-z])/", r"\1:/", p)
    return p


def mentions(text):
    return {norm(m.group(0)) for m in PATH_RE.finditer(text or "")}


def resolve(ms, known):
    """Map relative/basename mentions to a full path already seen in this session (no future information)."""
    out = set()
    for m in ms:
        if m in known:
            out.add(m)
            continue
        hits = [k for k in known if k.endswith("/" + m) or k == m]
        out.add(hits[0] if len(hits) == 1 else m)
    return out


def text_of(content):
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        if isinstance(c, dict):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "tool_result":
                parts.append(text_of(c.get("content")))
    return "\n".join(parts)


def parse_session(path):
    recs = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    turns, cur, known, t0 = [], None, set(), None

    def close(turn):
        """Resolve raw mentions once the turn ends; cues are only used after that point, so no leakage."""
        if turn is None:
            return
        known.update(turn["needed"])
        known.update(m for m in turn["env"] if "/" in m)
        turn["conv"] = resolve(turn["conv"], known)
        turn["env"] = resolve(turn["env"], known)

    for r in recs:
        if r.get("isSidechain") or r.get("type") not in ("user", "assistant"):
            continue
        t0 = t0 or r.get("timestamp")
        msg = r.get("message") or {}
        content = msg.get("content")
        is_tool_result = isinstance(content, list) and any(
            isinstance(c, dict) and c.get("type") == "tool_result" for c in content)
        if r["type"] == "user" and not is_tool_result and not r.get("isMeta"):
            txt = text_of(content)
            if not txt.strip() or txt.lstrip().startswith("<command-") or txt.lstrip().startswith("<local-command"):
                continue
            close(cur)
            cur = {"conv": set(mentions(txt)), "env": set(), "needed": set()}
            turns.append(cur)
        elif cur is None:
            continue
        elif r["type"] == "user":  # tool results = environment
            cur["env"] |= mentions(text_of(content))
        else:  # assistant
            for c in content if isinstance(content, list) else []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    cur["conv"] |= mentions(c.get("text"))
                elif c.get("type") == "tool_use" and c.get("name") in FILE_TOOLS:
                    p = (c.get("input") or {}).get("file_path") or (c.get("input") or {}).get("notebook_path")
                    if p:
                        cur["needed"].add(norm(p))
    close(cur)
    turns = [{k: sorted(v) for k, v in t.items()} for t in turns]
    return {"t0": t0 or "", "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--exclude-session", nargs="*", default=[])
    ap.add_argument("--min-turns", type=int, default=3)
    a = ap.parse_args()
    files = [p for p in Path(a.root).rglob("*.jsonl") if "subagents" not in p.parts
             and p.stem not in set(a.exclude_session)]
    sessions = [parse_session(p) for p in files]
    sessions = [s for s in sessions if len(s["turns"]) >= a.min_turns and any(t["needed"] for t in s["turns"])]
    out = Path(__file__).resolve().parent.parent / "data" / "private" / "claude_sessions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sessions), encoding="utf-8")
    n_turns = sum(len(s["turns"]) for s in sessions)
    n_need = sum(len(t["needed"]) for s in sessions for t in s["turns"])
    n_env = sum(len(t["env"]) for s in sessions for t in s["turns"])
    n_conv = sum(len(t["conv"]) for s in sessions for t in s["turns"])
    print(f"log files: {len(files)} | sessions kept: {len(sessions)} | turns: {n_turns} | "
          f"needed-file events: {n_need} | env cues: {n_env} | conv cues: {n_conv}")
    print(f"written to {out} (gitignored)")


if __name__ == "__main__":
    main()
