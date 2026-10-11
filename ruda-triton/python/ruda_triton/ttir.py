"""Parser for the textual TTIR emitted by ``module.str_nodebug()``.

Only the structure needed by the SIMT lowering is recovered: operations, their operands and
results, attribute text, trailing type annotations and nested regions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

VALUE = r"%[\w.$-]+(?:#\d+)?"
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_LOC = re.compile(r"\s+loc\((?:[^()]|\([^()]*\))*\)")


@dataclass(frozen=True)
class Type:
    shape: tuple | None
    elem: str
    pointee: str | None = None

    @property
    def is_tensor(self) -> bool:
        return self.shape is not None

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape or ():
            n *= d
        return n

    def scalar(self) -> "Type":
        return Type(None, self.elem, self.pointee)

    def with_elem(self, elem: str, pointee: str | None = None) -> "Type":
        return Type(self.shape, elem, pointee)


def parse_type(text: str) -> Type:
    text = text.strip()
    if text.startswith("tensor<") and text.endswith(">"):
        inner = text[len("tensor<"):-1]
        dims = []
        while True:
            m = re.match(r"(\d+)x", inner)
            if not m:
                break
            dims.append(int(m.group(1)))
            inner = inner[m.end():]
        scalar = parse_type(inner)
        return Type(tuple(dims), scalar.elem, scalar.pointee)
    if text.startswith("!tt.ptr<"):
        inner = text[len("!tt.ptr<"):-1].split(",")[0]
        return Type(None, "ptr", parse_type(inner).elem)
    if text == "index":
        return Type(None, "i64")
    if re.fullmatch(r"(f16|bf16|f32|f64|i1|i8|i16|i32|i64)", text):
        return Type(None, text)
    raise NotImplementedError(f"unsupported TTIR type {text!r}")


def split_top(text: str, sep: str = ",") -> list[str]:
    parts, depth, cur = [], 0, []
    for i, ch in enumerate(text):
        if ch in "<([{":
            depth += 1
        elif ch == ">" and i > 0 and text[i - 1] == "-":
            pass
        elif ch in ">)]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def split_tail_type(text: str) -> tuple[str, str]:
    """Split ``body : type`` at the last top-level colon that is followed by a space."""
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == ">" and i > 0 and text[i - 1] == "-":
            continue
        if ch in ">)]}":
            depth += 1
        elif ch in "<([{":
            depth -= 1
        elif ch == ":" and depth == 0 and i + 1 < len(text) and text[i + 1] == " " and i > 0 and text[i - 1] == " ":
            return text[:i].strip(), text[i + 1:].strip()
    return text.strip(), ""


@dataclass
class Block:
    args: list[tuple[str, Type]] = field(default_factory=list)
    ops: list["Op"] = field(default_factory=list)


@dataclass
class Op:
    name: str
    results: list[str]
    body: str
    tail: str
    regions: list[Block] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)

    @property
    def operands(self) -> list[str]:
        return re.findall(VALUE, self.body)

    def attr(self, key: str):
        m = re.search(rf"\b{re.escape(key)}\s*=\s*([^,}}>]+)", self.body)
        if not m:
            return None
        value = m.group(1).strip()
        if " : " in value:
            value = value.split(" : ")[0]
        return value

    def result_types(self) -> list[Type]:
        tail = self.tail
        if "->" in tail:
            out = tail.split("->")[-1].strip()
        elif " to " in tail:
            out = tail.split(" to ")[-1].strip()
        else:
            out = split_top(tail)[-1] if tail else ""
        if out.startswith("(") and out.endswith(")"):
            out = out[1:-1]
        return [parse_type(t) for t in split_top(out)] if out else []


@dataclass
class Func:
    name: str
    args: list[tuple[str, Type]]
    body: Block


def _strip(line: str) -> str:
    return _LOC.sub("", line).rstrip()


def parse_module(text: str) -> Func:
    lines = [_strip(l) for l in text.splitlines()]
    lines = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    pos = 0
    while pos < len(lines) and "tt.func" not in lines[pos]:
        pos += 1
    if pos == len(lines):
        raise ValueError("no tt.func in module")
    header = lines[pos]
    m = re.search(r"@([\w$.]+)\((.*)\)\s*(?:attributes\s*\{.*\})?\s*\{$", header)
    if not m:
        raise ValueError(f"cannot parse function header: {header}")
    args = []
    for part in split_top(m.group(2)):
        am = re.match(rf"({VALUE})\s*:\s*(.*?)(\s*\{{.*\}})?$", part)
        args.append((am.group(1), parse_type(am.group(2))))
    body, pos = _parse_block(lines, pos + 1)
    funcs = sum(1 for l in lines if "tt.func" in l)
    if funcs != 1:
        raise NotImplementedError("modules with several functions must be inlined first")
    return Func(m.group(1), args, body)


def _parse_block(lines: list[str], pos: int) -> tuple[Block, int]:
    block = Block()
    while pos < len(lines):
        line = lines[pos].strip()
        if line == "}" or line.startswith("})") or line.startswith("} else"):
            return block, pos
        if line.startswith("^bb"):
            m = re.match(r"\^bb\d+\((.*)\):", line)
            for part in split_top(m.group(1)) if m and m.group(1) else []:
                name, ty = part.split(":", 1)
                block.args.append((name.strip(), parse_type(ty)))
            pos += 1
            continue
        op, pos = _parse_op(lines, pos)
        block.ops.append(op)
    return block, pos


def _parse_op(lines: list[str], pos: int) -> tuple[Op, int]:
    line = lines[pos].strip()
    strings = _STRING.findall(line)
    results: list[str] = []
    m = re.match(r"((?:%[\w.$-]+(?::\d+)?)(?:\s*,\s*%[\w.$-]+(?::\d+)?)*)\s*=\s*", line)
    rest = line
    if m:
        for r in split_top(m.group(1)):
            name, _, count = r.partition(":")
            results += [f"{name}#{i}" for i in range(int(count))] if count else [name]
        rest = line[m.end():]
    if rest.startswith('"'):
        name_end = rest.index('"', 1)
        name = rest[1:name_end]
        rest = rest[name_end + 1:]
    else:
        name, _, rest = rest.partition(" ")
    if not line.endswith("{"):
        body, tail = split_tail_type(_STRING.sub('""', rest))
        return Op(name, results, body, tail, strings=strings), pos + 1

    # Region-holding operation.
    head = rest[:-1].rstrip()
    regions = []
    pos += 1
    region, pos = _parse_block(lines, pos)
    regions.append(region)
    tail = ""
    while True:
        closing = lines[pos].strip()
        if closing.startswith("} else"):
            region, pos = _parse_block(lines, pos + 1)
            regions.append(region)
            continue
        if closing.startswith("}) : "):
            tail = closing[len("}) : "):]
        elif closing.startswith("}) "):
            tail = split_tail_type(closing[2:])[1]
        elif closing.startswith("}, {"):
            region, pos = _parse_block(lines, pos + 1)
            regions.append(region)
            continue
        pos += 1
        break
    if head.endswith("({"):
        head = head[:-2]
    head = head.rstrip("(").rstrip()
    if not tail:
        head, tail = split_tail_type(head)
    return Op(name, results, _STRING.sub('""', head), tail, regions, strings), pos
