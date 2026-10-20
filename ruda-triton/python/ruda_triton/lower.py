"""Lower TTIR to the scalarized SIMT kernel description consumed by ``ruda-triton-ptx``.

Execution model: one Triton program is one CTA of ``N = num_warps * 32`` threads. A tensor
with ``P`` elements (row-major flattening) is distributed cyclically: thread ``t`` owns elements
``e = s * N + t`` for slots ``s < max(1, P // N)``; when ``P < N`` only threads ``t < P`` own an
element. Scalars are replicated in every thread. Because the flattening is row-major,
``expand_dims`` and ``reshape`` move no data. Ops that permute elements across threads either
recompute the value at the new index (when it is a pure function of the element index) or
exchange it through shared memory.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Callable

from .ttir import Block, Func, Op, Type, parse_module, split_top, parse_type

FLOATS = {"f16", "bf16", "f32", "f64"}
BYTES = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "f16": 2, "bf16": 2, "f32": 4, "f64": 8, "ptr": 8,
         "u8": 1, "u16": 2, "u32": 4, "u64": 8}
UNSIGNED = {"i8": "u8", "i16": "u16", "i32": "u32", "i64": "u64"}


class LoweringError(NotImplementedError):
    pass


def const(value, ty: str) -> dict:
    if ty in FLOATS:
        return {"c": "0x%016x" % struct.unpack("<Q", struct.pack("<d", float(value)))[0], "t": ty}
    if ty == "i1":
        return {"c": "1" if value else "0", "t": ty}
    return {"c": str(int(value)), "t": ty}


def official_registers(layout: dict, axis: int, R: int) -> list[tuple[int, int]]:
    """(rep, element) pairs one thread holds along ``axis``, in register order."""
    s, t, w = layout["spt"][axis], layout["tpw"][axis], layout["wpc"][axis]
    tile = s * t * w
    reps = max(1, -(-R // tile))
    return [(rep, j) for rep in range(reps) for j in range(s) if rep * tile + j < R]


def simt_type(t: Type) -> str:
    return "u64" if t.elem == "ptr" else t.elem


@dataclass
class Val:
    """A lowered SSA value. ``slots`` is None for scalars (use ``scalar``)."""
    ty: Type
    scalar: dict | None = None
    slots: list | None = None
    uniform: dict | None = None
    gen: Callable | None = None

    @property
    def is_tensor(self) -> bool:
        return self.ty.is_tensor


@dataclass
class Emitter:
    threads: int
    locals: list = field(default_factory=list)
    stack: list = field(default_factory=list)
    shared: list = field(default_factory=list)
    shared_pool: dict = field(default_factory=dict)
    elem_index: list = field(default_factory=list)
    externs: dict = field(default_factory=dict)
    contract: bool = False
    # GX1 target: no inline PTX or libdevice; every operation stays a generic SIMT statement.
    gx: bool = False

    def __post_init__(self):
        self.stack.append([])

    @property
    def stmts(self) -> list:
        return self.stack[-1]

    def local(self, ty: str) -> dict:
        self.locals.append(ty)
        return {"l": len(self.locals) - 1}

    def ty_of(self, operand: dict) -> str:
        if "l" in operand:
            return self.locals[operand["l"]]
        if "t" in operand:
            return operand["t"]
        if "b" in operand:
            return "u32"
        raise LoweringError(f"type of {operand}")

    def emit(self, stmt: dict):
        self.stmts.append(stmt)

    def bin(self, op: str, a: dict, b: dict, ty: str | None = None) -> dict:
        ty = ty or self.ty_of(a)
        if self.contract and not self.gx and op in ("add", "sub", "mul") and ty in ("f32", "f64"):
            # LLVM emits contractable float ops without `.rn`, letting ptxas form FMAs.
            return self.asm(f"{op}.{ty} $0, $1, $2;", ty, a, b, registers=False)
        out = self.local(ty)
        self.emit({"k": "bin", "op": op, "out": out["l"], "a": a, "b": b})
        return out

    def un(self, op: str, a: dict, ty: str | None = None) -> dict:
        out = self.local(ty or self.ty_of(a))
        self.emit({"k": "un", "op": op, "out": out["l"], "a": a})
        return out

    def asm(self, code: str, ty: str, *args: dict, registers: bool = True) -> dict:
        if registers:
            # LLVM materializes constant operands of NVVM intrinsics in registers.
            args = tuple(self.copy(a) if "c" in a else a for a in args)
        out = self.local(ty)
        self.emit({"k": "asm", "code": code, "out": out["l"], "args": list(args)})
        return out

    def libdevice(self, symbol: str, ty: str, *args: dict) -> dict:
        """Calls ``symbol`` from libdevice; the ``.func`` bodies are attached by the compiler."""
        arg_tys = tuple(self.ty_of(a) for a in args)
        name = f"ruda_{symbol.lstrip('_')}"
        self.externs[name] = (symbol, ty, arg_tys)
        lines = ["{", f".param .b{BYTES[ty] * 8} rp;"]
        for i, t in enumerate(arg_tys):
            lines += [f".param .b{BYTES[t] * 8} p{i};", f"st.param.b{BYTES[t] * 8} [p{i}], ${i + 1};"]
        lines += [f"call.uni (rp), {name}, ({', '.join(f'p{i}' for i in range(len(args)))});",
                  f"ld.param.b{BYTES[ty] * 8} $0, [rp];", "}"]
        out = self.local(ty)
        self.emit({"k": "asm", "code": "\n".join(lines), "out": out["l"], "args": list(args), "extern": name})
        return out

    def cmp(self, op: str, a: dict, b: dict) -> dict:
        out = self.local("i1")
        self.emit({"k": "cmp", "op": op, "out": out["l"], "a": a, "b": b})
        return out

    def cast(self, a: dict, ty: str) -> dict:
        if self.ty_of(a) == ty:
            return a
        if "c" in a and ty not in FLOATS and ty != "i1" and self.ty_of(a) not in FLOATS:
            return const(int(a["c"]), ty)
        out = self.local(ty)
        self.emit({"k": "cast", "out": out["l"], "a": a})
        return out

    def bitcast(self, a: dict, ty: str) -> dict:
        if self.ty_of(a) == ty:
            return a
        out = self.local(ty)
        self.emit({"k": "bitcast", "out": out["l"], "a": a})
        return out

    def sel(self, c: dict, a: dict, b: dict) -> dict:
        out = self.local(self.ty_of(a))
        self.emit({"k": "sel", "out": out["l"], "c": c, "a": a, "b": b})
        return out

    def mov(self, out: dict, a: dict):
        self.emit({"k": "mov", "out": out["l"], "a": a})

    def copy(self, a: dict) -> dict:
        out = self.local(self.ty_of(a))
        self.mov(out, a)
        return out

    def block(self) -> list:
        self.stack.append([])
        return self.stack[-1]

    def end(self) -> list:
        return self.stack.pop()

    def if_(self, cond: dict, then: Callable, otherwise: Callable | None = None):
        if "c" in cond:
            if cond["c"] == "1":
                then()
            elif otherwise:
                otherwise()
            return
        self.block()
        then()
        t = self.end()
        e = []
        if otherwise:
            self.block()
            otherwise()
            e = self.end()
        self.emit({"k": "if", "c": cond, "then": t, "else": e})

    def barrier(self):
        self.emit({"k": "bar"})

    def uindex(self, idx: dict) -> dict:
        if "c" in idx:
            return const(int(idx["c"]), "u32")
        return self.bitcast(idx, "u32")

    def shst(self, arr: int, idx: dict, val: dict):
        self.emit({"k": "shst", "sh": arr, "idx": self.uindex(idx), "val": val})

    def shld(self, arr: int, ty: str, idx: dict) -> dict:
        out = self.local(ty)
        self.emit({"k": "shld", "out": out["l"], "sh": arr, "idx": self.uindex(idx)})
        return out

    def shared_array(self, ty: str, length: int, slot: int) -> int:
        key = (ty, slot)
        if key not in self.shared_pool:
            self.shared_pool[key] = len(self.shared)
            self.shared.append({"ty": ty, "len": length})
        index = self.shared_pool[key]
        self.shared[index]["len"] = max(self.shared[index]["len"], length)
        return index

    # Element ownership -------------------------------------------------------------------
    def slots_for(self, numel: int) -> int:
        return max(1, numel // self.threads)

    def index(self, slot: int) -> dict:
        while len(self.elem_index) <= slot:
            s = len(self.elem_index)
            # Element indices live in the root scope so every nested block sees them.
            root = self.stack[0]
            if not self.elem_index:
                tid = self.local("i32")
                root.append({"k": "cast", "out": tid["l"], "a": {"b": "tid"}})
                self.elem_index.append(tid)
                continue
            out = self.local("i32")
            root.append({"k": "bin", "op": "add", "out": out["l"], "a": self.elem_index[0],
                         "b": const(s * self.threads, "i32")})
            self.elem_index.append(out)
        return self.elem_index[slot]

    def valid(self, numel: int, slot: int) -> dict:
        if numel >= self.threads:
            return const(1, "i1")
        return self.cmp("lt", self.index(slot), const(numel, "i32"))


class Lowering:
    def __init__(self, func: Func, num_warps: int, fp_fusion: bool = True, reduce_layouts: list | None = None,
                 gx: bool = False):
        self.func = func
        self.reduce_layouts = reduce_layouts
        self.reduce_count = 0
        self.e = Emitter(threads=num_warps * 32, contract=fp_fusion, gx=gx)
        self.env: dict[str, Val] = {}
        self.num_ptrs = 0
        self.scalar_types: list[str] = []
        self.arg_kinds: list[tuple[str, str]] = []
        self.fp_fusion = fp_fusion
        self.muls: dict[str, tuple[Op, int]] = {}
        self.negs: dict[str, str] = {}
        self.blocks: list[int] = []
        self.uses: dict[str, int] = {}
        self._count_uses(func.body)
        # LLVM Reassociate ranks, used to reproduce the operand order NVPTX sees when contracting.
        self.rank: dict[str, int] = {name: i + 2 for i, (name, _) in enumerate(func.args)}
        self.bb = 1
        self.bb_rank = 1 << 16

    def new_bb(self):
        self.bb += 1
        self.bb_rank = self.bb << 16

    def unmovable_rank(self) -> int:
        self.bb_rank += 1
        return self.bb_rank

    def assign_rank(self, op: Op):
        if op.name.startswith("scf.") or not op.results:
            return
        operands = [self.rank.get(n, 0) for n in op.operands]
        if op.name in ("tt.load", "tt.reduce", "tt.scan", "tt.atomic_rmw", "tt.atomic_cas") or (
                op.name in ("arith.divsi", "arith.divui", "arith.remsi", "arith.remui")
                and self.rank.get(op.operands[1], 0) != 0):
            rank = self.unmovable_rank()
        elif op.name == "arith.constant":
            rank = 0
        elif op.name in ("tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "arith.negf"):
            rank = max(operands, default=0)
        elif op.name == "tt.make_range":
            rank = 2
        else:
            rank = max(operands, default=0) + 1
        for r in op.results:
            self.rank[r] = rank

    def _count_uses(self, block: Block):
        for op in block.ops:
            for name in op.operands:
                self.uses[name] = self.uses.get(name, 0) + 1
            for region in op.regions:
                self._count_uses(region)

    # Values ------------------------------------------------------------------------------
    def get(self, name: str) -> Val:
        if name not in self.env:
            raise LoweringError(f"undefined value {name}")
        return self.env[name]

    def slot(self, v: Val, s: int) -> dict:
        if not v.is_tensor:
            return v.scalar
        if v.uniform is not None:
            return v.uniform
        return v.slots[s]

    def nslots(self, t: Type) -> int:
        return self.e.slots_for(t.numel) if t.is_tensor else 1

    def make(self, ty: Type, per_slot: Callable[[int], dict], gen: Callable | None = None) -> Val:
        if not ty.is_tensor:
            return Val(ty, scalar=per_slot(0))
        return Val(ty, slots=[per_slot(s) for s in range(self.nslots(ty))], gen=gen)

    def uniform(self, ty: Type, operand: dict) -> Val:
        if not ty.is_tensor:
            return Val(ty, scalar=operand)
        return Val(ty, uniform=operand, gen=lambda e, o=operand: o)

    def materialize(self, v: Val) -> Val:
        if not v.is_tensor or v.uniform is None:
            return v
        return Val(v.ty, slots=[v.uniform] * self.nslots(v.ty), gen=v.gen)

    # Elementwise -------------------------------------------------------------------------
    def elementwise(self, out_ty: Type, inputs: list[Val], fn: Callable[[list[dict]], dict]) -> Val:
        if not out_ty.is_tensor:
            return Val(out_ty, scalar=fn([v.scalar for v in inputs]))
        tensors = [v for v in inputs if v.is_tensor]
        if all(v.uniform is not None for v in tensors):
            return self.uniform(out_ty, fn([v.uniform if v.is_tensor else v.scalar for v in inputs]))
        gen = None
        if all(v.gen is not None or not v.is_tensor for v in inputs):
            gens = [v.gen if v.is_tensor else (lambda e, o=v.scalar: o) for v in inputs]
            gen = lambda e, gens=gens: fn([g(e) for g in gens])
        return self.make(out_ty, lambda s: fn([self.slot(v, s) for v in inputs]), gen)

    # Element movement --------------------------------------------------------------------
    def coords(self, e: dict, shape: tuple) -> list[dict]:
        strides = []
        stride = 1
        for d in reversed(shape):
            strides.append(stride)
            stride *= d
        strides.reverse()
        out = []
        for i, (d, st) in enumerate(zip(shape, strides)):
            if d == 1:
                out.append(const(0, "i32"))
                continue
            c = e if st == 1 else self.e.bin("div", e, const(st, "i32"))
            if i > 0:
                c = self.e.bin("rem", c, const(d, "i32"))
            out.append(c)
        return out

    def flatten(self, coords: list, shape: tuple) -> dict:
        acc = None
        stride = 1
        for c, d in zip(reversed(coords), reversed(shape)):
            if d != 1:
                term = c if stride == 1 else self.e.bin("mul", c, const(stride, "i32"))
                acc = term if acc is None else self.e.bin("add", acc, term)
            stride *= d
        return acc if acc is not None else const(0, "i32")

    def gather(self, srcs: list[Val], out_ty: Type, index_fn: Callable[[dict], list[dict]],
               combine: Callable[[dict, list[dict]], dict]) -> Val:
        """out[e] = combine(e, [src_i[index_fn(e)[i]]])."""
        if all(v.gen is not None for v in srcs):
            gens = [v.gen for v in srcs]
            gen = lambda e: combine(e, [g(i) for g, i in zip(gens, index_fn(e))])
            return self.make(out_ty, lambda s: gen(self.e.index(s)), gen)
        e = self.e
        arrays = []
        e.barrier()
        for i, v in enumerate(srcs):
            v = self.materialize(v)
            ty = simt_type(v.ty)
            arr = e.shared_array(ty, v.ty.numel, i)
            arrays.append((arr, ty, v))
            for s in range(self.nslots(v.ty)):
                write = lambda arr=arr, s=s, v=v: e.shst(arr, e.index(s), v.slots[s])
                e.if_(e.valid(v.ty.numel, s), write)
        e.barrier()

        def read(s):
            idx = index_fn(e.index(s))
            vals = []
            for (arr, ty, _), i in zip(arrays, idx):
                safe = i if out_ty.numel >= e.threads else e.sel(e.valid(out_ty.numel, s), i, const(0, "i32"))
                vals.append(e.shld(arr, ty, safe))
            return combine(e.index(s), vals)

        return self.make(out_ty, read)

    # Lowering driver ---------------------------------------------------------------------
    def run(self) -> dict:
        for name, ty in self.func.args:
            if ty.elem == "ptr" and not ty.is_tensor:
                self.env[name] = Val(ty, scalar={"p": self.num_ptrs})
                self.arg_kinds.append(("ptr", ty.pointee))
                self.num_ptrs += 1
            elif not ty.is_tensor:
                self.env[name] = Val(ty, scalar={"s": len(self.scalar_types)})
                self.scalar_types.append(ty.elem)
                self.arg_kinds.append(("scalar", ty.elem))
            else:
                raise LoweringError(f"tensor kernel argument {name}")
        # Scalar arguments live in the root scope; give them registers so the IR types match.
        for name, val in list(self.env.items()):
            if "s" in val.scalar:
                local = self.e.local(val.ty.elem)
                self.e.mov(local, {"s": val.scalar["s"]})
                self.env[name] = Val(val.ty, scalar=local)
            elif "p" in val.scalar:
                local = self.e.local("u64")
                self.e.mov(local, val.scalar)
                self.env[name] = Val(val.ty, scalar=local)
        self.block(self.func.body)
        return {
            "name": self.func.name,
            "threads": self.e.threads,
            "num_ptrs": self.num_ptrs,
            "scalars": self.scalar_types,
            "shared": self.e.shared,
            "locals": self.e.locals,
            "body": self.e.stack[0],
        }

    def block(self, block: Block, args: list[Val] | None = None) -> list[Val] | None:
        for (name, _), v in zip(block.args, args or []):
            self.env[name] = v
        # Every lowering of a region is its own basic block for FMA contraction purposes.
        self._next_block = getattr(self, "_next_block", 0) + 1
        self.blocks.append(self._next_block)
        try:
            for op in block.ops:
                if op.name in ("scf.yield", "scf.condition", "tt.reduce.return", "tt.scan.return"):
                    return [self.get(n) for n in op.operands]
                if op.name == "tt.return":
                    return None
                self.op(op)
                self.assign_rank(op)
                if op.name == "arith.mulf":
                    self.muls[op.results[0]] = (op, self.blocks[-1])
                elif op.name == "arith.negf":
                    self.negs[op.results[0]] = op.operands[0]
            return None
        finally:
            self.blocks.pop()

    def bind(self, op: Op, vals: list[Val]):
        for name, v in zip(op.results, vals):
            self.env[name] = v

    def op(self, op: Op):
        handler = getattr(self, "op_" + op.name.replace(".", "_"), None)
        if handler is not None:
            handler(op)
            return
        dialect, _, name = op.name.partition(".")
        if dialect in ("arith", "math") and name in ARITH:
            self.arith(op, name)
            return
        raise LoweringError(f"unsupported TTIR operation {op.name}")

    # Constants, ranges, ids --------------------------------------------------------------
    def op_arith_constant(self, op: Op):
        (ty,) = op.result_types() or [Type(None, "i1")]
        text = op.body.strip()
        if text.startswith("dense<"):
            inner = text[len("dense<"):text.rindex(">")]
            if inner.startswith("["):
                raise LoweringError("non-splat dense constants")
            value = inner
        else:
            value = text
        value = value.strip()
        if ty.elem == "i1":
            v = value in ("true", "1")
        elif ty.elem in FLOATS:
            v = _bits_to_float(int(value, 16), ty.elem) if value.lower().startswith("0x") else float(value)
        else:
            v = int(value)
        self.bind(op, [self.uniform(ty, const(v, ty.elem))])

    def op_tt_make_range(self, op: Op):
        (ty,) = op.result_types()
        start = int(op.attr("start"))
        e = self.e
        gen = (lambda i: i) if start == 0 else (lambda i: e.bin("add", i, const(start, "i32")))
        self.bind(op, [self.make(ty, lambda s: gen(e.index(s)), gen)])

    def op_tt_get_program_id(self, op: Op):
        axis = op.body.strip().split()[0]
        builtin = {"x": "pidx", "y": "pidy", "z": "pidz"}[axis]
        self.bind(op, [Val(Type(None, "i32"), scalar=self.e.cast({"b": builtin}, "i32"))])

    def op_tt_get_num_programs(self, op: Op):
        axis = op.body.strip().split()[0]
        builtin = {"x": "npx", "y": "npy", "z": "npz"}[axis]
        self.bind(op, [Val(Type(None, "i32"), scalar=self.e.cast({"b": builtin}, "i32"))])

    def op_tt_splat(self, op: Op):
        (ty,) = op.result_types()
        src = self.get(op.operands[0])
        self.bind(op, [self.uniform(ty, src.scalar)])

    def op_tt_assert(self, op: Op):
        pass

    def op_tt_print(self, op: Op):
        pass

    # Shape ops ---------------------------------------------------------------------------
    def _same_layout(self, op: Op):
        (ty,) = op.result_types()
        src = self.get(op.operands[0])
        if not src.is_tensor:
            self.bind(op, [self.uniform(ty, src.scalar)])
            return
        self.bind(op, [Val(ty, slots=src.slots, uniform=src.uniform, gen=src.gen)])

    op_tt_expand_dims = _same_layout

    def op_tt_reshape(self, op: Op):
        if "allow_reorder" in op.body and "efficient_layout" not in op.body:
            # Reordering is permitted but not required; keeping row-major order is valid.
            pass
        self._same_layout(op)

    def op_tt_broadcast(self, op: Op):
        (ty,) = op.result_types()
        src = self.get(op.operands[0])
        if not src.is_tensor or src.uniform is not None:
            self.bind(op, [self.uniform(ty, src.scalar if not src.is_tensor else src.uniform)])
            return
        if src.ty.shape == ty.shape:
            self.bind(op, [src])
            return

        def index(e):
            cs = self.coords(e, ty.shape)
            cs = [c if sd != 1 else const(0, "i32") for c, sd in zip(cs, src.ty.shape)]
            return [self.flatten(cs, src.ty.shape)]

        self.bind(op, [self.gather([src], ty, index, lambda e, v: v[0])])

    def op_tt_trans(self, op: Op):
        (ty,) = op.result_types()
        src = self.get(op.operands[0])
        order = [int(x) for x in op.attr("order").strip("array<i32: >").split(",")] if op.attr("order") else [1, 0]
        if src.uniform is not None:
            self.bind(op, [self.uniform(ty, src.uniform)])
            return

        def index(e):
            cs = self.coords(e, ty.shape)
            src_cs = [None] * len(order)
            for out_dim, in_dim in enumerate(order):
                src_cs[in_dim] = cs[out_dim]
            return [self.flatten(src_cs, src.ty.shape)]

        self.bind(op, [self.gather([src], ty, index, lambda e, v: v[0])])

    def op_tt_join(self, op: Op):
        (ty,) = op.result_types()
        a, b = (self.get(n) for n in op.operands)
        e = self.e

        def index(i):
            half = e.bin("div", i, const(2, "i32"))
            return [half, half]

        def combine(i, vals):
            odd = e.cmp("ne", e.bin("rem", i, const(2, "i32")), const(0, "i32"))
            return e.sel(odd, vals[1], vals[0])

        self.bind(op, [self.gather([a, b], ty, index, combine)])

    def op_tt_split(self, op: Op):
        tys = op.result_types()
        src = self.get(op.operands[0])
        e = self.e
        outs = []
        for part in (0, 1):
            index = lambda i, part=part: [e.bin("add", e.bin("mul", i, const(2, "i32")), const(part, "i32"))]
            outs.append(self.gather([src], tys[part], index, lambda i, v: v[0]))
        self.bind(op, outs)

    # Memory ------------------------------------------------------------------------------
    def op_tt_addptr(self, op: Op):
        ptr_name, off_name = op.operands[:2]
        ptr, off = self.get(ptr_name), self.get(off_name)
        ty = ptr.ty
        size = BYTES[ty.pointee]
        e = self.e

        def fn(xs):
            p, o = xs
            if e.gx:
                out = e.local(e.ty_of(p))
                e.emit({"k": "ptradd", "out": out["l"], "p": p, "o": o, "scale": size})
                return out
            o64 = e.cast(o, "i64")
            if size != 1:
                o64 = e.bin("mul", o64, const(size, "i64"))
            return e.bin("add", p, e.bitcast(o64, "u64"))

        self.bind(op, [self.elementwise(ty, [ptr, off], fn)])

    def op_tt_load(self, op: Op):
        names = op.operands
        ptr = self.get(names[0])
        mask = self.get(names[1]) if len(names) > 1 else None
        other = self.get(names[2]) if len(names) > 2 else None
        out_ty = ptr.ty.with_elem(ptr.ty.pointee)
        elem = out_ty.elem
        e = self.e

        def load(p):
            out = e.local(elem)
            e.emit({"k": "ld", "out": out["l"], "ptr": p})
            return out

        if not ptr.is_tensor:
            self.bind(op, [Val(out_ty, scalar=load(ptr.scalar))])
            return

        def per_slot(s):
            cond = e.valid(out_ty.numel, s)
            if mask is not None:
                m = self.slot(mask, s)
                cond = m if "c" in cond and cond["c"] == "1" else e.bin("and", cond, m)
            if "c" in cond and cond["c"] == "1":
                return load(self.slot(ptr, s))
            fallback = self.slot(other, s) if other is not None else const(0, elem)
            out = e.local(elem)
            e.mov(out, fallback)
            e.if_(cond, lambda: e.emit({"k": "ld", "out": out["l"], "ptr": self.slot(ptr, s)}))
            return out

        self.bind(op, [self.make(out_ty, per_slot)])

    def op_tt_store(self, op: Op):
        names = op.operands
        ptr, val = self.get(names[0]), self.get(names[1])
        mask = self.get(names[2]) if len(names) > 2 else None
        e = self.e
        if not ptr.is_tensor:
            first = e.cmp("eq", {"b": "tid"}, const(0, "u32"))
            cond = first if mask is None else e.bin("and", first, mask.scalar)
            e.if_(cond, lambda: e.emit({"k": "st", "ptr": ptr.scalar, "val": val.scalar}))
            return
        for s in range(self.nslots(ptr.ty)):
            cond = e.valid(ptr.ty.numel, s)
            if mask is not None:
                m = self.slot(mask, s)
                cond = m if "c" in cond and cond["c"] == "1" else e.bin("and", cond, m)
            e.if_(cond, lambda s=s: e.emit({"k": "st", "ptr": self.slot(ptr, s), "val": self.slot(val, s)}))

    # Reductions --------------------------------------------------------------------------
    def op_tt_reduce(self, op: Op):
        inputs = [self.materialize(self.get(n)) for n in op.operands]
        axis = int(op.attr("axis"))
        region = op.regions[0]
        out_tys = op.result_types()
        e = self.e
        shape = inputs[0].ty.shape
        R = shape[axis]
        inner = 1
        for d in shape[axis + 1:]:
            inner *= d
        P_in = inputs[0].ty.numel
        P_out = P_in // R
        N = e.threads
        tys = [simt_type(v.ty) for v in inputs]

        def combine(acc: list[dict], new: list[dict]) -> list[dict]:
            args = [Val(v.ty.scalar(), scalar=x) for v, x in zip(inputs * 2, acc + new)]
            res = self.block(region, args)
            return [r.scalar for r in res]

        def src_index(o: dict, r: dict) -> dict:
            if inner == 1:
                base = o if R == 1 else e.bin("mul", o, const(R, "i32"))
                return e.bin("add", base, r) if not ("c" in r and r["c"] == "0") else base
            outer = e.bin("div", o, const(inner, "i32"))
            in_ = e.bin("rem", o, const(inner, "i32"))
            idx = e.bin("add", e.bin("mul", outer, const(R * inner, "i32")), in_)
            if "c" in r and r["c"] == "0":
                return idx
            return e.bin("add", idx, e.bin("mul", r, const(inner, "i32")))

        layout = None
        if self.reduce_layouts is not None and self.reduce_count < len(self.reduce_layouts):
            layout = self.reduce_layouts[self.reduce_count]
            if layout is not None and tuple(layout["shape"]) != tuple(shape):
                layout = None
        self.reduce_count += 1
        factors = self.reduce_mul_factors(op) if layout is not None else None
        staged = inputs
        if factors is not None and 2 <= len(official_registers(layout, axis, R)) <= 256 and all(
                self.get(n).is_tensor and tuple(self.get(n).ty.shape) == tuple(shape) for n in factors):
            # LLVM contracts the producing fmul into the first adds of the per-thread tree.
            staged = [self.materialize(self.get(n)) for n in factors]
        else:
            factors = None

        # Stage inputs in shared memory.
        e.barrier()
        arrays = []
        for i, v in enumerate(staged):
            arr = e.shared_array(simt_type(v.ty), P_in, i)
            arrays.append(arr)
            for s in range(self.nslots(v.ty)):
                e.if_(e.valid(P_in, s), lambda arr=arr, s=s, v=v: e.shst(arr, e.index(s), v.slots[s]))
        e.barrier()
        shload = e.shld

        if layout is not None:
            results = self.reduce_official(layout, axis, R, P_out, arrays, tys, combine, src_index,
                                           fused=factors is not None)
            if results is not None:
                outs = [Val(ty, slots=results[i]) if ty.is_tensor else Val(ty, scalar=results[i][0])
                        for i, ty in enumerate(out_tys)]
                self.bind(op, outs)
                return

        def strided_reduce(o: dict, first: dict, stride: int) -> list[dict]:
            """acc = reduce over r = first, first+stride, ... < R."""
            acc = [e.copy(shload(a, t, src_index(o, first))) for a, t in zip(arrays, tys)]
            if R <= stride:
                return acc
            r = e.copy(first)
            e.block()
            e.mov(r, e.bin("add", r, const(stride, "i32")))
            done = e.cmp("ge", r, const(R, "i32"))
            e.if_(done, lambda: e.emit({"k": "brk"}))
            new = [shload(a, t, src_index(o, r)) for a, t in zip(arrays, tys)]
            res = combine(acc, new)
            for a_, r_ in zip(acc, res):
                e.mov(a_, r_)
            body = e.end()
            e.emit({"k": "loop", "body": body})
            return acc

        results: list[list[dict]] = [[] for _ in inputs]
        if P_out >= N:
            for s in range(e.slots_for(P_out)):
                acc = strided_reduce(e.index(s), const(0, "i32"), 1)
                for i, a in enumerate(acc):
                    results[i].append(a)
        else:
            G = min(N // P_out, R)
            tid = e.cast({"b": "tid"}, "i32")
            o = e.bin("rem", tid, const(P_out, "i32")) if P_out > 1 else const(0, "i32")
            g = e.bin("div", tid, const(P_out, "i32")) if P_out > 1 else tid
            active = e.cmp("lt", tid, const(G * P_out, "i32"))
            parts = [e.shared_array(t, G * P_out, len(inputs) + i) for i, t in enumerate(tys)]
            def partial():
                acc = strided_reduce(o, g, G)
                slot = e.bin("add", e.bin("mul", g, const(P_out, "i32")), o) if P_out > 1 else g
                for p, a in zip(parts, acc):
                    e.shst(p, slot, a)

            e.if_(active, partial)
            e.barrier()
            stride = G // 2
            while stride >= 1:
                def tree(stride=stride):
                    lo = e.bin("add", e.bin("mul", g, const(P_out, "i32")), o) if P_out > 1 else g
                    hi = e.bin("add", lo, const(stride * P_out, "i32"))
                    a = [shload(p, t, lo) for p, t in zip(parts, tys)]
                    b = [shload(p, t, hi) for p, t in zip(parts, tys)]
                    res = combine(a, b)
                    for p, r_ in zip(parts, res):
                        e.shst(p, lo, r_)
                e.if_(e.cmp("lt", tid, const(stride * P_out, "i32")), tree)
                e.barrier()
                stride //= 2
            for i, (p, t) in enumerate(zip(parts, tys)):
                results[i].append(shload(p, t, o))
            e.barrier()

        outs = []
        for i, ty in enumerate(out_tys):
            if ty.is_tensor:
                outs.append(Val(ty, slots=results[i]))
            else:
                outs.append(Val(ty, scalar=results[i][0]))
        self.bind(op, outs)

    def reduce_mul_factors(self, op: Op) -> list[str] | None:
        """Factors (x, y) when a float add-reduce consumes ``x * y`` from the same block."""
        if not self.fp_fusion or len(op.operands) != 1:
            return None
        region = op.regions[0]
        body = [o for o in region.ops if o.name != "tt.reduce.return"]
        args = {a for a, _ in region.args}
        if len(body) != 1 or body[0].name != "arith.addf" or set(body[0].operands) != args or len(args) != 2:
            return None
        found = self.muls.get(op.operands[0])
        if not found or found[1] != self.blocks[-1]:
            return None
        x, y = found[0].operands[:2]
        if x in self.negs or y in self.negs:
            return None
        return [x, y]

    def reduce_official(self, layout: dict, axis: int, R: int, P_out: int, arrays: list[int], tys: list[str],
                        combine: Callable, src_index: Callable, fused: bool = False) -> list[list[dict]] | None:
        """Reduce staged inputs in the order of Triton's NVIDIA lowering for a blocked layout: each
        thread tree-reduces its registers pairwise, then lanes and then warps combine with xor
        butterflies (lane/warp ``i`` takes ``combine(own, partner i + k)`` for k = n/2 .. 1).
        ``fused``: ``arrays`` hold the factors of a product; tree adds of a product become FMAs."""
        e = self.e
        s, t, w = layout["spt"][axis], layout["tpw"][axis], layout["wpc"][axis]
        t_u = min(t, max(1, -(-R // s)))
        w_u = min(w, max(1, -(-R // (s * t))))
        tile = s * t * w
        regs = official_registers(layout, axis, R)
        if len(regs) > 256:
            return None
        V = t_u * w_u
        N = e.threads
        n = len(arrays)
        tys = tys[:1] if fused else tys
        parts = [e.shared_array(ty, P_out * V, n + i) for i, ty in enumerate(tys)]
        tid = e.cast({"b": "tid"}, "i32")

        def tree_combine(a, b):
            if not fused:
                return combine(a, b)
            if a[0] == "mul":
                return ("val", _fma(e, [a[1], a[2], e.bin("mul", b[1], b[2]) if b[0] == "mul" else b[1]]))
            if b[0] == "mul":
                return ("val", _fma(e, [b[1], b[2], a[1]]))
            return ("val", e.bin("add", a[1], b[1]))

        def per_thread(item):
            o = e.bin("div", item, const(V, "i32")) if V > 1 else item
            v = e.bin("rem", item, const(V, "i32")) if V > 1 else const(0, "i32")
            wi = e.bin("div", v, const(t_u, "i32")) if t_u > 1 else v
            li = e.bin("rem", v, const(t_u, "i32")) if t_u > 1 else const(0, "i32")
            base = e.bin("add", e.bin("mul", wi, const(s * t, "i32")), e.bin("mul", li, const(s, "i32")))
            vals = []
            for rep, j in regs:
                pos = e.bin("add", base, const(rep * tile + j, "i32"))
                idx = src_index(o, pos)
                if fused:
                    vals.append(("mul", e.shld(arrays[0], tys[0], idx), e.shld(arrays[1], tys[0], idx)))
                else:
                    vals.append([e.shld(a, ty, idx) for a, ty in zip(arrays, tys)])
            while len(vals) > 1:
                nxt = [tree_combine(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
                if len(vals) % 2:
                    nxt.append(vals[-1])
                vals = nxt
            for p, x in zip(parts, [vals[0][1]] if fused else vals[0]):
                e.shst(p, item, x)

        self.counted_loop(tid, const(P_out * V, "i32"), per_thread, N)
        e.barrier()

        def butterfly(groups: int, k: int, stride: int):
            def level(item):
                o = e.bin("div", item, const(groups * k, "i32"))
                rem = e.bin("rem", item, const(groups * k, "i32"))
                g = e.bin("div", rem, const(k, "i32"))
                i = e.bin("rem", rem, const(k, "i32"))
                lo = e.bin("add", e.bin("mul", o, const(V, "i32")),
                           e.bin("add", e.bin("mul", g, const(t_u, "i32")), e.bin("mul", i, const(stride, "i32"))))
                hi = e.bin("add", lo, const(k * stride, "i32"))
                res = combine([e.shld(p, ty, lo) for p, ty in zip(parts, tys)],
                              [e.shld(p, ty, hi) for p, ty in zip(parts, tys)])
                for p, r in zip(parts, res):
                    e.shst(p, lo, r)

            self.counted_loop(tid, const(P_out * groups * k, "i32"), level, N)
            e.barrier()

        k = t_u // 2
        while k >= 1:
            butterfly(w_u, k, 1)
            k //= 2
        k = w_u // 2
        while k >= 1:
            butterfly(1, k, t_u)
            k //= 2
        results = [[] for _ in parts]
        for slot in range(e.slots_for(P_out)):
            o = self.safe_index(P_out, slot) if P_out > 1 else const(0, "i32")
            idx = e.bin("mul", o, const(V, "i32")) if V > 1 else o
            for i, (p, ty) in enumerate(zip(parts, tys)):
                results[i].append(e.shld(p, ty, idx))
        e.barrier()
        return results

    def counted_loop(self, start: dict, end: dict, body: Callable[[dict], None], step: int = 1):
        """Runtime loop ``for i in range(start, end, step): body(i)`` over i32 values."""
        e = self.e
        i = e.copy(start)
        e.block()
        e.if_(e.cmp("ge", i, end), lambda: e.emit({"k": "brk"}))
        body(i)
        e.mov(i, e.bin("add", i, const(step, "i32")))
        e.emit({"k": "loop", "body": e.end()})

    def stage(self, vals: list[Val], first_slot: int) -> list[tuple[int, str]]:
        """Write every element of each tensor to its own shared array, indexed by element."""
        e = self.e
        arrays = []
        e.barrier()
        for i, v in enumerate(vals):
            v = self.materialize(v)
            ty = simt_type(v.ty)
            arr = e.shared_array(ty, v.ty.numel, first_slot + i)
            arrays.append((arr, ty))
            for s in range(self.nslots(v.ty)):
                e.if_(e.valid(v.ty.numel, s), lambda arr=arr, s=s, v=v: e.shst(arr, e.index(s), v.slots[s]))
        e.barrier()
        return arrays

    def safe_index(self, numel: int, s: int) -> dict:
        e = self.e
        if numel >= e.threads:
            return e.index(s)
        return e.sel(e.valid(numel, s), e.index(s), const(0, "i32"))

    # Matrix multiply ---------------------------------------------------------------------
    def op_tt_dot(self, op: Op):
        a, b, c = (self.get(n) for n in op.operands[:3])
        (out_ty,) = op.result_types()
        e = self.e
        shape = out_ty.shape
        M, N = shape[-2], shape[-1]
        K = a.ty.shape[-1]
        acc_ty = simt_type(out_ty)
        precision = (op.attr("inputPrecision") or "ieee").strip()
        tf32 = precision == "tf32" and a.ty.elem == "f32" and b.ty.elem == "f32"
        integer = acc_ty not in FLOATS
        (sa, ta), (sb, tb) = self.stage([a, b], 0)

        def operand(x: dict) -> dict:
            if tf32:
                bits = e.bitcast(x, "u32")
                bits = e.bin("and", e.bin("add", bits, const(0x1000, "u32")), const(0xFFFFE000, "u32"))
                x = e.bitcast(bits, "f32")
            return e.cast(x, acc_ty)

        def per_slot(s):
            idx = self.safe_index(out_ty.numel, s)
            row = e.bin("div", idx, const(N, "i32"))
            col = e.bin("rem", idx, const(N, "i32"))
            a_base = e.bin("mul", row, const(K, "i32"))
            b_base = col
            if len(shape) == 3:
                batch = e.bin("div", idx, const(M * N, "i32"))
                b_base = e.bin("add", e.bin("mul", batch, const(K * N, "i32")), col)
            acc = e.copy(e.cast(self.slot(self.materialize(c), s), acc_ty))

            def step(k):
                x = operand(e.shld(sa, ta, e.bin("add", a_base, k)))
                y = operand(e.shld(sb, tb, e.bin("add", b_base, e.bin("mul", k, const(N, "i32")))))
                e.mov(acc, e.bin("add", e.bin("mul", x, y), acc) if integer else _fma(e, [x, y, acc]))

            self.counted_loop(const(0, "i32"), const(K, "i32"), step)
            return acc

        self.bind(op, [self.make(out_ty, per_slot)])

    # Atomics -----------------------------------------------------------------------------
    ATOMIC_RMW = {"add": ("add", False), "fadd": ("add", False), "and": ("and", False), "or": ("or", False),
                  "xor": ("xor", False), "max": ("max", False), "min": ("min", False), "umax": ("max", True),
                  "umin": ("min", True), "exch": ("xchg", False), "xchg": ("xchg", False)}

    def _atomic(self, op: Op, out_ty: Type, ptr: Val, values: list[Val], mask: Val | None,
                emit: Callable[[dict, list[dict], str], dict], unsigned: bool = False):
        """Per-element atomic; a scalar atomic runs once per program and its result is broadcast."""
        e = self.e
        elem = simt_type(out_ty)
        aty = UNSIGNED.get(elem, elem) if unsigned else elem

        def run(p, xs):
            address = e.bitcast(p, "atomic_" + aty)
            return e.bitcast(emit(address, [e.bitcast(x, aty) for x in xs], aty), elem)

        if not ptr.is_tensor:
            out = e.local(elem)
            e.mov(out, const(0, elem))
            first = e.cmp("eq", {"b": "tid"}, const(0, "u32"))
            cond = first if mask is None else e.bin("and", first, mask.scalar)
            arr = e.shared_array(elem, 1, 0)
            e.barrier()
            e.if_(cond, lambda: (e.mov(out, run(ptr.scalar, [v.scalar for v in values])),
                                 e.shst(arr, const(0, "i32"), out)))
            e.barrier()
            result = e.shld(arr, elem, const(0, "i32"))
            if mask is not None:
                result = e.sel(mask.scalar, result, const(0, elem))
            self.bind(op, [Val(out_ty, scalar=result)])
            return

        def per_slot(s):
            cond = e.valid(out_ty.numel, s)
            if mask is not None:
                m = self.slot(mask, s)
                cond = m if "c" in cond and cond["c"] == "1" else e.bin("and", cond, m)
            out = e.local(elem)
            e.mov(out, const(0, elem))
            e.if_(cond, lambda: e.mov(out, run(self.slot(ptr, s), [self.slot(v, s) for v in values])))
            return out

        self.bind(op, [self.make(out_ty, per_slot)])

    def op_tt_atomic_rmw(self, op: Op):
        kind = op.body.strip().split(",")[0].strip()
        if kind not in self.ATOMIC_RMW:
            raise LoweringError(f"unsupported atomic_rmw kind {kind}")
        name, unsigned = self.ATOMIC_RMW[kind]
        names = op.operands
        ptr, val = self.get(names[0]), self.get(names[1])
        mask = self.get(names[2]) if len(names) > 2 else None
        (out_ty,) = op.result_types()
        e = self.e

        def emit(address, xs, aty):
            out = e.local(aty)
            e.emit({"k": "atom", "op": name, "out": out["l"], "ptr": address, "val": xs[0]})
            return out

        self._atomic(op, out_ty, ptr, [val], mask, emit, unsigned)

    def op_tt_atomic_cas(self, op: Op):
        ptr, cmp, val = (self.get(n) for n in op.operands[:3])
        (out_ty,) = op.result_types()
        e = self.e

        def emit(address, xs, aty):
            out = e.local(aty)
            e.emit({"k": "cas", "out": out["l"], "ptr": address, "cmp": xs[0], "val": xs[1]})
            return out

        self._atomic(op, out_ty, ptr, [cmp, val], None, emit)

    # Scan --------------------------------------------------------------------------------
    def op_tt_scan(self, op: Op):
        """Chunked scan: each line is split into G chunks scanned sequentially, then every chunk
        is offset by the combined totals of the chunks before it."""
        inputs = [self.materialize(self.get(n)) for n in op.operands]
        axis = int(op.attr("axis"))
        reverse = (op.attr("reverse") or "false").strip() == "true"
        region = op.regions[0]
        out_tys = op.result_types()
        e = self.e
        shape = inputs[0].ty.shape
        R = shape[axis]
        inner = 1
        for d in shape[axis + 1:]:
            inner *= d
        P = inputs[0].ty.numel
        L = P // R
        T = e.threads
        G = max(1, min(T // L, R)) if L < T else 1
        C = -(-R // G)
        G = -(-R // C)
        n = len(inputs)
        src = self.stage(inputs, 0)
        tys = [t for _, t in src]
        outs = [e.shared_array(t, P, n + i) for i, t in enumerate(tys)]
        totals = [e.shared_array(t, L * G, 2 * n + i) for i, t in enumerate(tys)]

        def combine(acc, new):
            args = [Val(v.ty.scalar(), scalar=x) for v, x in zip(inputs * 2, acc + new)]
            return [r.scalar for r in self.block(region, args)]

        def index(line, r):
            pos = e.bin("sub", const(R - 1, "i32"), r) if reverse else r
            if inner == 1:
                return e.bin("add", e.bin("mul", line, const(R, "i32")), pos)
            outer = e.bin("div", line, const(inner, "i32"))
            within = e.bin("rem", line, const(inner, "i32"))
            base = e.bin("add", e.bin("mul", outer, const(R * inner, "i32")), within)
            return e.bin("add", base, e.bin("mul", pos, const(inner, "i32")))

        def chunk(w):
            line = e.bin("div", w, const(G, "i32")) if G > 1 else w
            g = e.bin("rem", w, const(G, "i32")) if G > 1 else const(0, "i32")
            r0 = e.bin("mul", g, const(C, "i32"))
            r1 = e.bin("min", e.bin("add", r0, const(C, "i32")), const(R, "i32"))
            return line, g, r0, r1

        def phase1(w):
            line, _, r0, r1 = chunk(w)
            i0 = index(line, r0)
            acc = [e.copy(e.shld(a, t, i0)) for (a, t) in src]
            for o, x in zip(outs, acc):
                e.shst(o, i0, x)

            def step(r):
                i = index(line, r)
                res = combine(acc, [e.shld(a, t, i) for (a, t) in src])
                for a_, r_, o in zip(acc, res, outs):
                    e.mov(a_, r_)
                    e.shst(o, i, a_)

            self.counted_loop(e.bin("add", r0, const(1, "i32")), r1, step)
            for tot, x in zip(totals, acc):
                e.shst(tot, w, x)

        tid = e.cast({"b": "tid"}, "i32")
        self.counted_loop(tid, const(L * G, "i32"), phase1, T)
        e.barrier()
        if G > 1:
            def phase2(w):
                line, g, r0, r1 = chunk(w)

                def offset():
                    first = e.bin("mul", line, const(G, "i32"))
                    prefix = [e.copy(e.shld(tot, t, first)) for tot, t in zip(totals, tys)]

                    def more(j):
                        res = combine(prefix, [e.shld(tot, t, e.bin("add", first, j)) for tot, t in zip(totals, tys)])
                        for p_, r_ in zip(prefix, res):
                            e.mov(p_, r_)

                    self.counted_loop(const(1, "i32"), g, more)

                    def apply(r):
                        i = index(line, r)
                        res = combine(prefix, [e.shld(o, t, i) for o, t in zip(outs, tys)])
                        for o, r_ in zip(outs, res):
                            e.shst(o, i, r_)

                    self.counted_loop(r0, r1, apply)

                e.if_(e.cmp("gt", g, const(0, "i32")), offset)

            self.counted_loop(tid, const(L * G, "i32"), phase2, T)
            e.barrier()
        results = []
        for (o, t), ty in zip(zip(outs, tys), out_tys):
            results.append(self.make(ty, lambda s, o=o, t=t, ty=ty: e.shld(o, t, self.safe_index(ty.numel, s))))
        e.barrier()
        self.bind(op, results)

    # Control flow ------------------------------------------------------------------------
    def op_scf_while(self, op: Op):
        import re
        m = re.match(r"\((.*)\)", op.body.strip())
        pairs = [tuple(x.strip() for x in p.split("=")) for p in split_top(m.group(1))] if m and m.group(1) else []
        out_tys = op.result_types()
        e = self.e

        def fresh(ty: Type) -> Val:
            if ty.is_tensor:
                return Val(ty, slots=[e.local(simt_type(ty)) for _ in range(self.nslots(ty))])
            return Val(ty, scalar=e.local(simt_type(ty)))

        def assign(dst: Val, src: Val):
            if dst.is_tensor:
                for s, d in enumerate(dst.slots):
                    e.mov(d, self.slot(src, s))
            else:
                e.mov(dst.scalar, src.scalar)

        carriers = []
        for _, init in pairs:
            v = self.get(init)
            c = fresh(v.ty)
            assign(c, v)
            carriers.append(c)
        results = [fresh(ty) for ty in out_tys]
        before, after = op.regions[0], op.regions[1]
        e.block()
        self.new_bb()
        names = [n for n, _ in before.args] or [p[0] for p in pairs]
        for name, c in zip(names, carriers):
            self.rank[name] = self.unmovable_rank()
            self.env[name] = c
        cond_vals = self.block(before) or []
        cond, vals = cond_vals[0], cond_vals[1:]
        snapshot = [Val(v.ty, slots=[e.copy(self.slot(v, s)) for s in range(self.nslots(v.ty))]) if v.is_tensor
                    else Val(v.ty, scalar=e.copy(v.scalar)) for v in vals]
        for r, v in zip(results, snapshot):
            assign(r, v)
        e.if_(e.un("not", cond.scalar), lambda: e.emit({"k": "brk"}))
        yielded = self.block(after, snapshot) or []
        temps = [Val(y.ty, slots=[e.copy(self.slot(y, s)) for s in range(self.nslots(y.ty))]) if y.is_tensor
                 else Val(y.ty, scalar=e.copy(y.scalar)) for y in yielded]
        for c, t in zip(carriers, temps):
            assign(c, t)
        e.emit({"k": "loop", "body": e.end()})
        self.new_bb()
        for name in op.results:
            self.rank[name] = self.unmovable_rank()
        self.bind(op, results)

    def op_scf_for(self, op: Op):
        import re
        m = re.match(r"(%\S+)\s*=\s*(%\S+)\s+to\s+(%\S+)\s+step\s+(%\S+)(?:\s+iter_args\((.*?)\)\s*(?:->|$))?", op.body.strip())
        if not m:
            raise LoweringError(f"cannot parse scf.for: {op.body}")
        iv_name, lb, ub, step, iters = m.groups()
        iv_ty = parse_type(op.tail.strip()) if op.tail else Type(None, "i32")
        pairs = [tuple(x.strip() for x in p.split("=")) for p in split_top(iters)] if iters else []
        e = self.e
        carriers = []
        for _, init in pairs:
            v = self.materialize(self.get(init))
            if v.is_tensor:
                cs = [e.copy(x) for x in v.slots]
                carriers.append(Val(v.ty, slots=cs))
            else:
                carriers.append(Val(v.ty, scalar=e.copy(v.scalar)))
        iv = e.copy(self.get(lb).scalar)
        ubv, stepv = self.get(ub).scalar, self.get(step).scalar
        e.block()
        done = e.cmp("ge", iv, ubv)
        e.if_(done, lambda: e.emit({"k": "brk"}))
        self.env[iv_name] = Val(iv_ty, scalar=iv)
        args = [Val(c.ty, slots=list(c.slots) if c.is_tensor else None, scalar=c.scalar) for c in carriers]
        for (name, _), a in zip(pairs, args):
            self.env[name] = a
        self.new_bb()
        for name in [iv_name] + [name for name, _ in pairs]:
            self.rank[name] = self.unmovable_rank()
        yielded = self.block(op.regions[0]) or []
        self.new_bb()
        for name in op.results:
            self.rank[name] = self.unmovable_rank()
        temps = []
        for c, y in zip(carriers, yielded):
            if c.is_tensor:
                temps.append([e.copy(self.slot(y, s)) for s in range(len(c.slots))])
            else:
                temps.append([e.copy(y.scalar)])
        for c, t in zip(carriers, temps):
            dst = c.slots if c.is_tensor else [c.scalar]
            for d, x in zip(dst, t):
                e.mov(d, x)
        e.mov(iv, e.bin("add", iv, stepv))
        body = e.end()
        e.emit({"k": "loop", "body": body})
        self.bind(op, carriers)

    def op_scf_if(self, op: Op):
        cond = self.get(op.operands[0]).scalar
        e = self.e
        out_tys = []
        if "->" in op.body:
            text = op.body.split("->", 1)[1].strip().strip("()")
            out_tys = [parse_type(t) for t in split_top(text)]
        outs = []
        for ty in out_tys:
            if ty.is_tensor:
                outs.append(Val(ty, slots=[e.local(simt_type(ty)) for _ in range(self.nslots(ty))]))
            else:
                outs.append(Val(ty, scalar=e.local(simt_type(ty))))

        def arm(region):
            def run():
                self.new_bb()
                ys = self.block(region) or []
                for o, y in zip(outs, ys):
                    if o.is_tensor:
                        for s, d in enumerate(o.slots):
                            e.mov(d, self.slot(y, s))
                    else:
                        e.mov(o.scalar, y.scalar)
            return run

        e.if_(cond, arm(op.regions[0]), arm(op.regions[1]) if len(op.regions) > 1 else None)
        self.new_bb()
        for name in op.results:
            self.rank[name] = self.unmovable_rank()
        self.bind(op, outs)

    # Arithmetic --------------------------------------------------------------------------
    def contract(self, op: Op, name: str) -> bool:
        """fadd/fsub of an fmul from the same block becomes one FMA (LLVM aggressive contraction)."""
        a_name, b_name = op.operands[:2]

        def mul(n):
            """(x, y, negated) for an fmul in this block, folding fmul(x, fneg y) = fneg(fmul(x, y))."""
            found = self.muls.get(n)
            if not found or found[1] != self.blocks[-1]:
                return None
            x, y = found[0].operands[:2]
            negated = False
            if x in self.negs:
                x, negated = self.negs[x], not negated
            if y in self.negs:
                y, negated = self.negs[y], not negated
            return x, y, negated

        ma, mb = mul(a_name), mul(b_name)
        if ma is None and mb is None:
            return False
        # InstCombine first rewrites a - (-m) to a + m and a + (-m) to a - m.
        if mb is not None and mb[2]:
            name = "subf" if name == "addf" else "addf"
            mb = (mb[0], mb[1], False)
        elif ma is not None and ma[2] and name == "addf":
            a_name, b_name, ma, mb, name = b_name, a_name, mb, (ma[0], ma[1], False), "subf"
        if ma is not None and ma[2]:
            ma = None
        if ma is None and mb is None:
            return False
        if ma is not None and mb is not None:
            # Reassociate puts the lower-ranked operand of a commutative op first; the DAG combiner
            # then folds the multiply with fewer uses, preferring the first operand on a tie.
            if name == "addf" and self.rank.get(b_name, 0) < self.rank.get(a_name, 0):
                a_name, b_name, ma, mb = b_name, a_name, mb, ma
            if self.uses.get(a_name, 0) > self.uses.get(b_name, 0):
                ma = None
        (out_ty,) = op.result_types()
        e = self.e

        def other(n, info):
            """The non-fused operand; a negated multiply is materialized without its negation."""
            if info is None:
                return self.get(n), False
            x, y = self.get(info[0]), self.get(info[1])
            return self.elementwise(out_ty, [x, y], lambda xs: e.bin("mul", xs[0], xs[1])), info[2]

        if ma is not None:
            x, y = self.get(ma[0]), self.get(ma[1])
            z, z_neg = other(b_name, mb)
            neg_z = (name == "subf") != z_neg
            fn = lambda xs: _fma(e, [xs[0], xs[1], e.un("neg", xs[2]) if neg_z else xs[2]])
        else:
            x, y = self.get(mb[0]), self.get(mb[1])
            z, z_neg = other(a_name, ma)
            if z_neg:
                z = self.elementwise(out_ty, [z], lambda xs: e.un("neg", xs[0]))
            neg_x = name == "subf"
            fn = lambda xs: _fma(e, [e.un("neg", xs[0]) if neg_x else xs[0], xs[1], xs[2]])
        self.bind(op, [self.elementwise(out_ty, [x, y, z], fn)])
        return True

    def arith(self, op: Op, name: str):
        if self.fp_fusion and name in ("addf", "subf") and self.contract(op, name):
            return
        out_ty = op.result_types()[0]
        if name in ("cmpi", "cmpf"):
            out_ty = out_ty.with_elem("i1")
        inputs = [self.get(n) for n in op.operands]
        if name == "select" and inputs[0].ty.is_tensor is False and out_ty.is_tensor:
            inputs[0] = self.uniform(out_ty.with_elem("i1"), inputs[0].scalar)
        pred = op.body.strip().split(",")[0].split()[0] if name in ("cmpi", "cmpf") else None
        in_elem = inputs[0].ty.elem if inputs else None
        fn = ARITH[name]
        e = self.e
        elem = simt_type(out_ty)
        src = simt_type(inputs[0].ty) if inputs else None
        self.bind(op, [self.elementwise(out_ty, inputs, lambda xs: fn(e, xs, elem, src, pred))])

    def op_tt_extern_elementwise(self, op: Op):
        symbol = next((s.strip('"') for s in reversed(op.strings) if s.strip('"')), "")
        (out_ty,) = op.result_types()
        inputs = [self.get(n) for n in op.operands]
        fn = extern_function(symbol)
        e = self.e
        self.bind(op, [self.elementwise(out_ty, inputs, lambda xs: fn(e, xs, simt_type(out_ty)))])

    def op_tt_precise_sqrt(self, op: Op):
        (out_ty,) = op.result_types()
        e = self.e
        self.bind(op, [self.elementwise(out_ty, [self.get(op.operands[0])], lambda xs: e.un("sqrt", xs[0]))])

    def op_tt_precise_divf(self, op: Op):
        (out_ty,) = op.result_types()
        e = self.e
        inputs = [self.get(n) for n in op.operands[:2]]
        self.bind(op, [self.elementwise(out_ty, inputs, lambda xs: e.bin("div", xs[0], xs[1]))])

    def op_tt_bitcast(self, op: Op):
        (out_ty,) = op.result_types()
        inputs = [self.get(op.operands[0])]
        e = self.e
        self.bind(op, [self.elementwise(out_ty, inputs, lambda xs: e.bitcast(xs[0], simt_type(out_ty)))])

    def op_tt_int_to_ptr(self, op: Op):
        (out_ty,) = op.result_types()
        e = self.e
        self.bind(op, [self.elementwise(out_ty, [self.get(op.operands[0])], lambda xs: e.bitcast(xs[0], "u64"))])

    def op_tt_ptr_to_int(self, op: Op):
        (out_ty,) = op.result_types()
        e = self.e
        self.bind(op, [self.elementwise(out_ty, [self.get(op.operands[0])], lambda xs: e.bitcast(xs[0], "i64"))])

    def op_tt_fp_to_fp(self, op: Op):
        (out_ty,) = op.result_types()
        e = self.e
        self.bind(op, [self.elementwise(out_ty, [self.get(op.operands[0])], lambda xs: e.cast(xs[0], simt_type(out_ty)))])

    def op_tt_mulhiui(self, op: Op):
        (out_ty,) = op.result_types()
        e = self.e
        def fn(xs):
            u = UNSIGNED[simt_type(out_ty)]
            r = e.bin("mulhi", e.bitcast(xs[0], u), e.bitcast(xs[1], u))
            return e.bitcast(r, simt_type(out_ty))
        self.bind(op, [self.elementwise(out_ty, [self.get(n) for n in op.operands], fn)])

    def op_tt_clampf(self, op: Op):
        (out_ty,) = op.result_types()
        e = self.e
        fn = lambda xs: e.bin("min", e.bin("max", xs[0], xs[1]), xs[2])
        self.bind(op, [self.elementwise(out_ty, [self.get(n) for n in op.operands[:3]], fn)])


def _bits_to_float(bits: int, elem: str) -> float:
    if elem == "f64":
        return struct.unpack("<d", struct.pack("<Q", bits))[0]
    if elem == "f32":
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    if elem == "f16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    if elem == "bf16":
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    raise LoweringError(elem)


def _unsigned(e: Emitter, op: str, xs, elem):
    u = UNSIGNED[elem]
    return e.bitcast(e.bin(op, e.bitcast(xs[0], u), e.bitcast(xs[1], u)), elem)


def _minmax_nan(e: Emitter, xs, op):
    r = e.bin(op, xs[0], xs[1])
    r = e.sel(e.cmp("isnan", xs[0], xs[0]), xs[0], r)
    return e.sel(e.cmp("isnan", xs[1], xs[1]), xs[1], r)


CMPI = {"eq": ("eq", False), "ne": ("ne", False), "slt": ("lt", False), "sle": ("le", False), "sgt": ("gt", False),
        "sge": ("ge", False), "ult": ("lt", True), "ule": ("le", True), "ugt": ("gt", True), "uge": ("ge", True)}


def _cmpi(e: Emitter, xs, elem, src, pred):
    op, unsigned = CMPI[pred]
    a, b = xs
    if src == "i1":
        if op in ("eq", "ne"):
            r = e.bin("xor", a, b)
            return r if op == "ne" else e.un("not", r)
        a, b = e.cast(a, "i32"), e.cast(b, "i32")
        if not unsigned:
            a, b = e.un("neg", a), e.un("neg", b)
        return e.cmp(op, a, b)
    if unsigned and src in UNSIGNED:
        a, b = e.bitcast(a, UNSIGNED[src]), e.bitcast(b, UNSIGNED[src])
    return e.cmp(op, a, b)


def _cmpf(e: Emitter, xs, elem, src, pred):
    a, b = xs
    if pred in ("false", "true"):
        return const(pred == "true", "i1")
    if pred in ("ord", "uno"):
        nan = e.bin("or", e.cmp("isnan", a, a), e.cmp("isnan", b, b))
        return nan if pred == "uno" else e.un("not", nan)
    base = {"eq": "eq", "gt": "gt", "ge": "ge", "lt": "lt", "le": "le", "ne": "ne"}[pred[1:]]
    if pred[0] == "o":
        if base == "ne":
            nan = e.bin("or", e.cmp("isnan", a, a), e.cmp("isnan", b, b))
            return e.bin("and", e.cmp("ne", a, b), e.un("not", nan))
        return e.cmp(base, a, b)
    # Unordered: true when either operand is NaN.
    if base == "ne":
        return e.cmp("ne", a, b)
    nan = e.bin("or", e.cmp("isnan", a, a), e.cmp("isnan", b, b))
    return e.bin("or", e.cmp(base, a, b), nan)


def _extui(e: Emitter, xs, elem, src, pred):
    x = xs[0]
    if src == "i1":
        return e.cast(x, elem)
    return e.bitcast(e.cast(e.bitcast(x, UNSIGNED[src]), UNSIGNED[elem]), elem)


def _extsi(e: Emitter, xs, elem, src, pred):
    x = xs[0]
    if src == "i1":
        return e.sel(x, const(-1, elem), const(0, elem))
    return e.cast(x, elem)


def _trunci(e: Emitter, xs, elem, src, pred):
    x = xs[0]
    if elem == "i1":
        return e.cmp("ne", e.bin("and", x, const(1, src)), const(0, src))
    return e.bitcast(e.cast(e.bitcast(x, UNSIGNED[src]), UNSIGNED[elem]), elem)


def _uitofp(e: Emitter, xs, elem, src, pred):
    x = xs[0]
    if src == "i1":
        return e.cast(e.cast(x, "u32"), elem)
    return e.cast(e.bitcast(x, UNSIGNED[src]), elem)


def _fptoui(e: Emitter, xs, elem, src, pred):
    return e.bitcast(e.cast(xs[0], UNSIGNED[elem]), elem)


LN2 = math.log(2.0)
LOG2E = 1.0 / LN2


def _native(op: str, f32=None, f64=None):
    """Matches the official NVIDIA lowering: f32/f64 use the given PTX instruction, libdevice
    symbol (``__nv_*``) or callable; other types use the generic RUDA operation."""

    def lower_one(e: Emitter, xs, t, s, p):
        impl = {"f32": f32, "f64": f64}.get(t)
        if e.gx and op == "exp2":
            return e.un("exp", e.bin("mul", xs[0], const(LN2, t)))
        if e.gx and op == "log2":
            return e.bin("mul", e.un("log", xs[0]), const(LOG2E, t))
        if impl is None or e.gx:
            return e.un(op, xs[0])
        if callable(impl):
            return impl(e, xs[0])
        if impl.startswith("__nv_"):
            return e.libdevice(impl, t, xs[0])
        return e.asm(f"{impl} $0, $1;", t, xs[0])

    return lower_one


ARITH = {
    "addi": lambda e, xs, t, s, p: e.bin("add", *xs),
    "addf": lambda e, xs, t, s, p: e.bin("add", *xs),
    "subi": lambda e, xs, t, s, p: e.bin("sub", *xs),
    "subf": lambda e, xs, t, s, p: e.bin("sub", *xs),
    "muli": lambda e, xs, t, s, p: e.bin("mul", *xs),
    "mulf": lambda e, xs, t, s, p: e.bin("mul", *xs),
    "divf": lambda e, xs, t, s, p: e.asm("div.full.f32 $0, $1, $2;", t, *xs) if t == "f32" and not e.gx
    else e.bin("div", *xs),
    "divsi": lambda e, xs, t, s, p: e.bin("div", *xs),
    "divui": lambda e, xs, t, s, p: _unsigned(e, "div", xs, t),
    "remsi": lambda e, xs, t, s, p: e.bin("rem", *xs),
    "remui": lambda e, xs, t, s, p: _unsigned(e, "rem", xs, t),
    "remf": lambda e, xs, t, s, p: e.bin("mod", *xs),
    "andi": lambda e, xs, t, s, p: e.bin("and", *xs),
    "ori": lambda e, xs, t, s, p: e.bin("or", *xs),
    "xori": lambda e, xs, t, s, p: e.bin("xor", *xs),
    "shli": lambda e, xs, t, s, p: e.bin("shl", *xs),
    "shrsi": lambda e, xs, t, s, p: e.bin("shr", *xs),
    "shrui": lambda e, xs, t, s, p: _unsigned(e, "shr", xs, t),
    "minsi": lambda e, xs, t, s, p: e.bin("min", *xs),
    "maxsi": lambda e, xs, t, s, p: e.bin("max", *xs),
    "minui": lambda e, xs, t, s, p: _unsigned(e, "min", xs, t),
    "maxui": lambda e, xs, t, s, p: _unsigned(e, "max", xs, t),
    "minnumf": lambda e, xs, t, s, p: e.bin("min", *xs),
    "maxnumf": lambda e, xs, t, s, p: e.bin("max", *xs),
    "minimumf": lambda e, xs, t, s, p: _minmax_nan(e, xs, "min"),
    "maximumf": lambda e, xs, t, s, p: _minmax_nan(e, xs, "max"),
    "negf": lambda e, xs, t, s, p: e.un("neg", xs[0]),
    "cmpi": _cmpi,
    "cmpf": _cmpf,
    "select": lambda e, xs, t, s, p: e.sel(*xs),
    "extf": lambda e, xs, t, s, p: e.cast(xs[0], t),
    "truncf": lambda e, xs, t, s, p: e.cast(xs[0], t),
    "extsi": _extsi,
    "extui": _extui,
    "trunci": _trunci,
    "sitofp": lambda e, xs, t, s, p: e.cast(e.cast(xs[0], "i32") if s == "i1" else xs[0], t),
    "uitofp": _uitofp,
    "fptosi": lambda e, xs, t, s, p: e.cast(xs[0], t),
    "fptoui": _fptoui,
    "bitcast": lambda e, xs, t, s, p: e.bitcast(xs[0], t),
    "index_cast": lambda e, xs, t, s, p: e.cast(xs[0], t),
    "exp": _native("exp", f32=lambda e, x: e.asm("ex2.approx.f32 $0, $1;", "f32", e.bin("mul", x, const(LOG2E, "f32"))),
                   f64="__nv_exp"),
    "exp2": _native("exp2", f32="ex2.approx.ftz.f32", f64="__nv_exp2"),
    "log": _native("log", f32="__nv_logf", f64="__nv_log"),
    "log2": _native("log2", f32="__nv_log2f", f64="__nv_log2"),
    "log1p": _native("log1p", f32="__nv_log1pf", f64="__nv_log1p"),
    "sqrt": _native("sqrt", f32="sqrt.approx.ftz.f32"),
    "rsqrt": _native("rsqrt", f32="rsqrt.approx.ftz.f32", f64="__nv_rsqrt"),
    "sin": _native("sin", f32="__nv_sinf", f64="__nv_sin"),
    "cos": _native("cos", f32="__nv_cosf", f64="__nv_cos"),
    "tanh": _native("tanh", f32="__nv_tanhf", f64="__nv_tanh"),
    "erf": _native("erf", f32="__nv_erff", f64="__nv_erf"),
    "floor": lambda e, xs, t, s, p: e.un("floor", xs[0]),
    "ceil": lambda e, xs, t, s, p: e.un("ceil", xs[0]),
    "trunc": lambda e, xs, t, s, p: e.un("trunc", xs[0]),
    "roundeven": lambda e, xs, t, s, p: e.un("round", xs[0]),
    "absf": lambda e, xs, t, s, p: e.un("abs", xs[0]),
    "absi": lambda e, xs, t, s, p: e.un("abs", xs[0]),
    "fma": lambda e, xs, t, s, p: _fma(e, xs),
    "powf": lambda e, xs, t, s, p: e.libdevice({"f32": "__nv_powf", "f64": "__nv_pow"}[t], t, *xs)
    if t in ("f32", "f64") and not e.gx else e.bin("pow", *xs),
}


def _fma(e: Emitter, xs):
    out = e.local(e.ty_of(xs[0]))
    e.emit({"k": "fma", "out": out["l"], "a": xs[0], "b": xs[1], "c": xs[2]})
    return out


_EXTERN_UNARY = {"exp": "exp", "log": "log", "sqrt": "sqrt", "rsqrt": "rsqrt", "tanh": "tanh", "erf": "erf",
                 "floor": "floor", "ceil": "ceil", "fabs": "abs", "abs": "abs", "sin": "sin", "cos": "cos",
                 "log1p": "log1p", "trunc": "trunc", "rint": "round", "nearbyint": "round", "frcp_rn": "recip",
                 "drcp_rn": "recip", "fsqrt_rn": "sqrt", "dsqrt_rn": "sqrt", "fast_expf": "exp",
                 "fast_logf": "log"}
_EXTERN_BINARY = {"fmul_rn": "mul", "dmul_rn": "mul", "fadd_rn": "add", "dadd_rn": "add", "fsub_rn": "sub",
                  "dsub_rn": "sub", "fdiv_rn": "div", "ddiv_rn": "div", "fmaxf": "max", "fmax": "max",
                  "fminf": "min", "fmin": "min", "pow": "pow", "powf": "pow", "fmod": "mod", "fmodf": "mod",
                  "fdividef": "div"}


def extern_function(symbol: str):
    if symbol.startswith("__nv_"):
        def call(e, xs, t):
            if e.gx:
                return extern_function(symbol[len("__nv_"):])(e, xs, t)
            return e.libdevice(symbol, t, *xs)
        return call
    name = symbol
    if name in _EXTERN_BINARY:
        return lambda e, xs, t, op=_EXTERN_BINARY[name]: e.bin(op, xs[0], xs[1])
    candidates = [name, name[:-1] if name.endswith("f") else name]
    for c in candidates:
        if c in _EXTERN_UNARY:
            return lambda e, xs, t, op=_EXTERN_UNARY[c]: e.un(op, xs[0])
    if name in ("exp2f", "exp2"):
        return lambda e, xs, t: e.un("exp", e.bin("mul", xs[0], const(LN2, t)))
    if name in ("log2f", "log2"):
        return lambda e, xs, t: e.bin("mul", e.un("log", xs[0]), const(1.0 / LN2, t))
    if name in ("fmaf", "fma", "fmaf_rn", "fma_rn"):
        return lambda e, xs, t: _fma(e, xs)
    if name in ("isnanf", "isnand", "isnan"):
        return lambda e, xs, t: e.cast(e.cmp("isnan", xs[0], xs[0]), t)
    if name in ("isinff", "isinfd", "isinf"):
        return lambda e, xs, t: e.cast(e.cmp("isinf", xs[0], xs[0]), t)
    raise LoweringError(f"unsupported extern function {symbol}")


def lower(ttir_text: str, num_warps: int, fp_fusion: bool = True,
          reduce_layouts: list | None = None, gx: bool = False) -> tuple[dict, list]:
    func = parse_module(ttir_text)
    lowering = Lowering(func, num_warps, fp_fusion, reduce_layouts, gx)
    kernel = lowering.run()
    kernel["externs"] = lowering.e.externs
    return kernel, lowering.arg_kinds
