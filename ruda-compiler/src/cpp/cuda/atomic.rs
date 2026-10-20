use std::fmt::{self, Formatter};

use crate::cpp::shared::{Component, Dialect, Elem, FmtLeft, Variable};

pub(super) fn reduction<D: Dialect>(
    f: &mut Formatter<'_>,
    pointer: &Variable<D>,
    value: &Variable<D>,
    out: &Variable<D>,
    operation: &str,
) -> fmt::Result {
    let item = out.item();
    let elem = out.elem();
    let width = item.vectorization;
    let (ty, register, constraint, half) = match elem {
        Elem::F16 => ("f16", "unsigned short", "h", true),
        Elem::BF16 => ("bf16", "unsigned short", "h", true),
        Elem::F32 => ("f32", "float", "f", false),
        Elem::F64 => ("f64", "double", "d", false),
        Elem::I32 => (
            if matches!(operation, "min" | "max") {
                "s32"
            } else {
                "u32"
            },
            "unsigned int",
            "r",
            false,
        ),
        Elem::U32 => ("u32", "unsigned int", "r", false),
        Elem::I64 => (
            if matches!(operation, "min" | "max") {
                "s64"
            } else {
                "u64"
            },
            "unsigned long long",
            "l",
            false,
        ),
        Elem::U64 => ("u64", "unsigned long long", "l", false),
        _ => panic!("Unsupported CUDA atomic arithmetic type: {elem}"),
    };
    let op = if operation == "sub" { "add" } else { operation };
    let noftz = if half { ".noftz" } else { "" };
    writeln!(f, "{} = [&]() {{", out.fmt_left())?;
    for lane in 0..width {
        writeln!(f, "{elem} value_{lane} = {};", value.index(lane))?;
        writeln!(
            f,
            "{register} input_{lane} = reinterpret_cast<const {register}&>(value_{lane});"
        )?;
        if operation == "sub" {
            if half {
                writeln!(f, "input_{lane} ^= 0x8000u;")?;
            } else {
                writeln!(f, "input_{lane} = -input_{lane};")?;
            }
        }
        writeln!(f, "{register} result_{lane};")?;
    }
    let vector = (half && matches!(width, 2 | 4 | 8))
        || (elem == Elem::F32 && matches!(width, 2 | 4) && op == "add");
    if vector {
        if op == "add" {
            writeln!(f, "#if __CUDA_ARCH__ >= 900\nif (__isGlobal({pointer})) {{")?;
        }
        let dst = (0..width)
            .map(|i| format!("%{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let src = (0..width)
            .map(|i| format!("%{}", width + 1 + i))
            .collect::<Vec<_>>()
            .join(", ");
        let outputs = (0..width)
            .map(|i| format!("\"={constraint}\"(result_{i})"))
            .collect::<Vec<_>>()
            .join(", ");
        let inputs = (0..width)
            .map(|i| format!("\"{constraint}\"(input_{i})"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            f,
            "asm volatile(\"atom.{op}{noftz}.v{width}.{ty} {{{dst}}}, [%{width}], {{{src}}};\" : {outputs} : \"l\"({pointer}), {inputs} : \"memory\");"
        )?;
        if op == "add" {
            writeln!(f, "}} else\n#endif\n{{")?;
        }
    }
    if !vector || op == "add" {
        for lane in 0..width {
            writeln!(
                f,
                "asm volatile(\"atom.{op}{noftz}.{ty} %0, [%1], %2;\" : \"={constraint}\"(result_{lane}) : \"l\"(reinterpret_cast<{elem}*>({pointer}) + {lane}), \"{constraint}\"(input_{lane}) : \"memory\");"
            )?;
        }
        if vector {
            writeln!(f, "}}")?;
        }
    }
    writeln!(f, "{item} result;")?;
    for lane in 0..width {
        writeln!(
            f,
            "reinterpret_cast<{elem}*>(&result)[{lane}] = reinterpret_cast<const {elem}&>(result_{lane});"
        )?;
    }
    writeln!(f, "return result;\n}}();")
}

pub(super) fn bits<D: Dialect>(
    f: &mut Formatter<'_>,
    pointer: &Variable<D>,
    value: Option<&Variable<D>>,
    compare: Option<&Variable<D>>,
    out: &Variable<D>,
    operation: &str,
) -> fmt::Result {
    let item = value.unwrap_or(out).item();
    let elem = item.elem;
    let bytes = item.size();
    let bits = match bytes {
        2 => "unsigned short",
        4 => "unsigned int",
        8 => "unsigned long long",
        16 => "AtomicBits128",
        _ => panic!("CUDA whole-value atomics require 2, 4, 8 or 16 bytes"),
    };
    if operation == "store" {
        writeln!(f, "[&]() {{")?;
    } else {
        writeln!(f, "{} = [&]() {{", out.fmt_left())?;
    }
    if bytes == 16 {
        writeln!(
            f,
            "struct alignas(16) AtomicBits128 {{ unsigned long long lo, hi; }};"
        )?;
    }
    writeln!(f, "{bits} desired{{}}, expected{{}};")?;
    for (name, input) in [("desired", value), ("expected", compare)] {
        if let Some(input) = input {
            for lane in 0..item.vectorization {
                writeln!(
                    f,
                    "reinterpret_cast<{elem}*>(&{name})[{lane}] = {};",
                    input.index(lane)
                )?;
            }
        }
    }
    writeln!(f, "auto pointer = reinterpret_cast<{bits}*>({pointer});")?;
    match operation {
        "load" | "cas" => writeln!(
            f,
            "{bits} previous = atomicCAS(pointer, expected, desired);"
        )?,
        "store" | "exch" if bytes == 2 => {
            writeln!(
                f,
                "{bits} previous = atomicCAS(pointer, expected, expected);\ndo {{\nexpected = previous;\nprevious = atomicCAS(pointer, expected, desired);\n}} while (previous != expected);"
            )?;
        }
        "store" | "exch" => writeln!(f, "{bits} previous = atomicExch(pointer, desired);")?,
        _ => unreachable!(),
    }
    if operation != "store" {
        writeln!(f, "return reinterpret_cast<const {item}&>(previous);")?;
    }
    writeln!(f, "}}();")
}
