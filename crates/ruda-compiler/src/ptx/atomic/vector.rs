use super::*;

impl Emitter {
    pub(super) fn atomic_vector(&mut self, operation: &AtomicOp, out: Variable) -> Result<bool> {
        let (pointer_var, value, compare, opcode) = match operation {
            AtomicOp::Load(op) => (op.input, None, None, "load"),
            AtomicOp::Store(op) => (out, Some(op.input), None, "exch"),
            AtomicOp::Swap(op) => (op.lhs, Some(op.rhs), None, "exch"),
            AtomicOp::CompareAndSwap(op) => (op.input, Some(op.val), Some(op.cmp), "cas"),
            AtomicOp::Add(op) | AtomicOp::Sub(op) => (op.lhs, Some(op.rhs), None, "add"),
            AtomicOp::Min(op) => (op.lhs, Some(op.rhs), None, "min"),
            AtomicOp::Max(op) => (op.lhs, Some(op.rhs), None, "max"),
            _ => return Ok(false),
        };
        let ty = element(pointer_var)?;
        let width = pointer_var.ty.vector_size();
        for arg in value.iter().chain(compare.iter()) {
            if Scalar::of(arg.ty.with_vector_size(1))? != ty {
                return Err(invalid("atomic vector element types differ"));
            }
        }
        if !is_atomic(out) && Scalar::of(out.ty.with_vector_size(1))? != ty {
            return Err(invalid("atomic vector output element type differs"));
        }
        let pointer = self.atomic_pointer(pointer_var)?;
        let semantics = self.atomic_semantics(&pointer);
        if matches!(opcode, "load" | "exch" | "cas") {
            let bytes = width * ty.bytes();
            if !matches!(bytes, 4 | 8 | 16) {
                return Err(unsupported(
                    "whole-value vector atomics require 4, 8 or 16 bytes",
                ));
            }
            if bytes == 16 && (self.target.sm < 90 || self.target.version < (8, 3)) {
                return Err(unsupported(
                    "128-bit atomics require SM >= 90 and PTX >= 8.3",
                ));
            }
            let values = if let Some(value) = value {
                self.vector_values(value)?
            } else {
                let zero = self.reg(ty);
                self.line(format!("mov.{} {zero}, 0;", ty.bits()));
                vec![zero; width]
            };
            let packed = self.pack_atomic_bits(ty, &values);
            let previous = self.atomic_bits_register(bytes);
            if opcode == "exch" {
                self.line(format!(
                    "atom{semantics}.exch.b{} {previous}, [{pointer}], {packed};",
                    bytes * 8
                ));
            } else {
                let expected = if let Some(compare) = compare {
                    let values = self.vector_values(compare)?;
                    self.pack_atomic_bits(ty, &values)
                } else {
                    packed.clone()
                };
                self.line(format!(
                    "atom{semantics}.cas.b{} {previous}, [{pointer}], {expected}, {packed};",
                    bytes * 8
                ));
            }
            if !is_atomic(out) {
                let destinations = self.vector_destination(out)?;
                self.unpack_atomic_bits(ty, &previous, &destinations);
            }
            return Ok(true);
        }
        let supported = (ty.half() && matches!(width, 2 | 4 | 8))
            || (ty == Scalar::F32 && matches!(width, 2 | 4) && opcode == "add");
        if !supported || self.target.sm < 90 || self.target.version < (8, 1) {
            if ty.half() && matches!(opcode, "min" | "max") {
                return Err(unsupported(
                    "half vector min/max requires global memory, SM >= 90 and PTX >= 8.1",
                ));
            }
            return Ok(false);
        }
        let mut values = self.vector_values(value.unwrap())?;
        if matches!(operation, AtomicOp::Sub(_)) {
            for value in &mut values {
                let negated = self.reg(ty);
                let sign = if ty.half() { "0x8000" } else { "0x80000000" };
                self.line(format!("xor.{} {negated}, {value}, {sign};", ty.bits()));
                *value = negated;
            }
        }
        let destinations = self.vector_destination(out)?;
        let noftz = if ty.half() { ".noftz" } else { "" };
        let fallback = if opcode == "add" {
            let global = self.reg(Scalar::Pred);
            let scalar = self.label();
            let done = self.label();
            self.line(format!("isspacep.global {global}, {pointer};"));
            self.line(format!("@!{global} bra {scalar};"));
            Some((scalar, done))
        } else {
            None
        };
        self.line(format!(
            "atom{semantics}.{opcode}{noftz}.v{width}.{} {{{}}}, [{pointer}], {{{}}};",
            ty.suffix(),
            destinations.join(", "),
            values.join(", ")
        ));
        if let Some((scalar, done)) = fallback {
            self.line(format!("bra {done};"));
            self.line(format!("{scalar}:"));
            for (lane, (value, destination)) in values.iter().zip(&destinations).enumerate() {
                let address = self.reg(Scalar::U64);
                self.line(format!(
                    "add.u64 {address}, {pointer}, {};",
                    lane * ty.bytes()
                ));
                if self.native_atomic_pointers.contains(&pointer) {
                    self.native_atomic_pointers.insert(address.clone());
                }
                self.atomic_reduce(ty, opcode, &address, value, destination)?;
            }
            self.line(format!("{done}:"));
        }
        Ok(true)
    }

    fn atomic_bits_register(&mut self, bytes: usize) -> String {
        match bytes {
            4 => self.reg(Scalar::U32),
            8 => self.reg(Scalar::U64),
            16 => {
                let register = format!("%atomic_{}", self.label());
                self.declarations
                    .push_str(&format!("    .reg .b128 {register};\n"));
                register
            }
            _ => unreachable!(),
        }
    }

    fn pack_atomic_bits(&mut self, ty: Scalar, values: &[String]) -> String {
        let bytes = values.len() * ty.bytes();
        let mut words = Vec::new();
        for chunk in values.chunks(8 / ty.bytes()) {
            let word = self.reg(Scalar::U64);
            self.line(format!("mov.u64 {word}, 0;"));
            for (lane, value) in chunk.iter().enumerate() {
                let part = self.reg(Scalar::U64);
                if ty.bytes() == 8 {
                    self.line(format!("mov.b64 {part}, {value};"));
                } else if ty.half() {
                    self.line(format!("cvt.u64.u16 {part}, {value};"));
                } else {
                    let bits = self.reg(Scalar::U32);
                    self.line(format!("mov.b32 {bits}, {value};"));
                    self.line(format!("cvt.u64.u32 {part}, {bits};"));
                    if ty.bytes() == 2 {
                        self.line(format!("and.b64 {part}, {part}, 65535;"));
                    }
                }
                self.line(format!(
                    "shl.b64 {part}, {part}, {};",
                    lane * ty.bytes() * 8
                ));
                self.line(format!("or.b64 {word}, {word}, {part};"));
            }
            words.push(word);
        }
        let packed = self.atomic_bits_register(bytes);
        match bytes {
            4 => self.line(format!("cvt.u32.u64 {packed}, {};", words[0])),
            8 => self.line(format!("mov.b64 {packed}, {};", words[0])),
            16 => self.line(format!(
                "mov.b128 {packed}, {{{}, {}}};",
                words[0], words[1]
            )),
            _ => unreachable!(),
        }
        packed
    }

    fn unpack_atomic_bits(&mut self, ty: Scalar, packed: &str, destinations: &[String]) {
        let bytes = destinations.len() * ty.bytes();
        let lo = self.reg(Scalar::U64);
        let hi = self.reg(Scalar::U64);
        match bytes {
            4 => self.line(format!("cvt.u64.u32 {lo}, {packed};")),
            8 => self.line(format!("mov.b64 {lo}, {packed};")),
            16 => self.line(format!("mov.b128 {{{lo}, {hi}}}, {packed};")),
            _ => unreachable!(),
        }
        for (lane, destination) in destinations.iter().enumerate() {
            let word = if lane * ty.bytes() < 8 { &lo } else { &hi };
            let bits = self.reg(Scalar::U64);
            self.line(format!(
                "shr.u64 {bits}, {word}, {};",
                (lane * ty.bytes() % 8) * 8
            ));
            if ty.bytes() == 8 {
                self.line(format!("mov.b64 {destination}, {bits};"));
            } else if ty.half() {
                self.line(format!("cvt.u16.u64 {destination}, {bits};"));
            } else {
                let narrow = self.reg(Scalar::U32);
                self.line(format!("cvt.u32.u64 {narrow}, {bits};"));
                self.line(format!("mov.b32 {destination}, {narrow};"));
                if ty.narrow() {
                    self.normalize_integer(ty, destination);
                }
            }
        }
    }
}
