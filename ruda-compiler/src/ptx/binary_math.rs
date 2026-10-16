use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{BinaryOperator, Variable};

impl Emitter {
    pub fn inverse_trigonometry(&mut self, out: Variable, input: Variable, operation: u32) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if !ty.float() || ty.fp8() || out.ty != input.ty {
            return Err(invalid("trigonometry requires matching floating operands"));
        }
        let x = self.binary_math_input(ty, input)?;
        let result = self.reg(Scalar::F64);
        if operation < 2 {
            let left = self.reg(Scalar::F64);
            let right = self.reg(Scalar::F64);
            let root = self.reg(Scalar::F64);
            self.line(format!("sub.rn.f64 {left}, 0d3ff0000000000000, {x};"));
            self.line(format!("add.rn.f64 {right}, 0d3ff0000000000000, {x};"));
            self.line(format!("mul.rn.f64 {root}, {left}, {right};"));
            self.line(format!("sqrt.rn.f64 {root}, {root};"));
            if operation == 0 { self.atan2_f64(&result, &x, &root); }
            else { self.atan2_f64(&result, &root, &x); }
        } else if operation == 2 {
            self.atan2_f64(&result, &x, "0d3ff0000000000000");
        } else {
            let sine = self.reg(Scalar::F64);
            let cosine = self.reg(Scalar::F64);
            self.trigonometry_f64(&sine, &x, false);
            self.trigonometry_f64(&cosine, &x, true);
            self.line(format!("div.rn.f64 {result}, {sine}, {cosine};"));
        }
        let destination = self.destination(out)?;
        if ty == Scalar::F64 {
            self.line(format!("mov.f64 {destination}, {result};"));
        } else if ty.half() {
            let narrowed = self.reg(Scalar::F32);
            self.line(format!("cvt.rn.f32.f64 {narrowed}, {result};"));
            self.line(format!("cvt.rn.{}.f32 {destination}, {narrowed};", ty.suffix()));
        } else {
            self.line(format!("cvt.rn.f32.f64 {destination}, {result};"));
        }
        Ok(())
    }

    fn binary_math_input(&mut self, ty: Scalar, value: Variable) -> Result<String> {
        let source = self.value(value)?;
        if ty == Scalar::F64 { return Ok(source); }
        let source = if ty.half() { self.half_to_f32(ty, &source)? } else { source };
        let promoted = self.reg(Scalar::F64);
        self.line(format!("cvt.f64.f32 {promoted}, {source};"));
        Ok(promoted)
    }

    pub fn binary_math(&mut self, out: Variable, op: BinaryOperator, angle: bool) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if !ty.float() || ty.fp8() || out.ty != op.lhs.ty || out.ty != op.rhs.ty {
            return Err(invalid("hypot/atan2 require matching floating operands"));
        }
        let a = self.binary_math_input(ty, op.lhs)?;
        let b = self.binary_math_input(ty, op.rhs)?;
        let result = self.reg(Scalar::F64);
        if angle { self.atan2_f64(&result, &a, &b); }
        else { self.hypot_f64(&result, &a, &b); }
        let destination = self.destination(out)?;
        if ty == Scalar::F64 {
            self.line(format!("mov.b64 {destination}, {result};"));
        } else if ty.half() {
            let narrowed = self.reg(Scalar::F32);
            self.line(format!("cvt.rn.f32.f64 {narrowed}, {result};"));
            self.line(format!("cvt.rn.{}.f32 {destination}, {narrowed};", ty.suffix()));
        } else {
            self.line(format!("cvt.rn.f32.f64 {destination}, {result};"));
        }
        Ok(())
    }

    fn hypot_f64(&mut self, result: &str, a: &str, b: &str) {
        let x = self.reg(Scalar::F64);
        let y = self.reg(Scalar::F64);
        let hi = self.reg(Scalar::F64);
        let lo = self.reg(Scalar::F64);
        let ratio = self.reg(Scalar::F64);
        let p = self.reg(Scalar::Pred);
        let end = self.label();
        self.line(format!("abs.f64 {x}, {a};"));
        self.line(format!("abs.f64 {y}, {b};"));
        self.line(format!("mov.f64 {result}, 0d7ff0000000000000;"));
        for v in [&x, &y] {
            self.line(format!("setp.eq.f64 {p}, {v}, 0d7ff0000000000000;"));
            self.line(format!("@{p} bra {end};"));
        }
        self.line(format!("add.rn.f64 {result}, {x}, {y};"));
        self.line(format!("setp.nan.f64 {p}, {x}, {y};"));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("max.f64 {hi}, {x}, {y};"));
        self.line(format!("min.f64 {lo}, {x}, {y};"));
        self.line(format!("setp.eq.f64 {p}, {hi}, 0d0000000000000000;"));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("div.rn.f64 {ratio}, {lo}, {hi};"));
        self.line(format!("fma.rn.f64 {ratio}, {ratio}, {ratio}, 0d3ff0000000000000;"));
        self.line(format!("sqrt.rn.f64 {ratio}, {ratio};"));
        self.line(format!("mul.rn.f64 {result}, {hi}, {ratio};"));
        self.line(format!("{end}:"));
    }

    fn atan2_f64(&mut self, result: &str, y: &str, x: &str) {
        let ax = self.reg(Scalar::F64);
        let ay = self.reg(Scalar::F64);
        let high = self.reg(Scalar::F64);
        let low = self.reg(Scalar::F64);
        let t = self.reg(Scalar::F64);
        let square = self.reg(Scalar::F64);
        let numerator = self.reg(Scalar::F64);
        let denominator = self.reg(Scalar::F64);
        let base = self.reg(Scalar::F64);
        let p = self.reg(Scalar::Pred);
        let swap = self.reg(Scalar::Pred);
        let bits = self.reg(Scalar::U64);
        let sign = self.reg(Scalar::U64);
        let special = self.label();
        let ratio_ready = self.label();
        let polynomial = self.label();
        let quadrant = self.label();
        let end = self.label();
        self.line(format!("add.rn.f64 {result}, {x}, {y};"));
        self.line(format!("setp.nan.f64 {p}, {x}, {y};"));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("abs.f64 {ax}, {x};"));
        self.line(format!("abs.f64 {ay}, {y};"));
        self.line(format!("setp.gt.f64 {swap}, {ay}, {ax};"));
        self.line(format!("max.f64 {high}, {ay}, {ax};"));
        self.line(format!("min.f64 {low}, {ay}, {ax};"));
        self.line(format!("setp.eq.f64 {p}, {high}, 0d0000000000000000;"));
        self.line(format!("mov.f64 {result}, 0d0000000000000000;"));
        self.line(format!("@{p} bra {quadrant};"));
        self.line(format!("setp.eq.f64 {p}, {high}, 0d7ff0000000000000;"));
        self.line(format!("@{p} bra {special};"));
        self.line(format!("div.rn.f64 {t}, {low}, {high};"));
        self.line(format!("bra {ratio_ready};"));
        self.line(format!("{special}:"));
        self.line(format!("setp.eq.f64 {p}, {low}, 0d7ff0000000000000;"));
        self.line(format!("selp.f64 {t}, 0d3ff0000000000000, 0d0000000000000000, {p};"));
        self.line(format!("{ratio_ready}:"));
        self.line(format!("mov.f64 {base}, 0d0000000000000000;"));
        self.line(format!("setp.le.f64 {p}, {t}, 0d{:016x};", (core::f64::consts::SQRT_2 - 1.0).to_bits()));
        self.line(format!("@{p} bra {polynomial};"));
        self.line(format!("sub.rn.f64 {numerator}, {t}, 0d3ff0000000000000;"));
        self.line(format!("add.rn.f64 {denominator}, {t}, 0d3ff0000000000000;"));
        self.line(format!("div.rn.f64 {t}, {numerator}, {denominator};"));
        self.line(format!("mov.f64 {base}, 0d{:016x};", core::f64::consts::FRAC_PI_4.to_bits()));
        self.line(format!("{polynomial}:"));
        self.line(format!("mul.rn.f64 {square}, {t}, {t};"));
        self.line(format!("mov.f64 {result}, 0d{:016x};", (1.0f64 / 49.0).to_bits()));
        for k in (0..24).rev() {
            let coefficient = if k % 2 == 0 { 1.0 } else { -1.0 } / (2 * k + 1) as f64;
            self.line(format!("fma.rn.f64 {result}, {result}, {square}, 0d{:016x};", coefficient.to_bits()));
        }
        self.line(format!("fma.rn.f64 {result}, {result}, {t}, {base};"));
        self.line(format!("@{swap} sub.rn.f64 {result}, 0d{:016x}, {result};", core::f64::consts::FRAC_PI_2.to_bits()));
        self.line(format!("{quadrant}:"));
        self.line(format!("mov.b64 {bits}, {x};"));
        self.line(format!("and.b64 {sign}, {bits}, 0x8000000000000000;"));
        self.line(format!("setp.ne.u64 {p}, {sign}, 0;"));
        self.line(format!("@{p} sub.rn.f64 {result}, 0d{:016x}, {result};", core::f64::consts::PI.to_bits()));
        self.line(format!("mov.b64 {bits}, {y};"));
        self.line(format!("and.b64 {sign}, {bits}, 0x8000000000000000;"));
        self.line(format!("mov.b64 {bits}, {result};"));
        self.line(format!("xor.b64 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b64 {result}, {bits};"));
        self.line(format!("{end}:"));
    }
}
