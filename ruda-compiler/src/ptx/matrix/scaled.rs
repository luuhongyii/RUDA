use super::*;
use ruda_core::ir::{ElemType, FloatKind};

impl Emitter {
    pub(super) fn matrix_scaled(
        &mut self, out: Variable, matrix: Matrix,
        a: Variable, b: Variable, c: Variable,
        scales_a: Variable, scales_b: Variable, factor: usize,
    ) -> Result<()> {
        if self.target.sm != 120 || self.target.version < (8, 7) {
            return Err(unsupported("NVFP4 scaled MMA requires SM 120 and PTX >= 8.7 (sm_120a instructions)"));
        }
        let f32_type = Type::scalar(ElemType::Float(FloatKind::F32));
        let scale_type = Type::scalar(ElemType::Float(FloatKind::E4M3));
        if (matrix.m, matrix.n, matrix.k) != (16, 8, 64) || factor != 4
            || !Scalar::is_packed_fp4(a.ty) || !Scalar::is_packed_fp4(b.ty)
            || c.ty.with_vector_size(1) != f32_type || out.ty.with_vector_size(1) != f32_type
            || scales_a.ty.with_vector_size(1) != scale_type
            || scales_b.ty.with_vector_size(1) != scale_type
        {
            return Err(unsupported("scaled MMA supports m16n8k64 E2M1x2 A/B, F32 C/D, E4M3 scales and scale_vec::4X"));
        }
        let a = self.matrix_read_registers(a, 4)?;
        let b = self.matrix_read_registers(b, 2)?;
        let c = self.matrix_read_registers(c, 4)?;
        let sa = self.matrix_read_registers(scales_a, 1)?;
        let sb = self.matrix_read_registers(scales_b, 1)?;
        let d = (0..4).map(|_| self.reg(Scalar::F32)).collect::<Vec<_>>();
        let selector = self.reg_b16();
        self.line(format!("mov.b16 {selector}, 0;"));
        self.line(format!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {{{}}}, {{{}}}, {{{}}}, {{{}}}, {}, {{{selector}, {selector}}}, {}, {{{selector}, {selector}}};",
            d.join(", "), a.join(", "), b.join(", "), c.join(", "), sa[0], sb[0],
        ));
        self.matrix_write_registers(out, &d)?;
        self.architecture_specific = true;
        Ok(())
    }
}
