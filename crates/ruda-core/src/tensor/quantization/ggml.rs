#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13,
    Q6_K = 14, Q8_K = 15, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18,
    IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29,
    BF16 = 30, TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39, NVFP4 = 40,
    Q1_0 = 41, Q2_0 = 42,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32, 1 => Self::F16, 2 => Self::Q4_0, 3 => Self::Q4_1,
            6 => Self::Q5_0, 7 => Self::Q5_1, 8 => Self::Q8_0, 9 => Self::Q8_1,
            10 => Self::Q2_K, 11 => Self::Q3_K, 12 => Self::Q4_K, 13 => Self::Q5_K,
            14 => Self::Q6_K, 15 => Self::Q8_K, 16 => Self::IQ2_XXS, 17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS, 19 => Self::IQ1_S, 20 => Self::IQ4_NL, 21 => Self::IQ3_S,
            22 => Self::IQ2_S, 23 => Self::IQ4_XS, 24 => Self::I8, 25 => Self::I16,
            26 => Self::I32, 27 => Self::I64, 28 => Self::F64, 29 => Self::IQ1_M,
            30 => Self::BF16, 34 => Self::TQ1_0, 35 => Self::TQ2_0,
            39 => Self::MXFP4, 40 => Self::NVFP4, 41 => Self::Q1_0, 42 => Self::Q2_0,
            _ => return None,
        })
    }

    pub const fn block_layout(self) -> (usize, usize) {
        match self {
            Self::F32 | Self::I32 => (1, 4),
            Self::F16 | Self::BF16 | Self::I16 => (1, 2),
            Self::I8 => (1, 1),
            Self::I64 | Self::F64 => (1, 8),
            Self::Q4_0 | Self::IQ4_NL => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::IQ2_XXS => (256, 66),
            Self::IQ2_XS => (256, 74),
            Self::IQ3_XXS => (256, 98),
            Self::IQ1_S => (256, 50),
            Self::IQ3_S => (256, 110),
            Self::IQ2_S => (256, 82),
            Self::IQ4_XS => (256, 136),
            Self::IQ1_M => (256, 56),
            Self::TQ1_0 => (256, 54),
            Self::TQ2_0 => (256, 66),
            Self::MXFP4 => (32, 17),
            Self::NVFP4 => (64, 36),
            Self::Q1_0 => (128, 18),
            Self::Q2_0 => (64, 18),
        }
    }

    pub fn row_bytes(self, width: usize) -> Option<usize> {
        let (elements, bytes) = self.block_layout();
        if width % elements != 0 { return None; }
        (width / elements).checked_mul(bytes)
    }

    pub const fn is_quantized(self) -> bool {
        self.block_layout().0 != 1
    }
}
