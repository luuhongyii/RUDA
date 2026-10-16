// Adapted from OpenLibm src/s_erf.c.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

use super::{emit::Emitter, types::Scalar};

fn c(x: f64) -> String { format!("0d{:016x}", x.to_bits()) }

impl Emitter {
    fn erf_polynomial(&mut self, x: &str, coefficients: &[f64]) -> String {
        let y = self.reg(Scalar::F64);
        self.line(format!("mov.f64 {y}, {};", c(*coefficients.last().unwrap())));
        for &coefficient in coefficients[..coefficients.len()-1].iter().rev() {
            self.line(format!("fma.rn.f64 {y}, {y}, {x}, {};", c(coefficient)));
        }
        y
    }

    pub(super) fn erf_f64(&mut self, destination: &str, source: &str) {
        let x=self.reg(Scalar::F64);let s=self.reg(Scalar::F64);let result=self.reg(Scalar::F64);
        let ratio=self.reg(Scalar::F64);let exponent=self.reg(Scalar::F64);
        let predicate=self.reg(Scalar::Pred);let bits=self.reg(Scalar::U64);let sign=self.reg(Scalar::U64);
        let small=self.label();let middle=self.label();let far=self.label();let exponential=self.label();
        let signed=self.label();let end=self.label();
        self.line(format!("abs.f64 {x}, {source};"));
        self.line(format!("setp.nan.f64 {predicate}, {x}, {x};"));
        self.line(format!("@{predicate} mov.f64 {destination}, {source};"));
        self.line(format!("@{predicate} bra {end};"));
        self.line(format!("setp.ge.f64 {predicate}, {x}, {};",c(6.0)));
        self.line(format!("@{predicate} mov.f64 {result}, {};",c(1.0)));
        self.line(format!("@{predicate} bra {signed};"));
        self.line(format!("setp.lt.f64 {predicate}, {x}, {};",c(0.84375)));
        self.line(format!("@{predicate} bra {small};"));
        self.line(format!("setp.lt.f64 {predicate}, {x}, {};",c(1.25)));
        self.line(format!("@{predicate} bra {middle};"));
        self.line(format!("mul.rn.f64 {s}, {x}, {x};"));
        self.line(format!("div.rn.f64 {s}, {}, {s};",c(1.0)));
        self.line(format!("setp.ge.f64 {predicate}, {x}, {};",c(1.0/0.35)));
        self.line(format!("@{predicate} bra {far};"));
        let p=self.erf_polynomial(&s,&[-9.86494403484714822705e-3,-6.93858572707181764372e-1,-1.05586262253232909814e1,-6.23753324503260060396e1,-1.62396669462573470355e2,-1.84605092906711035994e2,-8.12874355063065934246e1,-9.81432934416914548592]);
        let q=self.erf_polynomial(&s,&[1.0,1.96512716674392571292e1,1.37657754143519042600e2,4.34565877475229228821e2,6.45387271733267880336e2,4.29008140027567833386e2,1.08635005541779435134e2,6.57024977031928170135,-6.04244152148580987438e-2]);
        self.line(format!("div.rn.f64 {ratio}, {p}, {q};"));
        self.line(format!("bra {exponential};"));
        self.line(format!("{far}:"));
        let p=self.erf_polynomial(&s,&[-9.86494292470009928597e-3,-7.99283237680523006574e-1,-1.77579549177547519889e1,-1.60636384855821916062e2,-6.37566443368389627722e2,-1.02509513161107724954e3,-4.83519191608651397019e2]);
        let q=self.erf_polynomial(&s,&[1.0,3.03380607434824582924e1,3.25792512996573918826e2,1.53672958608443695994e3,3.19985821950859553908e3,2.55305040643316442583e3,4.74528541206955367215e2,-2.24409524465858183362e1]);
        self.line(format!("div.rn.f64 {ratio}, {p}, {q};"));
        self.line(format!("{exponential}:"));
        let truncated=self.reg(Scalar::F64);let correction=self.reg(Scalar::F64);let sum=self.reg(Scalar::F64);
        self.line(format!("mov.b64 {bits}, {x};"));
        self.line(format!("and.b64 {bits}, {bits}, 0xffffffff00000000;"));
        self.line(format!("mov.b64 {truncated}, {bits};"));
        self.line(format!("sub.rn.f64 {correction}, {truncated}, {x};"));
        self.line(format!("add.rn.f64 {sum}, {truncated}, {x};"));
        self.line(format!("fma.rn.f64 {correction}, {correction}, {sum}, {ratio};"));
        self.line(format!("neg.f64 {exponent}, {truncated};"));
        self.line(format!("fma.rn.f64 {exponent}, {exponent}, {truncated}, {};",c(-0.5625)));
        let e=self.reg(Scalar::F64);let f=self.reg(Scalar::F64);
        self.exponential_f64(&e,&exponent);self.exponential_f64(&f,&correction);
        self.line(format!("mul.rn.f64 {e}, {e}, {f};"));
        self.line(format!("div.rn.f64 {e}, {e}, {x};"));
        self.line(format!("sub.rn.f64 {result}, {}, {e};",c(1.0)));
        self.line(format!("bra {signed};"));
        self.line(format!("{small}:"));
        self.line(format!("mul.rn.f64 {s}, {x}, {x};"));
        let p=self.erf_polynomial(&s,&[1.28379167095512558561e-1,-3.25042107247001499370e-1,-2.84817495755985104766e-2,-5.77027029648944159157e-3,-2.37630166566501626084e-5]);
        let q=self.erf_polynomial(&s,&[1.0,3.97917223959155352819e-1,6.50222499887672944485e-2,5.08130628187576562776e-3,1.32494738004321644526e-4,-3.96022827877536812320e-6]);
        self.line(format!("div.rn.f64 {ratio}, {p}, {q};"));
        self.line(format!("fma.rn.f64 {result}, {x}, {ratio}, {x};"));
        self.line(format!("bra {signed};"));
        self.line(format!("{middle}:"));
        self.line(format!("sub.rn.f64 {s}, {x}, {};",c(1.0)));
        let p=self.erf_polynomial(&s,&[-2.36211856075265944077e-3,4.14856118683748331666e-1,-3.72207876035701323847e-1,3.18346619901161753674e-1,-1.10894694282396677476e-1,3.54783043256182359371e-2,-2.16637559486879084300e-3]);
        let q=self.erf_polynomial(&s,&[1.0,1.06420880400844228286e-1,5.40397917702171048937e-1,7.18286544141962662868e-2,1.26171219808761642112e-1,1.36370839120290507362e-2,1.19844998467991074170e-2]);
        self.line(format!("div.rn.f64 {ratio}, {p}, {q};"));
        self.line(format!("add.rn.f64 {result}, {ratio}, {};",c(8.45062911510467529297e-1)));
        self.line(format!("{signed}:"));
        self.line(format!("mov.b64 {sign}, {source};"));
        self.line(format!("and.b64 {sign}, {sign}, 0x8000000000000000;"));
        self.line(format!("mov.b64 {bits}, {result};"));
        self.line(format!("or.b64 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b64 {destination}, {bits};"));
        self.line(format!("{end}:"));
    }
}
