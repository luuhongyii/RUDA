use super::{emit::Emitter, types::Scalar, Result};
use ruda_core::ir::{BinaryOperator, Variable};

#[derive(Clone)]
struct Double { hi: String, lo: String }

fn f64c(x: f64) -> String { format!("0d{:016x}", x.to_bits()) }
fn pair(bits: (u64, u64)) -> Double {
    Double { hi: format!("0d{:016x}", bits.0), lo: format!("0d{:016x}", bits.1) }
}

const LN2: (u64, u64) = (0x3fe62e42fefa39ef, 0x3c7abc9e3b39803f);
const PIO2: (u64, u64) = (0x3ff921fb54442d18, 0x3c91a62633145c07);
const LOG_COEFFICIENTS: [(u64, u64); 25] = [
    (0x3ff0000000000000, 0x0000000000000000),
    (0x3fd5555555555555, 0x3c75555555555555),
    (0x3fc999999999999a, 0xbc6999999999999a),
    (0x3fc2492492492492, 0x3c62492492492492),
    (0x3fbc71c71c71c71c, 0x3c5c71c71c71c71c),
    (0x3fb745d1745d1746, 0xbc4745d1745d1746),
    (0x3fb3b13b13b13b14, 0xbc53b13b13b13b14),
    (0x3fb1111111111111, 0x3c31111111111111),
    (0x3fae1e1e1e1e1e1e, 0x3c2e1e1e1e1e1e1e),
    (0x3faaf286bca1af28, 0x3c4af286bca1af28),
    (0x3fa8618618618618, 0x3c48618618618618),
    (0x3fa642c8590b2164, 0x3c3642c8590b2164),
    (0x3fa47ae147ae147b, 0xbc2eb851eb851eb8),
    (0x3fa2f684bda12f68, 0x3c42f684bda12f68),
    (0x3fa1a7b9611a7b96, 0x3c21a7b9611a7b96),
    (0x3fa0842108421084, 0x3c30842108421084),
    (0x3f9f07c1f07c1f08, 0xbc2f07c1f07c1f08),
    (0x3f9d41d41d41d41d, 0x3c30750750750750),
    (0x3f9bacf914c1bad0, 0xbc3bacf914c1bad0),
    (0x3f9a41a41a41a41a, 0x3c30690690690690),
    (0x3f98f9c18f9c18fa, 0xbc2f3831f3831f38),
    (0x3f97d05f417d05f4, 0x3c17d05f417d05f4),
    (0x3f96c16c16c16c17, 0xbc2f49f49f49f49f),
    (0x3f95c9882b931057, 0x3c2310572620ae4c),
    (0x3f94e5e0a72f0539, 0x3c3e0a72f0539783),
];

const EXP_COEFFICIENTS: [(u64, u64); 25] = [
    (0x3ff0000000000000, 0x0000000000000000),
    (0x3ff0000000000000, 0x0000000000000000),
    (0x3fe0000000000000, 0x0000000000000000),
    (0x3fc5555555555555, 0x3c65555555555555),
    (0x3fa5555555555555, 0x3c45555555555555),
    (0x3f81111111111111, 0x3c01111111111111),
    (0x3f56c16c16c16c17, 0xbbef49f49f49f49f),
    (0x3f2a01a01a01a01a, 0x3b6a01a01a01a01a),
    (0x3efa01a01a01a01a, 0x3b3a01a01a01a01a),
    (0x3ec71de3a556c734, 0xbb6c154f8ddc6c00),
    (0x3e927e4fb7789f5c, 0x3b3cbbc05b4fa99a),
    (0x3e5ae64567f544e4, 0xbafc062e06d1f209),
    (0x3e21eed8eff8d898, 0xbac2aec959e14c06),
    (0x3de6124613a86d09, 0x3a8f28e0cc748ebe),
    (0x3da93974a8c07c9d, 0x3a305d6f8a2efd1f),
    (0x3d6ae7f3e733b81f, 0x39e1d8656b0ee8cb),
    (0x3d2ae7f3e733b81f, 0x39a1d8656b0ee8cb),
    (0x3ce952c77030ad4a, 0x398ac981465ddc6c),
    (0x3ca6827863b97d97, 0x394eec01221a8b0b),
    (0x3c62f49b46814157, 0x38f2650f61dbdcb4),
    (0x3c1e542ba4020225, 0x387ea72b4afe3c2f),
    (0x3bd71b8ef6dcf572, 0xb87d043ae40c4647),
    (0x3b90ce396db7f853, 0xb83aebcdbd20331c),
    (0x3b4761b41316381a, 0xb7d3423c7d91404f),
    (0x3aff2cf01972f578, 0xb789ada5fcc1ab14),
];

const SIN_COEFFICIENTS: [(u64, u64); 18] = [
    (0x3ff0000000000000, 0x0000000000000000),
    (0xbfc5555555555555, 0xbc65555555555555),
    (0x3f81111111111111, 0x3c01111111111111),
    (0xbf2a01a01a01a01a, 0xbb6a01a01a01a01a),
    (0x3ec71de3a556c734, 0xbb6c154f8ddc6c00),
    (0xbe5ae64567f544e4, 0x3afc062e06d1f209),
    (0x3de6124613a86d09, 0x3a8f28e0cc748ebe),
    (0xbd6ae7f3e733b81f, 0xb9e1d8656b0ee8cb),
    (0x3ce952c77030ad4a, 0x398ac981465ddc6c),
    (0xbc62f49b46814157, 0xb8f2650f61dbdcb4),
    (0x3bd71b8ef6dcf572, 0xb87d043ae40c4647),
    (0xbb4761b41316381a, 0x37d3423c7d91404f),
    (0x3ab3f3ccdd165fa9, 0xb7458ddadf344487),
    (0xba1d1ab1c2dccea3, 0xb6a054d0c78aea14),
    (0x398259f98b4358ad, 0x362eaf8c39dd9bc5),
    (0xb8e434d2e783f5bc, 0xb580b87b91be9aff),
    (0x3843981254dd0d52, 0xb4e2b1f4c8015a2f),
    (0xb7a0dc59c716d91f, 0xb43419e3fad3f031),
];

const COS_COEFFICIENTS: [(u64, u64); 18] = [
    (0x3ff0000000000000, 0x0000000000000000),
    (0xbfe0000000000000, 0x0000000000000000),
    (0x3fa5555555555555, 0x3c45555555555555),
    (0xbf56c16c16c16c17, 0x3bef49f49f49f49f),
    (0x3efa01a01a01a01a, 0x3b3a01a01a01a01a),
    (0xbe927e4fb7789f5c, 0xbb3cbbc05b4fa99a),
    (0x3e21eed8eff8d898, 0xbac2aec959e14c06),
    (0xbda93974a8c07c9d, 0xba305d6f8a2efd1f),
    (0x3d2ae7f3e733b81f, 0x39a1d8656b0ee8cb),
    (0xbca6827863b97d97, 0xb94eec01221a8b0b),
    (0x3c1e542ba4020225, 0x387ea72b4afe3c2f),
    (0xbb90ce396db7f853, 0x383aebcdbd20331c),
    (0x3aff2cf01972f578, 0xb789ada5fcc1ab14),
    (0xba688e85fc6a4e5a, 0x37071c37ebd16540),
    (0x39d0a18a2635085d, 0x366b9e2e28e1aa54),
    (0xb933932c5047d60e, 0xb5d832b7b530a627),
    (0x389434d2e783f5bc, 0x3530b87b91be9aff),
    (0xb7f2710231c0fd7a, 0xb473f8a2b4af9d6b),
];

const TWO_OVER_PI: [u32; 48] = [
    0xca73a8c9,
    0x06061556,
    0x4d732731,
    0x8dffd880,
    0x14a06840,
    0x6599855f,
    0x5ee61b08,
    0xa9e39161,
    0x9af4361d,
    0xf0cfbc20,
    0xfc7b6bab,
    0x56033046,
    0x1f8d5d08,
    0x6bfb5fb1,
    0x8a5292ea,
    0x3d0739f7,
    0xebe5f17b,
    0x7527bac7,
    0x9e5fea2d,
    0x4f463f66,
    0x27cb09b7,
    0x6d367ecf,
    0x5a0a6d1f,
    0xef2f118b,
    0xde05980f,
    0x1ff897ff,
    0xbdf9283b,
    0x9c845f8b,
    0x835339f4,
    0x3991d639,
    0xb45f7e41,
    0xe99c7026,
    0x2ebb4484,
    0xe88235f5,
    0xb129a73e,
    0xfe1deb1c,
    0x09d1921c,
    0x06492eea,
    0x424dd2e0,
    0xb7246e3a,
    0xdebbc561,
    0xfe5163ab,
    0x3c439041,
    0xdb629599,
    0xf534ddc0,
    0xfc2757d1,
    0x4e441529,
    0xa2f9836e,
];

impl Emitter {
    fn double_scalar(x: &str) -> Double { Double { hi: x.to_owned(), lo: f64c(0.0) } }

    fn d_op(&mut self, op: &str, a: &str, b: &str) -> String {
        let r = self.reg(Scalar::F64);
        self.line(format!("{op}.f64 {r}, {a}, {b};"));
        r
    }

    fn d_fma(&mut self, a: &str, b: &str, c: &str) -> String {
        let r = self.reg(Scalar::F64);
        self.line(format!("fma.rn.f64 {r}, {a}, {b}, {c};"));
        r
    }

    fn d_neg(&mut self, a: &str) -> String {
        let r = self.reg(Scalar::F64);
        self.line(format!("neg.f64 {r}, {a};"));
        r
    }

    fn dd_add(&mut self, a: &Double, b: &Double) -> Double {
        let s = self.d_op("add.rn", &a.hi, &b.hi);
        let v = self.d_op("sub.rn", &s, &a.hi);
        let u = self.d_op("sub.rn", &s, &v);
        let u = self.d_op("sub.rn", &a.hi, &u);
        let v = self.d_op("sub.rn", &b.hi, &v);
        let e = self.d_op("add.rn", &u, &v);
        let tails = self.d_op("add.rn", &a.lo, &b.lo);
        let e = self.d_op("add.rn", &e, &tails);
        let hi = self.d_op("add.rn", &s, &e);
        let v = self.d_op("sub.rn", &hi, &s);
        let lo = self.d_op("sub.rn", &e, &v);
        Double { hi, lo }
    }

    fn dd_mul(&mut self, a: &Double, b: &Double) -> Double {
        let p = self.d_op("mul.rn", &a.hi, &b.hi);
        let negative = self.d_neg(&p);
        let e = self.d_fma(&a.hi, &b.hi, &negative);
        let e = self.d_fma(&a.hi, &b.lo, &e);
        let e = self.d_fma(&a.lo, &b.hi, &e);
        let e = self.d_fma(&a.lo, &b.lo, &e);
        let hi = self.d_op("add.rn", &p, &e);
        let v = self.d_op("sub.rn", &hi, &p);
        let lo = self.d_op("sub.rn", &e, &v);
        Double { hi, lo }
    }

    fn dd_poly(&mut self, x: &Double, coefficients: &[(u64, u64)]) -> Double {
        let mut r = pair(*coefficients.last().unwrap());
        for &coefficient in coefficients[..coefficients.len()-1].iter().rev() {
            r = self.dd_mul(&r, x);
            r = self.dd_add(&r, &pair(coefficient));
        }
        r
    }

    fn dd_log_positive(&mut self, source: &str) -> Double {
        let x = self.reg(Scalar::F64);
        let bits = self.reg(Scalar::U64);
        let exponent = self.reg(Scalar::I32);
        let correction = self.reg(Scalar::I32);
        let p = self.reg(Scalar::Pred);
        let normal = self.label();
        self.line(format!("mov.f64 {x}, {source};"));
        self.line(format!("mov.s32 {correction}, 0;"));
        self.line(format!("setp.ge.f64 {p}, {x}, 0d0010000000000000;"));
        self.line(format!("@{p} bra {normal};"));
        self.line(format!("mul.rn.f64 {x}, {x}, 0d4350000000000000;"));
        self.line(format!("mov.s32 {correction}, -54;"));
        self.line(format!("{normal}:"));
        self.line(format!("mov.b64 {bits}, {x};"));
        let e = self.reg(Scalar::U64);
        self.line(format!("shr.u64 {e}, {bits}, 52;"));
        self.line(format!("cvt.s32.u64 {exponent}, {e};"));
        self.line(format!("sub.s32 {exponent}, {exponent}, 1023;"));
        self.line(format!("add.s32 {exponent}, {exponent}, {correction};"));
        self.line(format!("and.b64 {bits}, {bits}, 0x000fffffffffffff;"));
        self.line(format!("or.b64 {bits}, {bits}, 0x3ff0000000000000;"));
        self.line(format!("mov.b64 {x}, {bits};"));
        self.line(format!("setp.gt.f64 {p}, {x}, {};", f64c(core::f64::consts::SQRT_2)));
        self.line(format!("@{p} mul.rn.f64 {x}, {x}, 0d3fe0000000000000;"));
        self.line(format!("@{p} add.s32 {exponent}, {exponent}, 1;"));
        let f = self.d_op("sub.rn", &x, &f64c(1.0));
        let denominator = self.dd_add(&Self::double_scalar(&x), &Self::double_scalar(&f64c(1.0)));
        let q = self.d_op("div.rn", &f, &denominator.hi);
        let minus_q = self.d_neg(&q);
        let residual = self.d_fma(&minus_q, &denominator.hi, &f);
        let residual = self.d_fma(&minus_q, &denominator.lo, &residual);
        let tail = self.d_op("div.rn", &residual, &denominator.hi);
        let s = self.dd_add(&Self::double_scalar(&q), &Self::double_scalar(&tail));
        let square = self.dd_mul(&s, &s);
        let polynomial = self.dd_poly(&square, &LOG_COEFFICIENTS);
        let local = self.dd_mul(&s, &polynomial);
        let local = self.dd_mul(&local, &Self::double_scalar(&f64c(2.0)));
        let ef = self.reg(Scalar::F64);
        self.line(format!("cvt.rn.f64.s32 {ef}, {exponent};"));
        let scale = self.dd_mul(&Self::double_scalar(&ef), &pair(LN2));
        self.dd_add(&scale, &local)
    }

    fn dd_div(&mut self, a: &Double, b: &Double) -> Double {
        let q = self.d_op("div.rn", &a.hi, &b.hi);
        let minus_q = self.d_neg(&q);
        let residual = self.d_fma(&minus_q, &b.hi, &a.hi);
        let residual = self.d_op("add.rn", &residual, &a.lo);
        let residual = self.d_fma(&minus_q, &b.lo, &residual);
        let tail = self.d_op("div.rn", &residual, &b.hi);
        self.dd_add(&Self::double_scalar(&q), &Self::double_scalar(&tail))
    }

    fn small_expm1_f64(&mut self, x: &str) -> Double {
        let x = Self::double_scalar(x);
        let polynomial = self.dd_poly(&x, &EXP_COEFFICIENTS[1..]);
        self.dd_mul(&x, &polynomial)
    }

    fn sign_f64(&mut self, destination: &str, magnitude: &str, source: &str) {
        let bits = self.reg(Scalar::U64);
        let sign = self.reg(Scalar::U64);
        self.line(format!("mov.b64 {sign}, {source};"));
        self.line(format!("and.b64 {sign}, {sign}, 0x8000000000000000;"));
        self.line(format!("mov.b64 {bits}, {magnitude};"));
        self.line(format!("and.b64 {bits}, {bits}, 0x7fffffffffffffff;"));
        self.line(format!("or.b64 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b64 {destination}, {bits};"));
    }

    pub(super) fn tanh_f64(&mut self, destination: &str, source: &str) {
        let x = self.reg(Scalar::F64);
        let result = self.reg(Scalar::F64);
        let p = self.reg(Scalar::Pred);
        let end = self.label();
        let ordinary = self.label();
        let sign = self.label();
        self.line(format!("mov.f64 {destination}, {source};"));
        self.line(format!("setp.nan.f64 {p}, {source}, {source};"));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("abs.f64 {x}, {source};"));
        self.line(format!("setp.lt.f64 {p}, {x}, {};", f64c(2.0f64.powi(-28))));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("mov.f64 {result}, 0d3ff0000000000000;"));
        self.line(format!("setp.ge.f64 {p}, {x}, {};", f64c(20.0)));
        self.line(format!("@{p} bra {sign};"));
        let twice = self.d_op("add.rn", &x, &x);
        self.line(format!("setp.gt.f64 {p}, {x}, 0d3fe0000000000000;"));
        self.line(format!("@{p} bra {ordinary};"));
        let t = self.small_expm1_f64(&twice);
        let denominator = self.dd_add(&t, &Self::double_scalar(&f64c(2.0)));
        let quotient = self.dd_div(&t, &denominator);
        self.line(format!("add.rn.f64 {result}, {}, {};", quotient.hi, quotient.lo));
        self.line(format!("bra {sign};"));
        self.line(format!("{ordinary}:"));
        let exponential = self.reg(Scalar::F64);
        self.exponential_f64(&exponential, &twice);
        let denominator = self.d_op("add.rn", &exponential, &f64c(1.0));
        let inverse = self.d_op("div.rn", &f64c(2.0), &denominator);
        self.line(format!("sub.rn.f64 {result}, 0d3ff0000000000000, {inverse};"));
        self.line(format!("{sign}:"));
        self.sign_f64(destination, &result, source);
        self.line(format!("{end}:"));
    }

    pub(super) fn hyperbolic_f64(&mut self, destination: &str, source: &str, cosine: bool) {
        let x = self.reg(Scalar::F64);
        let result = self.reg(Scalar::F64);
        let p = self.reg(Scalar::Pred);
        let end = self.label();
        let ordinary = self.label();
        let scaled = self.label();
        let finish = self.label();
        self.line(format!("abs.f64 {x}, {source};"));
        self.line(format!("add.rn.f64 {destination}, {source}, {source};"));
        if cosine { self.line(format!("abs.f64 {destination}, {destination};")); }
        self.line(format!("setp.geu.f64 {p}, {x}, 0d7ff0000000000000;"));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("mov.f64 {result}, {};", if cosine { f64c(1.0) } else { x.clone() }));
        self.line(format!("setp.lt.f64 {p}, {x}, {};", f64c(2.0f64.powi(-28))));
        self.line(format!("@{p} bra {finish};"));
        self.line(format!("setp.gt.f64 {p}, {x}, 0d3fe0000000000000;"));
        self.line(format!("@{p} bra {ordinary};"));
        let t = self.small_expm1_f64(&x);
        let one = Self::double_scalar(&f64c(1.0));
        let denominator = self.dd_add(&t, &one);
        let value = if cosine {
            let square = self.dd_mul(&t, &t);
            let ratio = self.dd_div(&square, &denominator);
            let half = self.dd_mul(&ratio, &Self::double_scalar(&f64c(0.5)));
            self.dd_add(&one, &half)
        } else {
            let ratio = self.dd_div(&t, &denominator);
            let sum = self.dd_add(&t, &ratio);
            self.dd_mul(&sum, &Self::double_scalar(&f64c(0.5)))
        };
        self.line(format!("add.rn.f64 {result}, {}, {};", value.hi, value.lo));
        self.line(format!("bra {finish};"));
        self.line(format!("{ordinary}:"));
        self.line(format!("setp.gt.f64 {p}, {x}, {};", f64c(20.0)));
        self.line(format!("@{p} bra {scaled};"));
        let exponential = self.reg(Scalar::F64);
        self.exponential_f64(&exponential, &x);
        let inverse = self.d_op("div.rn", &f64c(1.0), &exponential);
        let sum = self.d_op(if cosine { "add.rn" } else { "sub.rn" }, &exponential, &inverse);
        self.line(format!("mul.rn.f64 {result}, {sum}, 0d3fe0000000000000;"));
        self.line(format!("bra {finish};"));
        self.line(format!("{scaled}:"));
        let minus_ln2 = pair((LN2.0 ^ (1u64 << 63), LN2.1 ^ (1u64 << 63)));
        let argument = self.dd_add(&Self::double_scalar(&x), &minus_ln2);
        self.exp_f64_pair(&result, &argument);
        self.line(format!("{finish}:"));
        if cosine { self.line(format!("mov.f64 {destination}, {result};")); }
        else { self.sign_f64(destination, &result, source); }
        self.line(format!("{end}:"));
    }

    pub(super) fn inverse_hyperbolic_f64(&mut self, destination: &str, source: &str, cosine: bool) {
        let x = self.reg(Scalar::F64);
        let result = self.reg(Scalar::F64);
        let p = self.reg(Scalar::Pred);
        let large = self.label();
        let invalid = self.label();
        let finish = self.label();
        let end = self.label();
        self.line(format!("mov.f64 {destination}, {source};"));
        self.line(format!("setp.nan.f64 {p}, {source}, {source};"));
        self.line(format!("@{p} bra {end};"));
        if cosine {
            self.line(format!("setp.lt.f64 {p}, {source}, 0d3ff0000000000000;"));
            self.line(format!("@{p} bra {invalid};"));
        }
        self.line(format!("abs.f64 {x}, {source};"));
        self.line(format!("setp.eq.f64 {p}, {x}, 0d7ff0000000000000;"));
        self.line(format!("@{p} bra {end};"));
        if !cosine {
            self.line(format!("setp.lt.f64 {p}, {x}, {};", f64c(2.0f64.powi(-28))));
            self.line(format!("@{p} bra {end};"));
        }
        self.line(format!("setp.ge.f64 {p}, {x}, {};", f64c(2.0f64.powi(28))));
        self.line(format!("@{p} bra {large};"));
        let root = self.reg(Scalar::F64);
        let argument = if cosine {
            let left = self.d_op("sub.rn", &x, &f64c(1.0));
            let right = self.d_op("add.rn", &x, &f64c(1.0));
            let radicand = self.d_op("mul.rn", &left, &right);
            self.line(format!("sqrt.rn.f64 {root}, {radicand};"));
            self.d_op("add.rn", &left, &root)
        } else {
            let radicand = self.d_fma(&x, &x, &f64c(1.0));
            self.line(format!("sqrt.rn.f64 {root}, {radicand};"));
            let denominator = self.d_op("add.rn", &root, &f64c(1.0));
            let square = self.d_op("mul.rn", &x, &x);
            let correction = self.d_op("div.rn", &square, &denominator);
            self.d_op("add.rn", &x, &correction)
        };
        self.log1p_f64(&result, &argument);
        self.line(format!("bra {finish};"));
        self.line(format!("{large}:"));
        let logarithm = self.dd_log_positive(&x);
        let sum = self.dd_add(&logarithm, &pair(LN2));
        self.line(format!("add.rn.f64 {result}, {}, {};", sum.hi, sum.lo));
        self.line(format!("{finish}:"));
        if cosine { self.line(format!("mov.f64 {destination}, {result};")); }
        else { self.sign_f64(destination, &result, source); }
        self.line(format!("bra {end};"));
        self.line(format!("{invalid}:"));
        self.line(format!("mov.f64 {destination}, 0d7ff8000000000000;"));
        self.line(format!("{end}:"));
    }

    pub(super) fn atanh_f64(&mut self, destination: &str, source: &str) {
        let x = self.reg(Scalar::F64);
        let result = self.reg(Scalar::F64);
        let p = self.reg(Scalar::Pred);
        let invalid = self.label();
        let finish = self.label();
        let end = self.label();
        self.line(format!("mov.f64 {destination}, {source};"));
        self.line(format!("setp.nan.f64 {p}, {source}, {source};"));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("abs.f64 {x}, {source};"));
        self.line(format!("setp.gt.f64 {p}, {x}, 0d3ff0000000000000;"));
        self.line(format!("@{p} bra {invalid};"));
        self.line(format!("setp.lt.f64 {p}, {x}, {};", f64c(2.0f64.powi(-28))));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("mov.f64 {result}, 0d7ff0000000000000;"));
        self.line(format!("setp.eq.f64 {p}, {x}, 0d3ff0000000000000;"));
        self.line(format!("@{p} bra {finish};"));
        let twice = self.d_op("add.rn", &x, &x);
        let denominator = self.d_op("sub.rn", &f64c(1.0), &x);
        let argument = self.d_op("div.rn", &twice, &denominator);
        self.log1p_f64(&result, &argument);
        self.line(format!("mul.rn.f64 {result}, {result}, 0d3fe0000000000000;"));
        self.line(format!("{finish}:"));
        self.sign_f64(destination, &result, source);
        self.line(format!("bra {end};"));
        self.line(format!("{invalid}:"));
        self.line(format!("mov.f64 {destination}, 0d7ff8000000000000;"));
        self.line(format!("{end}:"));
    }

    pub(super) fn logarithm_f64(&mut self, destination: &str, source: &str) {
        let p = self.reg(Scalar::Pred);
        let end = self.label();
        let zero = self.label();
        let invalid = self.label();
        self.line(format!("add.rn.f64 {destination}, {source}, {source};"));
        self.line(format!("setp.nan.f64 {p}, {source}, {source};"));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("setp.eq.f64 {p}, {source}, 0d0000000000000000;"));
        self.line(format!("@{p} bra {zero};"));
        self.line(format!("setp.lt.f64 {p}, {source}, 0d0000000000000000;"));
        self.line(format!("@{p} bra {invalid};"));
        self.line(format!("setp.eq.f64 {p}, {source}, 0d7ff0000000000000;"));
        self.line(format!("@{p} bra {end};"));
        let value = self.dd_log_positive(source);
        self.line(format!("add.rn.f64 {destination}, {}, {};", value.hi, value.lo));
        self.line(format!("bra {end};"));
        self.line(format!("{zero}:"));
        self.line(format!("mov.f64 {destination}, 0dfff0000000000000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{invalid}:"));
        self.line(format!("mov.f64 {destination}, 0d7ff8000000000000;"));
        self.line(format!("{end}:"));
    }

    pub(super) fn log1p_f64(&mut self, destination: &str, source: &str) {
        let p = self.reg(Scalar::Pred);
        let end = self.label();
        let zero = self.label();
        let invalid = self.label();
        self.line(format!("mov.f64 {destination}, {source};"));
        self.line(format!("setp.nan.f64 {p}, {source}, {source};"));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("setp.eq.f64 {p}, {source}, 0d7ff0000000000000;"));
        self.line(format!("@{p} bra {end};"));
        let magnitude = self.reg(Scalar::F64);
        self.line(format!("abs.f64 {magnitude}, {source};"));
        self.line(format!("setp.lt.f64 {p}, {magnitude}, {};", f64c(2.0f64.powi(-54))));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("setp.eq.f64 {p}, {source}, 0dbff0000000000000;"));
        self.line(format!("@{p} bra {zero};"));
        self.line(format!("setp.lt.f64 {p}, {source}, 0dbff0000000000000;"));
        self.line(format!("@{p} bra {invalid};"));
        let sum = self.dd_add(&Self::double_scalar(source), &Self::double_scalar(&f64c(1.0)));
        let logarithm = self.dd_log_positive(&sum.hi);
        let correction = self.d_op("div.rn", &sum.lo, &sum.hi);
        let square = self.d_op("mul.rn", &correction, &correction);
        let correction = Double { hi: correction, lo: self.d_op("mul.rn", &square, &f64c(-0.5)) };
        let result = self.dd_add(&logarithm, &correction);
        self.line(format!("add.rn.f64 {destination}, {}, {};", result.hi, result.lo));
        self.line(format!("bra {end};"));
        self.line(format!("{zero}:"));
        self.line(format!("mov.f64 {destination}, 0dfff0000000000000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{invalid}:"));
        self.line(format!("mov.f64 {destination}, 0d7ff8000000000000;"));
        self.line(format!("{end}:"));
    }

    pub(super) fn exponential_f64(&mut self, destination: &str, source: &str) {
        self.exp_f64_pair(destination, &Self::double_scalar(source));
    }

    fn exp_f64_pair(&mut self, destination: &str, source: &Double) {
        let p = self.reg(Scalar::Pred);
        let end = self.label();
        let overflow = self.label();
        let underflow = self.label();
        let subnormal = self.label();
        let upper = self.label();
        self.line(format!("add.rn.f64 {destination}, {}, {};", source.hi, source.lo));
        self.line(format!("setp.nan.f64 {p}, {}, {};", source.hi, source.lo));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("setp.ge.f64 {p}, {}, {};", source.hi, f64c(710.0)));
        self.line(format!("@{p} bra {overflow};"));
        self.line(format!("setp.le.f64 {p}, {}, {};", source.hi, f64c(-746.0)));
        self.line(format!("@{p} bra {underflow};"));
        let scaled = self.d_op("mul.rn", &source.hi, &f64c(core::f64::consts::LOG2_E));
        let k = self.reg(Scalar::I32);
        let kf = self.reg(Scalar::F64);
        self.line(format!("cvt.rni.s32.f64 {k}, {scaled};"));
        self.line(format!("cvt.rn.f64.s32 {kf}, {k};"));
        let negative_k = self.d_neg(&kf);
        let offset = self.dd_mul(&Self::double_scalar(&negative_k), &pair(LN2));
        let r = self.dd_add(source, &offset);
        let value = self.dd_poly(&r, &EXP_COEFFICIENTS);
        let rounded = self.d_op("add.rn", &value.hi, &value.lo);
        self.line(format!("setp.le.s32 {p}, {k}, -1022;"));
        self.line(format!("@{p} bra {subnormal};"));
        self.line(format!("setp.eq.s32 {p}, {k}, 1024;"));
        self.line(format!("@{p} bra {upper};"));
        let e = self.reg(Scalar::I32);
        let bits = self.reg(Scalar::U64);
        let factor = self.reg(Scalar::F64);
        self.line(format!("add.s32 {e}, {k}, 1023;"));
        self.line(format!("cvt.u64.s32 {bits}, {e};"));
        self.line(format!("shl.b64 {bits}, {bits}, 52;"));
        self.line(format!("mov.b64 {factor}, {bits};"));
        self.line(format!("mul.rn.f64 {destination}, {rounded}, {factor};"));
        self.line(format!("bra {end};"));
        self.line(format!("{upper}:"));
        self.line(format!("mul.rn.f64 {destination}, {rounded}, 0d7fe0000000000000;"));
        self.line(format!("mul.rn.f64 {destination}, {destination}, 0d4000000000000000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{subnormal}:"));
        self.line(format!("add.s32 {e}, {k}, 2097;"));
        self.line(format!("cvt.u64.s32 {bits}, {e};"));
        self.line(format!("shl.b64 {bits}, {bits}, 52;"));
        self.line(format!("mov.b64 {factor}, {bits};"));
        let units = self.d_op("mul.rn", &value.hi, &factor);
        let tail = self.d_op("mul.rn", &value.lo, &factor);
        let integer = self.reg(Scalar::U64);
        let integer_f = self.reg(Scalar::F64);
        let odd = self.reg(Scalar::U64);
        let half = self.reg(Scalar::Pred);
        let positive_tail = self.reg(Scalar::Pred);
        let tie = self.reg(Scalar::Pred);
        self.line(format!("cvt.rmi.u64.f64 {integer}, {units};"));
        self.line(format!("cvt.rn.f64.u64 {integer_f}, {integer};"));
        let fraction = self.d_op("sub.rn", &units, &integer_f);
        self.line(format!("setp.gt.f64 {p}, {fraction}, 0d3fe0000000000000;"));
        self.line(format!("setp.eq.f64 {half}, {fraction}, 0d3fe0000000000000;"));
        self.line(format!("setp.gt.f64 {positive_tail}, {tail}, 0d0000000000000000;"));
        self.line(format!("and.b64 {odd}, {integer}, 1;"));
        self.line(format!("setp.ne.u64 {tie}, {odd}, 0;"));
        let zero_tail = self.reg(Scalar::Pred);
        self.line(format!("setp.eq.f64 {zero_tail}, {tail}, 0d0000000000000000;"));
        self.line(format!("and.pred {tie}, {tie}, {zero_tail};"));
        self.line(format!("or.pred {positive_tail}, {positive_tail}, {tie};"));
        self.line(format!("and.pred {half}, {half}, {positive_tail};"));
        self.line(format!("or.pred {p}, {p}, {half};"));
        self.line(format!("@{p} add.u64 {integer}, {integer}, 1;"));
        self.line(format!("mov.b64 {destination}, {integer};"));
        self.line(format!("bra {end};"));
        self.line(format!("{overflow}:"));
        self.line(format!("mov.f64 {destination}, 0d7ff0000000000000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{underflow}:"));
        self.line(format!("mov.f64 {destination}, 0d0000000000000000;"));
        self.line(format!("{end}:"));
    }

    pub(super) fn floating_power_f64(&mut self, out: Variable, op: BinaryOperator) -> Result<()> {
        let x = self.value(op.lhs)?;
        let y = self.value(op.rhs)?;
        let destination = self.destination(out)?;
        let result = self.reg(Scalar::F64);
        let ax = self.reg(Scalar::F64);
        let ay = self.reg(Scalar::F64);
        let p = self.reg(Scalar::Pred);
        let odd = self.reg(Scalar::Pred);
        let integral = self.reg(Scalar::Pred);
        let n = self.reg(Scalar::U64);
        let bit = self.reg(Scalar::U64);
        let one = self.label();
        let nan = self.label();
        let zero = self.label();
        let infinity = self.label();
        let infinite_y = self.label();
        let signed = self.label();
        let finish = self.label();
        let parity = self.label();
        self.line(format!("setp.eq.f64 {p}, {y}, 0d0000000000000000;"));
        self.line(format!("@{p} bra {one};"));
        self.line(format!("setp.eq.f64 {p}, {x}, 0d3ff0000000000000;"));
        self.line(format!("@{p} bra {one};"));
        self.line(format!("setp.nan.f64 {p}, {x}, {y};"));
        self.line(format!("@{p} bra {nan};"));
        self.line(format!("abs.f64 {ax}, {x};"));
        self.line(format!("abs.f64 {ay}, {y};"));
        self.line(format!("setp.eq.f64 {p}, {ay}, 0d7ff0000000000000;"));
        self.line(format!("@{p} bra {infinite_y};"));
        let rounded = self.reg(Scalar::F64);
        self.line(format!("cvt.rzi.f64.f64 {rounded}, {y};"));
        self.line(format!("setp.eq.f64 {integral}, {rounded}, {y};"));
        self.line(format!("setp.ne.u32 {odd}, 0, 0;"));
        self.line(format!("setp.ge.f64 {p}, {ay}, 0d4340000000000000;"));
        self.line(format!("@{p} bra {parity};"));
        self.line(format!("cvt.rzi.u64.f64 {n}, {ay};"));
        self.line(format!("and.b64 {bit}, {n}, 1;"));
        self.line(format!("setp.ne.u64 {odd}, {bit}, 0;"));
        self.line(format!("and.pred {odd}, {odd}, {integral};"));
        self.line(format!("{parity}:"));
        self.line(format!("setp.eq.f64 {p}, {ax}, 0d0000000000000000;"));
        self.line(format!("@{p} bra {zero};"));
        self.line(format!("setp.eq.f64 {p}, {ax}, 0d7ff0000000000000;"));
        self.line(format!("@{p} bra {infinity};"));
        self.line(format!("setp.ge.f64 {p}, {x}, 0d0000000000000000;"));
        self.line(format!("or.pred {p}, {p}, {integral};"));
        self.line(format!("@!{p} bra {nan};"));
        let logarithm = self.dd_log_positive(&ax);
        let estimate = self.d_op("mul.rn", &logarithm.hi, &y);
        let large = self.label();
        let small = self.label();
        self.line(format!("setp.ge.f64 {p}, {estimate}, {};", f64c(710.0)));
        self.line(format!("@{p} bra {large};"));
        self.line(format!("setp.le.f64 {p}, {estimate}, {};", f64c(-746.0)));
        self.line(format!("@{p} bra {small};"));
        let exponent = self.dd_mul(&logarithm, &Self::double_scalar(&y));
        self.exp_f64_pair(&result, &exponent);
        self.line(format!("bra {signed};"));
        self.line(format!("{large}:"));
        self.line(format!("mov.f64 {result}, 0d7ff0000000000000;"));
        self.line(format!("bra {signed};"));
        self.line(format!("{small}:"));
        self.line(format!("mov.f64 {result}, 0d0000000000000000;"));
        self.line(format!("bra {signed};"));
        self.line(format!("{zero}:"));
        self.line(format!("setp.lt.f64 {p}, {y}, 0d0000000000000000;"));
        self.line(format!("selp.f64 {result}, 0d7ff0000000000000, 0d0000000000000000, {p};"));
        self.line(format!("bra {signed};"));
        self.line(format!("{infinity}:"));
        self.line(format!("setp.lt.f64 {p}, {y}, 0d0000000000000000;"));
        self.line(format!("selp.f64 {result}, 0d0000000000000000, 0d7ff0000000000000, {p};"));
        self.line(format!("bra {signed};"));
        self.line(format!("{infinite_y}:"));
        self.line(format!("setp.eq.f64 {p}, {ax}, 0d3ff0000000000000;"));
        self.line(format!("@{p} bra {one};"));
        self.line(format!("setp.gt.f64 {p}, {ax}, 0d3ff0000000000000;"));
        self.line(format!("setp.lt.f64 {integral}, {y}, 0d0000000000000000;"));
        self.line(format!("xor.pred {p}, {p}, {integral};"));
        self.line(format!("selp.f64 {result}, 0d7ff0000000000000, 0d0000000000000000, {p};"));
        self.line(format!("bra {finish};"));
        self.line(format!("{signed}:"));
        self.line(format!("mov.b64 {bit}, {x};"));
        self.line(format!("and.b64 {bit}, {bit}, 0x8000000000000000;"));
        self.line(format!("@!{odd} mov.u64 {bit}, 0;"));
        self.line(format!("mov.b64 {n}, {result};"));
        self.line(format!("or.b64 {n}, {n}, {bit};"));
        self.line(format!("mov.b64 {result}, {n};"));
        self.line(format!("bra {finish};"));
        self.line(format!("{nan}:"));
        self.line(format!("mov.f64 {result}, 0d7ff8000000000000;"));
        self.line(format!("bra {finish};"));
        self.line(format!("{one}:"));
        self.line(format!("mov.f64 {result}, 0d3ff0000000000000;"));
        self.line(format!("{finish}:"));
        self.line(format!("mov.f64 {destination}, {result};"));
        Ok(())
    }

    fn reduce_angle_f64(&mut self, magnitude: &str, quadrant: &str, remainder: &Double) {
        let bits = self.reg(Scalar::U64);
        let wide = self.reg(Scalar::U64);
        let ml = self.reg(Scalar::U32);
        let mh = self.reg(Scalar::U32);
        let exponent = self.reg(Scalar::I32);
        let position = self.reg(Scalar::I32);
        let shift = self.reg(Scalar::U32);
        let index = self.reg(Scalar::I32);
        let p = self.reg(Scalar::Pred);
        self.line(format!("mov.b64 {bits}, {magnitude};"));
        self.line(format!("shr.u64 {wide}, {bits}, 52;"));
        self.line(format!("cvt.s32.u64 {exponent}, {wide};"));
        self.line(format!("sub.s32 {exponent}, {exponent}, 1023;"));
        self.line(format!("sub.s32 {position}, 1588, {exponent};"));
        self.line(format!("and.b32 {shift}, {position}, 31;"));
        self.line(format!("shr.s32 {index}, {position}, 5;"));
        self.line(format!("sub.s32 {index}, {index}, 6;"));
        self.line(format!("and.b64 {bits}, {bits}, 0x000fffffffffffff;"));
        self.line(format!("or.b64 {bits}, {bits}, 0x0010000000000000;"));
        self.line(format!("cvt.u32.u64 {ml}, {bits};"));
        self.line(format!("shr.u64 {wide}, {bits}, 32;"));
        self.line(format!("cvt.u32.u64 {mh}, {wide};"));
        let window: Vec<_> = (0..8).map(|_| self.reg(Scalar::U32)).collect();
        for w in &window { self.line(format!("mov.u32 {w}, 0;")); }
        let targets: Vec<_> = (0..8).map(|j| {
            let t = self.reg(Scalar::I32);
            self.line(format!("add.s32 {t}, {index}, {j};"));
            t
        }).collect();
        let carry = self.reg(Scalar::U64);
        let low = self.reg(Scalar::U64);
        let high = self.reg(Scalar::U64);
        let overflow = self.reg(Scalar::U64);
        let word = self.reg(Scalar::U32);
        self.line(format!("mov.u64 {carry}, 0;"));
        for i in 0..50 {
            let a = TWO_OVER_PI.get(i).copied().unwrap_or(0);
            let b = if i == 0 { 0 } else { TWO_OVER_PI.get(i-1).copied().unwrap_or(0) };
            self.line(format!("mul.wide.u32 {low}, {ml}, 0x{a:08x};"));
            self.line(format!("mul.wide.u32 {high}, {mh}, 0x{b:08x};"));
            self.line(format!("add.cc.u64 {low}, {low}, {carry};"));
            self.line(format!("addc.u64 {overflow}, 0, 0;"));
            self.line(format!("add.cc.u64 {low}, {low}, {high};"));
            self.line(format!("addc.u64 {overflow}, {overflow}, 0;"));
            self.line(format!("cvt.u32.u64 {word}, {low};"));
            self.line(format!("shr.u64 {carry}, {low}, 32;"));
            self.line(format!("shl.b64 {overflow}, {overflow}, 32;"));
            self.line(format!("add.u64 {carry}, {carry}, {overflow};"));
            for (target, w) in targets.iter().zip(&window) {
                self.line(format!("setp.eq.s32 {p}, {target}, {i};"));
                self.line(format!("@{p} mov.u32 {w}, {word};"));
            }
        }
        let fraction: Vec<_> = (0..6).map(|i| {
            let r = self.reg(Scalar::U32);
            self.line(format!("shf.r.clamp.b32 {r}, {}, {}, {shift};", window[i], window[i+1]));
            r
        }).collect();
        self.line(format!("shf.r.clamp.b32 {quadrant}, {}, {}, {shift};", window[6], window[7]));
        self.line(format!("setp.ge.u32 {p}, {}, 0x80000000;", fraction[5]));
        self.line(format!("@{p} add.u32 {quadrant}, {quadrant}, 1;"));
        self.line(format!("and.b32 {quadrant}, {quadrant}, 3;"));
        for f in &fraction { self.line(format!("@{p} not.b32 {f}, {f};")); }
        self.line(format!("@{p} add.cc.u32 {}, {}, 1;", fraction[0], fraction[0]));
        for f in &fraction[1..] { self.line(format!("@{p} addc.cc.u32 {f}, {f}, 0;")); }
        let mut value = Self::double_scalar(&f64c(0.0));
        for i in (0..6).rev() {
            let r = self.reg(Scalar::F64);
            self.line(format!("cvt.rn.f64.u32 {r}, {};", fraction[i]));
            self.line(format!("mul.rn.f64 {r}, {r}, {};", f64c(2.0f64.powi(-32*(6-i as i32)))));
            value = self.dd_add(&value, &Self::double_scalar(&r));
        }
        value = self.dd_mul(&value, &pair(PIO2));
        self.line(format!("mov.f64 {}, {};", remainder.hi, value.hi));
        self.line(format!("mov.f64 {}, {};", remainder.lo, value.lo));
        self.line(format!("@{p} neg.f64 {}, {};", remainder.hi, remainder.hi));
        self.line(format!("@{p} neg.f64 {}, {};", remainder.lo, remainder.lo));
    }

    pub(super) fn trigonometry_f64(&mut self, destination: &str, source: &str, cosine: bool) {
        let magnitude = self.reg(Scalar::F64);
        let p = self.reg(Scalar::Pred);
        let end = self.label();
        let invalid = self.label();
        let reduced = self.label();
        let local = self.label();
        let use_cosine = self.label();
        let sign = self.label();
        let quadrant = self.reg(Scalar::U32);
        let remainder = Double { hi: self.reg(Scalar::F64), lo: self.reg(Scalar::F64) };
        self.line(format!("abs.f64 {magnitude}, {source};"));
        self.line(format!("setp.geu.f64 {p}, {magnitude}, 0d7ff0000000000000;"));
        self.line(format!("@{p} bra {invalid};"));
        self.line(format!("mov.f64 {destination}, {};", if cosine { f64c(1.0) } else { source.to_owned() }));
        self.line(format!("setp.lt.f64 {p}, {magnitude}, {};", f64c(2.0f64.powi(-27))));
        self.line(format!("@{p} bra {end};"));
        self.line(format!("setp.le.f64 {p}, {magnitude}, {};", f64c(core::f64::consts::FRAC_PI_4)));
        self.line(format!("@{p} bra {local};"));
        self.reduce_angle_f64(&magnitude, &quadrant, &remainder);
        self.line(format!("setp.lt.f64 {p}, {source}, 0d0000000000000000;"));
        self.line(format!("@{p} neg.f64 {}, {};", remainder.hi, remainder.hi));
        self.line(format!("@{p} neg.f64 {}, {};", remainder.lo, remainder.lo));
        self.line(format!("@{p} sub.u32 {quadrant}, 4, {quadrant};"));
        self.line(format!("and.b32 {quadrant}, {quadrant}, 3;"));
        self.line(format!("bra {reduced};"));
        self.line(format!("{local}:"));
        self.line(format!("mov.f64 {}, {source};", remainder.hi));
        self.line(format!("mov.f64 {}, 0d0000000000000000;", remainder.lo));
        self.line(format!("mov.u32 {quadrant}, 0;"));
        self.line(format!("{reduced}:"));
        let odd = self.reg(Scalar::U32);
        self.line(format!("and.b32 {odd}, {quadrant}, 1;"));
        self.line(format!("setp.eq.u32 {p}, {odd}, {};", if cosine { 0 } else { 1 }));
        let square = self.dd_mul(&remainder, &remainder);
        self.line(format!("@{p} bra {use_cosine};"));
        let sin_poly = self.dd_poly(&square, &SIN_COEFFICIENTS);
        let sine = self.dd_mul(&remainder, &sin_poly);
        self.line(format!("add.rn.f64 {destination}, {}, {};", sine.hi, sine.lo));
        self.line(format!("bra {sign};"));
        self.line(format!("{use_cosine}:"));
        let cosine_value = self.dd_poly(&square, &COS_COEFFICIENTS);
        self.line(format!("add.rn.f64 {destination}, {}, {};", cosine_value.hi, cosine_value.lo));
        self.line(format!("{sign}:"));
        self.line(format!("add.u32 {odd}, {quadrant}, {};", if cosine { 1 } else { 0 }));
        self.line(format!("and.b32 {odd}, {odd}, 2;"));
        self.line(format!("setp.ne.u32 {p}, {odd}, 0;"));
        self.line(format!("@{p} neg.f64 {destination}, {destination};"));
        self.line(format!("bra {end};"));
        self.line(format!("{invalid}:"));
        self.line(format!("sub.rn.f64 {destination}, {source}, {source};"));
        self.line(format!("{end}:"));
    }
}
