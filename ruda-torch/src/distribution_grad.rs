// Adapted from ATen/native/Distributions.h.
// PyTorch and Cephes licenses retained in bessel.rs and gamma.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn digamma<F: Float>(input: F) -> F {
    let zero = F::new(0.0);
    let one = F::new(1.0);
    let mut x = input;
    let mut result = F::new(f32::INFINITY);
    if x != zero && !(x < zero && x == x.floor()) {
        let mut additional = zero;
        if x < zero {
            let pi = F::cast_from(3.14159265358979323846f64);
            additional = -pi / (pi * x).tan();
            x = one - x;
        }
        result = zero;
        while x < F::new(10.0) { result -= one / x; x += one; }
        if x == F::new(10.0) {
            result = result + F::cast_from(2.25175258906672110764f64) + additional;
        } else {
            let mut correction = zero;
            if x < F::cast_from(1.0e17f32) {
                let z = F::cast_from(1.0f64 / f64::cast_from(x * x));
                let mut poly = F::cast_from(8.33333333333333333333e-2f64);
                poly = poly * z + F::cast_from(-2.10927960927960927961e-2f64);
                poly = poly * z + F::cast_from(7.57575757575757575758e-3f64);
                poly = poly * z + F::cast_from(-4.16666666666666666667e-3f64);
                poly = poly * z + F::cast_from(3.96825396825396825397e-3f64);
                poly = poly * z + F::cast_from(-8.33333333333333333333e-3f64);
                poly = poly * z + F::cast_from(8.33333333333333333333e-2f64);
                correction = z * poly;
            }
            result = result + x.ln() - F::new(0.5) / x - correction + additional;
        }
    }
    result
}

#[ruda]
fn gamma_grad<F: Float>(alpha: F, x: F) -> F {
    let one = F::new(1.0);
    let mut result = F::new(0.0);
    if x < F::cast_from(0.8f32) {
        let mut numerator = one;
        let mut denominator = alpha;
        let mut series1 = numerator / denominator;
        let mut series2 = numerator / (denominator * denominator);
        for i in 1u32..6u32 {
            numerator *= -x / F::cast_from(i);
            denominator += one;
            series1 += numerator / denominator;
            series2 += numerator / (denominator * denominator);
        }
        let power = x.powf(alpha);
        let pdf = x.powf(alpha - one) * (-x).exp();
        let cdf = power * series1;
        let cdf_alpha = (x.ln() - digamma(alpha)) * cdf - power * series2;
        result = -cdf_alpha / pdf;
        if result != result { result = F::new(0.0); }
    } else if alpha > F::new(8.0) {
        if F::cast_from(0.9f32) * alpha <= x && x <= F::cast_from(1.1f32) * alpha {
            let numerator1 = one + F::new(24.0) * alpha * (one + F::new(12.0) * alpha);
            let numerator2 = F::new(1440.0) * (alpha * alpha)
                + F::new(6.0) * x * (F::new(53.0) - F::new(120.0) * x)
                - F::new(65.0) * x * x / alpha + alpha * (F::new(107.0) + F::new(3600.0) * x);
            let denominator = F::new(1244160.0) * (alpha * alpha) * (alpha * alpha);
            result = numerator1 * numerator2 / denominator;
        } else {
            let denominator = (F::new(8.0) * alpha).sqrt();
            let term2 = denominator / (alpha - x);
            let term3 = (x - alpha - alpha * (x / alpha).ln()).powf(F::new(-1.5));
            let mut term23 = term2 + term3;
            if x < alpha { term23 = term2 - term3; }
            let term1 = (x / alpha).ln() * term23
                - (F::new(2.0) / alpha).sqrt() * (alpha + x) / ((alpha - x) * (alpha - x));
            let stirling = one + one / (F::new(12.0) * alpha) * (one + one / (F::new(24.0) * alpha));
            result = -stirling * (x * term1) / denominator;
        }
    } else {
        let u = (x / alpha).ln();
        let v = alpha.ln();
        let coefficients = Array::<F>::from_data(comptime!(GAMMA_COEFFICIENTS.to_vec()));
        let mut cv = Array::<F>::new(8usize);
        for i in 0usize..8usize {
            cv[i] = coefficients[i] + u * (coefficients[8 + i] + u * coefficients[16 + i]);
        }
        let p = cv[0] + v * (cv[1] + v * (cv[2] + v * cv[3]));
        let q = cv[4] + v * (cv[5] + v * (cv[6] + v * cv[7]));
        result = (p / q).exp();
    }
    result
}

#[ruda]
fn beta_alpha_small<F: Float>(x: F, alpha: F, beta: F) -> F {
    let one = F::new(1.0);
    let factor = digamma(alpha) - digamma(alpha + beta) - x.ln();
    let mut numerator = one;
    let mut series = numerator / alpha * (factor + one / alpha);
    for i in 1u32..11u32 {
        let index = F::cast_from(i);
        numerator *= (index - beta) * x / index;
        let denominator = alpha + index;
        series += numerator / denominator * (factor + one / denominator);
    }
    let mut result = x * (one - x).powf(-beta) * series;
    if result != result { result = F::new(0.0); }
    result
}

#[ruda]
fn beta_beta_small<F: Float>(x: F, alpha: F, beta: F) -> F {
    let one = F::new(1.0);
    let factor = digamma(alpha + beta) - digamma(beta);
    let mut numerator = one;
    let mut betas = one;
    let mut dbetas = F::new(0.0);
    let mut series = factor / alpha;
    for i in 1u32..9u32 {
        let index = F::cast_from(i);
        numerator *= -x / index;
        dbetas = dbetas * (beta - index) + betas;
        betas *= beta - index;
        series += numerator / (alpha + index) * (dbetas + factor * betas);
    }
    let mut result = -(one - x).powf(one - beta) * series;
    if result != result { result = F::new(0.0); }
    result
}

#[ruda]
fn beta_alpha_mid<F: Float>(x: F, alpha: F, beta: F) -> F {
    let one = F::new(1.0);
    let total = alpha + beta;
    let mean = alpha / total;
    let std = (alpha * beta / (total + one)).sqrt() / total;
    let mut result = F::new(0.0);
    if f64::cast_from(mean) - 0.1f64 * f64::cast_from(std) <= f64::cast_from(x)
        && f64::cast_from(x) <= f64::cast_from(mean) + 0.1f64 * f64::cast_from(std) {
        let poly = F::new(47.0) * x * (beta * beta) * (beta * beta) + alpha * (
            (F::new(43.0) + F::new(20.0) * (F::new(16.0) + F::new(27.0) * beta) * x) * (beta * beta) * beta + alpha * (
            F::new(3.0) * (F::new(59.0) + F::new(180.0) * beta - F::new(90.0) * x) * (beta * beta) + alpha * (
            (F::new(453.0) + F::new(1620.0) * beta * (one - x) - F::new(455.0) * x) * beta + alpha * (
            F::new(8.0) * (one - x) * (F::new(135.0) * beta - F::new(11.0))))));
        let prefactor_num = (one + F::new(12.0) * alpha) * (one + F::new(12.0) * beta) / (total * total);
        let prefactor_den = F::new(12960.0) * alpha * alpha * alpha * beta * beta * (one + F::new(12.0) * total);
        result = prefactor_num / (one - x) * poly / prefactor_den;
    } else {
        let prefactor = -x / (F::new(2.0) * alpha * beta / total).sqrt();
        let stirling = (one + one / (F::new(12.0) * alpha) + one / (F::new(288.0) * alpha * alpha))
            * (one + one / (F::new(12.0) * beta) + one / (F::new(288.0) * beta * beta))
            / (one + one / (F::new(12.0) * total) + one / (F::new(288.0) * total * total));
        let term1_num = F::new(2.0) * (alpha * alpha) * (x - one) + alpha * beta * (x - one) - x * (beta * beta);
        let axbx = alpha * (x - one) + beta * x;
        let term1_den = (F::new(2.0) * alpha / beta).sqrt() * total.powf(F::new(1.5)) * axbx * axbx;
        let term1 = term1_num / term1_den;
        let term2 = F::new(0.5) * (alpha / (total * x)).ln();
        let term3 = (F::new(8.0) * alpha * beta / total).sqrt() / (beta * x + alpha * (x - one));
        let term4_base = beta * (beta / (total * (one - x))).ln() + alpha * (alpha / (total * x)).ln();
        let mut term4 = term4_base.powf(F::new(-1.5));
        if !(x < mean) { term4 = -term4; }
        result = stirling * prefactor * (term1 + term2 * (term3 + term4));
    }
    result
}

#[ruda]
fn dirichlet_grad<F: Float>(x: F, alpha: F, total: F) -> F {
    let one = F::new(1.0);
    let beta = total - alpha;
    let boundary = total * x * (one - x);
    let mut result = F::new(0.0);
    if x <= F::new(0.5) && boundary < F::new(2.5) {
        result = beta_alpha_small(x, alpha, beta);
    } else if x >= F::new(0.5) && boundary < F::new(0.75) {
        result = -beta_beta_small(one - x, beta, alpha);
    } else if alpha > F::new(6.0) && beta > F::new(6.0) {
        result = beta_alpha_mid(x, alpha, beta);
    } else {
        let coefficients = Array::<F>::from_data(comptime!(DIRICHLET_COEFFICIENTS.to_vec()));
        let u = x.ln();
        let a = alpha.ln() - u;
        let b = total.ln() - a;
        let mut powers_u = Array::<F>::new(3usize);
        let mut powers_a = Array::<F>::new(3usize);
        powers_u[0] = one; powers_u[1] = u; powers_u[2] = u * u;
        powers_a[0] = one; powers_a[1] = a; powers_a[2] = a * a;
        let mut p = F::new(0.0);
        let mut q = F::new(0.0);
        for i in 0usize..3usize {
            for j in 0usize..3usize {
                let index = (i * 3 + j) * 4;
                let ua = powers_u[i] * powers_a[j];
                p += ua * (coefficients[index] + b * (coefficients[index + 1]
                    + b * (coefficients[index + 2] + b * coefficients[index + 3])));
                q += ua * (coefficients[index + 36] + b * (coefficients[index + 37]
                    + b * (coefficients[index + 38] + b * coefficients[index + 39])));
            }
        }
        let approximation = x * (digamma(total) - digamma(alpha)) / beta;
        result = p / q * approximation;
    }
    result
}

#[ruda(launch)]
fn gamma_gradient<S: Float + RudaElement, F: Float + RudaElement>(
    alpha: &Tensor<S>, samples: &Tensor<S>, output: &mut Tensor<S>,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let a = F::cast_from(alpha[address(alpha, p)]);
        let x = F::cast_from(samples[address(samples, p)]);
        output[address(output, p)] = S::cast_from(gamma_grad(a, x));
    }
}

#[ruda(launch)]
fn dirichlet_gradient<F: Float + RudaElement>(
    samples: &Tensor<F>, alpha: &Tensor<F>, total: &Tensor<F>, output: &mut Tensor<F>,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let x = samples[address(samples, p)];
        let a = alpha[address(alpha, p)];
        let t = total[address(total, p)];
        output[address(output, p)] = dirichlet_grad(x, a, t);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View]) {
    assert!(matches!(op, 428 | 429));
    assert_eq!(xs.len(), if op == 428 { 3 } else { 4 });
    let output = xs.last().unwrap();
    assert!(if op == 428 { matches!(output.dtype, 0 | 1 | 2 | 9) } else { matches!(output.dtype, 0 | 9) });
    assert!(xs.iter().all(|x| x.shape == output.shape && x.dtype == output.dtype));
    if output.len == 0 { return; }
    let c = client();
    let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("distribution gradient launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! gamma { ($s:ty, $f:ty) => { unsafe {
        gamma_gradient::launch::<$s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), output.arg());
    }}; }
    macro_rules! dirichlet { ($f:ty) => { unsafe {
        dirichlet_gradient::launch::<$f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), output.arg());
    }}; }
    if op == 428 {
        match output.dtype {
            0 => gamma!(f32, f32), 1 => gamma!(f16, f32), 2 => gamma!(bf16, f32), 9 => gamma!(f64, f64), _ => unreachable!(),
        }
    } else if output.dtype == 0 { dirichlet!(f32); }
    else { dirichlet!(f64); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}

const GAMMA_COEFFICIENTS: [f64; 24] = [
    0.16009398, -0.094634809, 0.025146376, -0.0030648343,
    1.0, 0.32668115, 0.10406089, 0.0014179084,
    0.53487893, 0.1298071, 0.065735949, -0.0015649758,
    0.16639465, 0.020070113, -0.0035938915, -0.00058392623,
    0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
    0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-7,
];

const DIRICHLET_COEFFICIENTS: [f64; 72] = [
    1.003668233, -0.01061107488, -0.0657888334, 0.01201642863,
    0.6336835991, -0.3557432599, 0.05486251648, -0.001465281033,
    -0.03276231906, 0.004474107445, 0.002429354597, -0.0001557569013,
    0.221950385, -0.3187676331, 0.01799915743, 0.01074823814,
    -0.2951249643, 0.06219954479, 0.01535556598, 0.001550077057,
    0.02155310298, 0.004170831599, 0.001292462449, 6.976601077e-5,
    -0.05980841433, 0.008441916499, 0.01085618172, 0.002319392565,
    0.02911413504, 0.01400243777, -0.002721828457, 0.000751041181,
    0.005900514878, -0.001936558688, -9.495446725e-6, 5.385558597e-5,
    1.0, -0.02924021934, -0.04438342661, 0.007285809825,
    0.6357567472, -0.3473456711, 0.05454656494, -0.002407477521,
    -0.03301322327, 0.004845219414, 0.00231480583, -0.0002307248149,
    0.5925320577, -0.1757678135, 0.01505928619, 0.000564515273,
    0.1014815858, -0.06589186703, 0.01272886114, -0.0007316646956,
    -0.007258481865, 0.001096195486, 0.0003934994223, -4.12701925e-5,
    0.06469649321, -0.0236701437, 0.002902096474, -5.896963079e-5,
    0.001925008108, -0.002869809258, 0.0008000589141, -6.063713228e-5,
    -0.0003477407336, 6.959756487e-5, 1.097287507e-5, -1.650964693e-6,
];
