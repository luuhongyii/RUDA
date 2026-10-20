/* Complex logarithm adapted from Thrust/FreeBSD clog and clogf.
 * Copyright 2008-2021 NVIDIA Corporation; Copyright 2013 Filipe RNC Maia.
 * Licensed under the Apache License, Version 2.0, available at
 * http://www.apache.org/licenses/LICENSE-2.0 . Distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>.
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

template <typename T> int64_t power_pack(T value) {
  uint64_t word = 0; std::memcpy(&word, &value, sizeof(T));
  int64_t bits; std::memcpy(&bits, &word, sizeof(bits)); return bits;
}

template <typename T> T power_log_trim(T value) {
  if constexpr (sizeof(T) == 8) {
    uint64_t bits; std::memcpy(&bits, &value, 8); bits &= UINT64_C(0xfffffff800000000);
    std::memcpy(&value, &bits, 8);
  } else {
    uint32_t bits; std::memcpy(&bits, &value, 4); bits &= 0xffff0000u;
    std::memcpy(&value, &bits, 4);
  }
  return value;
}

template <typename T> c10::complex<T> power_host_log(T x, T y) {
  const T angle = std::atan2(y, x), half = T(0.5), e = T(2.7182818284590452354);
  T ax = std::abs(x), ay = std::abs(y);
  if (x != x || y != y) return {std::log(x*x + y*y), angle};
  if (ax < ay) std::swap(ax, ay);
  const T high = sizeof(T) == 8 ? T(5e307) : T(1e34);
  const T tiny = sizeof(T) == 8 ? T(1e-150) : T(1e-19);
  const T lower = sizeof(T) == 8 ? T(1e-50) : T(1e-6);
  const T upper = sizeof(T) == 8 ? T(1e50) : T(1e6);
  if (ay > high) return {std::log(std::hypot(x/e, y/e)) + T(1), angle};
  if (ax == T(1)) return {ay < tiny ? (ay*half)*ay : std::log1p(ay*ay)*half, angle};
  if (ax < lower || ay < lower || ax > upper || ay > upper) return {std::log(std::hypot(x,y)), angle};
  if (ax >= T(1)) return {std::log1p((ax-T(1))*(ax+T(1)) + ay*ay)*half, angle};
  if (ax*ax + ay*ay <= T(0.7)) return {std::log(ax*ax + ay*ay)*half, angle};
  const T x0 = power_log_trim(ax), y0 = power_log_trim(ay);
  ax -= x0; ay -= y0;
  const T x1 = power_log_trim(ax), y1 = power_log_trim(ay), x2 = ax-x1, y2 = ay-y1;
  T parts[12]{x0*x0,y0*y0,T(2)*x0*x1,T(2)*y0*y1,x1*x1,y1*y1,
              T(2)*x0*x2,T(2)*y0*y2,T(2)*x1*x2,T(2)*y1*y2,x2*x2,y2*y2};
  bool sorted = false;
  while (!sorted) {
    sorted = true;
    for (int i=0; i<11; ++i) if (parts[i] < parts[i+1]) { std::swap(parts[i],parts[i+1]); sorted=false; }
  }
  T sum = T(-1); for (T value : parts) sum += value;
  return {half*std::log1p(sum),angle};
}

std::vector<int64_t> power_scalar_parameters(pybind11::object value, at::ScalarType dtype, bool tensor_scalar, bool integer_base) {
  at::Scalar exponent;
  if (tensor_scalar) {
    const auto tensor = pybind11::cast<at::Tensor>(value);
    TORCH_CHECK(tensor.device().is_cpu() && tensor.dim()==0, "expected a CPU scalar exponent");
    exponent = tensor.resolve_conj().resolve_neg().to(dtype).item();
  } else {
    static torch::PythonArgParser parser({"power_scalar(Scalar exponent)"});
    torch::ParsedArgs<1> parsed; auto args=pybind11::make_tuple(value);
    exponent=parser.parse(args.ptr(),nullptr,parsed).scalar(0);
    TORCH_CHECK(!(integer_base && exponent.isIntegral(true) && exponent.toLong()<0), "Integers to negative integer powers are not allowed");
    if (exponent.equal(0.0) || exponent.equal(false)) return {1,0,0};
    if (exponent.equal(1.0) || exponent.equal(true)) return {2,0,0};
  }
  TORCH_CHECK(dtype != at::kBool, "pow is not implemented for Bool computation");
  int64_t special=0;
  if (!exponent.isComplex()) {
    if (exponent.equal(0.5)) special=3;
    else if (exponent.equal(-0.5)) special=4;
    else if (exponent.equal(-1.0)) special=5;
  }
  TORCH_CHECK(!(special>=3 && special<=5 && at::isIntegralType(dtype,true)), "power specialization requires floating or complex storage");
  TORCH_CHECK(!(special==5 && dtype==at::kComplexHalf), "reciprocal power is not implemented for ComplexHalf");
  if (special) return {special,0,0};
  if (at::isComplexType(dtype)) {
    if (dtype!=at::kComplexHalf && exponent.equal(2.0)) special=6;
    if (dtype==at::kComplexDouble) {
      const auto z=exponent.to<c10::complex<double>>(); return {special,power_pack(z.real()),power_pack(z.imag())};
    }
    const auto z=exponent.to<c10::complex<float>>(); return {special,power_pack(z.real()),power_pack(z.imag())};
  }
  int64_t bits=0;
  AT_DISPATCH_ALL_TYPES_AND2(at::kHalf,at::kBFloat16,dtype,"power_scalar",[&] {
    const auto converted=exponent.to<scalar_t>(); const double d=static_cast<double>(converted);
    if (d==2) special=6; else if (d==3) special=7; else if (d==-2) special=8;
    if constexpr (std::is_same_v<scalar_t,at::Half> || std::is_same_v<scalar_t,at::BFloat16>) bits=power_pack(static_cast<float>(converted));
    else bits=power_pack(converted);
  });
  return {special,bits,0};
}

std::vector<int64_t> power_scalar_base_log(const at::Tensor& input, at::ScalarType dtype) {
  TORCH_CHECK(input.device().is_cpu() && input.dim()==0 && at::isComplexType(dtype), "expected a CPU scalar complex base");
  const auto scalar=input.resolve_conj().resolve_neg().to(dtype).item();
  if (dtype==at::kComplexDouble) {
    const auto z=scalar.to<c10::complex<double>>(); const auto value=power_host_log(z.real(),z.imag());
    return {power_pack(value.real()),power_pack(value.imag())};
  }
  const auto z=scalar.to<c10::complex<float>>(); const auto value=power_host_log(z.real(),z.imag());
  return {power_pack(value.real()),power_pack(value.imag())};
}
