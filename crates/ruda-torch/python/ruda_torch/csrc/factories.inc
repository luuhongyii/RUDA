using Arange = int (*)(const Descriptor*, uint64_t, uint64_t);
using Normal = int (*)(const Descriptor*, float, float, const uint32_t*);
Arange arange_native = nullptr;
Normal normal_native = nullptr;

void kaiser_window_kernel(at::TensorIteratorBase& iter, int64_t window_length, double beta_value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, iter.dtype(), "kaiser_window_ruda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t inv_alpha = static_cast<opmath_t>(2.0 / (window_length - 1));
    const opmath_t beta = static_cast<opmath_t>(beta_value);
    const opmath_t inv_i0_beta = 1.0 / calc_i0(beta);
    scientific_call(493, {iter.input(), iter.output()}, {},
                    {static_cast<double>(inv_alpha), static_cast<double>(beta), static_cast<double>(inv_i0_beta)});
  });
}

template <typename T>
auto linspace_delta(T start, T end, int64_t steps) {
  if constexpr (std::is_integral_v<T>) {
    return (static_cast<float>(end) - static_cast<float>(start)) / (steps - 1);
  } else {
    return (end - start) / static_cast<T>(steps - 1);
  }
}

at::Tensor& linspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps, at::Tensor& out) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  validate(out.device());
  if (out.numel() != steps) resize_tensor(out, {steps}, std::nullopt);
  at::assert_no_internal_overlap(out);
  if (!steps) return out;
  if (steps == 1) { (fill)(out, start); return out; }
  auto result = out.is_conj() || out.is_neg() ? at::empty(out.sizes(), out.options()) : out;
  auto first = at::empty({}, out.options());
  auto last = at::empty({}, out.options());
  auto delta = at::empty({}, out.options().dtype(at::isIntegralType(out.scalar_type(), false) ? at::kFloat : out.scalar_type()));
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "linspace_ruda", [&] {
    auto a = start.to<scalar_t>();
    auto b = end.to<scalar_t>();
    (fill)(first, a);
    (fill)(last, b);
    (fill)(delta, linspace_delta(a, b, steps));
  });
  scientific_call(248, {first, last, delta, result}, {}, {});
  if (!result.is_same(out)) copy_from(result, out, false);
  return out;
}

template <typename T>
auto logspace_delta(T start, T end, int64_t steps) {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<float>(end - start) / (steps - 1);
  } else {
    return (end - start) / static_cast<T>(steps - 1);
  }
}

/*
 * Copyright 2008-2013 NVIDIA Corporation
 * Copyright 2013 Filipe RNC Maia
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright (c) 2011 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
c10::complex<double> logspace_scalar_power(double base, c10::complex<double> exponent) {
  const double logarithm = std::log(base);
  const double x = logarithm * exponent.real(), y = logarithm * exponent.imag();
  if (y == 0.0) return {std::exp(x), y};
  if (x == 0.0) return {std::cos(y), std::sin(y)};
  if (!std::isfinite(y)) {
    if (!std::isinf(x)) return {y - y, y - y};
    if (std::signbit(x)) return {0.0, 0.0};
    return {x, y - y};
  }
  uint64_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  const auto high = bits >> 32;
  if (high >= 0x40862e42 && high <= 0x4096b8e4) {
    double scaled = std::exp(x - 1246.97177782734161156);
    std::memcpy(&bits, &scaled, sizeof(bits));
    const int power = int(bits >> 52) - 2046 + 1799;
    bits = (bits & UINT64_C(0x000fffffffffffff)) | (UINT64_C(2046) << 52);
    std::memcpy(&scaled, &bits, sizeof(scaled));
    const int half = power / 2;
    double scale1, scale2;
    bits = uint64_t(1023 + half) << 52;
    std::memcpy(&scale1, &bits, sizeof(scale1));
    bits = uint64_t(1023 + power - half) << 52;
    std::memcpy(&scale2, &bits, sizeof(scale2));
    return {std::cos(y) * scaled * scale1 * scale2, std::sin(y) * scaled * scale1 * scale2};
  }
  const double exp_x = std::exp(x);
  return {exp_x * std::cos(y), exp_x * std::sin(y)};
}

at::Tensor& logspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps,
                        double base, at::Tensor& out) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  validate(out.device());
  if (out.numel() != steps) resize_tensor(out, {steps}, std::nullopt);
  at::assert_no_internal_overlap(out);
  if (!steps) return out;
  if (steps == 1) {
    if (at::isComplexType(out.scalar_type())) {
      (fill)(out, logspace_scalar_power(base, start.to<c10::complex<double>>()));
    } else {
      (fill)(out, std::pow(base, start.to<double>()));
    }
    return out;
  }
  auto result = out.is_conj() || out.is_neg() ? at::empty(out.sizes(), out.options()) : out;
  auto first = at::empty({}, out.options());
  auto last = at::empty({}, out.options());
  auto arithmetic_options = out.options().dtype(at::isIntegralType(out.scalar_type(), false) ? at::kFloat : out.scalar_type());
  auto delta = at::empty({}, arithmetic_options);
  auto radix = at::empty({}, arithmetic_options);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "logspace_ruda", [&] {
    auto a = start.to<scalar_t>();
    auto b = end.to<scalar_t>();
    (fill)(first, a);
    (fill)(last, b);
    (fill)(delta, logspace_delta(a, b, steps));
    using base_t = std::conditional_t<std::is_integral_v<scalar_t>, float, scalar_t>;
    (fill)(radix, static_cast<base_t>(base));
  });
  scientific_call(413, {first, last, delta, radix, result}, {}, {});
  if (!result.is_same(out)) copy_from(result, out, false);
  return out;
}

at::Tensor& normal_(at::Tensor& self, double mean, double std, std::optional<at::Generator> generator) {
  TORCH_CHECK(normal_native, "RUDA factory API 1 unavailable; rebuild both native libraries");
  TORCH_CHECK(self.is_floating_point() && std>=0, "normal_ requires floating storage and std >= 0");
  at::assert_no_internal_overlap(self);
  Argument out(self);
  auto gen = at::get_generator_or_default<Generator>(generator,default_generator());
  if (!self.numel()) return self;
  uint32_t seeds[4];
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    for (auto& seed:seeds) seed=gen->random();
  }
  check(normal_native(&out.desc,float(mean),float(std),seeds));
  return self;
}

at::Tensor& arange_out(const at::Scalar& start, const at::Scalar& end,
                       const at::Scalar& step, at::Tensor& out) {
  TORCH_CHECK(arange_native, "RUDA factory API 1 unavailable; rebuild both native libraries");
  validate(out.device());
  const auto dtype = dtype_code(out.scalar_type());
  TORCH_CHECK(dtype!=3 && dtype<10, "arange does not support bool or complex storage");
  at::native::arange_check_bounds(start,end,step);
  int64_t size;
  if (out.scalar_type()==at::kLong) {
    auto a=start.to<int64_t>(), b=end.to<int64_t>(), s=step.to<int64_t>();
    TORCH_CHECK(s!=0, "step must be nonzero");
    TORCH_CHECK(s>0 ? b>=a : b<=a, "upper bound and lower bound inconsistent with step sign");
    uint64_t distance = s>0 ? uint64_t(b)-uint64_t(a) : uint64_t(a)-uint64_t(b);
    uint64_t magnitude = s>0 ? uint64_t(s) : uint64_t(0)-uint64_t(s);
    uint64_t count = distance/magnitude + (distance%magnitude!=0);
    TORCH_CHECK(count<=uint64_t(std::numeric_limits<int64_t>::max()), "invalid size, possible overflow?");
    size=int64_t(count);
  } else {
    auto count = std::ceil((end.to<double>()-start.to<double>())/step.to<double>());
    TORCH_CHECK(count>=0 && count<double(std::numeric_limits<int64_t>::max()), "invalid size, possible overflow?");
    size=int64_t(count);
  }
  if (out.numel()!=size) {
    if (out.numel()) TORCH_WARN("arange out tensor will be resized to match the computed number of elements");
    resize_tensor(out,{size},std::nullopt);
  }
  at::assert_no_internal_overlap(out);
  Argument result(out);
  auto a = dtype==9 ? scalar_bits<double>(start) : dtype<=2 ? scalar_bits<float>(start) : scalar_bits<int64_t>(start);
  auto s = dtype==9 ? scalar_bits<double>(step) : dtype<=2 ? scalar_bits<float>(step) : scalar_bits<int64_t>(step);
  check(arange_native(&result.desc,a,s));
  return out;
}

at::Tensor& range_out(const at::Scalar& start, const at::Scalar& end,
                      const at::Scalar& step, at::Tensor& out) {
  TORCH_CHECK(arange_native, "RUDA factory API 1 unavailable; rebuild both native libraries");
  validate(out.device());
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, out.scalar_type(), "range_ruda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto a = start.to<accscalar_t>();
    const auto b = end.to<accscalar_t>();
    const auto s = step.to<accscalar_t>();
    at::native::arange_check_bounds(start, end, step);
    int64_t size;
    if constexpr (std::is_integral_v<accscalar_t>) {
      const auto minimum = std::numeric_limits<int64_t>::min();
      const auto maximum = std::numeric_limits<int64_t>::max();
      TORCH_CHECK(s != 0, "step must be nonzero");
      TORCH_CHECK(!(a > 0 && b < minimum + a) && !(a < 0 && b > maximum + a),
                  "invalid size, possible overflow?");
      const auto distance = b - a;
      TORCH_CHECK(!(distance == minimum && s == -1), "invalid size, possible overflow?");
      const auto intervals = distance / s;
      TORCH_CHECK(intervals >= -1 && intervals < maximum, "invalid size, possible overflow?");
      size = intervals + 1;
    } else {
      const accscalar_t count = (b - a) / s + accscalar_t(1);
      TORCH_CHECK(std::isfinite(count) && count > accscalar_t(-1)
                      && count < static_cast<accscalar_t>(std::numeric_limits<int64_t>::max()),
                  "invalid size, possible overflow?");
      size = static_cast<int64_t>(count);
    }
    if (out.numel() != size) resize_tensor(out, {size}, std::nullopt);
    at::assert_no_internal_overlap(out);
    if (!size) return;
    auto result = !out.is_contiguous() || out.is_conj() || out.is_neg()
        ? at::empty(out.sizes(), out.options()) : out;
    Argument output(result);
    uint64_t start_bits = 0, step_bits = 0;
    std::memcpy(&start_bits, &a, sizeof(a));
    std::memcpy(&step_bits, &s, sizeof(s));
    check(arange_native(&output.desc, start_bits, step_bits));
    if (!result.is_same(out)) copy_from(result, out, false);
  });
  return out;
}
