std::vector<int64_t> random_bounds(const at::Tensor& out, bool bounded,
                                   int64_t from, std::optional<int64_t> to) {
  validate(out.device());
  if (!bounded && !out.numel()) return {0, 0, 0, 0};
  at::assert_no_internal_overlap(out);
  const auto dtype = out.scalar_type();
  uint64_t range = 0;
  bool wide = false;
  if (!bounded) {
    from = 0;
    AT_DISPATCH_ALL_TYPES_AND3(at::kHalf, at::kBFloat16, at::kBool, dtype, "random_ruda_bounds", [&] {
      if constexpr (std::is_floating_point_v<scalar_t> ||
                    std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
        range = (uint64_t(1) << std::numeric_limits<scalar_t>::digits) + 1;
      } else {
        range = uint64_t(std::numeric_limits<scalar_t>::max()) + 1;
      }
    });
    wide = dtype == at::kLong || dtype == at::kDouble;
  } else if (!to && from == std::numeric_limits<int64_t>::lowest()) {
    if (out.numel()) {
      TORCH_CHECK(dtype == at::kLong || dtype == at::kFloat || dtype == at::kDouble || dtype == at::kBFloat16,
                  "full-range random_ supports int64, float32, float64 and bfloat16");
    }
    wide = true;
  } else {
    int64_t upper;
    if (to) {
      upper = *to;
      TORCH_CHECK(from < upper, "random_ requires from < to");
      if (at::isFloatingType(dtype)) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dtype, "random_ruda_adjust", [&] {
          from = at::native::templates::update_from<scalar_t>(from);
          upper = at::native::templates::update_to<scalar_t>(upper);
        });
        TORCH_CHECK(from < upper, "random_ bounds collapse after dtype conversion");
      }
      at::native::templates::check_from_to_in_range(from, upper - 1, out.dtype());
      range = uint64_t(upper) - uint64_t(from);
    } else {
      upper = 0;
      AT_DISPATCH_ALL_TYPES_AND3(at::kHalf, at::kBFloat16, at::kBool, dtype, "random_ruda_upper", [&] {
        if constexpr (std::is_floating_point_v<scalar_t> ||
                      std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
          upper = int64_t(1) << std::numeric_limits<scalar_t>::digits;
          from = at::native::templates::update_from<scalar_t>(from);
          TORCH_CHECK(from < upper, "random_ lower bound must be below the dtype upper bound");
        } else {
          upper = int64_t(std::numeric_limits<scalar_t>::max());
        }
      });
      at::native::templates::check_from_to_in_range(from, upper, out.dtype());
      range = uint64_t(upper) - uint64_t(from) + 1;
    }
    wide = range >= (uint64_t(1) << 28);
  }
  return {from, int64_t(uint32_t(range)), int64_t(uint32_t(range >> 32)), int64_t(wide)};
}

std::vector<double> uniform_parameters(const at::Tensor& out, double from, double to) {
  validate(out.device());
  std::vector<double> result;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "uniform_ruda_bounds", [&] {
    const double minimum = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
    const double maximum = static_cast<double>(std::numeric_limits<scalar_t>::max());
    TORCH_CHECK(from >= minimum && from <= maximum, "uniform_ from is out of bounds for ", out.scalar_type());
    TORCH_CHECK(to >= minimum && to <= maximum, "uniform_ to is out of bounds for ", out.scalar_type());
    TORCH_CHECK(from <= to, "uniform_ requires from <= to");
    TORCH_CHECK(to - from <= maximum, "uniform_ range exceeds the output dtype limit");
    const auto lower = static_cast<scalar_t>(std::min(std::max(from, minimum), maximum));
    const auto upper = static_cast<scalar_t>(std::max(std::min(to, maximum), minimum));
    result = {static_cast<double>(lower), static_cast<double>(upper), static_cast<double>(upper - lower)};
  });
  return result;
}

std::vector<int64_t> random_launch_seed(std::optional<at::Generator> generator) {
  TORCH_CHECK(generator.has_value(), "explicit RUDA torch.Generator required");
  auto gen = at::check_generator<Generator>(generator);
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return {int64_t(gen->random()), int64_t(gen->random()), 0, 0};
}

int64_t randperm_prepare(int64_t n, at::Tensor out) {
  TORCH_CHECK(n >= 0, "randperm requires nonnegative n");
  validate(out.device());
  TORCH_CHECK(out.layout() == c10::Layout::Strided, "randperm requires strided output");
  TORCH_CHECK(uint64_t(n) <= std::numeric_limits<uint32_t>::max(), "randperm exceeds native U32 indexing");
  at::native::check_supported_max_int_with_precision(n, out);
  const auto code = dtype_code(out.scalar_type());
  TORCH_CHECK(code <= 9 && code != 3, "randperm requires real floating or standard integer output");
  if (out.dim() != 1 || out.size(0) != n) resize_tensor(out, {n}, std::nullopt);
  at::assert_no_internal_overlap(out);
  if (n == 0) return 0;
  TORCH_CHECK(code != 2, "nonempty CUDA randperm does not support bfloat16");
  const double count = static_cast<double>(n);
  const double threshold = std::log(0.9) * 12;
  return std::min(64, static_cast<int>(std::ceil(std::log2(count - (6 * count * count + 1) / threshold))));
}
