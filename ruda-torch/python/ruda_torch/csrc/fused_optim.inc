double fused_optimizer_cpu_lr(const at::Tensor& lr) {
  TORCH_CHECK(lr.device().is_cpu(), "CPU learning rate tensor required");
  return lr.item<double>();
}

bool fused_optimizer_check(const std::vector<std::vector<at::Tensor>>& groups,
                           bool allow_mixed, bool warn_no_momentum) {
  TORCH_CHECK(groups.size() >= 2 && !groups[0].empty(), "fused optimizer requires nonempty tensor lists");
  const auto count = groups[0].size();
  std::vector<at::TensorList> lists;
  lists.reserve(groups.size());
  for (const auto& group : groups) {
    TORCH_CHECK(group.size() == count, "fused optimizer tensor lists must have equal lengths");
    lists.emplace_back(group);
  }
  validate(groups[0][0].device());
  const bool mixed = allow_mixed && groups.size() > 2 &&
      groups[0][0].scalar_type() != groups[2][0].scalar_type();
  bool fast = at::native::_check_tensors_share_sizes_and_strides(lists);
  for (const auto& list : lists) {
    fast = fast && at::native::check_fast_path_restrictions({list}) &&
        list[0].device() == groups[0][0].device() &&
        (mixed || list[0].scalar_type() == groups[0][0].scalar_type());
  }
  TORCH_CHECK(fast,
              "fused optimizer tensors must share device, dtype within each list, dense layout and corresponding sizes/strides");
  if (warn_no_momentum) {
    TORCH_WARN_ONCE("`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
  }
  return mixed;
}
