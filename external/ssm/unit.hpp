#pragma once

#include "vector.hpp"

namespace ssm {
template <typename T>
class unit {
public:
  explicit unit(const T& value) : value(ssm::normalize(value)) {}

  operator T() const { return value; }

private:
  T value;
};
} // namespace ssm
