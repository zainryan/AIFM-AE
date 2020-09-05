#pragma once

#include <cstdint>
#include <type_traits>

namespace far_memory {

enum DataFrameTypeID { Char = 0, Short, Int, Long, LongLong, Float, Double };

template <typename T> constexpr int8_t get_dataframe_type_id() {
  if (std::is_same<T, char>::value) {
    return DataFrameTypeID::Char;
  }
  if (std::is_same<T, short>::value) {
    return DataFrameTypeID::Short;
  }
  if (std::is_same<T, int>::value) {
    return DataFrameTypeID::Int;
  }
  if (std::is_same<T, long>::value) {
    return DataFrameTypeID::Long;
  }
  if (std::is_same<T, long long>::value) {
    return DataFrameTypeID::LongLong;
  }
  if (std::is_same<T, float>::value) {
    return DataFrameTypeID::Float;
  }
  if (std::is_same<T, double>::value) {
    return DataFrameTypeID::Double;
  }
  return -1;
}

template <typename T> constexpr bool is_basic_dataframe_types() {
  return get_dataframe_type_id<T>() != -1;
}
} // namespace far_memory
