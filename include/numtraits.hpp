//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include "meta.hpp"

namespace autodiff {
namespace detail {

template <typename T>
struct ArithmeticTraits {
    static constexpr bool isArithmetic = std::is_arithmetic_v<T>;
};

template <typename T>
constexpr bool isArithmetic = ArithmeticTraits<PlainType<T>>::isArithmetic;

// An auxiliary template to indicate NumberTraits has not been defined for a
// type.
template <typename T>
struct NumericTypeInfoNotDefinedFor {
    using type = T;
};

template <typename T>
struct NumberTraits {
    using NumericType = std::conditional_t<isArithmetic<T>, T, NumericTypeInfoNotDefinedFor<T>>;
    static constexpr auto Order = 0;
};

template <typename T>
using NumericType = typename NumberTraits<PlainType<T>>::NumericType;

template <typename T>
constexpr auto Order = NumberTraits<PlainType<T>>::Order;

}  // namespace detail

}  // namespace autodiff
