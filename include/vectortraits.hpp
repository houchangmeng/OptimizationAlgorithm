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
#include <vector>
#include "meta.hpp"

namespace autodiff {
namespace detail {

template <typename V>
struct VectorTraitsNotDefinedFor {};

template <typename V>
struct VectorReplaceValueTypeNotSupportedFor {};

template <typename V, class Enable = void>
struct VectorTraits {
    using ValueType = VectorTraitsNotDefinedFor<V>;

    using ReplaceValueType = VectorReplaceValueTypeNotSupportedFor<V>;
};

template <typename V>
using VectorValueType = typename VectorTraits<PlainType<V>>::ValueType;

template <typename V, typename NewValueType>
using VectorReplaceValueType =
    typename VectorTraits<PlainType<V>>::template ReplaceValueType<NewValueType>;

template <typename V>
constexpr bool isVector =
    !std::is_same_v<VectorValueType<PlainType<V>>, VectorTraitsNotDefinedFor<PlainType<V>>>;

template <typename T, template <typename> class Allocator>
struct VectorTraits<std::vector<T, Allocator<T>>> {
    using ValueType = T;
    template <typename NewValueType>
    using ReplaceValueTYpe = std::vector<NewValueType, Allocator<NewValueType>>;
};

}  // namespace detail
}  // namespace autodiff