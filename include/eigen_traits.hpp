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
#include <Eigen/Core>

#include "vectortraits.hpp"

namespace autodiff {
namespace detail {

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////  PARTIAL SPECIALAZATION ////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct VectorTraits<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
    using ValueType = Scalar;

    template <typename NewValueType>
    using ReplaceValueType = Eigen::Matrix<NewValueType, Rows, Cols, Options, MaxRows, MaxCols>;
};

static_assert(std::is_same_v<VectorTraits<Eigen::MatrixX2d>::ValueType, double>);
static_assert(std::is_same_v<VectorTraits<Eigen::Vector2f>::ValueType, float>);
static_assert(std::is_same_v<VectorTraits<Eigen::MatrixX2d>::template ReplaceValueType<float>,
                             Eigen::MatrixX2f>);

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct VectorTraits<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
    using ValueType = Scalar;

    template <typename NewValueType>
    using ReplaceValueType = Eigen::Array<NewValueType, Rows, Cols, Options, MaxRows, MaxCols>;
};

static_assert(std::is_same_v<VectorTraits<Eigen::Array22d>::ValueType, double>);
static_assert(std::is_same_v<VectorTraits<Eigen::Array2f>::ValueType, float>);
static_assert(std::is_same_v<VectorTraits<Eigen::Array22d>::template ReplaceValueType<float>,
                             Eigen::Array22f>);

template <typename VectorType, int Size>
struct VectorTraits<Eigen::VectorBlock<VectorType, Size>> {
    using ValueType = typename PlainType<VectorType>::Scalar;

    template <typename NewValueType>
    using ReplaceValueType = VectorReplaceValueType<VectorType, NewValueType>;
};

static_assert(
    std::is_same_v<VectorTraits<Eigen::VectorBlock<Eigen::VectorXd, 2>>::ValueType, double>);
static_assert(
    std::is_same_v<VectorTraits<Eigen::VectorBlock<Eigen::VectorXf, 2>>::ValueType, float>);

static_assert(
    std::is_same_v<
        VectorTraits<Eigen::VectorBlock<Eigen::VectorXf, 2>>::template ReplaceValueType<double>,
        Eigen::VectorXd>);

template <typename VectorType, typename IndicesType>
struct VectorTraits<Eigen::IndexedView<VectorType, IndicesType, Eigen::internal::SingleRange>> {
    using ValueType = typename PlainType<VectorType>::Scalar;

    template <typename NewValueType>
    using ReplaceValueType = VectorReplaceValueType<VectorType, NewValueType>;
};

template <typename VectorType, typename IndicesType>
struct VectorTraits<Eigen::IndexedView<VectorType, Eigen::internal::SingleRange, IndicesType>> {
    using ValueType = typename PlainType<VectorType>::Scalar;

    template <typename NewValueType>
    using ReplaceValueType = VectorReplaceValueType<VectorType, NewValueType>;
};

// static_assert(std::is_same_v<VectorTraits<Eigen::IndexedView<
//                                  Eigen::Matrix2d,
//                                  Eigen::internal::SingleRange,
//                                  Eigen::internal::SingleRange>>::ValueType,
//                              double>);

// static_assert(std::is_same_v<
//               VectorTraits<Eigen::IndexedView<Eigen::MatrixXd,
//                                               Eigen::internal::SingleRange,
//                                               Eigen::internal::SingleRange>>::
//                   template ReplaceValueType<float>,
//               Eigen::MatrixXf>);

template <typename MatrixType>
struct VectorTraits<Eigen::Ref<MatrixType>> {
    using ValueType = VectorValueType<MatrixType>;  // == VectorTraits<Matrixtype>

    template <typename NewValueType>
    using ReplaceValueType = VectorReplaceValueType<MatrixType, NewValueType>;
};

template <typename VectorType, int MapOptions, typename StrideType>
struct VectorTraits<Eigen::Map<VectorType, MapOptions, StrideType>> {
    using ValueType = VectorValueType<VectorType>;

    template <typename NewValueType>
    using ReplaceValueType =
        Eigen::Map<VectorReplaceValueType<VectorType, NewValueType>, MapOptions, StrideType>;
};

//
}  // namespace detail
}  // namespace autodiff
