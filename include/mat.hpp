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

#include "eigen_traits.hpp"
#include "meta.hpp"
#include "var.hpp"

namespace Eigen {
template <typename T>
struct NumTraits;

template <typename T>
struct NumTraits;

template <typename T>
struct NumTraits<autodiff::Variable<T>> : NumTraits<T> {
    typedef autodiff::Variable<T> Real;
    typedef autodiff::Variable<T> NonInteger;
    typedef autodiff::Variable<T> Nested;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3,
    };
};  // permits to get the epsilon, dummy_precision, lowest, highest functions

template <typename T, typename Binop>
struct ScalarBinaryOpTraits<autodiff::Variable<T>, T, Binop> {
    typedef autodiff::Variable<T> ReturnType;
};

template <typename T, typename Binop>
struct ScalarBinaryOpTraits<T, autodiff::Variable<T>, Binop> {
    typedef autodiff::Variable<T> ReturnType;
};

template <typename T>
struct NumTraits<autodiff::detail::ExprPtr<T>> : NumTraits<T> {
    typedef autodiff::Variable<T> Real;
    typedef autodiff::Variable<T> NonInteger;
    typedef autodiff::Variable<T> Nested;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };

};  // permits to get the epsilon, dummy_precision, lowest, highest functions

template <typename T, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::detail::ExprPtr<T>, T, BinOp> {
    typedef autodiff ::Variable<T> ReturnType;
};

template <typename T, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::detail::ExprPtr<T>, BinOp> {
    typedef autodiff ::Variable<T> ReturnType;
};

}  // namespace Eigen

namespace autodiff {

namespace detail {
template <typename T, int Rows, int MaxRows>
using Vec = Eigen::Matrix<T, Rows, 1, 0, MaxRows, 1>;

template <typename T, int Rows, int Cols, int MaxRows, int MaxCols>
using Mat = Eigen::Matrix<T, Rows, Cols, 0, MaxRows, MaxCols>;

template <typename T, typename X>
auto gradient(const Variable<T>& y, Eigen::DenseBase<X>& x) {
    // Return the gradient vector of variable y with respect to variables x.
    using U = VariableValueType<T>;
    using ScalarX = typename X::Scalar;
    static_assert(isVariable<ScalarX>, "Argument x is not a vector with Variable<T> objects..");
    constexpr auto isVec = X::IsVectorAtCompileTime;
    static_assert(isVec, "Argument x is not a vector. ");

    constexpr auto Rows = X::RowsAtCompileTime;
    constexpr auto MaxRows = X::MaxRowsAtCompileTime;

    const auto n = x.size();
    using Gradient = Vec<U, Rows, MaxRows>;
    Gradient g = Gradient::Zero(n);

    for (auto i = 0; i < n; ++i) {
        x[i].expr->bind_value(&g[i]);
    }
    y.expr->propagate(1.0);

    for (auto i = 0; i < n; ++i) {
        x[i].expr->bind_value(nullptr);
    }
    return g;
}

template <typename Y, typename X>
auto jacbian(const Eigen::DenseBase<Y>& y, Eigen::DenseBase<X>& x) {
    // Return the gradient vector of variable y with respect to variables x.
    // using U = VariableValueType<T>;
    using ScalarX = typename X::Scalar;
    static_assert(isVariable<ScalarX>, "Argument x is not a vector with Variable<T> objects..");

    using ScalarY = typename Y::Scalar;
    static_assert(std::is_same_v<ScalarY, ScalarX>,
                  "Argument y is not a vector with Variable<T> objects..");

    constexpr auto isVecX = X::IsVectorAtCompileTime;
    static_assert(isVecX, "Argument x is not a vector. ");

    constexpr auto isVecY = Y::IsVectorAtCompileTime;
    static_assert(isVecY, "Argument y is not a vector. ");

    using U = typename ScalarY::ArithmeticType;

    constexpr auto RowsX = X::RowsAtCompileTime;
    constexpr auto MaxRowsX = X::MaxRowsAtCompileTime;

    constexpr auto RowsY = Y::RowsAtCompileTime;
    constexpr auto MaxRowsY = Y::MaxRowsAtCompileTime;

    const auto n_row = y.size();
    const auto n_col = x.size();

    using Jacbian = Mat<U, RowsY, RowsX, MaxRowsY, MaxRowsX>;
    Jacbian J = Jacbian::Zero(n_row, n_col);

    using Gradient = Vec<U, RowsX, MaxRowsX>;
    Gradient g = Gradient::Zero(n_col);

    for (auto i = 0; i < n_row; ++i) {
        g = gradient(y[i], x);
        J.row(i) = g.transpose();
    }

    return J;
}

template <typename T, typename X, typename GradientVec>
auto hessian(const Variable<T>& y, Eigen::DenseBase<X>& x, GradientVec& g) {
    // Return the Hessian matrix of variable y with respect to variable x.
    using U = VariableValueType<T>;

    using ScalarX = typename X::Scalar;
    static_assert(isVariable<ScalarX>,
                  "Argument x is not a vector with Variable<T> (aka var) objects");

    using ScalarG = typename GradientVec::Scalar;
    static_assert(std::is_same_v<U, ScalarG>,
                  "Argument g oes not have the same arithmetic type as y.");

    constexpr auto Rows = X::RowsAtCompileTime;
    constexpr auto MaxRows = X::MaxRowsAtCompileTime;

    const auto n = x.size();

    // Form a vector containing gradient expressions for each variable
    using ExpressionGradient = Vec<ScalarX, Rows, MaxRows>;
    ExpressionGradient G(n);

    for (auto k = 0; k < n; ++k) {
        x[k].expr->bind_expr(&G(k).expr);
    }
    // Build a full gradient expression in DFS tree traversal,
    // updating gradient expressions when encountering variables.
    y.expr->propagatex(constant<T>(1.0));

    for (auto k = 0; k < n; ++k) {
        x[k].expr->bind_expr(nullptr);
    }

    // Read the gradient value form gradient expression's cached values
    g.resize(n);
    for (auto i = 0; i < n; ++i) {
        g[i] = val(G[i]);
    }

    using Hessian = Mat<U, Rows, Rows, MaxRows, MaxRows>;
    Hessian H = Hessian::Zero(n, n);

    for (auto i = 0; i < n; ++i) {
        for (auto k = 0; k < n; ++k) {
            x[k].expr->bind_value(&H(i, k));
        }

        // Propagate a second derivate value calculation down
        // the gradient expression tree for x(i)
        // i.e. G[i] is 1st order expression, when the expression propagate
        // a variable, propagate() function will calculate the derivites of G[i]
        // respect the variable x(k) and save result in bind_value(&H).
        G[i].expr->propagate(1.0);

        for (auto k = 0; k < n; ++k) {
            x[k].expr->bind_expr(nullptr);
        }
    }

    return H;
}

template <typename T, typename X>
auto hessian(const Variable<T>& y, Eigen::DenseBase<X>& x) {
    using U = VariableValueType<T>;
    constexpr auto Rows = X::RowsAtCompileTime;
    constexpr auto MaxRows = X::MaxRowsAtCompileTime;
    Vec<U, Rows, MaxRows> g;
    return hessian(y, x, g);
}

}  // namespace detail

using detail::gradient;
using detail::hessian;
using detail::jacbian;

#define AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)                 \
    using Array##SizeSuffix##SizeSuffix##TypeSuffix = Eigen::Array<Type, Size, Size>;      \
    using Array##SizeSuffix##TypeSuffix = Eigen::Array<Type, Size, 1>;                     \
    using Matrix##SizeSuffix##TypeSuffix = Eigen::Matrix<Type, Size, Size, 0, Size, Size>; \
    using Vector##SizeSuffix##TypeSuffix = Eigen::Matrix<Type, Size, 1, 0, Size, 1>;       \
    using RowVector##SizeSuffix##TypeSuffix = Eigen::Matrix<Type, 1, Size, 1, 1, Size>;

#define AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS(Type, TypeSuffix, Size)                \
    using Array##Size##X##TypeSuffix = Eigen::Array<Type, Size, -1>;                \
    using Array##X##Size##TypeSuffix = Eigen::Array<Type, -1, Size>;                \
    using Matrix##Size##X##TypeSuffix = Eigen::Matrix<Type, Size, -1, 0, Size, -1>; \
    using Matrix##X##Size##TypeSuffix = Eigen::Matrix<Type, -1, Size, 0, -1, Size>;

#define AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
    AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 2, 2)         \
    AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 3, 3)         \
    AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 4, 4)         \
    AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, -1, X)        \
    AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS(Type, TypeSuffix, 2)      \
    AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS(Type, TypeSuffix, 3)      \
    AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(autodiff::var, var)

#undef AUTODIFF_DEFINE_EIGEN_TYPEDEFS
#undef AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS
#undef AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES
}  // namespace autodiff
