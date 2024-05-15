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

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>

#include "meta.hpp"
#include "numtraits.hpp"

namespace autodiff {

namespace detail {

// clang-format off
using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::cosh;
using std::erf;
using std::exp;
using std::hypot;
using std::log;
using std::log10;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

template <typename T> struct Expr;
template <typename T> struct VaribleExpr;
template <typename T> struct IndependentVariableExpr;
template <typename T> struct DependentVariableExpr;
template <typename T> struct ConstantExpr;
template <typename T> struct UnaryExpr;
template <typename T> struct NegativeExpr;
template <typename T> struct BinaryExpr;
template <typename T> struct TernaryExpr;
template <typename T> struct AddExpr;
template <typename T> struct SubExpr;
template <typename T> struct MulExpr;
template <typename T> struct DivExpr;
template <typename T> struct SinExpr;
template <typename T> struct CosExpr;
template <typename T> struct TanExpr;
template <typename T> struct SinhExpr;
template <typename T> struct CoshExpr;
template <typename T> struct TanhExpr;
template <typename T> struct ArcSinExpr;
template <typename T> struct ArcCosExpr;
template <typename T> struct ArcTanExpr;
template <typename T> struct ArcTan2Expr;
template <typename T> struct ExpExpr;
template <typename T> struct LogExpr;
template <typename T> struct Log10Expr;
template <typename T> struct PowExpr;
template <typename T> struct SqrtExpr;
template <typename T> struct AbsExpr;
template <typename T> struct ErfExpr;
template <typename T> struct Hypot2Expr;
template <typename T> struct Hypot3Expr;

template <typename T> using ExprPtr = std::shared_ptr<Expr<T>>;
template <typename T> struct Variable;

// clang-format on
namespace traits {

template <typename T>
struct VariableValueTypeNotDefinedFor {};

template <typename T>
struct isVariable {
    constexpr static bool value = false;
};

template <typename T>
struct isVariable<Variable<T>> {
    constexpr static bool value = true;
};

template <typename T>
struct VariableValueType {
    using type = std::conditional_t<isArithmetic<T>, T, VariableValueTypeNotDefinedFor<T>>;
};

template <typename T>
struct VariableValueType<ExprPtr<T>> {
    using type = typename VariableValueType<T>::type;
};

template <typename T>
struct VariableOrder {
    constexpr static auto value = 0;
};

template <typename T>
struct VariableOrder<Variable<T>> {
    constexpr static auto value = 1 + VariableOrder<T>::value;
};

}  // namespace traits

template <typename T>
using VariableValueType = typename traits::VariableValueType<T>::type;

template <typename T>
constexpr auto VariableOrder = traits::VariableOrder<T>::value;

template <typename T>
constexpr auto isVariable = traits::isVariable<T>::value;

// The abstract type of any node type in the expression tree.
template <typename T>
struct Expr {
    /// The value of this expression node.
    T val = {};

    /// Construct an Expr object with given value.
    explicit Expr(const T& v) : val(v) {}

    /// Destructor (to avoid warning)
    virtual ~Expr() {}

    /// Bind a value pointer for writing the derivate during propagation
    virtual void bind_value(T* /* grad */) {}

    /// Bind an expression pointer for writing the derivate expression during
    /// propagation
    virtual void bind_expr(ExprPtr<T>* /*gradx*/){};

    /// Update the contribution of this expression in the derivate of the father
    /// node of the expression.
    /// @param wprime the derivate of the father expression node w.r.t the child
    /// expression of this expression node.
    /// ./doc folder $v_{j}^{-}$
    virtual void propagate(const T& wprime) = 0;

    /// Update the contribution of this expression in the derivate of the father
    /// node of the expression.
    /// @param wprime the derivate of the father expression node w.r.t the child
    /// expression of this expression node(as an expression).
    virtual void propagatex(const ExprPtr<T>& wprime) = 0;

    /// Update the value of this expression
    virtual void update() = 0;
};

/// The node in the expression tree representing either an independent or
/// dependent variable.
template <typename T>
struct VariableExpr : Expr<T> {
    /// The derivative value of the root expression node w.r.t. this variable.
    T* gradPtr = {};

    ExprPtr<T>* gradxPtr = {};

    /// Construct a VariableExpr object with given value.
    VariableExpr(const T& v) : Expr<T>(v) {}

    virtual void bind_value(T* grad) { gradPtr = grad; }
    virtual void bind_expr(ExprPtr<T>* gradx) { gradxPtr = gradx; }
};

// The node in the expression tree representing an independent variable.
template <typename T>
struct IndependentVariableExpr : VariableExpr<T> {
    // Using declarations for data members of base class
    using VariableExpr<T>::gradPtr;
    using VariableExpr<T>::gradxPtr;

    IndependentVariableExpr(const T& v) : VariableExpr<T>(v) {}
    void propagate(const T& wprime) override {
        if (gradPtr) {
            *gradPtr += wprime;
        }
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        if (gradxPtr) {
            *gradxPtr = *gradxPtr + wprime;
        }
    }

    void update() override {}
};

/// The node int the expression tree representing a dependent vaiable.
template <typename T>
struct DependentVariableExpr : VariableExpr<T> {
    // Using declarations for data members of base class
    using VariableExpr<T>::gradPtr;
    using VariableExpr<T>::gradxPtr;

    /// The expression tree that defines how the depend variable is calculated.
    ExprPtr<T> expr;

    DependentVariableExpr(const ExprPtr<T>& e) : VariableExpr<T>(e->val), expr(e) {}

    void propagate(const T& wprime) override {
        if (gradPtr) {
            *gradPtr += wprime;
        }
        expr->propagate(wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        if (gradxPtr) {
            *gradxPtr = *gradxPtr + wprime;
        }
        expr->propagatex(wprime);
    }

    void update() override {
        expr->update();
        this->val = expr->val;
    }
};

template <typename T>
struct ConstantExpr : Expr<T> {
    using Expr<T>::Expr;

    void propagate(const T& wprime) override{};

    void propagatex(const ExprPtr<T>& wprime) override{};

    void update() override {}
};

template <typename T>
ExprPtr<T> constant(const T& val) {
    return std::make_shared<ConstantExpr<T>>(val);
}

template <typename T>
struct UnaryExpr : Expr<T> {
    ExprPtr<T> x;
    UnaryExpr(const T& v, const ExprPtr<T>& e) : Expr<T>(v), x(e) {}
};

// @ref runoob.com/w3cnote/cpp11-inheritance-constructor.html

template <typename T>
struct NegativeExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using UnaryExpr<T>::UnaryExpr;  // 声明基类构造函数为派生类构造函数的一部分

    void propagate(const T& wprime) override { x->propagate(-wprime); }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(-wprime); }

    void update() override {
        x->update();
        this->val = -(x->val);
    }
};

template <typename T>
struct BinaryExpr : Expr<T> {
    ExprPtr<T> l, r;

    BinaryExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : Expr<T>(v), l(ll), r(rr) {}
};

template <typename T>
struct TernaryExpr : Expr<T> {
    ExprPtr<T> l, c, r;
    TernaryExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& cc, const ExprPtr<T>& rr)
        : Expr<T>(v), l(ll), c(cc), r(rr) {}
};

template <typename T>
struct AddExpr : BinaryExpr<T> {
    // Using declarations for data members of base class.
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    void propagate(const T& wprime) override {
        l->propagate(wprime);
        r->propagate(wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        l->propagatex(wprime);
        r->propagatex(wprime);
    }

    void update() override {
        l->update();
        r->update();
        this->val = l->val + r->val;
    }
};

template <typename T>
struct SubExpr : BinaryExpr<T> {
    // Using declarations for data members of base class.
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    void propagate(const T& wprime) override {
        l->propagate(wprime);
        r->propagate(-wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        l->propagatex(wprime);
        r->propagatex(-wprime);
    }

    void update() override {
        l->update();
        r->update();
        this->val = l->val - r->val;
    }
};

template <typename T>
struct MulExpr : BinaryExpr<T> {
    // Using declarations for data members of base class.
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    void propagate(const T& wprime) override {
        l->propagate(wprime * r->val);
        r->propagate(wprime * l->val);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        l->propagatex(wprime * r);
        r->propagatex(wprime * l);
    }

    void update() override {
        l->update();
        r->update();
        this->val = l->val * r->val;
    }
};

template <typename T>
struct DivExpr : BinaryExpr<T> {
    // Using declarations for data members of base class.
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    void propagate(const T& wprime) override {
        const auto aux1 = 1.0 / r->val;
        const auto aux2 = -l->val * aux1 * aux1;
        l->propagate(wprime * aux1);
        r->propagate(wprime * aux2);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        const auto aux1 = 1.0 / r;
        const auto aux2 = -l * aux1 * aux1;
        l->propagatex(wprime * aux1);
        r->propagatex(wprime * aux2);
    }

    void update() override {
        l->update();
        r->update();
        this->val = l->val / r->val;
    }
};

template <typename T>
struct SinExpr : UnaryExpr<T> {
    using UnaryExpr<T>::x;

    SinExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override { x->propagate(wprime * cos(x->val)); }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(wprime * cos(x)); }

    void update() override {
        x->update();
        this->val = sin(x->val);
    }
};

template <typename T>
struct CosExpr : UnaryExpr<T> {
    using UnaryExpr<T>::x;

    CosExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override { x->propagate(-wprime * sin(x->val)); }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(-wprime * sin(x)); }

    void update() override {
        x->update();
        this->val = cos(x->val);
    }
};

template <typename T>
struct TanExpr : UnaryExpr<T> {
    using UnaryExpr<T>::x;

    TanExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override {
        const auto aux = 1.0 / cos(x->val);
        x->propagate(wprime * aux * aux);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        const auto aux = 1.0 / cos(x);
        x->propagatex(wprime * aux * aux);
    }

    void update() override {
        x->update();
        this->val = tan(x->val);
    }
};

template <typename T>
struct SinhExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    SinhExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override { x->propagate(wprime * cosh(x->val)); }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(wprime * cosh(x)); }

    void update() override {
        x->update();
        this->val = sinh(x->val);
    }
};

template <typename T>
struct CoshExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    CoshExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override { x->propagate(wprime * sinh(x->val)); }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(wprime * sinh(x)); }

    void update() override {
        x->update();
        this->val = cosh(x->val);
    }
};

template <typename T>
struct TanhExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    TanhExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override {
        const auto aux = 1.0 / cosh(x->val);
        x->propagate(wprime * aux * aux);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        const auto aux = 1.0 / cosh(x);
        x->propagatex(wprime * aux * aux);
    }

    void update() override {
        x->update();
        this->val = tanh(x->val);
    }
};

template <typename T>
struct ArcSinExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcSinExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override { x->propagate(wprime / sqrt(1.0 - x->val * x->val)); }

    void propagatex(const ExprPtr<T>& wprime) override {
        x->propagatex(wprime / sqrt(1.0 - x * x));
    }

    void update() override {
        x->update();
        this->val = asin(x->val);
    }
};

template <typename T>
struct ArcCosExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcCosExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override {
        x->propagate(-wprime / sqrt(1.0 - x->val * x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        x->propagatex(-wprime / sqrt(1.0 - x * x));
    }

    void update() override {
        x->update();
        this->val = acos(x->val);
    }
};

template <typename T>
struct ArcTanExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcTanExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override { x->propagate(wprime / (1.0 + x->val * x->val)); }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(wprime / (1.0 + x * x)); }

    void update() override {
        x->update();
        this->val = atan(x->val);
    }
};

template <typename T>
struct ArcTan2Expr : BinaryExpr<T> {
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    ArcTan2Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr)
        : BinaryExpr<T>(v, ll, rr) {}

    void propagate(const T& wprime) override {
        const auto aux = wprime / (l->val * l->val + r->val * r->val);
        l->propagate(r->val * aux);
        r->propagate(-l->val * aux);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        const auto aux = wprime / (l * l + r * r);
        l->propagatex(r * aux);
        r->propagatex(-l * aux);
    }

    void update() override {
        l->update();
        r->update();
        this->val = atan2(l->val, r->val);
    }
};

template <typename T>
struct ExpExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::UnaryExpr;
    using UnaryExpr<T>::val;
    using UnaryExpr<T>::x;

    void propagate(const T& wprime) override {
        x->propagate(wprime * val);  // exp(x)' = exp(x) * x'
    }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(wprime * exp(x)); }

    void update() override {
        x->update();
        this->val = exp(x->val);
    }
};

template <typename T>
struct LogExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using UnaryExpr<T>::UnaryExpr;

    void propagate(const T& wprime) override {
        x->propagate(wprime / x->val);  // log(x)' = x'/x
    }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(wprime / x); }

    void update() override {
        x->update();
        this->val = log(x->val);
    }
};

template <typename T>
struct Log10Expr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto ln10 =
        static_cast<VariableValueType<T>>(2.3025850929940456840179914546843);

    Log10Expr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override { x->propagate(wprime / (ln10 * x->val)); }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(wprime / (ln10 * x)); }

    void update() override {
        x->update();
        this->val = log10(x->val);
    }
};
template <typename T>
struct PowExpr : BinaryExpr<T> {
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    T log_l;

    PowExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr)
        : BinaryExpr<T>(v, ll, rr), log_l(log(ll->val)) {}

    void propagate(const T& wprime) override {
        using U = VariableValueType<T>;
        constexpr auto zero = U(0.0);
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * pow(lval, rval - 1);
        l->propagate(aux * rval);
        const auto auxr = lval == zero ? 0.0 : lval * log(lval);  // since x*log(x) -> 0 as x -> 0
        r->propagate(aux * auxr);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        using U = VariableValueType<T>;
        constexpr auto zero = U(0.0);
        const auto aux = wprime * pow(l, r - 1);
        l->propagatex(aux * r);
        const auto auxr = l == zero ? 0.0 * l : l * log(l);  // since x*log(x) -> 0 as x -> 0
        r->propagatex(aux * auxr);
    }

    void update() override {
        l->update();
        r->update();
        this->val = pow(l->val, r->val);
    }
};

template <typename T>
struct PowConstantLeftExpr : BinaryExpr<T> {
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    PowConstantLeftExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr)
        : BinaryExpr<T>(v, ll, rr) {}

    void propagate(const T& wprime) override {
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * pow(lval, rval - 1);
        const auto auxr = lval == 0.0 ? 0.0 : lval * log(lval);  // since x*log(x) -> 0 as x -> 0
        r->propagate(aux * auxr);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        const auto aux = wprime * pow(l, r - 1);
        const auto auxr = l == 0.0 ? 0.0 * l : l * log(l);  // since x*log(x) -> 0 as x -> 0
        r->propagatex(aux * auxr);
    }

    void update() override {
        r->update();
        this->val = pow(l->val, r->val);
    }
};

template <typename T>
struct PowConstantRightExpr : BinaryExpr<T> {
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    PowConstantRightExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr)
        : BinaryExpr<T>(v, ll, rr) {}

    void propagate(const T& wprime) override {
        l->propagate(wprime * pow(l->val, r->val - 1) *
                     r->val);  // pow(l, r)'l = r * pow(l, r - 1) * l'
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        l->propagatex(wprime * pow(l, r - 1) * r);
    }

    void update() override {
        l->update();
        this->val = pow(l->val, r->val);
    }
};

template <typename T>
struct SqrtExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    SqrtExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override {
        x->propagate(wprime / (2.0 * sqrt(x->val)));  // sqrt(x)' = 1/2 * 1/sqrt(x) * x'
    }

    void propagatex(const ExprPtr<T>& wprime) override { x->propagatex(wprime / (2.0 * sqrt(x))); }

    void update() override {
        x->update();
        this->val = sqrt(x->val);
    }
};

template <typename T>
struct AbsExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using U = VariableValueType<T>;

    AbsExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override {
        if (x->val < 0.0) x->propagate(-wprime);
        else if (x->val > 0.0)
            x->propagate(wprime);
        else
            x->propagate(T(0));
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        if (x->val < 0.0) x->propagatex(-wprime);
        else if (x->val > 0.0)
            x->propagatex(wprime);
        else
            x->propagate(T(0));
    }

    void update() override {
        x->update();
        this->val = abs(x->val);
    }
};

template <typename T>
struct ErfExpr : UnaryExpr<T> {
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto sqrt_pi =
        static_cast<VariableValueType<T>>(1.7724538509055160272981674833411451872554456638435);

    ErfExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override {
        const auto aux =
            2.0 / sqrt_pi * exp(-(x->val) * (x->val));  // erf(x)' = 2/sqrt(pi) * exp(-x * x) * x'
        x->propagate(wprime * aux);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        const auto aux = 2.0 / sqrt_pi * exp(-x * x);
        x->propagatex(wprime * aux);
    }

    void update() override {
        x->update();
        this->val = erf(x->val);
    }
};

template <typename T>
struct Hypot2Expr : BinaryExpr<T> {
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    Hypot2Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    void propagate(const T& wprime) override {
        l->propagate(wprime * l->val / val);  // sqrt(l*l + r*r)'l = 1/2 * 1/sqrt(l*l + r*r) *
                                              // (2*l*l') = (l*l')/sqrt(l*l + r*r)
        r->propagate(wprime * r->val / val);  // sqrt(l*l + r*r)'r = 1/2 * 1/sqrt(l*l + r*r) *
                                              // (2*r*r') = (r*r')/sqrt(l*l + r*r)
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        l->propagatex(wprime * l / hypot(l, r));
        r->propagatex(wprime * r / hypot(l, r));
    }

    void update() override {
        l->update();
        r->update();
        this->val = hypot(l->val, r->val);
    }
};

template <typename T>
struct Hypot3Expr : TernaryExpr<T> {
    // Using declarations for data members of base class
    using TernaryExpr<T>::val;
    using TernaryExpr<T>::l;
    using TernaryExpr<T>::c;
    using TernaryExpr<T>::r;

    Hypot3Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& cc, const ExprPtr<T>& rr)
        : TernaryExpr<T>(v, ll, cc, rr) {}

    void propagate(const T& wprime) override {
        l->propagate(wprime * l->val / val);
        c->propagate(wprime * c->val / val);
        r->propagate(wprime * r->val / val);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        l->propagatex(wprime * l / hypot(l, c, r));
        c->propagatex(wprime * c / hypot(l, c, r));
        r->propagatex(wprime * r / hypot(l, c, r));
    }

    void update() override {
        l->update();
        c->update();
        r->update();
        this->val = hypot(l->val, c->val, r->val);
    }
};

/// Any expression yielding a boolean depending on arithmetic subexpressions
struct BooleanExpr {
    std::function<bool()> expr;
    bool val = {};

    explicit BooleanExpr(std::function<bool()> expression) : expr(std::move(expression)) {
        update();
    }
    operator bool() const { return val; }

    void update() { val = expr(); }

    auto operator!() const {
        return BooleanExpr([this]() { return !(expr()); });
    }
};

/// Capture numeric comparison between two expression trees
template <typename T, typename Comparator>
auto expr_comparison(const ExprPtr<T>& l, const ExprPtr<T>& r, Comparator&& compare) {
    return BooleanExpr([=]() mutable -> bool {
        l->update();
        r->update();
        return compare(l->val, r->val);
    });
}

template <typename Op>
auto bool_expr_op(BooleanExpr& l, BooleanExpr& r, Op op) {
    return BooleanExpr([=]() mutable -> bool {
        l.update();  // l and r can implicit conversion to bool.
        r.update();
        return op(l, r);
    });
}

inline auto operator&&(BooleanExpr&& l, BooleanExpr&& r) {
    return bool_expr_op(l, r, std::logical_and<>{});
}

inline auto operator||(BooleanExpr&& l, BooleanExpr&& r) {
    return bool_expr_op(l, r, std::logical_or<>{});
}

// Select between expression branches depending on a boolean expression
template <typename T>
struct ConditionalExpr : Expr<T> {
    // Using declarations for data members of base class
    BooleanExpr predicate;
    using Expr<T>::val;
    ExprPtr<T> l, r;

    ConditionalExpr(const BooleanExpr& wrappedPred, const ExprPtr<T>& ll, const ExprPtr<T>& rr)
        : Expr<T>(wrappedPred ? ll->val : rr->val), predicate(wrappedPred), l(ll), r(rr) {}

    ExprPtr<T> derive(const ExprPtr<T> left, const ExprPtr<T>& right) const {
        return std::make_shared<ConditionalExpr>(predicate, left, right);
    }

    void propagate(const T& wprime) override {
        if (predicate.val) l->propagate(wprime);
        else
            r->propagate(wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override {
        l->propagatex(derive(wprime, constant<T>(0.0)));
        r->propagatex(derive(constant<T>(0.0), wprime));
    }

    void update() override {
        predicate.update();
        if (predicate.val) {
            l->update();
            this->val = l->val;
        } else {
            r->update();
            this->val = r->val;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
////////////////////// ARITHMETIC OPERATORS FOR EXPRESSION/////////////////////
///////////////////////////////////////////////////////////////////////////////

/// Unary Operators
template <typename T>
ExprPtr<T> operator+(const ExprPtr<T>& r) {
    return r;
}

template <typename T>
ExprPtr<T> operator-(const ExprPtr<T>& r) {
    return std::make_shared<NegativeExpr<T>>(-r->val, r);
}

//-----------------------------------------------------------------------------
// MACRO FOR EXPRESSION ARITHMETIC OPERATOR
//-----------------------------------------------------------------------------

#define OPERATOR_FOR_EXPR_OP_EXPR(OP, OPERATOR)                              \
    template <typename T>                                                    \
    ExprPtr<T> operator OPERATOR(const ExprPtr<T>& l, const ExprPtr<T>& r) { \
        return std::make_shared<OP##Expr<T>>(l->val OPERATOR r->val, l, r);  \
    }                                                                        \
    //

OPERATOR_FOR_EXPR_OP_EXPR(Add, +)
OPERATOR_FOR_EXPR_OP_EXPR(Sub, -)
OPERATOR_FOR_EXPR_OP_EXPR(Mul, *)
OPERATOR_FOR_EXPR_OP_EXPR(Div, /)

#undef OPERATOR_FOR_EXPR_OP_EXPR

/// Arithmetic OPERATOR Expration

#define OPERATOR_FOR_CONSTANT_OP_EXPR(OP, OPERATOR)                             \
    template <typename T, typename U, detail::Requires<isArithmetic<U>> = true> \
    ExprPtr<T> operator OPERATOR(const U& l, const ExprPtr<T>& r) {             \
        return constant<T>(l) OPERATOR r;                                       \
    }

OPERATOR_FOR_CONSTANT_OP_EXPR(Add, +)
OPERATOR_FOR_CONSTANT_OP_EXPR(Sub, -)
OPERATOR_FOR_CONSTANT_OP_EXPR(Mul, *)
OPERATOR_FOR_CONSTANT_OP_EXPR(Div, /)

#undef OPERATOR_FOR_CONSTANT_OP_EXPR

// template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
// ExprPtr<T> operator+(const U& l, const ExprPtr<T>& r) {
//   return constant<T>(l) + r;
// }

// template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
// ExprPtr<T> operator-(const U& l, const ExprPtr<T>& r) {
//   return constant<T>(l) - r;
// }

// template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
// ExprPtr<T> operator*(const U& l, const ExprPtr<T>& r) {
//   return constant<T>(l) * r;
// }

// template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
// ExprPtr<T> operator/(const U& l, const ExprPtr<T>& r) {
//   return constant<T>(l) / r;
// }

/// Expration OPERATOR Arithmetic

#define OPERATOR_FOR_EXPR_OP_CONSTANT(OP, OPERATOR)                             \
    template <typename T, typename U, detail::Requires<isArithmetic<U>> = true> \
    ExprPtr<T> operator OPERATOR(const ExprPtr<T>& l, const U& r) {             \
        return l OPERATOR constant<T>(r);                                       \
    }                                                                           \
//
OPERATOR_FOR_EXPR_OP_CONSTANT(Add, +)
OPERATOR_FOR_EXPR_OP_CONSTANT(Sub, -)
OPERATOR_FOR_EXPR_OP_CONSTANT(Mul, *)
OPERATOR_FOR_EXPR_OP_CONSTANT(Div, /)

#undef OPERATOR_FOR_EXPR_OP_CONSTANT

// template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
// ExprPtr<T> operator+(const ExprPtr<T>& l, const U& r) {
//   return l + constant<T>(r);
// }

// template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
// ExprPtr<T> operator-(const ExprPtr<T>& l, const U& r) {
//   return l - constant<T>(r);
// }

// template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
// ExprPtr<T> operator*(const ExprPtr<T>& l, const U& r) {
//   return l * constant<T>(r);
// }

// template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
// ExprPtr<T> operator/(const ExprPtr<T>& l, const U& r) {
//   return l / constant<T>(r);
// }

//-----------------------------------------------------------------------------
// MACRO FOR EXPRESSION UNARY OPERATOR
//-----------------------------------------------------------------------------

#define OPERATOR_FOR_EXPR_UNARY(OP, OPERATOR)                      \
    template <typename T>                                          \
    ExprPtr<T> OPERATOR(const ExprPtr<T>& x) {                     \
        return std::make_shared<OP##Expr<T>>(OPERATOR(x->val), x); \
    }                                                              \
                                                                   \
// TRIGONOMETRIC
OPERATOR_FOR_EXPR_UNARY(Sin, sin)
OPERATOR_FOR_EXPR_UNARY(Cos, cos)
OPERATOR_FOR_EXPR_UNARY(Tan, tan)
OPERATOR_FOR_EXPR_UNARY(ArcSin, asin)
OPERATOR_FOR_EXPR_UNARY(ArcCos, acos)
OPERATOR_FOR_EXPR_UNARY(ArcTan, atan)

/// HYPERBOLIC OPERATOR
OPERATOR_FOR_EXPR_UNARY(Sinh, sinh)
OPERATOR_FOR_EXPR_UNARY(Cosh, cosh)
OPERATOR_FOR_EXPR_UNARY(Tanh, tanh)

/// EXPONENTIAL AND LOGARITHMIC
OPERATOR_FOR_EXPR_UNARY(Exp, exp)
OPERATOR_FOR_EXPR_UNARY(Log, log)
OPERATOR_FOR_EXPR_UNARY(Log10, log10)

/// ABS AND ERF
OPERATOR_FOR_EXPR_UNARY(Abs, abs)
OPERATOR_FOR_EXPR_UNARY(Erf, erf)

/// SQRT
OPERATOR_FOR_EXPR_UNARY(Sqrt, sqrt)

#undef OPERATOR_FOR_EXPR_UNARY

//-----------------------------------------------------------------------------
// ATAN2 OPERATOR
//-----------------------------------------------------------------------------

template <typename T>
ExprPtr<T> atan2(const ExprPtr<T>& l, const ExprPtr<T>& r) {
    return std::make_shared<ArcTan2Expr<T>>(atan2(l->val, r->val), l, r);
}
template <typename T, typename U, Requires<isArithmetic<U>> = true>
ExprPtr<T> atan2(const U& l, const ExprPtr<T>& r) {
    return std::make_shared<ArcTan2Expr<T>>(atan2(l, r->val), constant<T>(l), r);
}
template <typename T, typename U, Requires<isArithmetic<U>> = true>
ExprPtr<T> atan2(const ExprPtr<T>& l, const U& r) {
    return std::make_shared<ArcTan2Expr<T>>(atan2(l->val, r), l, constant<T>(r));
}

//-----------------------------------------------------------------------------
// POWER OPERATOR
//-----------------------------------------------------------------------------
template <typename T>
ExprPtr<T> pow(const ExprPtr<T>& l, const ExprPtr<T>& r) {
    return std::make_shared<PowExpr<T>>(pow(l->val, r->val), l, r);
}

template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> pow(const U& l, const ExprPtr<T>& r) {
    return std::make_shared<PowConstantLeftExpr<T>>(pow(l, r->val), constant<T>(l), r);
}

template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> pow(const ExprPtr<T>& l, const U& r) {
    return std::make_shared<PowConstantRightExpr<T>>(pow(l->val, r), l, constant<T>(r));
}

//-----------------------------------------------------------------------------
// ABS2
//-----------------------------------------------------------------------------

template <typename T>
ExprPtr<T> abs2(const ExprPtr<T>& x) {
    return x * x;
}

//-----------------------------------------------------------------------------
// Complex
//-----------------------------------------------------------------------------
template <typename T>
ExprPtr<T> conj(const ExprPtr<T>& x) {
    return x;
}

template <typename T>
ExprPtr<T> real(const ExprPtr<T>& x) {
    return x;
}

template <typename T>
ExprPtr<T> imag(const ExprPtr<T>& x) {
    return constant<T>(0.0);
}

//-----------------------------------------------------------------------------
// HYPOT2 OPERATOR
//-----------------------------------------------------------------------------

template <typename T>
ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& r) {
    return std::make_shared<Hypot2Expr<T>>(hypot(l->val, r->val), l, r);
}
template <typename T, typename U, Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const U& l, const ExprPtr<T>& r) {
    return std::make_shared<Hypot2Expr<T>>(hypot(l, r->val), constant<T>(l), r);
}
template <typename T, typename U, Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const ExprPtr<T>& l, const U& r) {
    return std::make_shared<Hypot2Expr<T>>(hypot(l->val, r), l, constant<T>(r));
}

//-----------------------------------------------------------------------------
// HYPOT3 OPERATOR
//-----------------------------------------------------------------------------

template <typename T>
ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& c, const ExprPtr<T>& r) {
    return std::make_shared<Hypot3Expr<T>>(hypot(l->val, c->val, r->val), l, c, r);
}

template <typename T, typename U, Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& c, const U& r) {
    return std::make_shared<Hypot3Expr<T>>(hypot(l->val, c->val, r), l, c, constant<T>(r));
}

template <typename T, typename U, Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const U& l, const ExprPtr<T>& c, const ExprPtr<T>& r) {
    return std::make_shared<Hypot3Expr<T>>(hypot(l, c->val, r->val), constant<T>(l), c, r);
}

template <typename T, typename U, Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const ExprPtr<T>& l, const U& c, const ExprPtr<T>& r) {
    return std::make_shared<Hypot3Expr<T>>(hypot(l->val, c, r->val), l, constant<T>(c), r);
}

template <typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true>
ExprPtr<T> hypot(const ExprPtr<T>& l, const U& c, const V& r) {
    return std::make_shared<Hypot3Expr<T>>(hypot(l->val, c, r), l, constant<T>(c), constant<T>(r));
}

template <typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true>
ExprPtr<T> hypot(const U& l, const ExprPtr<T>& c, const V& r) {
    return std::make_shared<Hypot3Expr<T>>(hypot(l, c->val, r), constant<T>(l), c, constant<T>(r));
}

template <typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true>
ExprPtr<T> hypot(const V& l, const U& c, const ExprPtr<T>& r) {
    return std::make_shared<Hypot3Expr<T>>(hypot(l, c, r->val), constant<T>(l), constant<T>(c), r);
}

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Variable //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Variable {
    /// The pointer to the expression tree of variable operations
    ExprPtr<T> expr;

    using ArithmeticType = T;

    /// Default ctor
    Variable() : Variable(0.0) {}

    /// Copy ctor
    Variable(const Variable& other) : Variable(other.expr) {}

    /// Template ctor with arithmetic value.
    template <typename U, detail::Requires<isArithmetic<U>> = true>
    Variable(const U& val) : expr(std::make_shared<IndependentVariableExpr<T>>(val)) {}

    /// Expr ctor
    Variable(const ExprPtr<T>& e) : expr(std::make_shared<DependentVariableExpr<T>>(e)) {}

    /// Default copy assignment;
    Variable& operator=(const Variable&) = default;

    /// Update the value of this variable with changes in its expression tree
    void update() { expr->update(); }

    void update(T value) {
        if (auto independentExpr = std::dynamic_pointer_cast<IndependentVariableExpr<T>>(expr)) {
            independentExpr->val = value;
            independentExpr->update();
        } else {
            throw std::logic_error(
                "Cannot update the value of a dependent expression stored in a "
                "variable.");
        }
    }

    /// Implicitly convert this Variable object into an expression pointer.
    operator const ExprPtr<T> &() const { return expr; }

    /// Assign an arithmetic value to this variable.
    // clang-format off
  template < typename U, detail::Requires<isArithmetic<U>> = true> 
  auto operator=(const U& val)->Variable& {
    *this = Variable(val);
    return *this;
  }
    // clang-format on

    auto operator=(const ExprPtr<T>& x) -> Variable& {
        *this = Variable(x);
        return *this;
    }

    /// Assignment operators
    Variable& operator+=(const ExprPtr<T>& x) {
        *this = Variable(expr + x);
        return *this;
    }

    Variable& operator-=(const ExprPtr<T>& x) {
        *this = Variable(expr - x);
        return *this;
    }

    Variable& operator*=(const ExprPtr<T>& x) {
        *this = Variable(expr * x);
        return *this;
    }

    Variable& operator/=(const ExprPtr<T>& x) {
        *this = Variable(expr / x);
        return *this;
    }

    /// Assignment operators with arithmetic values

    template <typename U, detail::Requires<isArithmetic<U>> = true>
    Variable& operator+=(const U& x) {
        *this = Variable(expr + x);
        return *this;
    }

    template <typename U, detail::Requires<isArithmetic<U>> = true>
    Variable& operator-=(const U& x) {
        *this = Variable(expr - x);
        return *this;
    }

    template <typename U, detail::Requires<isArithmetic<U>> = true>
    Variable& operator*=(const U& x) {
        *this = Variable(expr * x);
        return *this;
    }

    template <typename U, detail::Requires<isArithmetic<U>> = true>
    Variable& operator/=(const U& x) {
        *this = Variable(expr / x);
        return *this;
    }
    ///  Implicit Conversion
    operator T() const { return expr->val; }

    template <typename U>
    operator U() const {
        return static_cast<U>(expr->val);
    }

    // Explicit Conversion
    // explicit operator T() const { return expr->val; }

    // template <typename U>
    // explicit operator U() const {
    //   return static_cast<U>(expr->val);
    // }
};

// template <typename T>
// Variable() -> Variable<double>;

///////////////////////////////////////////////////////////////////////////////
//////////////////////////////EXPRESSION TRAITS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename T, detail::Requires<isArithmetic<T>> = true>
T expr_value(const T& t) {
    return t;
}

template <typename T>
T expr_value(const ExprPtr<T>& t) {
    return t->val;
}

template <typename T>
T expr_value(const Variable<T>& t) {
    return t.expr->val;
}

template <typename T, typename U>
using expr_common_t = std::common_type_t<decltype(expr_value(std::declval<T>())),
                                         decltype(expr_value(std::declval<T>()))>;

template <typename>
struct sfinae_true : std::true_type {};

template <typename T>
static auto is_expr_test(int) -> sfinae_true<decltype(expr_value(std::declval<T>()))>;

template <typename T>
static auto is_expr_test(long) -> std::false_type;

template <typename T>
struct is_expr : decltype(is_expr_test<T>(0)) {};

template <typename T>
constexpr bool is_expr_v = is_expr<T>::value;

static_assert(is_expr_v<ExprPtr<double>>);
static_assert(is_expr_v<Variable<double>>);
// static_assert(!is_expr_v<AddExpr<double>>);

// template <typename T>
// using AddPtr = std::shared_ptr<AddExpr<T>>;
// static_assert(!is_expr<AddPtr<double>>::value);

// template <typename T, typename U, Requires<isArithmetic<U>> = true>
// ExprPtr<T> coerce_expr(const U& t) {
//   return constant<T>(t);
// }

template <typename T, Requires<isArithmetic<T>> = true>
ExprPtr<T> coerce_expr(const T& t) {
    return constant<T>(t);
}

template <typename T>
ExprPtr<T> coerce_expr(const ExprPtr<T>& t) {
    return t;
}

template <typename T>
ExprPtr<T> coerce_expr(const Variable<T>& t) {
    return t.expr;
}

// static_assert(is_expr_v<decltype(coerce_expr(double(1.1)))>);
static_assert(is_expr_v<decltype(coerce_expr(std::declval<ExprPtr<double>>()))>);

// static_assert(
//     is_expr_v<decltype(coerce_expr(std::declval<AddExpr<double>>()))>);

static_assert(is_expr_v<decltype(coerce_expr(std::declval<Variable<double>>()))>);

template <typename T, typename U>
struct is_binary_expr
    : std::conditional_t<!(isArithmetic<T> && isArithmetic<U>)&&is_expr_v<T> && is_expr_v<U>,
                         std::true_type, std::false_type> {};

template <typename T, typename U>
constexpr bool is_binary_expr_v = is_binary_expr<T, U>::value;

static_assert(is_binary_expr_v<Variable<double>, ExprPtr<double>>);
static_assert(is_binary_expr_v<ExprPtr<double>, ExprPtr<double>>);
static_assert(is_binary_expr_v<Variable<double>, Variable<double>>);

///////////////////////////////////////////////////////////////////////////////
////////////////////// COMPARISION OPERATORS //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename Comparator, typename T, typename U>
auto comparison_operator(const T& t, const U& u) {
    using C = expr_common_t<T, U>;

    return expr_comparison(coerce_expr<C>(t), coerce_expr<C>(u), Comparator{});

};  /// -> Return BooleanExpr

//

//-----------------------------------------------------------------------------
// MACRO FOR VARIABLE BINARY COMPARISON
//-----------------------------------------------------------------------------
#define OPERATOR_FOR_BINARY_COMPARISON(OP, OPERATOR)                                   \
    template <typename T, typename U, detail::Requires<is_binary_expr_v<T, U>> = true> \
    auto operator OPERATOR(const T& t, const U& u) {                                   \
        return comparison_operator<OP<>>(t, u);                                        \
    }

//
OPERATOR_FOR_BINARY_COMPARISON(std::equal_to, ==)
OPERATOR_FOR_BINARY_COMPARISON(std::not_equal_to, !=)
OPERATOR_FOR_BINARY_COMPARISON(std::less_equal, <=)
OPERATOR_FOR_BINARY_COMPARISON(std::greater_equal, >=)
OPERATOR_FOR_BINARY_COMPARISON(std::less, <)
OPERATOR_FOR_BINARY_COMPARISON(std::greater, >)

#undef OPERATOR_FOR_BINARY_COMPARISON

///////////////////////////////////////////////////////////////////////////////
////////////////////// CONDITION OPERATORS ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, Requires<is_expr_v<T> && is_expr_v<U>> = true>
auto condition(BooleanExpr&& p, const T& t, const U& u) {
    using C = expr_common_t<T, U>;
    ExprPtr<C> expr = std::make_shared<ConditionalExpr<C>>(std::forward<BooleanExpr>(p),
                                                           coerce_expr<C>(t), coerce_expr<C>(u));
    return expr;
}

template <typename T, typename U, detail::Requires<is_binary_expr_v<T, U>> = true>
auto min(const T& x, const U& y) {
    return condition(x < y, x, y);
}

template <typename T, typename U, detail::Requires<is_binary_expr_v<T, U>> = true>
auto max(const T& x, const U& y) {
    return condition(x > y, x, y);
}

template <typename T>
ExprPtr<T> sgn(const ExprPtr<T>& x) {
    return condition(x < 0, -1.0, condition(x > 0, 1.0, 0.0));
}

template <typename T>
ExprPtr<T> sgn(const Variable<T>& x) {
    return condition(x.expr < 0, -1.0, condition(x.expr > 0, 1.0, 0.0));
}

///////////////////////////////////////////////////////////////////////////////
////////////////////// ARITHMETIC OPERATORS FOR VARIABLE //////////////////////
///////////////////////////////////////////////////////////////////////////////

// Unary Operator

template <typename T>
const ExprPtr<T>& operator+(const Variable<T>& r) {
    return r.expr;
}

template <typename T>
ExprPtr<T> operator-(const Variable<T>& r) {
    return -r.expr;
}

//-----------------------------------------------------------------------------
// MACRO FOR VARIABLE BINARY OPERATOR
//-----------------------------------------------------------------------------

#define OPERATOR_FOR_VAR_OP_VAR(OP, OPERATOR)                                  \
    template <typename T>                                                      \
    ExprPtr<T> operator OPERATOR(const Variable<T>& l, const Variable<T>& r) { \
        return l.expr OPERATOR r.expr;                                         \
    }                                                                          \
//
OPERATOR_FOR_VAR_OP_VAR(Add, +)
OPERATOR_FOR_VAR_OP_VAR(Sub, -)
OPERATOR_FOR_VAR_OP_VAR(Mul, *)
OPERATOR_FOR_VAR_OP_VAR(Div, /)

#undef OPERATOR_FOR_VAR_OP_VAR

#define OPERATOR_FOR_EXPR_OP_VAR(OP, OPERATOR)                                \
    template <typename T>                                                     \
    ExprPtr<T> operator OPERATOR(const ExprPtr<T>& l, const Variable<T>& r) { \
        return l OPERATOR r.expr;                                             \
    }

OPERATOR_FOR_EXPR_OP_VAR(Add, +)
OPERATOR_FOR_EXPR_OP_VAR(Sub, -)
OPERATOR_FOR_EXPR_OP_VAR(Mul, *)
OPERATOR_FOR_EXPR_OP_VAR(Div, /)

#undef OPERATOR_FOR_EXPR_OP_VAR

#define OPERATOR_FOR_VAR_OP_EXPR(OP, OPERATOR)                                \
    template <typename T>                                                     \
    ExprPtr<T> operator OPERATOR(const Variable<T>& l, const ExprPtr<T>& r) { \
        return l.expr OPERATOR r;                                             \
    }

OPERATOR_FOR_VAR_OP_EXPR(Add, +)
OPERATOR_FOR_VAR_OP_EXPR(Sub, -)
OPERATOR_FOR_VAR_OP_EXPR(Mul, *)
OPERATOR_FOR_VAR_OP_EXPR(Div, /)

#undef OPERATOR_FOR_VAR_OP_EXPR

#define OPERATOR_FOR_VAR_OP_CONSTANT(OP, OPERATOR)                              \
    template <typename T, typename U, detail::Requires<isArithmetic<U>> = true> \
    ExprPtr<T> operator OPERATOR(const Variable<T>& l, const U& r) {            \
        return l.expr OPERATOR r;                                               \
    }

OPERATOR_FOR_VAR_OP_CONSTANT(Add, +)
OPERATOR_FOR_VAR_OP_CONSTANT(Sub, -)
OPERATOR_FOR_VAR_OP_CONSTANT(Mul, *)
OPERATOR_FOR_VAR_OP_CONSTANT(Div, /)

#undef OPERATOR_FOR_VAR_OP_CONSTANT

#define OPERATOR_FOR_CONSTANT_OP_VAR(OP, OPERATOR)                              \
    template <typename T, typename U, detail::Requires<isArithmetic<U>> = true> \
    ExprPtr<T> operator OPERATOR(const U& l, const Variable<T>& r) {            \
        return l OPERATOR r.expr;                                               \
    }

OPERATOR_FOR_CONSTANT_OP_VAR(Add, +)
OPERATOR_FOR_CONSTANT_OP_VAR(Sub, -)
OPERATOR_FOR_CONSTANT_OP_VAR(Mul, *)
OPERATOR_FOR_CONSTANT_OP_VAR(Div, /)

#undef OPERATOR_FOR_CONSTANT_OP_VAR

//-----------------------------------------------------------------------------
// ATAN2 OPERATOR
//-----------------------------------------------------------------------------

template <typename T>
ExprPtr<T> atan2(const Variable<T>& l, const Variable<T>& r) {
    return atan2(l.expr, r.expr);
}

template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> atan2(const U& l, const Variable<T>& r) {
    return atan2(l, r.expr);
}

template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> atan2(const Variable<T>& l, const U& r) {
    return atan2(l.expr, r);
}

//-----------------------------------------------------------------------------
// HYPOT2 OPERATOR
//-----------------------------------------------------------------------------

template <typename T>
ExprPtr<T> hypot(const Variable<T>& l, const Variable<T>& r) {
    return hypot(l.expr, r.expr);
}
template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const U& l, const Variable<T>& r) {
    return hypot(l, r.expr);
}
template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const Variable<T>& l, const U& r) {
    return hypot(l.expr, r);
}

//-----------------------------------------------------------------------------
// HYPOT3 OPERATOR
//-----------------------------------------------------------------------------

template <typename T>
ExprPtr<T> hypot(const Variable<T>& l, const Variable<T>& c, const Variable<T>& r) {
    return hypot(l.expr, c.expr, r.expr);
}
template <typename T, typename U, typename V,
          detail::Requires<isArithmetic<U> && isArithmetic<V>> = true>
ExprPtr<T> hypot(const Variable<T>& l, const U& c, const V& r) {
    return hypot(l.expr, c, r);
}
template <typename T, typename U, typename V,
          detail::Requires<isArithmetic<U> && isArithmetic<V>> = true>
ExprPtr<T> hypot(const U& l, const Variable<T>& c, const V& r) {
    return hypot(l, c.expr, r);
}
template <typename T, typename U, typename V,
          detail::Requires<isArithmetic<U> && isArithmetic<V>> = true>
ExprPtr<T> hypot(const U& l, const V& c, const Variable<T>& r) {
    return hypot(l, c, r.expr);
}
template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const Variable<T>& l, const Variable<T>& c, const U& r) {
    return hypot(l.expr, c.expr, r);
}
template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const U& l, const Variable<T>& c, const Variable<T>& r) {
    return hypot(l, c.expr, r.expr);
}
template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> hypot(const Variable<T>& l, const U& c, const Variable<T>& r) {
    return hypot(l.expr, c, r.expr);
}

//-----------------------------------------------------------------------------
// POW OPERATOR
//-----------------------------------------------------------------------------

template <typename T>
ExprPtr<T> pow(const Variable<T>& l, const Variable<T>& r) {
    return pow(l.expr, r.expr);
}

template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> pow(const U& l, const Variable<T>& r) {
    return pow(l, r.expr);
}

template <typename T, typename U, detail::Requires<isArithmetic<U>> = true>
ExprPtr<T> pow(const Variable<T>& l, const U& r) {
    return pow(l.expr, r);
}

//-----------------------------------------------------------------------------
// MACRO FOR EXPRESSION UNARY OPERATOR
//-----------------------------------------------------------------------------

#define OPERATOR_FOR_VARIABLE_UNARY(OP, OPERATOR) \
    template <typename T>                         \
    ExprPtr<T> OPERATOR(const Variable<T>& x) {   \
        return OPERATOR(x.expr);                  \
    }                                             \
                                                  \
// TRIGONOMETRIC
OPERATOR_FOR_VARIABLE_UNARY(Sin, sin)
OPERATOR_FOR_VARIABLE_UNARY(Cos, cos)
OPERATOR_FOR_VARIABLE_UNARY(Tan, tan)
OPERATOR_FOR_VARIABLE_UNARY(ArcSin, asin)
OPERATOR_FOR_VARIABLE_UNARY(ArcCos, acos)
OPERATOR_FOR_VARIABLE_UNARY(ArcTan, atan)

/// HYPERBOLIC OPERATOR
OPERATOR_FOR_VARIABLE_UNARY(Sinh, sinh)
OPERATOR_FOR_VARIABLE_UNARY(Cosh, cosh)
OPERATOR_FOR_VARIABLE_UNARY(Tanh, tanh)

/// EXPONENTIAL AND LOGARITHMIC
OPERATOR_FOR_VARIABLE_UNARY(Exp, exp)
OPERATOR_FOR_VARIABLE_UNARY(Log, log)
OPERATOR_FOR_VARIABLE_UNARY(Log10, log10)

/// SQRT
OPERATOR_FOR_VARIABLE_UNARY(Sqrt, sqrt)

/// ABS
OPERATOR_FOR_VARIABLE_UNARY(Abs, abs)
OPERATOR_FOR_VARIABLE_UNARY(Abs2, abs2)

/// Efr
OPERATOR_FOR_VARIABLE_UNARY(Erf, erf)

/// Complex
OPERATOR_FOR_VARIABLE_UNARY(Conj, conj)
OPERATOR_FOR_VARIABLE_UNARY(Real, real)
OPERATOR_FOR_VARIABLE_UNARY(Imag, imag)

#undef OPERATOR_FOR_VARIABLE_UNARY

template <typename T, Requires<is_expr_v<T>> = true>
auto val(const T& t) {
    return expr_value(t);
}

template <typename... Vars>
struct Wrt {
    std::tuple<Vars...> args;
};

template <typename... Args>
auto wrt(Args&&... args) {
    return Wrt<Args...>{std::forward_as_tuple(std::forward<Args>(args)...)};
}

template <typename T, typename... Vars, size_t N, size_t... indexSeq>
void derivatives_bind(const Wrt<Vars...>& wrt, std::array<T, N>& values,
                      std::index_sequence<indexSeq...>) {
    (std::get<indexSeq>(wrt.args).expr->bind_value(&(values.at(indexSeq))), ...);
}

template <typename T, typename... Vars, size_t N, size_t... indexSeq>
void derivatives_bind(const Wrt<Vars...>& wrt, std::array<Variable<T>, N>& values,
                      std::index_sequence<indexSeq...>) {
    (std::get<indexSeq>(wrt.args).expr->bind_expr(&(values.at(indexSeq).expr)), ...);
}

template <typename T, typename... Vars, size_t N, size_t... indexSeq>
void derivatives_clear(const Wrt<Vars...>& wrt, std::array<T, N>& values,
                       std::index_sequence<indexSeq...>) {
    (std::get<indexSeq>(wrt.args).expr->bind_value(nullptr), ...);
}

template <typename T, typename... Vars, size_t N, size_t... indexSeq>
void derivatives_clear(const Wrt<Vars...>& wrt, std::array<Variable<T>, N>& values,
                       std::index_sequence<indexSeq...>) {
    (std::get<indexSeq>(wrt.args).expr->bind_expr(nullptr), ...);
}

template <typename T, typename... Vars>
auto derivatives(const Variable<T>& y, const Wrt<Vars...>& wrt) {
    constexpr auto N = sizeof...(Vars);

    std::array<T, N> values;
    values.fill(0.0);

    derivatives_bind(wrt, values, std::make_index_sequence<N>{});

    y.expr->propagate(1.0);

    derivatives_clear(wrt, values, std::make_index_sequence<N>{});

    return values;
}

template <typename T, typename... Vars>
auto derivativesx(const Variable<T>& y, const Wrt<Vars...>& wrt) {
    constexpr auto N = sizeof...(Vars);

    std::array<Variable<T>, N> values;

    derivatives_bind(wrt, values, std::make_index_sequence<N>{});

    y.expr->propagatex(constant<T>(1.0));
    derivatives_clear(wrt, values, std::make_index_sequence<N>{});

    return values;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Variable<T>& x) {
    out << val(x);
    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const ExprPtr<T>& x) {
    out << val(x);
    return out;
}
//=============================================================================
// HIGHER-ORDER VAR NUMBERS
//=============================================================================

template <size_t N, typename T>
struct AuxHigherOrderVariable;

template <typename T>
struct AuxHigherOrderVariable<0, T> {
    using type = T;
};

template <size_t N, typename T>
struct AuxHigherOrderVariable {
    using type = Variable<typename AuxHigherOrderVariable<N - 1, T>::type>;
};

template <size_t N, typename T>
using HigherOrderVariable = typename AuxHigherOrderVariable<N, T>::type;

};  // namespace detail

using detail::derivatives;
using detail::val;
using detail::Variable;
using detail::wrt;

using var = Variable<double>;
// using namespace detail;

inline detail::BooleanExpr boolref(const bool& v) {
    return detail::BooleanExpr([&]() { return v; });
}

}  // namespace autodiff
