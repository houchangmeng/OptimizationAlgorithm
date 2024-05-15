// MIT License

// Copyright © 2024 HouChangmeng

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef UNCONSTRAINTSOLVER_h
#define UNCONSTRAINTSOLVER_h

#include <iomanip>  // 设置输出格式
#include <sstream>
#include <string>
#include "common.h"

namespace OptSolver {

typedef std::function<bool(Eigen::VectorXd&, Eigen::VectorXd&, Eigen::MatrixXd&)>
    UnconstraintIterFunction;

class UnconstraintSolver {
   public:
    enum class SolverType {
        GradientDescend,
        GaussNewton,
        QuasiNewton_BFGS,
        QuasiNewton_DFP,
        ConjugateGradient
    };

    struct Options {
        int max_iternum_ = 1000;
        double stop_grad_norm_ = 0.01;
        SolverType solvertype_ = SolverType::GaussNewton;
        ObjectiveFunction obj_func_ = nullptr;
        GradientFunction grad_func_ = nullptr;
        HessianFunction hess_func_ = nullptr;
    };

   private:
    ObjectiveFunction obj_func_;
    GradientFunction grad_func_;
    HessianFunction hess_func_;

    UnconstraintIterFunction iter_func_;
    LineSearchFunction linesearch_func_;

    Eigen::VectorXd x0_, grad0_;
    Eigen::MatrixXd hess0_;
    Eigen::MatrixXd cg_direc_mat0_;
    std::vector<Eigen::VectorXd> x_solution_traj_;
    bool is_initialized;

    UnconstraintSolver::Options options_;

   public:
    UnconstraintSolver(ObjectiveFunction objective_func, GradientFunction gradient_func,
                       HessianFunction hessian_func = nullptr);
    UnconstraintSolver(UnconstraintSolver::Options opts);
    void Initialize(Eigen::VectorXd x0);
    bool Solve();
    std::vector<Eigen::VectorXd> get_solution_trajectory() { return x_solution_traj_; };

    ~UnconstraintSolver();
};

bool gradient_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::MatrixXd& hess,
                                   LineSearchFunction linesearch_func,
                                   GradientFunction gradient_func, HessianFunction hessian_func);

bool gauss_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad,
                                       Eigen::MatrixXd& hess, LineSearchFunction linesearch_func,
                                       GradientFunction gradient_func,
                                       HessianFunction hessian_func);

bool bfgs_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::MatrixXd& B,
                                      LineSearchFunction linesearch_func,
                                      GradientFunction gradient_func, HessianFunction hessian_func);

bool dfp_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::MatrixXd& G,
                                     LineSearchFunction linesearch_func,
                                     GradientFunction gradient_func, HessianFunction hessian_func);

bool conjugate_gradient_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad,
                                             Eigen::MatrixXd& direc_mat,
                                             LineSearchFunction linesearch_func,
                                             GradientFunction gradient_func,
                                             HessianFunction hessian_func);
}  // namespace OptSolver

#endif