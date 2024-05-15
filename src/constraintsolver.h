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

#ifndef CONSTRAINTSOLVER_h
#define CONSTRAINTSOLVER_h

#include <iomanip>
#include <sstream>
#include <string>
#include "common.h"

namespace OptSolver {

typedef std::function<double(Eigen::VectorXd)> ObjectiveFunction;
typedef std::function<Eigen::VectorXd(Eigen::VectorXd)> GradientFunction;
typedef std::function<Eigen::MatrixXd(Eigen::VectorXd)> HessianFunction;
typedef std::function<void(Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&)>
    LineSearchFunction;

typedef std::function<Eigen::VectorXd(Eigen::VectorXd)> ConstraintsFunction;
typedef std::function<Eigen::MatrixXd(Eigen::VectorXd)> JacbianConstraintsFunction;

typedef std::function<double(Eigen::VectorXd, Eigen::VectorXd)> DualLineSearchFunction;

typedef std::function<bool(Eigen::VectorXd&, Eigen::MatrixXd&, Eigen::MatrixXd&)>
    ConstraintIterFunction;

class ConstraintSolver {
   public:
    enum class SolverType {
        AugmentedLagrangian,
        InteriorPoint,
        KKTSystemSolver,
    };

    struct Options {
        int max_iternum_ = 1000;
        double stop_x_norm_ = 0.001;
        SolverType solvertype_ = SolverType::AugmentedLagrangian;
        ObjectiveFunction obj_func_ = nullptr;
        GradientFunction grad_func_ = nullptr;
        HessianFunction hess_func_ = nullptr;
        ConstraintsFunction cons_func_ = nullptr;
        JacbianConstraintsFunction jac_cons_func_ = nullptr;
    };

   private:
    ObjectiveFunction obj_func_;
    GradientFunction grad_func_;
    HessianFunction hess_func_;
    ConstraintsFunction cons_func_;
    JacbianConstraintsFunction jac_cons_func_;

    ConstraintIterFunction iter_func_;
    LineSearchFunction linesearch_func_;
    DualLineSearchFunction dual_linesearch_func_;

    Eigen::VectorXd x0_;
    Eigen::MatrixXd lambda0_, rho0_, slack0_;
    std::vector<Eigen::VectorXd> x_solution_traj_;
    std::vector<Eigen::VectorXd> lambda_solution_traj_;
    bool is_initialized;

    ConstraintSolver::Options options_;

   public:
    ConstraintSolver(ObjectiveFunction obj_func, GradientFunction grad_func,
                     HessianFunction hess_func, ConstraintsFunction cons_func,
                     JacbianConstraintsFunction jac_cons_func);
    ConstraintSolver(ConstraintSolver::Options opts);
    void Initialize(Eigen::VectorXd x0);
    bool Solve();
    std::vector<Eigen::VectorXd> get_solution_trajectory() { return x_solution_traj_; };

    static double dual_linesearch(Eigen::VectorXd variable, Eigen::VectorXd delta);
    ~ConstraintSolver();
};

bool augmentedLagrangian_step(Eigen::VectorXd& x, Eigen::MatrixXd& lambda, Eigen::MatrixXd& rho,
                              GradientFunction gradient_func, HessianFunction hessian_func,
                              ConstraintsFunction constraints_func,
                              JacbianConstraintsFunction jac_constraints_func);

bool augmentedLagrangian_step_with_linesearch(Eigen::VectorXd& x, Eigen::MatrixXd& lambda,
                                              Eigen::MatrixXd& rho, ObjectiveFunction obj_func,
                                              GradientFunction gradient_func,
                                              HessianFunction hessian_func,
                                              ConstraintsFunction constraints_func,
                                              JacbianConstraintsFunction jac_constraints_func);

bool primal_dual_interior_step(Eigen::VectorXd& x, Eigen::MatrixXd& lambda, Eigen::MatrixXd& slack,
                               GradientFunction gradient_func, HessianFunction hessian_func,
                               ConstraintsFunction constraints_func,
                               JacbianConstraintsFunction jac_constraints_func);

bool primal_dual_interior_step_with_linesearch(Eigen::VectorXd& x, Eigen::MatrixXd& lambda,
                                               Eigen::MatrixXd& slack,
                                               GradientFunction gradient_func,
                                               HessianFunction hessian_func,
                                               ConstraintsFunction constraints_func,
                                               JacbianConstraintsFunction jac_constraints_func,
                                               DualLineSearchFunction linesearch_func);

bool kkt_newton_step(Eigen::VectorXd& x, Eigen::MatrixXd& lambda, Eigen::MatrixXd& rho,
                     GradientFunction gradient_func, HessianFunction hessian_func,
                     ConstraintsFunction constraints_func,
                     JacbianConstraintsFunction jac_constraints_func);

}  // end of namespace OptSolver

#endif