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

#include <cxxabi.h>
#include "constraintsolver.h"
#include "mat.hpp"
#include "unconstraintsolver.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////
/////////////////////////autodiff intergration /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * @example
 * Objective function.
 * f(x) = (x-v)^T * Q * (x - v)
 */

// clang-format off

// autodiff::var objective_function(autodiff::Vector2var x) {
//     Eigen::Matrix2d Q;
//     Q << 0.5, 0.0, 
//          0.0, 1.0;

//     Eigen::Vector2d v;
//     v << 1.0, 
//          0.0;

//     return (x - v).transpose() *(x - v)  ;
// }

autodiff::var objective_function(autodiff::Vector2var x) {

    return sin(x(0)) * cos(x(1));
}

Eigen::Vector2d gradient_function(autodiff::Vector2var x) {
    /*
     * shape: num_variable * 1
     */
    autodiff::var y = objective_function(x);
    return autodiff::gradient(y, x);
}

Eigen::Matrix2d hessian_function(autodiff::Vector2var x) {
    /*
     * shape: num_variable * num_variable
     */
    autodiff::var y = objective_function(x);
    return autodiff::hessian(y, x);
}

/**
 * @brief
 * equality_constraints
 * h(x) = 0
 */
autodiff::VectorXvar equality_constraint(autodiff::Vector2var x) {
    autodiff::VectorXvar eq_value_vec(1);
    eq_value_vec << x(0) * x(0) + 2 * x(0) - x(1);
    return eq_value_vec;
}

Eigen::MatrixXd jacbian_eq_constraint(autodiff::Vector2var x) {
    /**
     * num_constraint * num_variable
     */

    autodiff::VectorXvar y(1);
    y = equality_constraint(x);
    return autodiff::jacbian(y,x);
}

/**
 * @brief
 * inequality_constraints
 * G(x) <= b
 */

autodiff::VectorXvar inequality_constraint(autodiff::Vector2var x) {
    autodiff::Vector2var G;
    G << 1.0, -1.0;

    Eigen::Matrix<double, 1, 1> b;
    b << -1.0;
    // double b = 1.0;
    
    autodiff::VectorXvar ineq_value_vec(1);
    ineq_value_vec << G.transpose() * x - b; /// note Eigen::operator and Variable;
    return ineq_value_vec;
}

Eigen::MatrixXd jacbian_ineq_constraint(autodiff::Vector2var x) {
    autodiff::VectorXvar y(1);
    y = inequality_constraint(x);
    return autodiff::jacbian(y,x);
}

/**
 * @brief
 * inequality_constraints
 * g(x) <= b
 * equality_constraints
 * h(x) = b
 * =>   h(x) <= b
 * and -h(x) <= -b
 *
 * then, all constraints can be expression by
 * g(x) <= b
 * h(x) <= 0
 * -h(x)<= 0
 */

autodiff::Vector3var eq_ineq_constraints(autodiff::Vector2var x) {
    autodiff::Vector2var  G;
    G << 1.0, -1.0;
    // double b = 1.0;
    Eigen::Matrix<double, 1, 1> b;
    b << -1.0;

    autodiff::Vector3var ineq_value_vec(3);
   
    ineq_value_vec << G.transpose() * x - b, 
                  x(0) * x(0) + 2 * x(0) - x(1), 
                  -(x(0) * x(0) + 2 * x(0) - x(1));

    return ineq_value_vec;
}

Eigen::MatrixXd jacbian_eq_ineq_constraints(autodiff::Vector2var x) {
    autodiff::VectorXvar y(1);
    y = eq_ineq_constraints(x);
    return autodiff::jacbian(y,x);
}


/**
 * GradientDescend,
 * GaussNewton,
 * QuasiNewton_BFGS,
 * QuasiNewton_DFP,
 * ConjugateGradient
*/
void unconstraint_test() {
    printf("\033[32mUnconstraintSolver.\033[0m\n");

    plot_landscape(objective_function);
    plt::pause(0.1);

    OptSolver::UnconstraintSolver::Options opts;
    opts.max_iternum_ = 1000;
    opts.stop_grad_norm_ = 0.01;

    opts.solvertype_ = OptSolver::UnconstraintSolver::SolverType::QuasiNewton_BFGS;
    opts.grad_func_ = gradient_function;
    opts.hess_func_ = hessian_function;
    opts.obj_func_ = objective_function;

    OptSolver::UnconstraintSolver solver(opts);
    Eigen::Vector2d x0;
    x0 << -4, 2;
    solver.Initialize(x0);
    solver.Solve();

    plot_solution_path(solver.get_solution_trajectory(), objective_function,"QuasiNewton_BFGS");
    plt::pause(1.0);
    plt::clf();
}


/**
 * AugmentedLagrangian
 * InteriorPoint
 * EqualityConstraintSolver
*/

void constraint_test_augmented_lagrangian() {
    printf("\033[32mAugmentedLagrangian.\033[0m\n");

    plot_landscape(objective_function);
    plot_ineq_constraint();
    plot_eq_constraint();
    
    OptSolver::ConstraintSolver::Options opts;
    opts.max_iternum_ = 1000;
    opts.stop_x_norm_ = 0.001;

    opts.solvertype_ = OptSolver::ConstraintSolver::SolverType::AugmentedLagrangian;
    opts.grad_func_ = gradient_function;
    opts.hess_func_ = hessian_function;
    opts.obj_func_ = objective_function;

    opts.cons_func_ = eq_ineq_constraints;
    opts.jac_cons_func_ = jacbian_eq_ineq_constraints;

    // opts.cons_func_ = equality_constraint;
    // opts.jac_cons_func_ = jacbian_eq_constraint;

    // opts.cons_func_ = inequality_constraint;
    // opts.jac_cons_func_ = jacbian_ineq_constraint;

    OptSolver::ConstraintSolver solver(opts);
    Eigen::Vector2d x0;
    x0 << 2, -2;
    // x0 << -3, 2;
    // x0 << 3, 3;
    x0 << -1, 4;
    // x0 << 0.5, 0.0;
    // x0 << 1.0, 0.0;

    solver.Initialize(x0);
    solver.Solve();

    plot_solution_path(solver.get_solution_trajectory(), objective_function,"AugmentedLagrangian");
    plt::pause(1.0);
    plt::clf();

}

void constraint_test_interior() {
    printf("\033[32mInteriorPoint.\033[0m\n");

    plot_landscape(objective_function);
    plot_ineq_constraint();
    plot_eq_constraint();

    OptSolver::ConstraintSolver::Options opts;
    opts.max_iternum_ = 1000;
    opts.stop_x_norm_ = 0.001;

    opts.solvertype_ = OptSolver::ConstraintSolver::SolverType::InteriorPoint;
    opts.grad_func_ = gradient_function;
    opts.hess_func_ = hessian_function;
    opts.obj_func_ = objective_function;


    opts.cons_func_ = eq_ineq_constraints;
    opts.jac_cons_func_ = jacbian_eq_ineq_constraints;
    

    OptSolver::ConstraintSolver solver(opts);
    Eigen::Vector2d x0;
    x0 << 2, -2;
    // x0 << -3, 2;
    // x0 << 3, 3;
    // x0 << -1, 4;
    x0 << -3.0, 0.0;
    x0 << -1.0, 0.0;

    solver.Initialize(x0);
    solver.Solve();

    plot_solution_path(solver.get_solution_trajectory(), objective_function,"InteriorPoint");
    plt::pause(1.0);
    plt::clf();
}

void constraint_test_kkt_system_solver() {
    printf("\033[32mKKTSystemSolver.\033[0m\n");
    
    plot_landscape(objective_function);
    plot_eq_constraint();
    
    OptSolver::ConstraintSolver::Options opts;
    opts.max_iternum_ = 1000;
    opts.stop_x_norm_ = 0.01;

    opts.solvertype_ = OptSolver::ConstraintSolver::SolverType::KKTSystemSolver;
    opts.grad_func_ = gradient_function;
    opts.hess_func_ = hessian_function;
    opts.obj_func_ = objective_function;

    opts.cons_func_ = equality_constraint;
    opts.jac_cons_func_ = jacbian_eq_constraint;
    

    OptSolver::ConstraintSolver solver(opts);
    Eigen::Vector2d x0;
    x0 << -1, 4;
 
    solver.Initialize(x0);
    solver.Solve();

    plot_solution_path(solver.get_solution_trajectory(), objective_function, "EqualityConstraint");
    plt::pause(1.0);
    plt::clf();
}



int main() {

    unconstraint_test();
    constraint_test_augmented_lagrangian();
    constraint_test_interior();
    constraint_test_kkt_system_solver();
    plt::show();
    return 0;
}