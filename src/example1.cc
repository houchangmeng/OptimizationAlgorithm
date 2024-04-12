#include "constraintsolver.h"
#include "unconstraintsolver.h"
#include "utils.h"

/**
 * @example
 * Objective function.
 * f(x) = x(0) * x(1)
 */

// clang-format off

double objective_function(Eigen::Vector2d x) {

    return x(0) * x(1);
}

Eigen::Vector2d gradient_function(Eigen::Vector2d x) {
    /*
     * shape: num_variable * 1
     */
    Eigen::Vector2d grad;
    grad << x(1), 
            x(0);

    return grad;
}

Eigen::Matrix2d hessian_function(Eigen::Vector2d x) {
    /*
     * shape: num_variable * num_variable
     */
    Eigen::Matrix2d H;
    H << 0.0, 1.0, 
         1.0, 0.0;

    return H;
}

/**
 * @brief
 * equality_constraints
 * h(x) = 0
 */
Eigen::VectorXd equality_constraint(Eigen::Vector2d x) {
    Eigen::VectorXd eq_value_vec(1);
    eq_value_vec << x(0) * x(0) + 2 * x(0) - x(1);
    return eq_value_vec;
}

Eigen::MatrixXd jacbian_eq_constraint(Eigen::Vector2d x) {
    /**
     * num_constraint * num_variable
     */
    Eigen::MatrixXd jacbian_eq(1, 2);
    jacbian_eq << 2 * x(0) + 2, -1;
    return jacbian_eq;
}

/**
 * @brief
 * inequality_constraints
 * G(x) <= b
 */

Eigen::VectorXd inequality_constraint(Eigen::Vector2d x) {
    Eigen::Vector2d G;
    G << 1.0, -1.0;
    double b = -1.0;

    Eigen::VectorXd ineq_value_vec(1);
    ineq_value_vec << G.transpose() * x - b;
    return ineq_value_vec;
}

Eigen::MatrixXd jacbian_ineq_constraint(Eigen::Vector2d x) {
    Eigen::Vector2d jacbian_ineq;
    jacbian_ineq << 1.0, -1.0;

    return jacbian_ineq.transpose();
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

Eigen::VectorXd eq_ineq_constraints(Eigen::Vector2d x) {
    Eigen::Vector2d G;
    G << 1.0, -1.0;
    double b = -1.0;
    Eigen::VectorXd ineq_value_vec(3);

    ineq_value_vec << G.dot(x) - b, 
                  x(0) * x(0) + 2 * x(0) - x(1), 
                  -(x(0) * x(0) + 2 * x(0) - x(1));

    return ineq_value_vec;
}

Eigen::MatrixXd jacbian_eq_ineq_constraints(Eigen::Vector2d x) {
    Eigen::Vector2d G;
    G << 1.0, -1.0;

    Eigen::MatrixXd jacbian_eq_ineq;
    jacbian_eq_ineq.setZero(3, 2);
    jacbian_eq_ineq.row(0) = G.transpose();
    jacbian_eq_ineq.row(1) = Eigen::Vector2d({2 * x(0) + 2, -1}).transpose();
    jacbian_eq_ineq.row(2) = -Eigen::Vector2d({2 * x(0) + 2, -1}).transpose();

    return jacbian_eq_ineq;
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

    opts.solvertype_ = OptSolver::UnconstraintSolver::SolverType::ConjugateGradient;
    opts.grad_func_ = gradient_function;
    opts.hess_func_ = hessian_function;
    opts.obj_func_ = objective_function;

    OptSolver::UnconstraintSolver solver(opts);
    Eigen::Vector2d x0;
    x0 << -4, 2;
    solver.Initialize(x0);
    solver.Solve();

    plot_solution_path(solver.get_solution_trajectory(), objective_function,"ConjugateGradient");
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
    opts.stop_grad_norm_ = 0.01;

    opts.solvertype_ = OptSolver::ConstraintSolver::SolverType::AugmentedLagrangian;
    opts.grad_func_ = gradient_function;
    opts.hess_func_ = hessian_function;
    opts.obj_func_ = objective_function;

    opts.cons_func_ = eq_ineq_constraints;
    opts.jac_cons_func_ = jacbian_eq_ineq_constraints;

    OptSolver::ConstraintSolver solver(opts);
    Eigen::Vector2d x0;
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
    opts.stop_grad_norm_ = 0.01;

    opts.solvertype_ = OptSolver::ConstraintSolver::SolverType::InteriorPoint;
    opts.grad_func_ = gradient_function;
    opts.hess_func_ = hessian_function;
    opts.obj_func_ = objective_function;


    opts.cons_func_ = eq_ineq_constraints;
    opts.jac_cons_func_ = jacbian_eq_ineq_constraints;
    

    OptSolver::ConstraintSolver solver(opts);
    Eigen::Vector2d x0;
    // x0 << 2, -2;
    // x0 << -3, 2;
    // x0 << 3, 3;
    x0 << -1, 4;
    // x0 << 0.5, 0.0;
    // x0 << 1.0, 0.0;

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
    opts.stop_grad_norm_ = 0.01;

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

// clang-format on
int main() {
    // unconstraint_test();

    constraint_test_augmented_lagrangian();
    constraint_test_interior();
    constraint_test_kkt_system_solver();
    plt::show();
    plt::close();
    return 0;
}