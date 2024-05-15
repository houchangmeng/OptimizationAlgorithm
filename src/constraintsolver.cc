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

#include "constraintsolver.h"

namespace OptSolver {

ConstraintSolver::ConstraintSolver(ObjectiveFunction obj_func, GradientFunction grad_func,
                                   HessianFunction hess_func, ConstraintsFunction cons_func,
                                   JacbianConstraintsFunction jac_cons_func)
    : obj_func_(obj_func),
      grad_func_(grad_func),
      hess_func_(hess_func),
      cons_func_(cons_func),
      jac_cons_func_(jac_cons_func) {
    options_ = ConstraintSolver::Options();
}

ConstraintSolver::ConstraintSolver(ConstraintSolver::Options opts) {
    options_ = opts;
    obj_func_ = opts.obj_func_;
    grad_func_ = opts.grad_func_;
    hess_func_ = opts.hess_func_;
    cons_func_ = opts.cons_func_;
    jac_cons_func_ = opts.jac_cons_func_;
}

void ConstraintSolver::Initialize(Eigen::VectorXd x0) {
    x0_ = x0;

    Eigen::VectorXd cons_value = cons_func_(x0);
    Eigen::MatrixXd jac_value = jac_cons_func_(x0);

    int num_cons = jac_value.rows();
    int num_vars = jac_value.cols();

    rho0_.setIdentity(num_cons, num_cons);
    slack0_ = (-cons_value).asDiagonal();
    lambda0_.setIdentity(num_cons, num_cons);

    double alpha_p = -(-cons_value).minCoeff();  // - slack value

    if (alpha_p < 0) {
        slack0_ = (-cons_value).asDiagonal();
    } else {
        slack0_ = ((-cons_value).array() + 1 + alpha_p).matrix().asDiagonal();
    }

    double alpha_d = -cons_value.minCoeff();
    if (alpha_d >= 0) {
        lambda0_ = (cons_value.array() + 1 + alpha_d).matrix().asDiagonal();
    } else {
        lambda0_ = cons_value.asDiagonal();
    }
    switch (options_.solvertype_) {
        case SolverType::AugmentedLagrangian:
            rho0_.setIdentity(num_cons, num_cons);
            iter_func_ = std::bind(augmentedLagrangian_step, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, grad_func_,
                                   hess_func_, cons_func_, jac_cons_func_);

            // iter_func_ = std::bind(augmentedLagrangian_step_with_linesearch,
            // std::placeholders::_1,
            //                        std::placeholders::_2, std::placeholders::_3, obj_func_,
            //                        grad_func_, hess_func_, cons_func_, jac_cons_func_);

            break;
        case SolverType::InteriorPoint:
            rho0_ = slack0_;
            dual_linesearch_func_ = ConstraintSolver::dual_linesearch;
            // iter_func_ = std::bind(primal_dual_interior_step, std::placeholders::_1,
            //                        std::placeholders::_2, std::placeholders::_3, grad_func_,
            //                        hess_func_, cons_func_, jac_cons_func_);
            iter_func_ = std::bind(primal_dual_interior_step_with_linesearch, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, grad_func_,
                                   hess_func_, cons_func_, jac_cons_func_, dual_linesearch_func_);
            break;

        case SolverType::KKTSystemSolver:

            iter_func_ = std::bind(kkt_newton_step, std::placeholders::_1, std::placeholders::_2,
                                   std::placeholders::_3, grad_func_, hess_func_, cons_func_,
                                   jac_cons_func_);
            break;

        default:
            rho0_ = slack0_;
            dual_linesearch_func_ = ConstraintSolver::dual_linesearch;
            iter_func_ = std::bind(augmentedLagrangian_step, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, grad_func_,
                                   hess_func_, cons_func_, jac_cons_func_);
            break;
    }

    x_solution_traj_.push_back(x0);
    lambda_solution_traj_.push_back(lambda0_.diagonal());

    is_initialized = true;
}

bool ConstraintSolver::Solve() {
    if (!is_initialized) {
        printf("File : %s \nFunction: %s \nLine : %d\n", __FILE__, __func__, __LINE__);
        printf("\033[31mSolver should initialized before iterate.\033[0m\n");
        std::exit(0);
    }

    int step = 0;

    while (1) {
        if (iter_func_(x0_, lambda0_, rho0_)) {
            break;
        };

        if ((x0_ - x_solution_traj_.back()).norm() < options_.stop_x_norm_) {
            break;
        };

        if (step > options_.max_iternum_) {
            printf("\033[31mSolver has reached maximum iterate step.\033[0m\n");
            return false;
        }

        x_solution_traj_.push_back(x0_);
        lambda_solution_traj_.push_back(lambda0_.diagonal());
        step++;
    }

    x_solution_traj_.push_back(x0_);
    lambda_solution_traj_.push_back(lambda0_.diagonal());
    printf("\033[32mSuccess.\033[0m\n");
    return true;
}

ConstraintSolver::~ConstraintSolver() {}

double ConstraintSolver::dual_linesearch(Eigen::VectorXd variable, Eigen::VectorXd delta) {
    Eigen::VectorXd alpha_vec = variable.array() / delta.array();
    double alpha = 1.0;

    if (-alpha_vec.minCoeff() > 0) {
        alpha = -alpha_vec.minCoeff();
    }
    return alpha;
}

bool augmentedLagrangian_step(Eigen::VectorXd& x, Eigen::MatrixXd& lambda, Eigen::MatrixXd& rho,
                              GradientFunction gradient_func, HessianFunction hessian_func,
                              ConstraintsFunction constraints_func,
                              JacbianConstraintsFunction jac_constraints_func) {
    // allocate memory
    Eigen::VectorXd cons_value = constraints_func(x);

    // if (cons_value.maxCoeff() < 0.01) {
    //     return true;
    // }

    Eigen::VectorXd penalty_value = cons_value;
    penalty_value.setZero();
    Eigen::MatrixXd jac_cons_value = jac_constraints_func(x);
    jac_cons_value.setZero();
    Eigen::VectorXd al_kkt_grad = gradient_func(x);
    al_kkt_grad.setZero();
    Eigen::MatrixXd al_kkt_hess = hessian_func(x);
    al_kkt_hess.setZero();
    Eigen::VectorXd delta_x = al_kkt_grad;
    delta_x.setZero();

    Eigen::VectorXd rho_diag = rho.diagonal();
    Eigen::VectorXd lambda_vec = lambda.diagonal();

    int num_vars = jac_cons_value.cols();
    int num_cons = jac_cons_value.rows();
    int iter_count = 0;
    int sigma = 5;
    while (1) {
        cons_value = constraints_func(x);

        cons_value.unaryExpr(
            [](double cons_value) { return (cons_value > 0.0) ? cons_value : 0.0; });

        jac_cons_value = jac_constraints_func(x);

        for (int i = 0; i < num_cons; i++) {
            if (cons_value(i) <= 0.0) {
                jac_cons_value.row(i).setZero();
            }
        }

        penalty_value = (cons_value.array() * rho_diag.array()).matrix();

        al_kkt_grad = gradient_func(x) + jac_cons_value.transpose() * (lambda_vec + penalty_value);
        if (al_kkt_grad.norm() < 0.01) {
            break;
        } else {
            al_kkt_hess = hessian_func(x) + jac_cons_value.transpose() * rho * jac_cons_value;
            delta_x =
                -al_kkt_hess.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(al_kkt_grad);

            x += delta_x;
        };

        if (iter_count > 1000) {
            break;
        }
        iter_count++;
    }

    lambda = (lambda_vec + rho * cons_value).asDiagonal();
    rho = (rho_diag * sigma).asDiagonal();

    return false;
}

bool augmentedLagrangian_step_with_linesearch(Eigen::VectorXd& x, Eigen::MatrixXd& lambda,
                                              Eigen::MatrixXd& rho, ObjectiveFunction obj_func,
                                              GradientFunction gradient_func,
                                              HessianFunction hessian_func,
                                              ConstraintsFunction constraints_func,
                                              JacbianConstraintsFunction jac_constraints_func) {
    // allocate memory
    Eigen::VectorXd cons_value = constraints_func(x);

    // if (cons_value.maxCoeff() < 0.01) {
    //     return true;
    // }

    Eigen::VectorXd penalty_value = cons_value;
    penalty_value.setZero();
    Eigen::MatrixXd jac_cons_value = jac_constraints_func(x);
    jac_cons_value.setZero();
    Eigen::VectorXd al_kkt_grad = gradient_func(x);
    al_kkt_grad.setZero();
    Eigen::MatrixXd al_kkt_hess = hessian_func(x);
    al_kkt_hess.setZero();
    Eigen::VectorXd delta_x = al_kkt_grad;
    delta_x.setZero();

    Eigen::VectorXd rho_diag = rho.diagonal();
    Eigen::VectorXd lambda_vec = lambda.diagonal();

    int num_vars = jac_cons_value.cols();
    int num_cons = jac_cons_value.rows();
    int iter_count = 0;
    int sigma = 5;

    ObjectiveFunction mertfunc = [obj_func, constraints_func, num_vars,
                                  num_cons](Eigen::VectorXd x_and_lambda) {
        Eigen::VectorXd x = x_and_lambda.head(num_vars);
        Eigen::VectorXd lambda = x_and_lambda.tail(num_cons);
        return obj_func(x) + lambda.transpose() * constraints_func(x) +
               0.5 * constraints_func(x).transpose() * constraints_func(x);
    };
    while (1) {
        cons_value = constraints_func(x);

        cons_value.unaryExpr(
            [](double cons_value) { return (cons_value > 0.0) ? cons_value : 0.0; });

        jac_cons_value = jac_constraints_func(x);

        for (int i = 0; i < num_cons; i++) {
            if (cons_value(i) < 0) {
                jac_cons_value.row(i).setZero();
            }
        }

        penalty_value = (cons_value.array() * rho_diag.array()).matrix();

        al_kkt_grad = gradient_func(x) + jac_cons_value.transpose() * (lambda_vec + penalty_value);
        if (al_kkt_grad.norm() < 0.01) {
            break;
        } else {
            al_kkt_hess = hessian_func(x) + jac_cons_value.transpose() * rho * jac_cons_value;
            delta_x =
                -al_kkt_hess.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(al_kkt_grad);

            Eigen::VectorXd merit_grad;
            merit_grad.setZero(num_vars + num_cons);
            merit_grad << al_kkt_grad, cons_value;
            Eigen::VectorXd vec_x_lambda(num_vars + num_cons);
            vec_x_lambda << x, lambda_vec;
            Eigen::VectorXd delta_x_lambda;
            delta_x_lambda.setZero(num_vars + num_cons);
            delta_x_lambda.head(num_vars) = delta_x;
            backtracking_linesearch(vec_x_lambda, delta_x_lambda, merit_grad, mertfunc);
            x = vec_x_lambda.head(num_vars);
        };

        if (iter_count > 1000) {
            break;
        }

        iter_count++;
    }

    lambda = (lambda_vec + rho * cons_value).asDiagonal();
    rho = (rho_diag * sigma).asDiagonal();

    return false;
}

bool primal_dual_interior_step(Eigen::VectorXd& x, Eigen::MatrixXd& lambda_mat,
                               Eigen::MatrixXd& slack_mat, GradientFunction gradient_func,
                               HessianFunction hessian_func, ConstraintsFunction constraints_func,
                               JacbianConstraintsFunction jac_constraints_func) {
    Eigen::VectorXd cons_value = constraints_func(x);
    Eigen::MatrixXd jac_cons_value = jac_constraints_func(x);
    Eigen::MatrixXd jac_cons_transpose = jac_cons_value.transpose();

    Eigen::VectorXd grad_objfunc_value = gradient_func(x);
    Eigen::MatrixXd hess_objfunc_value = hessian_func(x);

    int num_vars = x.size();
    int num_slack = cons_value.size();
    int num_lambda = cons_value.size();

    Eigen::VectorXd slack_vec = slack_mat.diagonal();
    Eigen::VectorXd lambda_vec = lambda_mat.diagonal();

    double dual_gap = slack_vec.dot(lambda_vec);

    if (std::abs(dual_gap) < 0.001) {
        return true;
    }

    // Build KKT System.

    Eigen::VectorXd kkt_rhs_vec;
    kkt_rhs_vec.setZero(num_vars + num_slack + num_lambda);
    kkt_rhs_vec.block(0, 0, num_vars, 1) = -grad_objfunc_value - jac_cons_transpose * lambda_vec;
    kkt_rhs_vec.block(num_vars, 0, num_slack, 1) = -slack_mat * lambda_vec;
    kkt_rhs_vec.block(num_vars + num_slack, 0, num_lambda, 1) = -cons_value - slack_vec;

    if (kkt_rhs_vec.norm() < 0.01) {
        return true;
    }

    Eigen::MatrixXd cons_eye = Eigen::MatrixXd::Identity(num_lambda, num_lambda);

    Eigen::MatrixXd kkt_lhs_mat;
    kkt_lhs_mat.setZero(num_vars + num_slack + num_lambda, num_vars + num_slack + num_lambda);
    kkt_lhs_mat.block(0, 0, num_vars, num_vars) = hess_objfunc_value;
    kkt_lhs_mat.block(0, num_vars + num_slack, num_vars, num_lambda) = jac_cons_transpose;
    kkt_lhs_mat.block(num_vars, num_vars, num_slack, num_slack) = lambda_mat;
    kkt_lhs_mat.block(num_vars, num_vars + num_slack, num_slack, num_slack) = slack_mat;
    kkt_lhs_mat.block(num_vars + num_slack, 0, num_lambda, num_vars) = jac_cons_value;
    kkt_lhs_mat.block(num_vars + num_slack, num_vars, num_lambda, num_lambda) = cons_eye;

    Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> kktSVD;
    kktSVD.compute(kkt_lhs_mat);

    // solve affine.
    Eigen::VectorXd delta_aff = kktSVD.solve(kkt_rhs_vec);

    Eigen::MatrixXd delta_slack_aff = delta_aff.block(num_vars, 0, num_slack, 1);
    Eigen::MatrixXd delta_slack_aff_matrix = delta_slack_aff.asDiagonal();
    Eigen::VectorXd delta_lambda_aff = delta_aff.block(num_vars + num_slack, 0, num_slack, 1);

    double mu, sigma;
    mu = slack_vec.dot(lambda_vec) / num_lambda;
    sigma = 0.5;
    Eigen::VectorXd mu_vec = Eigen::VectorXd::Ones(num_slack) * mu * sigma;

    Eigen::VectorXd center_rhs_vec;
    center_rhs_vec.setZero(num_vars + num_slack + num_lambda);
    center_rhs_vec.block(num_vars, 0, num_slack, 1) =
        mu_vec - delta_slack_aff_matrix * delta_lambda_aff;

    // solve centering-corrector
    Eigen::VectorXd delta_cc = kktSVD.solve(center_rhs_vec);

    Eigen::VectorXd delta = delta_aff + delta_cc;

    x += delta.block(0, 0, num_vars, 1);

    slack_mat = (slack_vec + delta.block(num_vars, 0, num_slack, 1)).asDiagonal();
    lambda_mat = (lambda_vec + delta.block(num_vars + num_slack, 0, num_lambda, 1)).asDiagonal();

    return false;
}

/**
 * @ref https://web.stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf
 */

bool primal_dual_interior_step_with_linesearch(Eigen::VectorXd& x, Eigen::MatrixXd& lambda_mat,
                                               Eigen::MatrixXd& slack_mat,
                                               GradientFunction gradient_func,
                                               HessianFunction hessian_func,
                                               ConstraintsFunction constraints_func,
                                               JacbianConstraintsFunction jac_constraints_func,
                                               DualLineSearchFunction linesearch_func) {
    Eigen::VectorXd cons_value = constraints_func(x);
    Eigen::MatrixXd jac_cons_value = jac_constraints_func(x);
    Eigen::MatrixXd jac_cons_transpose = jac_cons_value.transpose();

    Eigen::VectorXd grad_objfunc_value = gradient_func(x);
    Eigen::MatrixXd hess_objfunc_value = hessian_func(x);

    int num_vars = x.size();
    int num_slack = cons_value.size();
    int num_lambda = cons_value.size();

    Eigen::VectorXd slack_vec = slack_mat.diagonal();
    Eigen::VectorXd lambda_vec = lambda_mat.diagonal();

    double dual_gap = slack_vec.dot(lambda_vec);

    if (std::abs(dual_gap) < 0.001) {
        return true;
    }

    // Build KKT System. Origin KKT matrix is unstable,

    Eigen::VectorXd kkt_rhs_vec;
    kkt_rhs_vec.setZero(num_vars + num_slack + num_lambda);
    kkt_rhs_vec.block(0, 0, num_vars, 1) = -grad_objfunc_value - jac_cons_transpose * lambda_vec;
    kkt_rhs_vec.block(num_vars, 0, num_slack, 1) = -lambda_vec;  // modify
    kkt_rhs_vec.block(num_vars + num_slack, 0, num_lambda, 1) = -cons_value - slack_vec;

    if (kkt_rhs_vec.norm() < 0.001) {
        return true;
    }

    Eigen::MatrixXd cons_eye = Eigen::MatrixXd::Identity(num_lambda, num_lambda);

    Eigen::MatrixXd kkt_lhs_mat;
    kkt_lhs_mat.setZero(num_vars + num_slack + num_lambda, num_vars + num_slack + num_lambda);
    kkt_lhs_mat.block(0, 0, num_vars, num_vars) = hess_objfunc_value;
    kkt_lhs_mat.block(0, num_vars + num_slack, num_vars, num_lambda) = jac_cons_transpose;
    kkt_lhs_mat.block(num_vars, num_vars, num_slack, num_slack) = slack_mat.inverse() * lambda_mat;
    kkt_lhs_mat.block(num_vars, num_vars + num_slack, num_slack, num_slack) = cons_eye;
    kkt_lhs_mat.block(num_vars + num_slack, 0, num_lambda, num_vars) = jac_cons_value;
    kkt_lhs_mat.block(num_vars + num_slack, num_vars, num_lambda, num_lambda) = cons_eye;

    // Eigen::LDLT<Eigen::MatrixXd> ldlt; SVD is more stable.
    Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> ldlt;
    ldlt.compute(kkt_lhs_mat);

    // affine delta solve
    Eigen::VectorXd delta_aff = ldlt.solve(kkt_rhs_vec);

    // centering-plus-corrector directions
    double mu, sigma, alpha;
    mu = dual_gap / num_lambda;
    Eigen::MatrixXd delta_slack_aff = delta_aff.block(num_vars, 0, num_slack, 1);
    Eigen::VectorXd delta_lambda_aff = delta_aff.block(num_vars + num_slack, 0, num_slack, 1);
    alpha = std::min(linesearch_func(slack_vec, delta_slack_aff),
                     linesearch_func(lambda_vec, delta_lambda_aff));
    sigma =
        (slack_vec + alpha * delta_slack_aff).dot(lambda_vec + alpha * delta_lambda_aff) / dual_gap;
    sigma = std::max(std::pow(sigma, 3), 0.8);

    Eigen::VectorXd mu_vec = Eigen::VectorXd::Ones(num_slack) * mu * sigma;

    Eigen::VectorXd center_rhs_vec;
    center_rhs_vec.setZero(num_vars + num_slack + num_lambda);
    center_rhs_vec.block(num_vars, 0, num_slack, 1) =
        slack_mat.inverse() * (mu_vec - delta_slack_aff.asDiagonal() * delta_lambda_aff);

    Eigen::VectorXd delta_cc = ldlt.solve(center_rhs_vec);

    Eigen::VectorXd delta = delta_aff + delta_cc;
    Eigen::MatrixXd delta_x = delta.block(0, 0, num_vars, 1);
    Eigen::MatrixXd delta_slack = delta.block(num_vars, 0, num_slack, 1);
    Eigen::VectorXd delta_lambda = delta.block(num_vars + num_slack, 0, num_slack, 1);

    alpha = std::min(linesearch_func(slack_vec, delta_slack),
                     linesearch_func(lambda_vec, delta_lambda));
    alpha = std::min(1.0, 0.99 * alpha);

    x += alpha * delta_x;
    slack_vec += alpha * delta_slack;
    lambda_vec += alpha * delta_lambda;

    lambda_mat = lambda_vec.asDiagonal();
    slack_mat = slack_vec.asDiagonal();

    return false;
}

bool kkt_newton_step(Eigen::VectorXd& x, Eigen::MatrixXd& lambda, Eigen::MatrixXd& lambda_mat,
                     GradientFunction gradient_func, HessianFunction hessian_func,
                     ConstraintsFunction constraints_func,
                     JacbianConstraintsFunction jac_constraints_func) {
    Eigen::VectorXd gradient_value = gradient_func(x);
    Eigen::VectorXd constraint_value = constraints_func(x);

    Eigen::MatrixXd hess = hessian_func(x);
    Eigen::MatrixXd jac_value = jac_constraints_func(x);
    Eigen::VectorXd lambda_vec = lambda_mat.diagonal();
    Eigen::MatrixXd jac_value_transpose = jac_value.transpose();

    int num_vars = x.size();
    int num_cons = lambda_vec.size();

    // Build KKT System.

    Eigen::VectorXd kkt_rhs;
    kkt_rhs.setZero(num_vars + num_cons);
    kkt_rhs.head(num_vars) = -gradient_value - jac_value_transpose * lambda_vec;
    kkt_rhs.tail(num_cons) = -constraint_value;

    if (kkt_rhs.norm() < 0.001) {
        return true;
    }

    Eigen::MatrixXd kkt_lhs;
    kkt_lhs.setZero(num_vars + num_cons, num_vars + num_cons);
    kkt_lhs.topLeftCorner(num_vars, num_vars) = hess;
    kkt_lhs.topRightCorner(num_vars, num_cons) = jac_value_transpose;
    kkt_lhs.bottomLeftCorner(num_cons, num_vars) = jac_value;

    Eigen::VectorXd delta;
    delta.setZero(num_vars + num_cons);
    delta = kkt_lhs.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(kkt_rhs);

    x += delta.head(num_vars);
    lambda_vec += delta.tail(num_cons);
    lambda_mat = lambda_vec.asDiagonal();

    return false;
}

}  // namespace OptSolver
