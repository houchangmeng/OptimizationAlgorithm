#include "unconstraintsolver.h"

namespace OptSolver {
UnconstraintSolver::UnconstraintSolver(ObjectiveFunction objective_func,
                                       GradientFunction gradient_func, HessianFunction hessian_func)
    : obj_func_(objective_func), grad_func_(gradient_func), hess_func_(hessian_func) {
    options_ = Options();
}

UnconstraintSolver::UnconstraintSolver(UnconstraintSolver::Options opts) : options_(opts) {
    obj_func_ = opts.obj_func_;
    grad_func_ = opts.grad_func_;
    hess_func_ = opts.hess_func_;
}

void UnconstraintSolver::Initialize(Eigen::VectorXd x0) {
    x0_ = x0;
    grad0_ = grad_func_(x0);

    if (hess_func_ == nullptr) {
        hess0_ = x0.asDiagonal();
        hess0_.setIdentity();

    } else {
        hess0_ = hess_func_(x0);
    }
    cg_direc_mat0_ = (-grad0_).asDiagonal();

    linesearch_func_ = std::bind(backtracking_linesearch, std::placeholders::_1,
                                 std::placeholders::_2, std::placeholders::_3, obj_func_);

    switch (options_.solvertype_) {
        case SolverType::GradientDescend:
            iter_func_ = std::bind(gradient_step_with_linesearch, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, linesearch_func_,
                                   grad_func_, hess_func_);
            break;
        case SolverType::GaussNewton:
            iter_func_ = std::bind(gauss_newton_step_with_linesearch, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, linesearch_func_,
                                   grad_func_, hess_func_);

            if (hess_func_ == nullptr) {
                printf("File : %s \nFunction: %s \nLine : %d\n", __FILE__, __func__, __LINE__);
                printf("\033[31mGauss newton must provide hessian function. \033[0m\n");
                std::exit(0);
            }

            break;
        case SolverType::QuasiNewton_BFGS:
            iter_func_ = std::bind(bfgs_newton_step_with_linesearch, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, linesearch_func_,
                                   grad_func_, hess_func_);
            break;
        case SolverType::QuasiNewton_DFP:
            iter_func_ = std::bind(dfp_newton_step_with_linesearch, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, linesearch_func_,
                                   grad_func_, hess_func_);
            break;
        case SolverType::ConjugateGradient:
            iter_func_ = std::bind(conjugate_gradient_step_with_linesearch, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, linesearch_func_,
                                   grad_func_, hess_func_);
            hess0_ = cg_direc_mat0_;
            break;

        default:
            iter_func_ = std::bind(gauss_newton_step_with_linesearch, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3, linesearch_func_,
                                   grad_func_, hess_func_);
            if (hess_func_ == nullptr) {
                printf("File : %s \nFunction: %s \nLine : %d\n", __FILE__, __func__, __LINE__);
                printf("\033[31mGauss newton must provide hessian function. \033[0m\n");
                std::exit(0);
            }
            break;
    }

    x_solution_traj_.push_back(x0_);
    is_initialized = true;
}

bool UnconstraintSolver::Solve() {
    if (!is_initialized) {
        printf("File : %s \nFunction: %s \nLine : %d\n", __FILE__, __func__, __LINE__);
        printf("\033[31mSolver should initialized before iterate.\033[0m\n");
        std::exit(0);
    }

    int step = 0;

    while (1) {
        if (iter_func_(x0_, grad0_, hess0_)) {
            break;
        };
        if (step > options_.max_iternum_) {
            printf("\033[31mSolver has reached maximum iterate step.\033[0m\n");
            return false;
        }

        x_solution_traj_.push_back(x0_);
        step++;
    }

    x_solution_traj_.push_back(x0_);
    printf("\033[32mSuccess.\033[0m\n");
    return true;
}

UnconstraintSolver::~UnconstraintSolver() {}

bool gradient_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::MatrixXd& hess,
                                   LineSearchFunction linesearchfunc,
                                   GradientFunction gradient_func, HessianFunction hessian_func) {
    if (grad.norm() < 0.01) {
        return true;
    }
    Eigen::VectorXd delta_x = -grad;
    linesearchfunc(x, delta_x, grad);
    grad = gradient_func(x);
    return false;
}

bool gauss_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad,
                                       Eigen::MatrixXd& hess, LineSearchFunction linesearchfunc,
                                       GradientFunction gradient_func,
                                       HessianFunction hessian_func) {
    if (grad.norm() < 0.01) {
        return true;
    }
    Eigen::VectorXd delta_x = -hess.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(grad);
    linesearchfunc(x, delta_x, grad);

    grad = gradient_func(x);
    hess = hessian_func(x);

    return false;
}

bool bfgs_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::MatrixXd& B,
                                      LineSearchFunction linesearchfunc,
                                      GradientFunction gradient_func,
                                      HessianFunction hessian_func) {
    if (grad.norm() < 0.01) {
        return true;
    }
    Eigen::VectorXd delta_x = -B.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(grad);
    auto delta_xt = delta_x.transpose();
    // linesearchfunc(x, delta_x, grad);
    Eigen::VectorXd xnew = x + delta_x;

    Eigen::VectorXd grad_new = gradient_func(x);

    Eigen::VectorXd delta_y = grad_new - grad;
    auto delta_yt = delta_y.transpose();

    B += (delta_y * delta_yt) / (delta_yt * delta_x) -
         (B * delta_x * delta_xt * B) / (delta_xt * B * delta_x);

    x = xnew;
    grad = grad_new;

    return false;
}

bool dfp_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::MatrixXd& G,
                                     LineSearchFunction linesearchfunc,
                                     GradientFunction gradient_func, HessianFunction hessian_func) {
    if (grad.norm() < 0.01) {
        return true;
    }
    Eigen::VectorXd delta_x = -G.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(grad);
    // linesearchfunc(x, delta_x, grad);
    auto delta_xt = delta_x.transpose();
    Eigen::VectorXd xnew = x + delta_x;
    Eigen::VectorXd grad_new = gradient_func(xnew);

    Eigen::VectorXd delta_y = grad_new - grad;
    auto delta_yt = delta_y.transpose();

    G += (delta_x * delta_xt) / (delta_xt * delta_y) -
         (G * delta_y * delta_yt * G) / (delta_yt * G * delta_y);

    x = xnew;
    grad = grad_new;

    return false;
}

bool conjugate_gradient_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad,
                                             Eigen::MatrixXd& direc_mat,
                                             LineSearchFunction linesearchfunc,
                                             GradientFunction gradient_func,
                                             HessianFunction hessian_func) {
    if (grad.norm() < 0.01) {
        return true;
    }
    Eigen::VectorXd direc_vec = direc_mat.diagonal();
    Eigen::VectorXd delta_x = direc_vec;
    Eigen::VectorXd xnew = x;
    // xk+1 = xk + alpha * △x
    linesearchfunc(xnew, delta_x, grad);

    Eigen::VectorXd grad_new = gradient_func(xnew);

    // FR
    double beta = grad_new.transpose().dot(grad_new) / grad.transpose().dot(grad);
    // PRP
    // double beta = grad_new.transpose().dot(grad_new - grad) / grad.transpose().dot(grad);
    // dk+1 = -grad(xk+1) + beta * dk
    Eigen::VectorXd direc_vec_new = -grad_new + beta * direc_vec;

    x = xnew;
    grad = grad_new;
    direc_mat = direc_vec_new.asDiagonal();

    return false;
}

}  // namespace OptSolver
