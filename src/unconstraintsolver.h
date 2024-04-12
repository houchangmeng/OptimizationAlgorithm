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
                                   LineSearchFunction linesearchfunc,
                                   GradientFunction gradient_func, HessianFunction hessian_func);

bool gauss_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad,
                                       Eigen::MatrixXd& hess, LineSearchFunction linesearchfunc,
                                       GradientFunction gradient_func,
                                       HessianFunction hessian_func);

bool bfgs_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::MatrixXd& B,
                                      LineSearchFunction linesearchfunc,
                                      GradientFunction gradient_func, HessianFunction hessian_func);

bool dfp_newton_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::MatrixXd& G,
                                     LineSearchFunction linesearchfunc,
                                     GradientFunction gradient_func, HessianFunction hessian_func);

bool conjugate_gradient_step_with_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& grad,
                                             Eigen::MatrixXd& direc_mat,
                                             LineSearchFunction linesearchfunc,
                                             GradientFunction gradient_func,
                                             HessianFunction hessian_func);
}  // namespace OptSolver

#endif