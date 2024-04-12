#ifndef COMMON_h
#define COMMON_h

#include <Eigen/Dense>
#include <functional>

namespace OptSolver {
typedef std::function<double(Eigen::VectorXd)> ObjectiveFunction;
typedef std::function<Eigen::VectorXd(Eigen::VectorXd)> GradientFunction;
typedef std::function<Eigen::MatrixXd(Eigen::VectorXd)> HessianFunction;
typedef std::function<void(Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&)>
    LineSearchFunction;

bool matrix_is_PSD(Eigen::MatrixXd A);
void backtracking_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& delta_x, Eigen::VectorXd& grad,
                             ObjectiveFunction f);

}  // namespace OptSolver

#endif