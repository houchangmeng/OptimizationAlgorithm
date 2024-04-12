#include "common.h"

namespace OptSolver {
bool matrix_is_PSD(Eigen::MatrixXd A) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A);
    if (eigensolver.info() != Eigen::Success) {
        // Eigenvalue computation failed
        return false;
    }

    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();

    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < 0) {
            return false;
        }
    }
    return true;
}

void backtracking_linesearch(Eigen::VectorXd& x, Eigen::VectorXd& delta_x, Eigen::VectorXd& grad,
                             ObjectiveFunction f) {
    double alpha = 1.0;

    while (f(x + alpha * delta_x) > f(x) + 0.001 * alpha * grad.transpose() * delta_x) {
        alpha = 0.5 * alpha;
    }

    delta_x = alpha * delta_x;
    x = x + delta_x;
}
}  // namespace OptSolver
