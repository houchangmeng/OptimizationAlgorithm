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
