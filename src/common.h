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