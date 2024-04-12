#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iomanip>  // 设置输出格式
#include <sstream>
#include <string>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void meshgrid(Eigen::VectorXd &vecX, Eigen::VectorXd &vecY, Eigen::MatrixXd &meshX,
              Eigen::MatrixXd &meshY) {
    int vecXLength = vecX.size();
    int vecYLength = vecY.size();
    // meshX.resize(vecXLength, vecYLength);
    // meshY.resize(vecXLength, vecYLength);
    // std::cout << "meshX.size()\n" << meshX.size() << "\n" << "meshY.size()\n" << meshY.size() <<
    // std::endl;
    for (int i = 0; i < vecYLength; ++i) {
        meshX.row(i) = vecX;
        // std::cout << "meshX.row("<< i <<")\n" << meshX.row(i)<< std::endl;
    }

    for (int i = 0; i < vecXLength; ++i) {
        meshY.col(i) = vecY.transpose();
        // std::cout << "meshX.row(" << i << ")\n" << meshY.col(i) << std::endl;
    }
}

void plot_landscape(std::function<double(Eigen::VectorXd)> f) {
    int Nsamp = 100;
    Eigen::VectorXd X(Nsamp);
    X = Eigen::VectorXd::LinSpaced(Nsamp, -6, 6);
    Eigen::VectorXd Y(Nsamp);
    Y = Eigen::VectorXd::LinSpaced(Nsamp, -6, 6);

    Eigen::MatrixXd meshX(Nsamp, Nsamp);
    Eigen::MatrixXd meshY(Nsamp, Nsamp);
    Eigen::MatrixXd meshZ(Nsamp, Nsamp);

    meshgrid(X, Y, meshX, meshY);

    for (int ix = 0; ix < Nsamp; ix++) {
        for (int iy = 0; iy < Nsamp; iy++) {
            double x = meshX(ix, iy);
            double y = meshY(ix, iy);
            meshZ(ix, iy) = f(Eigen::Vector2d({x, y}));
        }
    }
    plt::axis("equal");
    plt::levels_contour(meshX, meshY, meshZ, 15);
    plt::draw();
    plt::xlim(-4, 4);
    plt::ylim(-4, 4);
    plt::pause(0.1);
}

void plot_ineq_constraint() {
    Eigen::VectorXd xc = Eigen::VectorXd::LinSpaced(200, -4, 3);
    Eigen::VectorXd xcc = xc.unaryExpr([](double x) { return x + 1.0; });

    plt::plot(xc, xcc, "g--", {{"label", "inequality constraint"}});
    plt::xlabel("x[0]");
    plt::ylabel("x[1]");
    plt::legend();
    plt::xlim(-4, 4);
    plt::ylim(-4, 4);
    plt::pause(0.1);
}

void plot_eq_constraint() {
    Eigen::VectorXd xc = Eigen::VectorXd::LinSpaced(200, -4, 3);
    Eigen::VectorXd xcc = xc.unaryExpr([](double x) { return x * x + 2.0 * x; });
    plt::plot(xc, xcc, "b.", {{"label", "equality constraint"}});

    plt::xlabel("x[0]");
    plt::ylabel("x[1]");
    plt::legend();
    plt::xlim(-4, 4);
    plt::ylim(-4, 4);
    plt::pause(0.1);
}

void plot_solution_path(std::vector<Eigen::VectorXd> x_solve_traj,
                        std::function<double(Eigen::VectorXd)> f, std::string title = "") {
    std::string xlabel;
    plt::title(title);
    for (int i = 0; i < x_solve_traj.size(); i++) {
        Eigen::Vector2d x = x_solve_traj.at(i);

        xlabel = "Step: " + std::to_string(i + 1) + ", x: [ " + std::to_string(x(0)) + ", " +
                 std::to_string(x(1)) + " ], y: " + std::to_string(f(x)) + " .";

        std::vector<double> x0({x(0)});
        std::vector<double> x1({x(1)});
        plt::plot(x0, x1, "rx");
        plt::xlabel(xlabel);

        plt::pause(0.5);
    }

    xlabel = "Finished, " + xlabel;
    plt::xlabel(xlabel);
    plt::pause(0.5);
}

#endif