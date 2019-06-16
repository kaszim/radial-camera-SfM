#pragma once
#include "closed.h"

using namespace Eigen;

//int p5p_radial(const Eigen::Matrix<double, 2, 5> &x, const Eigen::Matrix<double, 3, 5> &X, Eigen::Matrix<double, 8, 4> &P_out)
void p5p_radial(Matrix<double, 2, 5>& x, Matrix<double, 3, 5>& X, Matrix<double, 8, 4>& P);