#include "p5p_radial.h"
#include "closed.h"

using namespace Eigen;

//int p5p_radial(const Eigen::Matrix<double, 2, 5> &x, const Eigen::Matrix<double, 3, 5> &X, Eigen::Matrix<double, 8, 4> &P_out)
void p5p_radial(Matrix<double, 2, 5>& x, Matrix<double, 3, 5>& X, Matrix<double, 8, 4>& P)
{
	// Setup nullspace
	Matrix<double, 8, 5> cc;
	for (int i = 0; i < 5; i++) {
		cc(0, i) = -x(1, i)*X(0, i);
		cc(1, i) = -x(1, i)*X(1, i);
		cc(2, i) = -x(1, i)*X(2, i);
		cc(3, i) = -x(1, i);
		cc(4, i) = x(0, i)*X(0, i);
		cc(5, i) = x(0, i)*X(1, i);
		cc(6, i) = x(0, i)*X(2, i);
		cc(7, i) = x(0, i);
	}
	Matrix<double, 8, 8> Q = cc.householderQr().householderQ();
	Matrix<double, 8, 3> N = Q.rightCols(3);

	//Matrix<double, 8, 3> N = cc.householderQr().householderQ().rightCols(3);

	// Compute coefficients for sylvester resultant
	double c11_1 = N(0, 1)*N(4, 1) + N(1, 1)*N(5, 1) + N(2, 1)*N(6, 1);
	double c12_1 = N(0, 1)*N(4, 2) + N(0, 2)*N(4, 1) + N(1, 1)*N(5, 2) + N(1, 2)*N(5, 1) + N(2, 1)*N(6, 2) + N(2, 2)*N(6, 1);
	double c12_2 = N(0, 0)*N(4, 1) + N(0, 1)*N(4, 0) + N(1, 0)*N(5, 1) + N(1, 1)*N(5, 0) + N(2, 0)*N(6, 1) + N(2, 1)*N(6, 0);
	double c13_1 = N(0, 2)*N(4, 2) + N(1, 2)*N(5, 2) + N(2, 2)*N(6, 2);
	double c13_2 = N(0, 0)*N(4, 2) + N(0, 2)*N(4, 0) + N(1, 0)*N(5, 2) + N(1, 2)*N(5, 0) + N(2, 0)*N(6, 2) + N(2, 2)*N(6, 0);
	double c13_3 = N(0, 0)*N(4, 0) + N(1, 0)*N(5, 0) + N(2, 0)*N(6, 0);
	double c21_1 = N(0, 1)*N(0, 1) + N(1, 1)*N(1, 1) + N(2, 1)*N(2, 1) - N(4, 1)*N(4, 1) - N(5, 1)*N(5, 1) - N(6, 1)*N(6, 1);
	double c22_1 = 2 * N(0, 1)*N(0, 2) + 2 * N(1, 1)*N(1, 2) + 2 * N(2, 1)*N(2, 2) - 2 * N(4, 1)*N(4, 2) - 2 * N(5, 1)*N(5, 2) - 2 * N(6, 1)*N(6, 2);
	double c22_2 = 2 * N(0, 0)*N(0, 1) + 2 * N(1, 0)*N(1, 1) + 2 * N(2, 0)*N(2, 1) - 2 * N(4, 0)*N(4, 1) - 2 * N(5, 0)*N(5, 1) - 2 * N(6, 0)*N(6, 1);
	double c23_1 = N(0, 2)*N(0, 2) + N(1, 2)*N(1, 2) + N(2, 2)*N(2, 2) - N(4, 2)*N(4, 2) - N(5, 2)*N(5, 2) - N(6, 2)*N(6, 2);
	double c23_2 = 2 * N(0, 0)*N(0, 2) + 2 * N(1, 0)*N(1, 2) + 2 * N(2, 0)*N(2, 2) - 2 * N(4, 0)*N(4, 2) - 2 * N(5, 0)*N(5, 2) - 2 * N(6, 0)*N(6, 2);
	double c23_3 = N(0, 0)*N(0, 0) + N(1, 0)*N(1, 0) + N(2, 0)*N(2, 0) - N(4, 0)*N(4, 0) - N(5, 0)*N(5, 0) - N(6, 0)*N(6, 0);

	double a0 = c11_1 * c11_1*c23_3*c23_3 - c11_1 * c12_2*c22_2*c23_3 - 2 * c11_1*c13_3*c21_1*c23_3 + c11_1 * c13_3*c22_2*c22_2 + c12_2 * c12_2*c21_1*c23_3 - c12_2 * c13_3*c21_1*c22_2 + c13_3 * c13_3*c21_1*c21_1;
	double a1 = c11_1 * c13_2*c22_2*c22_2 + 2 * c13_2*c13_3*c21_1*c21_1 + c12_2 * c12_2*c21_1*c23_2 + 2 * c11_1*c11_1*c23_2*c23_3 - c11_1 * c12_1*c22_2*c23_3 - c11_1 * c12_2*c22_1*c23_3 - c11_1 * c12_2*c22_2*c23_2 - 2 * c11_1*c13_2*c21_1*c23_3 - 2 * c11_1*c13_3*c21_1*c23_2 + 2 * c11_1*c13_3*c22_1*c22_2 + 2 * c12_1*c12_2*c21_1*c23_3 - c12_1 * c13_3*c21_1*c22_2 - c12_2 * c13_2*c21_1*c22_2 - c12_2 * c13_3*c21_1*c22_1;
	double a2 = c11_1 * c11_1*c23_2*c23_2 + c13_2 * c13_2*c21_1*c21_1 + c11_1 * c13_1*c22_2*c22_2 + c11_1 * c13_3*c22_1*c22_1 + 2 * c13_1*c13_3*c21_1*c21_1 + c12_2 * c12_2*c21_1*c23_1 + c12_1 * c12_1*c21_1*c23_3 + 2 * c11_1*c11_1*c23_1*c23_3 - c11_1 * c12_1*c22_1*c23_3 - c11_1 * c12_1*c22_2*c23_2 - c11_1 * c12_2*c22_1*c23_2 - c11_1 * c12_2*c22_2*c23_1 - 2 * c11_1*c13_1*c21_1*c23_3 - 2 * c11_1*c13_2*c21_1*c23_2 + 2 * c11_1*c13_2*c22_1*c22_2 - 2 * c11_1*c13_3*c21_1*c23_1 + 2 * c12_1*c12_2*c21_1*c23_2 - c12_1 * c13_2*c21_1*c22_2 - c12_1 * c13_3*c21_1*c22_1 - c12_2 * c13_1*c21_1*c22_2 - c12_2 * c13_2*c21_1*c22_1;
	double a3 = c11_1 * c13_2*c22_1*c22_1 + 2 * c13_1*c13_2*c21_1*c21_1 + c12_1 * c12_1*c21_1*c23_2 + 2 * c11_1*c11_1*c23_1*c23_2 - c11_1 * c12_1*c22_1*c23_2 - c11_1 * c12_1*c22_2*c23_1 - c11_1 * c12_2*c22_1*c23_1 - 2 * c11_1*c13_1*c21_1*c23_2 + 2 * c11_1*c13_1*c22_1*c22_2 - 2 * c11_1*c13_2*c21_1*c23_1 + 2 * c12_1*c12_2*c21_1*c23_1 - c12_1 * c13_1*c21_1*c22_2 - c12_1 * c13_2*c21_1*c22_1 - c12_2 * c13_1*c21_1*c22_1;
	double a4 = c11_1 * c11_1*c23_1*c23_1 - c11_1 * c12_1*c22_1*c23_1 - 2 * c11_1*c13_1*c21_1*c23_1 + c11_1 * c13_1*c22_1*c22_1 + c12_1 * c12_1*c21_1*c23_1 - c12_1 * c13_1*c21_1*c22_1 + c13_1 * c13_1*c21_1*c21_1;

	std::complex<double> roots[4];

	// This gives us the value for x
	solve_quartic(a1 / a0, a2 / a0, a3 / a0, a4 / a0, roots);

	int n_sols = 0;
	for (int i = 0; i < 4; i++) {
		if (std::abs(roots[i].imag()) > 1e-6)
			continue;

		// We have two quadratic polynomials in y after substituting x
		double a = roots[i].real();
		double c1a = c11_1;
		double c1b = c12_1 + c12_2 * a;
		double c1c = c13_1 + c13_2 * a + c13_3 * a*a;

		double c2a = c21_1;
		double c2b = c22_1 + c22_2 * a;
		double c2c = c23_1 + c23_2 * a + c23_3 * a*a;

		// we solve the first one
		std::complex<double> bb[2];
		solve_quadratic(c1a, c1b, c1c, bb);

		if (std::abs(bb[0].imag()) > 1e-6)
			continue;

		// and check the residuals of the other
		double res1 = c2a * bb[0].real()*bb[0].real() + c2b * bb[0].real() + c2c;
		double res2 = c2a * bb[1].real()*bb[1].real() + c2b * bb[1].real() + c2c;

		// Note: this is chosen such that it is correct when res2 is nan which happens for the degenerate case.
		double b = (std::abs(res1) > std::abs(res2)) ? bb[1].real() : bb[0].real();

		// Save output
		for (int j = 0; j < 8; j++)
			P(j,n_sols) = N(j, 0)*a + N(j, 1)*b + N(j, 2);
		n_sols++;
	}
}