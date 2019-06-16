#pragma once

#include <Eigen/Dense>
#include "GRANSAC/include/AbstractModel.hpp"
#include "p5p_radial.h"

using namespace Eigen;

inline Matrix<double, 4, 1> resize(const Matrix<double, 3, 1>& a) {
	double p[4];
	auto data = a.data();
	for (auto i = 0; i < 3; ++i){
		p[i] = data[i];
	}
	p[3] = 1;
	return Matrix<double, 4, 1>(p);
}

typedef Matrix<GRANSAC::VPFloat, 2, 1> Vector2;
typedef Matrix<GRANSAC::VPFloat, 3, 1> Vector3;
typedef Matrix<GRANSAC::VPFloat, 4, 1> Vector4;
typedef Matrix<GRANSAC::VPFloat, 2, 4> Matrix2x4;

class Correspondence
	: public GRANSAC::AbstractParameter
{
public:
	Correspondence(const Vector2& feature, const Vector3& point, const int id)
		: _feature(feature), _point(point), _id(id)
	{};

	const Vector2 _feature;
	const Vector3 _point;
	const int _id;
};

class PoseModel
	: public GRANSAC::AbstractModel<5>
{
protected:
	Matrix2x4 pose;

	virtual GRANSAC::VPFloat ComputeDistanceMeasure(std::shared_ptr<GRANSAC::AbstractParameter> Param) override
	{
		auto corr = std::dynamic_pointer_cast<Correspondence>(Param);
		auto P = resize(corr->_point);
		auto proj = pose * P;
		auto proj_normalized = proj.normalized();

		auto A = proj_normalized.dot(corr->_feature) * proj_normalized;
		auto diff = A - corr->_feature;
		return diff.norm();
	};

public:
	PoseModel(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> &InputParams)
	{
		Initialize(InputParams);
	};

	virtual void Initialize(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> &InputParams) override
	{
		if (InputParams.size() != 5)
			throw std::runtime_error("PoseModel - Number of input parameters does not match minimum number required for this model.");

		// Check for AbstractParamter types
		std::shared_ptr<Correspondence> corr[5] = {nullptr};
		for (int i = 0; i < 5; ++i) {
			corr[i] = std::dynamic_pointer_cast<Correspondence>(InputParams[i]);
			if (corr[i] == nullptr) {
				throw std::runtime_error("PoseModel - InputParams type mismatch. It is not a Correspondence.");
			}
		}
		std::copy(InputParams.begin(), InputParams.end(), m_MinModelParams.begin());
		Matrix<double, 2, 5> x;
		for (auto i = 0; i < 5; ++i) {
			x(0, i) = corr[i]->_feature(0, 0);
			x(1, i) = corr[i]->_feature(1, 0);
		}
		Matrix<double, 3, 5> X;
		for (auto i = 0; i < 5; ++i) {
			X(0, i) = corr[i]->_point(0, 0);
			X(1, i) = corr[i]->_point(1, 0);
			X(2, i) = corr[i]->_point(2, 0);
		}

		Matrix<double, 8, 4> P;
		P.setZero();
		p5p_radial(x, X, P);
		for (auto i = 0; i < 8; ++i) {
			pose((i / 4), (i % 4)) = P(i, 0);
		}
	};

	virtual std::pair<GRANSAC::VPFloat, std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>> Evaluate(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>& EvaluateParams, GRANSAC::VPFloat Threshold) override
	{
		std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> Inliers;
		int nTotalParams = EvaluateParams.size();
		int nInliers = 0;

		for (auto& Param : EvaluateParams)
		{
			if (ComputeDistanceMeasure(Param) < Threshold)
			{
				Inliers.push_back(Param);
				nInliers++;
			}
		}

		GRANSAC::VPFloat InlierFraction = GRANSAC::VPFloat(nInliers) / GRANSAC::VPFloat(nTotalParams); // This is the inlier fraction

		return std::make_pair(InlierFraction, Inliers);
	};

	Matrix2x4& Pose() {
		return pose;
	}
};

