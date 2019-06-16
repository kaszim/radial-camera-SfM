#ifndef SFM_H
#define SFM_H

#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <map>
#include <limits>
#include "GRANSAC/include/GRANSAC.hpp"
#include "RansacModel.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

struct Feature {
    Vector2 point;
    int id;
};

struct WorldPoint {
    Vector3 point;
    int id;
};

struct Pose {
    Matrix2x4 pose;
    int id;
};

struct FeaturePosePair {
    Feature* feature;
    int view_id;
};

struct ReprojectionError
{
    ReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {
        T P[4];
        std::copy(point, point + 3, P);
        P[3] = T(1.0);
        Matrix<T, 4, 1> p(P);
        Quaternion<T> quaternion;
        quaternion.x() = camera[0];
        quaternion.y() = camera[1];
        quaternion.z() = camera[2];
        quaternion.w() = camera[3];
        // Might need normalization
        auto rotMatrix = quaternion.toRotationMatrix();
        Matrix<T, 2, 4> cam;
        cam <<  rotMatrix.data()[0], rotMatrix.data()[3], rotMatrix.data()[6], camera[4],
                rotMatrix.data()[1], rotMatrix.data()[4], rotMatrix.data()[7], camera[5];

        auto proj = cam * p;
        auto proj_normalized = proj.normalized();

        Matrix<T, 2, 1> observation;
        observation << T(observed_x), T(observed_y);

        auto A = proj_normalized.dot(observation) * proj_normalized;
        auto diff = A - observation;
        residuals[0] = diff.squaredNorm();
        if (residuals[0] == T(0.0)) {
            residuals[0] = T(std::numeric_limits<double>::min());
        }

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 1, 6, 3>(
            new ReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
};

struct ReprojectionErrorPoints
{
    ReprojectionErrorPoints(double observed_x, double observed_y, const double* camera)
        : observed_x(observed_x), observed_y(observed_y), camera(camera) {}

    template <typename T>
    bool operator()(const T *const point,
                    T *residuals) const
    {
        T P[4];
        std::copy(point, point + 3, P);
        P[3] = T(1.0);
        Matrix<T, 2, 4> cam;
        cam << T(camera[0]), T(camera[2]), T(camera[4]), T(camera[6])
            ,  T(camera[1]), T(camera[3]), T(camera[5]), T(camera[7]);
        Matrix<T, 4, 1> p(P);

        auto proj = cam * p;
        auto proj_normalized = proj.normalized();

        Matrix<T, 2, 1> observation;
        observation << T(observed_x), T(observed_y);

        auto A = proj_normalized.dot(observation) * proj_normalized;
        auto diff = A - observation;
        residuals[0] = diff.squaredNorm();
        if (residuals[0] == T(0.0)) {
            residuals[0] = T(std::numeric_limits<double>::min());
        }

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y, const double* camera)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorPoints, 1, 3>(
            new ReprojectionErrorPoints(observed_x, observed_y, camera)));
    }

    double observed_x;
    double observed_y;
    const double* camera;
};

struct ReprojectionErrorPose
{
    ReprojectionErrorPose(double observed_x, double observed_y, double* point)
        : observed_x(observed_x), observed_y(observed_y), point(point) {}

    template <typename T>
    bool operator()(const T *const camera,
                    T *residuals) const
    {
        Quaternion<T> quaternion;
        quaternion.x() = camera[0];
        quaternion.y() = camera[1];
        quaternion.z() = camera[2];
        quaternion.w() = camera[3];
        // Might need normalization
        auto rotMatrix = quaternion.toRotationMatrix();
        T P[4];
        for (int i = 0; i < 3; ++i) {
            P[i] = T(point[i]);
        }
        P[3] = T(1.0);
        Matrix<T, 2, 4> cam;
        cam <<  rotMatrix.data()[0], rotMatrix.data()[3], rotMatrix.data()[6], camera[4],
                rotMatrix.data()[1], rotMatrix.data()[4], rotMatrix.data()[7], camera[5];
        Matrix<T, 4, 1> p(P);
        auto proj = cam * p;
        auto proj_normalized = proj.normalized();

        Matrix<T, 2, 1> observation;
        observation << T(observed_x), T(observed_y);

        auto A = proj_normalized.dot(observation) * proj_normalized;
        auto diff = A - observation;
        residuals[0] = diff.squaredNorm();
        if (residuals[0] == T(0.0)) {
            residuals[0] = T(std::numeric_limits<double>::min());
        }

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y,
                                       double* point)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorPose, 1, 6>(
            new ReprojectionErrorPose(observed_x, observed_y, point)));
    }

    double observed_x;
    double observed_y;
    double* point;
};

class SfM {
public:
    SfM() {
        _correspondences = std::map<int, std::vector<FeaturePosePair>>();
        _poses = std::vector<Pose>();
        _points3D = std::vector<WorldPoint>();
    }

    void add_features(const std::vector<double>& x, const std::vector<double>& y, const std::vector<int>& ids, double ppx, double ppy);

    void add_view(int view);
    int get_nviews() const;
    void print_views() const;

    void initializePoints(const std::vector<WorldPoint>& initial_points);
    void initialize(Matrix<double, 3, 4>& pose1, Matrix<double, 3, 4>& pose2);
    void print_currentPoints();
    void print_currentPoses();
    void print_finalPoints() const;

    int get_nimages() const;
    int get_npoints() const;

    int matches_between(int i, int j) const;

    void get_initial_poses(int& view1, int& view2);

    int determineNextCandidate();
    void ransacPose(const int view, const double threshold, const int iterations);
    void triangulate_new_points();
    void triangulate_all_points();
    void pointsBA();
    void posesBA();
    void fullBA();
    int minNumFeatures_;

private:
    Pose& get_pose(int id);
    WorldPoint& get_point(int id);
    void add_matches(const int new_view);
    void intersecting_points(std::vector<Feature>& features, std::vector<Feature*>& outFeatures, std::vector<WorldPoint*>& outPoints);
    bool contains_3d_point(int id);
    double* get_ceres_param(Pose& pose, std::vector<double*>& params, std::vector<Pose*>& poses);
    // current used views in the pipeline
    std::vector<int> _current_views;
    // features for every image
    std::vector<std::vector<Feature> > _features;
    // point cloud that is grown
    std::vector<WorldPoint> _points3D;
    std::vector<WorldPoint> _all_points3D;
    // view i to view j gives a pair of features that are matching (id wise) 
    std::vector<std::vector<std::vector<std::pair<Feature*, Feature*>>>> _matches;
    // pose estimates
    std::vector<Pose> _poses;
    // point_id => features that are the same 3d_point (features corresponding to a 3d_point)
    std::map<int, std::vector<FeaturePosePair>> _correspondences;
};
#endif
