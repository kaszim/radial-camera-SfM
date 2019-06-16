#include "SfM.h"
#include <iostream>

// Add recorded features for view. This should be done in order
void SfM::add_features(const std::vector<double>& x, const std::vector<double>& y, const std::vector<int>& ids,
 double ppx, double ppy) {
    std::vector<Feature> view(x.size());
    for (auto i = 0; i < x.size(); ++i) {
        view[i].point(0,0) = x[i] - ppx;
        view[i].point(1,0) = y[i] - ppy;
        view[i].id = ids[i];
    }
    _features.emplace_back(std::move(view));
    add_matches(_features.size()-1);
}

// Adds a view
void SfM::add_view(int view){
    _current_views.emplace_back(view);
    std::sort(_current_views.begin(), _current_views.end());
}

int SfM::get_nviews() const {
    return _current_views.size();
}

void SfM::print_views() const {
    for(auto i = 0; i < get_nviews(); ++i){
        std::cout << _current_views[i] << ", ";
    }
}

// Adds 3d points that are used for initialization
void SfM::initializePoints(const std::vector<WorldPoint>& initial_points) {
    for (auto&& point: initial_points){
        _points3D.emplace_back(point);
    }
}

void SfM::print_currentPoints() {
    std::sort(_points3D.begin(), _points3D.end(), [](auto&& a, auto&& b) {
        return a.id > b.id;
    });
    for(auto i = 0; i < _points3D.size(); ++i){
        std::cout << _points3D[i].point(0) << ", " << _points3D[i].point(1) << ", " << _points3D[i].point(2) << ", " << _points3D[i].id<< std::endl;
    }
}

void SfM::print_currentPoses() {
    std::sort(_poses.begin(), _poses.end(), [](auto&& a, auto&& b) {
        return a.id < b.id;
    });
    for(auto i = 0; i < get_nviews(); ++i){
        std::cerr << _poses[i].id << std::endl;
        std::cout << _poses[i].pose << std::endl;
    }
}

int SfM::get_nimages() const {
    return _features.size();
}

int SfM::get_npoints() const {
    return _points3D.size();
}

int SfM::matches_between(int i, int j) const {
    return _matches[i][j].size();
}

void SfM::get_initial_poses(int& view1, int& view2){
    int max = 0;
    for(int i = 0; i < get_nimages(); i++){
        for(int j = i+1; j < get_nimages(); j++){
            if (matches_between(i,j) > max){
                max = matches_between(i,j);
                view1 = i;
                view2 = j;
            }
        }
    }
}

Pose& SfM::get_pose(int id) {
    for (auto&& pose : _poses) {
        if (pose.id == id) {
            return pose;
        }
    }
    throw std::out_of_range("No such pose");
}

WorldPoint& SfM::get_point(int id) {
    for (auto&& point : _points3D) {
        if (point.id == id) {
            return point;
        }
    }
    throw std::out_of_range("No such point");
}

bool SfM::contains_3d_point(int id) {
    for (auto&& point : _points3D) {
        if (point.id == id) {
            return true;
        }
    }
    return false;
}

// Adds features to _matches for the new view
void SfM::add_matches(const int new_view) {
    auto& newf = _features[new_view];
    _matches.emplace_back(std::vector<std::vector<std::pair<Feature*, Feature*>>>(get_nimages()));
    for (auto i = 0; i < new_view; ++i) {
        _matches[i].emplace_back(std::vector<std::pair<Feature*, Feature*>>());
        auto& old = _features[i];
        int n = 0;
        int o = 0;
        while ( n < newf.size() && o < old.size()) {
            if (newf[n].id == old[o].id) {
                _matches[i][new_view].emplace_back(std::make_pair(&old[o], &newf[n]));
                _matches[new_view][i].emplace_back(std::make_pair(&newf[n], &old[o]));
                ++n;
                ++o;
            } else if (newf[n].id < old[o].id) {
                ++n;
            } else if (newf[n].id > old[o].id) {
                ++o;
            }
        }
    }
}

// Returns the points and features that have already been triangulated
void SfM::intersecting_points(std::vector<Feature>& features, std::vector<Feature*>& outFeatures, std::vector<WorldPoint*>& outPoints) {
    for (auto&& point : _points3D) {
        for (auto&& feature : features) {
            if (point.id == feature.id) {
                outFeatures.emplace_back(&feature);
                outPoints.emplace_back(&point);
            }
        }
    }
}

// returns unused view with maximal matches to current views
int SfM::determineNextCandidate() {
    std::vector<int> candidates(get_nimages());
    std::iota(candidates.begin(), candidates.end(), 0); // fills 0.. end
    std::vector<int> diff;
    std::set_difference(candidates.begin(), candidates.end(), _current_views.begin(), _current_views.end(), 
                        std::inserter(diff, diff.begin()));
    int maxVal = 0;
    int nextBestView = -1;
    for (auto view : _current_views) {
        for (auto cand : diff) {
            int matches = matches_between(view, cand);
            if (matches > maxVal) {
                maxVal = matches;
                nextBestView = cand;
            }
        }
    }
    std::cerr << "Next View: " << nextBestView << std::endl;
    return nextBestView;
}

// estimate pose of view using available points3D
void SfM::ransacPose(const int view, const double threshold, const int iterations){
    Pose pose;
    std::vector<Feature*> correspondencesF;
    std::vector<WorldPoint*> correspondencesP;
    intersecting_points(_features[view], correspondencesF, correspondencesP);

    std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandCorrs;
    for (int i = 0; i < correspondencesF.size(); ++i) {
        auto&& f = correspondencesF[i];
        Vector2 feature;
        feature(0,0) = f->point(0);
        feature(1,0) = f->point(1);
        
        auto&& p = correspondencesP[i];
        Vector3 point;
        point(0,0) = p->point(0);
        point(1,0) = p->point(1);
        point(2,0) = p->point(2);
        CandCorrs.push_back(std::make_shared<Correspondence>(feature, point, f->id));
    }
    GRANSAC::RANSAC<PoseModel, 5> Estimator;
    Estimator.Initialize(threshold, iterations); // Threshold, iterations
    Estimator.Estimate(CandCorrs);

    auto bestPoseModel = std::dynamic_pointer_cast<PoseModel>(Estimator.GetBestModel());
    if (bestPoseModel) {
        // std::cout << bestPoseModel->Pose() << std::endl;
        pose.pose = bestPoseModel->Pose();
        pose.id = view;
        _poses.emplace_back(pose);
        for (auto&& feature : _features[view]) {
            _correspondences[feature.id].emplace_back(FeaturePosePair{
                            &feature, view
                        });
        }
    }
}

// triangulate points3D from available views
void SfM::triangulate_new_points() {
    std::cerr << "Triangulating new Points ..." << std::endl;
    for (auto&& pair : _correspondences) {
        if (contains_3d_point(pair.first)) {
            // point is already in the pipeline, skip it
            continue;
        }
        auto&& track = pair.second;
        auto sizeTrack = static_cast<int>(track.size());
        if (sizeTrack > minNumFeatures_-1) {
            // std::cerr << sizeTrack << std::endl;
            // Enough features to triangulate
            MatrixXd A(sizeTrack, 3);
            MatrixXd b(sizeTrack, 1);
            for (int i = 0; i < sizeTrack; ++i) {
                auto&& point = track[i];
                auto&& pose = get_pose(point.view_id);
                auto feature = point.feature->point;
                feature(0, 0) = -point.feature->point(1, 0);
                feature(1, 0) = point.feature->point(0, 0); // -y, x
                auto featureT = feature.transpose();
                auto M = featureT * pose.pose;
                for (int y = 0; y < 3; ++y) {
                    A(i, y) = M(0, y);
                }
                b(i, 0) = -M(0, 3);
            }
            MatrixXd point_3d = (A.transpose() * A).ldlt().solve(A.transpose() * b);
            _points3D.emplace_back(WorldPoint{point_3d, pair.first});
            // double norm = point_3d.norm();
            // if(norm < 10000){
            //     _points3D.emplace_back(WorldPoint{point_3d, pair.first});
            // }
        }
    }
}

// Returns a param that is ceres friendly
double* SfM::get_ceres_param(Pose& pose, std::vector<double*>& params, std::vector<Pose*>& poses) {
    auto&& poseData = pose.pose.data();
    double* param = nullptr;
    auto poseOrend= std::find(poses.begin(), poses.end(), &pose);
    if (poseOrend != poses.end()) {
        // this pose has already been added, thus use those params instead
        // Get index of element from iterator
        int i = std::distance(poses.begin(), poseOrend);
        param = params[i];
    } else {
        param = new double[6];
        Eigen::Matrix<double,1,3> row1;
        Eigen::Matrix<double,1,3> row2;
        Eigen::Matrix<double,1,3> row3;
        row1 << poseData[0], poseData[2], poseData[4];
        row2 << poseData[1], poseData[3], poseData[5];
        auto row1n = row1.normalized();
        auto row2n = row2.normalized();
        row3 = row1n.cross(row2n);
        Eigen::Matrix<double,3,3> rotation;
        rotation << row1n.data()[0], row1n.data()[1], row1n.data()[2],
                    row2n.data()[0], row2n.data()[1], row2n.data()[2],
                    row3.data()[0], row3.data()[1], row3.data()[2];
        auto quaternion = Eigen::Quaterniond(rotation);
        // Might need normalization
        param[0] = quaternion.x();
        param[1] = quaternion.y();
        param[2] = quaternion.z();
        param[3] = quaternion.w();
        param[4] = poseData[6] / row1.norm();
        param[5] = poseData[7] / row1.norm();
    }
    return param;
}

// Bundle Adjustment on both points3D and poses
void SfM::fullBA(){
    ceres::Problem problem;

    // First Quaternion and then position, 4 params + 2 
    std::vector<double*> params;
    std::vector<Pose*> poses;

    for(auto&& co: _correspondences){
        if(contains_3d_point(co.first)){
            for(auto&& features: co.second) {
                auto&& pose = get_pose(features.view_id);
                auto&& poseData = pose.pose.data();
                ceres::CostFunction *cost_function = ReprojectionError::Create(features.feature->point(0,0),
                    features.feature->point(1,0));
                auto poseOrend= std::find(poses.begin(), poses.end(), &pose);
                double* param = get_ceres_param(pose, params, poses);
                problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                                    param,
                                    get_point(co.first).point.data());

                if (poseOrend == poses.end()) {
                    params.emplace_back(param);
                    poses.emplace_back(&pose);
                    auto camera_parameterization =
                        new ceres::ProductParameterization(
                            new ceres::EigenQuaternionParameterization(),
                            new ceres::IdentityParameterization(2));
                    problem.SetParameterization(param, camera_parameterization);
                }
            }
        }
    }
    std::cerr << "Bundle Adjustment - Full..." << std::endl;

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.function_tolerance = 1e-4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (auto i = 0; i < params.size(); ++i) {
        Quaterniond quaternion;
        quaternion.x() = params[i][0];
        quaternion.y() = params[i][1];
        quaternion.z() = params[i][2];
        quaternion.w() = params[i][3];
        // Might need normalization
        auto rotMatrix = quaternion.toRotationMatrix();
        poses[i]->pose <<  rotMatrix.data()[0], rotMatrix.data()[3], rotMatrix.data()[6], params[i][4],
                rotMatrix.data()[1], rotMatrix.data()[4], rotMatrix.data()[7], params[i][5];
        delete[] params[i];
    }

    std::cout << summary.FullReport() << "\n";
}

// Bundle Adjustment on available points3D
void SfM::pointsBA(){
    ceres::Problem problem;
    for(auto&& co: _correspondences){
        if(contains_3d_point(co.first)){
            for(auto&& features: co.second){
                ceres::CostFunction *cost_function = ReprojectionErrorPoints::Create(features.feature->point(0,0),
                 features.feature->point(1,0), get_pose(features.view_id).pose.data());
                problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                                    get_point(co.first).point.data());
                // std::cout << "New feature added" << std::endl;
        
            }
        }
    }
    std::cerr << "Bundle Adjustment - Points..." << std::endl;
    // std::cout << "Done adding" << std::endl;
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

// Bundle Adjustment on available poses
void SfM::posesBA(){
    ceres::Problem problem;

    // First Quaternion and then position, 4 params + 2 
    std::vector<double*> params;
    std::vector<Pose*> poses;

    for(auto&& co: _correspondences){
        if(contains_3d_point(co.first)){
            for(auto&& features: co.second){
                auto&& pose = get_pose(features.view_id);
                auto&& poseData = pose.pose.data();
                ceres::CostFunction *cost_function = ReprojectionErrorPose::Create(features.feature->point(0,0),
                    features.feature->point(1,0), get_point(co.first).point.data());
                auto poseOrend= std::find(poses.begin(), poses.end(), &pose);
                double* param = get_ceres_param(pose, params, poses);
                problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                                    param);
                if (poseOrend == poses.end()) {
                    params.emplace_back(param);
                    poses.emplace_back(&pose);
                    auto camera_parameterization =
                        new ceres::ProductParameterization(
                            new ceres::EigenQuaternionParameterization(),
                            new ceres::IdentityParameterization(2));
                    problem.SetParameterization(param, camera_parameterization);
                }
            }
        }
    }
    std::cerr << "Bundle Adjustment - Poses..." << std::endl;

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 8;
    options.function_tolerance = 1e-4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (auto i = 0; i < params.size(); ++i) {
        Quaterniond quaternion;
        quaternion.x() = params[i][0];
        quaternion.y() = params[i][1];
        quaternion.z() = params[i][2];
        quaternion.w() = params[i][3];
        // Might need normalization
        auto rotMatrix = quaternion.toRotationMatrix();
        poses[i]->pose <<  rotMatrix.data()[0], rotMatrix.data()[3], rotMatrix.data()[6], params[i][4],
                rotMatrix.data()[1], rotMatrix.data()[4], rotMatrix.data()[7], params[i][5];
        delete[] params[i];
    }

    std::cout << summary.FullReport() << "\n";
}