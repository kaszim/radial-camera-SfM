#include <iostream>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "SfM.h"

template<typename T>
inline std::vector<T> readRow(std::ifstream& is) {
    std::string line;
    std::getline(is, line);
    std::stringstream ss(line);
    std::vector<T> X;
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        X.emplace_back(std::stod(cell));
    }
    return X;
}

int main(int argc, char *argv[]) {
    std::string dataset(argv[1]);
    std::string thresholdString(argv[2]);
    std::string iterationsString(argv[3]);
    std::string imagesString(argv[4]);
    std::string minNumFeaturesString(argv[5]);
    double threshold = std::stod(thresholdString);
    int iterations = std::stoi(iterationsString);
    int images = std::stoi(imagesString);
    int minNumFeatures = std::stoi(minNumFeaturesString);
    std::ifstream imgcenters(dataset + "/im_centers.csv");

    SfM sfm;
    sfm.minNumFeatures_ = minNumFeatures;
    // Add our features and views to the pipeline
    for (auto img = 0; img < images; ++img) {
        std::ifstream features(dataset + "/features" + std::to_string(img) + ".csv");
        std::ifstream indices(dataset + "/indices" + std::to_string(img) + ".csv");

        auto x = readRow<double>(features);
        auto y = readRow<double>(features);
        auto idx = readRow<int>(indices);
        auto pp = readRow<double>(imgcenters);

        sfm.add_features(x, y, idx, pp[0], pp[1]);
    }

    std::ifstream points(dataset + "/matlab_points.csv");
    auto X = readRow<double>(points);
    auto Y = readRow<double>(points);
    auto Z = readRow<double>(points);
    auto idx = readRow<int>(points);
    std::vector<WorldPoint> initial_points(X.size());
    for (auto i = 0; i < X.size(); ++i) {
        initial_points[i].point(0,0) = X[i];
        initial_points[i].point(1,0) = Y[i];
        initial_points[i].point(2,0) = Z[i];
        initial_points[i].id = idx[i];
    }

    int view1;
    int view2;
    sfm.get_initial_poses(view1, view2);

    std::cout << "-------------Initial Poses------------" << std::endl;
    std::cout << "View 1 : " << view1 << std::endl;
    std::cout << "View 2 : " << view2 << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    sfm.add_view(view1);
    sfm.add_view(view2);
    sfm.initializePoints(initial_points);
    sfm.ransacPose(view1, threshold, iterations);
    sfm.ransacPose(view2, threshold, iterations);

    std::cout << "-------------Initial Points-------------" << std::endl;
    sfm.print_currentPoints();
    std::cout << "--------------------------------------" << std::endl;

    while (sfm.get_nviews() < sfm.get_nimages()) {
        // get next view to be added
        int nextView = sfm.determineNextCandidate();
        sfm.add_view(nextView);
        // estimate nextView pose
        std::cout << std::endl;
        std::cout << "Pose estimate " << nextView << std::endl;
        sfm.ransacPose(nextView, threshold, iterations);
        std::cout << "Points pre-triangulation" << std::endl;
        int pre_triangulation_points = sfm.get_npoints();
        std::cout << pre_triangulation_points << std::endl;
        // triangulate new points from views
        sfm.triangulate_new_points();
        std::cout << "Points post-triangulation" << std::endl;
        int post_triangulation_points = sfm.get_npoints();
        std::cout << post_triangulation_points << std::endl;
        if (pre_triangulation_points < post_triangulation_points){
            sfm.fullBA();
        }
        std::cerr << sfm.get_nviews() << std::endl;
    }
    std::cerr << "Printing points..." << std::endl;
    sfm.print_currentPoints();
    std::cerr << "Printing poses..." << std::endl;
    sfm.print_currentPoses();
}