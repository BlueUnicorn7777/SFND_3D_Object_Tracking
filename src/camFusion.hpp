
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"


    void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);
    void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);
    int matchBoundingBoxes(boost::circular_buffer<DataFrame> *dataBuffer);
    void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true);
    float computeTTCCamera(double sensorFrameRate,boost::circular_buffer<DataFrame> *dataBuffer,BoundingBox *prevBB,BoundingBox *currBB);
    float computeTTCLidar( double sensorFrameRate,BoundingBox *prevBB,BoundingBox *currBB);
    float meanLidarPoint(std::vector<LidarPoint> &lidarPoints , std::string windowName);

#endif /* camFusion_hpp */
