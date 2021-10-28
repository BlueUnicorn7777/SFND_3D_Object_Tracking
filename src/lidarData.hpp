
#ifndef lidarData_hpp
#define lidarData_hpp

#include <stdio.h>
#include <fstream>
#include <string>
#include "dataStructures.h"

class lidar{
    cv::Mat RT1 ;

public:
    lidar();
int median(int l, int r);
    void cropLidarPoints(std::vector<LidarPoint> &lidarPoints, float minX, float maxX, float maxY, float minZ, float maxZ, float minR);
    int loadLidarFromFile(std::string filename, boost::circular_buffer<DataFrame> *dataBuffer);
    void showLidarTopview(std::vector<LidarPoint> &lidarPoints, cv::Size worldSize, cv::Size imageSize, bool bWait=true);
    void showLidarImgOverlay(cv::Mat &img, std::vector<LidarPoint> &lidarPoints,  cv::Mat *extVisImg=nullptr);
};


#endif /* lidarData_hpp */
