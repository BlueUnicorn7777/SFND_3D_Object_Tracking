#ifndef HELPER_H
#define HELPER_H
#include <iomanip>
#include <boost/circular_buffer.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "dataStructures.h"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

class helper{

    ObjectDetection2D objDetector_;
    lidar Lidar_;
    double sensorFrameRate;
    int imgStepWidth = 1;
    std::vector<string>cameraFilesNames;
    std::vector<string>LidarFilesNames;

public:
    helper(std::string dataPath, int imgcount);
    int readImage(int imgIndex, boost::circular_buffer<DataFrame> *dataBuffer);

    int detectKeypoints(boost::circular_buffer<DataFrame> *dataBuffer,int detType);
    int descKeypoints_helper(boost::circular_buffer<DataFrame> *dataBuffer,  int descType);
    int matchDescriptors_helper( boost::circular_buffer<DataFrame>  *dataBuffer,  int descType, int matcherType, int selectorType);
    int matchBoundingBoxes_helper(boost::circular_buffer<DataFrame> *dataBuffer);
    int estimateTTC(boost::circular_buffer<DataFrame> *dataBuffer, bool bVis);
};


#endif // HELPER_H
