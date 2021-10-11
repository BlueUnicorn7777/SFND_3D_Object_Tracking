
#ifndef objectDetection2D_hpp
#define objectDetection2D_hpp

#include <stdio.h>
#include <opencv2/core.hpp>
#include <boost/circular_buffer.hpp>

#include "dataStructures.h"
using namespace std;

class ObjectDetection2D
{
    float confThreshold = 0.2;
    float nmsThreshold = 0.4;
    vector<string> classes;
    string yoloModelConfiguration ;
    string yoloModelWeights ;
    string dataPath;
public:
   ObjectDetection2D(std::string dataPath);
  int detectObjects(cv::Mat& img,boost::circular_buffer<DataFrame> *dataBuffer);
};



#endif /* objectDetection2D_hpp */
