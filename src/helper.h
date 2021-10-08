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



void initData(std::string dataPath, int imgcount);
int readImage(std::string dataPath, int imgIndex, boost::circular_buffer<DataFrame> *dataBuffer);
int detectKeypoints(boost::circular_buffer<DataFrame> *dataBuffer,int detType);
int descKeypoints_helper(boost::circular_buffer<DataFrame> *dataBuffer,  int descType);
int matchDescriptors_helper( boost::circular_buffer<DataFrame>  *dataBuffer,  int descType, int matcherType, int selectorType);
int matchBoundingBoxes(boost::circular_buffer<DataFrame> *dataBuffer);
int estimateTTC(boost::circular_buffer<DataFrame> *dataBuffer, bool bVis);
float estimateTTC_Lidar( BoundingBox *prevBB, BoundingBox *currBB);
float estimateTTC_Camera(boost::circular_buffer<DataFrame> *dataBuffer,BoundingBox *prevBB,BoundingBox *currBB);
#endif // HELPER_H
