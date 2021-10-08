#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <boost/circular_buffer.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

int detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
int detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
int detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, int detectorType, bool bVis=false);
int descKeypoints(boost::circular_buffer<DataFrame> *dataBuffer,  int descType);
int matchDescriptors( boost::circular_buffer<DataFrame>  *dataBuffer,  int descType, int matcherType, int selectorType);
double show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, std::string windowName);
int detectObjects(std::string dataPath, cv::Mat& img,boost::circular_buffer<DataFrame> *dataBuffer);
void showLidarImgOverlay(cv::Mat &img, std::vector<LidarPoint> &lidarPoints, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT, cv::Mat *extVisImg);
//void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
//                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

void inityoloData(std::string dataPath, int imgcount);

#endif /* matching2D_hpp */
