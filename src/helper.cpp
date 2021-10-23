#include "helper.h"
#include "matching2D.hpp"
#include "lidarData.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

helper::helper(string dataPath, int imgcount ):objDetector_(ObjectDetection2D(dataPath)){

    sensorFrameRate = 10.0 / imgStepWidth;
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)
    string imgPrefix = "images/KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    string lidarPrefix = "images/KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    for (int imgIndex = 0 ; imgIndex<=imgcount;imgIndex+=imgStepWidth){
        ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgIndex;
        cameraFilesNames.push_back(dataPath + imgPrefix + imgNumber.str() + imgFileType);
        LidarFilesNames.push_back(dataPath + lidarPrefix + imgNumber.str() + lidarFileType);
    }

}

int helper::readImage(int imgIndex,boost::circular_buffer<DataFrame> *dataBuffer){
    double t = (double)cv::getTickCount();
    cv::Mat img ,imgGray;
    img = cv::imread(cameraFilesNames[imgIndex]);
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = imgGray;
    dataBuffer->push_back(frame);
    objDetector_.detectObjects(img,dataBuffer);
    Lidar_.loadLidarFromFile(LidarFilesNames[imgIndex],dataBuffer);
    //show3DObjects((dataBuffer->end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    return 1000 * t / 1.0;
}

int helper::detectKeypoints(boost::circular_buffer<DataFrame> *dataBuffer,int detType){

    std::vector<cv::KeyPoint> keypoints;
    double t = (double)cv::getTickCount();
    switch(detType){
    case 0: // SHITOMASI:
        detKeypointsShiTomasi(keypoints, (dataBuffer->end() - 1)->cameraImg, false);break;
    case 1: //HARRIS:
        detKeypointsHarris(keypoints, (dataBuffer->end() - 1)->cameraImg, false);break;
    case 2: case 3: case 4: case 5: case 6: //FAST: case BRISK: case ORB: case SIFT: case AKAZE:
        detKeypointsModern(keypoints, (dataBuffer->end() - 1)->cameraImg, detType, false); break;
    default: ;
    }

    (dataBuffer->end()-1)->keypoints = keypoints;
    return (((double)cv::getTickCount() - t)*1000000.0) / cv::getTickFrequency();

}

int helper::descKeypoints_helper(boost::circular_buffer<DataFrame> *dataBuffer,  int descType){
    return descKeypoints(dataBuffer,   descType);
}
int helper::matchDescriptors_helper( boost::circular_buffer<DataFrame>  *dataBuffer,  int descType, int matcherType, int selectorType){
    return matchDescriptors( dataBuffer,  descType,  matcherType,  selectorType);
}

int helper::matchBoundingBoxes_helper(boost::circular_buffer<DataFrame> *dataBuffer){
    return matchBoundingBoxes(dataBuffer);
}

int helper::estimateTTC(boost::circular_buffer<DataFrame> *dataBuffer , bool bVis){
    double t = (double)cv::getTickCount();
    for (auto it1 = (dataBuffer->end() - 1)->bbMatches.begin(); it1 != (dataBuffer->end() - 1)->bbMatches.end(); ++it1)
    {
        // find bounding boxes associates with current match
        BoundingBox *prevBB, *currBB;
        for (auto it2 = (dataBuffer->end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer->end() - 1)->boundingBoxes.end(); ++it2)
        {
            if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
            {
                currBB = &(*it2);
            }
        }

        for (auto it2 = (dataBuffer->end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer->end() - 2)->boundingBoxes.end(); ++it2)
        {
            if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
            {
                prevBB = &(*it2);
            }
        }

        // compute TTC for current match
        if( currBB->lidarPoints.size()>100 && prevBB->lidarPoints.size()>100 ) // only compute TTC if we have Lidar points
        {
            (dataBuffer->end()-1)->ttcLidar=computeTTCLidar( sensorFrameRate,prevBB,currBB);
            (dataBuffer->end()-1)->ttcCamera = computeTTCCamera(sensorFrameRate, dataBuffer,prevBB,currBB);

            if (bVis)
            {
                cv::Mat visImg = (dataBuffer->end() - 1)->cameraImg.clone();
                Lidar_.showLidarImgOverlay(visImg, currBB->lidarPoints, &visImg);
                cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                char str[200];
                sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", (dataBuffer->end() - 1)->ttcLidar, (dataBuffer->end() - 1)->ttcCamera);
                putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                string windowName = "Final Results : TTC";
                cv::namedWindow(windowName, cv::WINDOW_NORMAL);

                cv::moveWindow(windowName, 20, 20);
                cv::resizeWindow(windowName, 800, 600);
                cv::imshow(windowName, visImg);
                cv::waitKey(100);

            }

        }

    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    return 1000 * t / 1.0;
}




