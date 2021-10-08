#include "helper.h"
#include "matching2D.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
cv::Mat RT1 (3,4,cv::DataType<double>::type); // rotation matrix and translation vector
double sensorFrameRate;
int imgStepWidth = 1;
std::vector<string>cameraFilesNames;
std::vector<string>LidarFilesNames;


void initData(string dataPath, int imgcount ){
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;

    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;

    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;
    RT1 = P_rect_00 * R_rect_00 * RT ;

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
    inityoloData(dataPath, imgcount);

}



int loadLidarPoints(string filename, boost::circular_buffer<DataFrame> *dataBuffer){
    double t = (double)cv::getTickCount();
    const static float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    unsigned long num = 1000000;
    float *data = (float*)malloc(num*sizeof(float));
    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;
    // load point cloud
    FILE *stream;
    stream = fopen (filename.c_str(),"rb");
    num = fread(data,sizeof(float),num,stream)/4;
    for (int32_t i=0; i<num; i++) {
        if( *px>=minX && *px<=maxX && *pz>=minZ && *pz<=maxZ && *pz<=0.0 && abs(*py)<=maxY && *pr>=minR ){
            LidarPoint lpt;
            lpt.x = *px; lpt.y = *py; lpt.z = *pz; lpt.r = *pr;
            (dataBuffer->end()-1)->lidarPoints.push_back(lpt);
        }
        px+=4; py+=4; pz+=4; pr+=4;
    }
    fclose(stream);
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = (dataBuffer->end()-1)->lidarPoints.begin(); it1 != (dataBuffer->end()-1)->lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
       // Y = P_rect_00 * R_rect_00 * RT * X;
         Y = RT1 * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point

        for (vector<BoundingBox>::iterator it2 = (dataBuffer->end()-1)->boundingBoxes.begin(); it2 != (dataBuffer->end()-1)->boundingBoxes.end(); ++it2)
            if (it2->shrinkroi.contains(pt)) enclosingBoxes.push_back(it2);

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);

        }

    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    return 1000 * t / 1.0;
}


int readImage(string dataPath, int imgIndex,boost::circular_buffer<DataFrame> *dataBuffer){

    double t = (double)cv::getTickCount();
    cv::Mat img ,imgGray;
    img = cv::imread(cameraFilesNames[imgIndex]);
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = imgGray;
    dataBuffer->push_back(frame);
    detectObjects(dataPath,img,dataBuffer);
    loadLidarPoints(LidarFilesNames[imgIndex],dataBuffer);
    //show3DObjects((dataBuffer->end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    return 1000 * t / 1.0;
}

int detectKeypoints(boost::circular_buffer<DataFrame> *dataBuffer,int detType){

    //// STUDENT ASSIGNMENT
    //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
    //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
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

{
//        std::string windowName = detectorType_to_string(detType)+"  keypoints on  camera images";
//        cv::namedWindow(windowName, 7);
//        cv::Mat visImage = img.clone();
//        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//        cv::imshow(windowName, visImage);
//        std::cout << "Press key to continue to next image" << std::endl;
//        cv::waitKey(0); // wait for key to be pressed
//        continue;
}

    //// STUDENT ASSIGNMENT
    //// TASK MP.3 -> only keep keypoints on the preceding vehicle

    // only keep keypoints on the preceding vehicle

    bool bFocusOnVehicle = false;
    cv::Rect vehicleRect(535, 180, 180, 150);
    if (bFocusOnVehicle)
    {
        std::vector<cv::KeyPoint> vehiclekeypoints;
        for (auto kp : keypoints) {
            if (vehicleRect.contains(kp.pt)) vehiclekeypoints.push_back(kp);
        }
        keypoints = vehiclekeypoints;
{
//            std::string windowName = detectorType_to_string(detType)+" keypoint detection results";
//            cv::namedWindow(windowName);
//            cv::Mat visImage = img.clone();
//            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//            cv::imshow(windowName, visImage);
//            cv::waitKey(0);
//            continue;
}
    }

    //// EOF STUDENT ASSIGNMENT

    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    if (bLimitKpts)
    {
        int maxKeypoints = 50;

        if (detType==0)//SHITOMASI
        { // there is no response info, so keep the first 50 as they are sorted in descending quality order
            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
        }
        cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
//      std::cout << " NOTE: Keypoints have been limited!" << std::endl;

    }

    // push keypoints and descriptor for current frame to end of data buffer

    (dataBuffer->end()-1)->keypoints = keypoints;

    return (((double)cv::getTickCount() - t)*1000000.0) / cv::getTickFrequency();

    //// EOF STUDENT ASSIGNMENT
}

int descKeypoints_helper(boost::circular_buffer<DataFrame> *dataBuffer,  int descType){
return descKeypoints(dataBuffer,   descType);
}
int matchDescriptors_helper( boost::circular_buffer<DataFrame>  *dataBuffer,  int descType, int matcherType, int selectorType){
    return matchDescriptors( dataBuffer,  descType,  matcherType,  selectorType);
}
int  matchBoundingBoxes(boost::circular_buffer<DataFrame> *dataBuffer)
{
    double t = (double)cv::getTickCount();
    for(auto bbx:(dataBuffer->end()-2)->boundingBoxes){bbx.kptMatches.clear();}
    for(auto bbx:(dataBuffer->end()-1)->boundingBoxes){bbx.kptMatches.clear();}
    int prev_to_curr[(dataBuffer->end()-2)->boundingBoxes.size()][(dataBuffer->end()-1)->boundingBoxes.size()];
    memset( prev_to_curr, 0, sizeof(prev_to_curr) );
//    for(unsigned long k = 0 ; k< (dataBuffer->end()-2)->boundingBoxes.size();k++)
//        for(unsigned long l = 0 ; l< (dataBuffer->end()-1)->boundingBoxes.size(); l++)
//            prev_to_curr[k][l]=0;

    for (cv::DMatch match : (dataBuffer->end()-1)->kptMatches) {
        unsigned long bb_prev =0 ,bb_curr =0;
        vector<vector<BoundingBox>::iterator> enclosingBoxes1,enclosingBoxes2;
        vector<int> v1,v2;
        for (vector<BoundingBox>::iterator it2 = (dataBuffer->end()-2)->boundingBoxes.begin(); it2 != (dataBuffer->end()-2)->boundingBoxes.end(); ++it2)
        {
            if (it2->shrinkroi.contains((dataBuffer->end()-2)->keypoints[match.queryIdx].pt)) {
               enclosingBoxes1.push_back(it2); v1.push_back(bb_prev);
            }
            bb_prev+=1;
        }
        for (vector<BoundingBox>::iterator it2 = (dataBuffer->end()-1)->boundingBoxes.begin(); it2 != (dataBuffer->end()-1)->boundingBoxes.end(); ++it2)
        {
            if (it2->shrinkroi.contains((dataBuffer->end()-1)->keypoints[match.trainIdx].pt)) {
               enclosingBoxes2.push_back(it2);v2.push_back(bb_curr);
            }
            bb_curr+=1;
        }
        if (enclosingBoxes1.size() == 1 && enclosingBoxes2.size()==1)
        {
            enclosingBoxes1[0]->kptMatches.push_back(match);
            enclosingBoxes2[0]->kptMatches.push_back(match);
            prev_to_curr[v1[0]][v2[0]]+=1;
        }

    }{
    //Debugging code
    //for(int k = 0 ; k< prevFrame.boundingBoxes.size();k++)cout<<prevFrame.boundingBoxes[k].kptMatches.size()<<" ";
    //cout<<endl;
    //for(int k = 0 ; k< currFrame.boundingBoxes.size();k++)cout<<currFrame.boundingBoxes[k].kptMatches.size()<<" ";
    //cout<<endl;

    //    for(int k = 0 ; k< prevFrame.boundingBoxes.size();k++){
    //        for(int l = 0 ; l< currFrame.boundingBoxes.size(); l++){
    //            cout<< prev_to_curr[k][l]<<"-";}
    //        cout<<"xxxx"<<endl;
    //    }
}
    for(unsigned long k = 0 ; k< (dataBuffer->end()-2)->boundingBoxes.size();k++){
        int matchindex = 0 , matchpoints = 0,m=0;
        for(unsigned long l = 0 ; l< (dataBuffer->end()-1)->boundingBoxes.size(); l++){
            if(matchpoints < prev_to_curr[k][l]){
                matchpoints=prev_to_curr[k][l];
                matchindex=(dataBuffer->end()-1)->boundingBoxes[l].boxID;
                m=l;
            }
            // cout<<matchpoints<<"-"<<matchindex<<endl;
        }

        (dataBuffer->end()-1)->bbMatches.insert({(dataBuffer->end()-2)->boundingBoxes[k].boxID,matchindex});
        //   cout<<prevFrame.boundingBoxes[k].boxID<<"-"<<matchindex<<"-"<<prevFrame.boundingBoxes[k].lidarPoints.size()<<
        //     "-"<<currFrame.boundingBoxes[m].lidarPoints.size()<<endl;
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    return 1000 * t / 1.0;
}

int estimateTTC(boost::circular_buffer<DataFrame> *dataBuffer , bool bVis){
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
      if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>100 ) // only compute TTC if we have Lidar points
      {
         (dataBuffer->end()-1)->ttcLidar=estimateTTC_Lidar( prevBB,currBB);
         (dataBuffer->end()-1)->ttcCamera = estimateTTC_Camera(dataBuffer,prevBB,currBB);

      if (bVis)
      {
          cv::Mat visImg = (dataBuffer->end() - 1)->cameraImg.clone();
          showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
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

float meanLidarPoint(std::vector<LidarPoint> &lidarPoints , string windowName)
{

    double dataSum{0};
    for(LidarPoint P : lidarPoints){dataSum+=P.x;}
    double dataMean = dataSum / lidarPoints.size();
    double dataVariance{0};
    double dataStd{0};
    for(LidarPoint P : lidarPoints){
        dataVariance += pow((P.x - dataMean), 2);
    }
     dataVariance = dataVariance / (lidarPoints.size() - 1);
    dataStd = sqrt(dataVariance);
    double upperLimit = dataMean + dataStd;
    double lowerLimit = dataMean - dataStd;
    dataSum=0;
    int  count = 0 ;
    float min =1e8;
    std::vector<BoundingBox> boundingBoxes;
    boundingBoxes.push_back(*new BoundingBox);

    for (LidarPoint P : lidarPoints)
    {
        if((P.x >upperLimit)||(P.x<lowerLimit))continue;
        boundingBoxes[0].lidarPoints.push_back(P);
        dataSum+=P.x;
        count +=1;
        if (min>P.x)min=P.x;

    }
//   cout<<  show3DObjects(boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), windowName) <<
//          "\t"<<(dataSum/count)<<"\t";

   return dataSum /count;
   // return min;

}

float estimateTTC_Lidar( BoundingBox *prevBB,BoundingBox *currBB){
   //cout<< " prev ";
   double d0 = meanLidarPoint(prevBB->lidarPoints,"PREV");
   //cout << "curr ";
   double d1 = meanLidarPoint(currBB->lidarPoints,"CURR");
  // cout << endl;
   return d1 * (1.0 / sensorFrameRate) / (d0 - d1);

}
float estimateTTC_Camera(boost::circular_buffer<DataFrame> *dataBuffer,BoundingBox *prevBB,BoundingBox *currBB){

      vector<double> distRatios;
       for (auto it1 = currBB->kptMatches.begin(); it1 != currBB->kptMatches.end() - 1; ++it1)
       {
           // get current keypoint and its matched partner in the prev. frame

           cv::KeyPoint kpOuterCurr = (dataBuffer->end()-1)->keypoints.at(it1->trainIdx);
           cv::KeyPoint kpOuterPrev = (dataBuffer->end()-2)->keypoints.at(it1->queryIdx);

           for (auto it2 = currBB->kptMatches.begin() + 1; it2 != currBB->kptMatches.end(); ++it2)
           { // inner kpt.-loop

               double minDist = 100.0; // min. required distance

               // get next keypoint and its matched partner in the prev. frame
               cv::KeyPoint kpInnerCurr = (dataBuffer->end()-1)->keypoints.at(it2->trainIdx);
               cv::KeyPoint kpInnerPrev = (dataBuffer->end()-2)->keypoints.at(it2->queryIdx);

               // compute distances and distance ratios
               double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
               double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

               if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
               { // avoid division by zero

                   double distRatio = distCurr / distPrev;
                   if (distRatio>1.001)
                   distRatios.push_back(distRatio);
               }
           } // eof inner loop over all matched kpts
       }     // eof outer loop over all matched kpts

       // only continue if list of distance ratios is not empty
       if (distRatios.size() == 0)
       {
           return NAN;

       }

       std::sort(distRatios.begin(), distRatios.end());
       long medIndex = floor(distRatios.size() / 2.0);
       double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
       double  dT = 1 / sensorFrameRate;
       return -dT / (1 - medDistRatio);

}
