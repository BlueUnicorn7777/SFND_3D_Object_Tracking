
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ... No code needed here. the clustering is already done in matchbounding boxes function.
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images

int median(int l, int r)
{
    return (r - l + 2) / 2 - 1 + l;
}

float IQR_median(vector<double> &distRatios)
{

    std::sort(distRatios.begin(), distRatios.end());
    int mid_index = median(0, distRatios.size());
    double Q1 = distRatios[median(0, mid_index)];
    double Q3 = distRatios[median(mid_index + 1, distRatios.size())];
    double lowerLimit = Q1 - 1.5 * (Q3-Q1);
    double upperLimit = Q3 + 1.5 * (Q3-Q1);
    std::vector<double> xPoints;
    for (double P : distRatios)
    {
        if((P >upperLimit)||(P<lowerLimit))continue;
        xPoints.push_back(P);
    }
    if (xPoints.size()==0) return NAN;
    mid_index = median(0, xPoints.size());
    float medDist = xPoints.size() % 2 == 0 ? (xPoints.at(mid_index - 1) + xPoints.at(mid_index) )/ 2.0 : xPoints.at(mid_index); // compute median dist. ratio to remove outlier influence
    return medDist;
}
float IQR_median(std::vector<LidarPoint> &lidarPoints)
{
    std::vector<double> xPoints;
    for (LidarPoint P : lidarPoints)
    {
        xPoints.push_back(P.x);
    }
    return IQR_median(xPoints);
}

float computeTTCCamera(double sensorFrameRate,boost::circular_buffer<DataFrame> *dataBuffer,BoundingBox *prevBB,BoundingBox *currBB){

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
       double medDistRatio=IQR_median(distRatios);
       double  dT = 1 / sensorFrameRate;
       return -dT / (1 - medDistRatio);
}

float computeTTCLidar(double sensorFrameRate, BoundingBox *prevBB,BoundingBox *currBB){
   //cout<< " prev ";
   double d0 = IQR_median(prevBB->lidarPoints);
   //cout << "curr ";
   double d1 = IQR_median(currBB->lidarPoints);
  // cout << endl;
   return d1 * (1.0 / sensorFrameRate) / (d0 - d1);

}

int matchBoundingBoxes(boost::circular_buffer<DataFrame> *dataBuffer)
{
    double t = (double)cv::getTickCount();
    for(auto bbx:(dataBuffer->end()-2)->boundingBoxes){bbx.kptMatches.clear();}
    for(auto bbx:(dataBuffer->end()-1)->boundingBoxes){bbx.kptMatches.clear();}
    int prev_to_curr[(dataBuffer->end()-2)->boundingBoxes.size()][(dataBuffer->end()-1)->boundingBoxes.size()];
    memset( prev_to_curr, 0, sizeof(prev_to_curr) );
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

// Original code of outlier removal and getting average lidar distance
//float meanLidarPoint(std::vector<LidarPoint> &lidarPoints , string windowName)
//{

//    double dataSum{0};
//    for(LidarPoint P : lidarPoints){dataSum+=P.x;}
//    double dataMean = dataSum / lidarPoints.size();
//    double dataVariance{0};
//    double dataStd{0};
//    for(LidarPoint P : lidarPoints){
//        dataVariance += pow((P.x - dataMean), 2);
//    }
//     dataVariance = dataVariance / (lidarPoints.size() - 1);
//    dataStd = sqrt(dataVariance);
//    double upperLimit = dataMean + dataStd;
//    double lowerLimit = dataMean - dataStd;
//    dataSum=0;
//    int  count = 0 ;
//    float min =1e8;

//    std::vector<double> xPoints; // Lidar 3D points which project into 2D image roi

//    for (LidarPoint P : lidarPoints)
//    {
//        if((P.x >upperLimit)||(P.x<lowerLimit))continue;
//        xPoints.push_back(P.x);
//        dataSum+=P.x;
//        count +=1;
//        if (min>P.x)min=P.x;

//    }
//   //cout<<  show3DObjects(boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), windowName) <<
//   //       "\t"<<(dataSum/count)<<"\t";

//    //return dataSum /count;
//   // return min;
//    std::sort(xPoints.begin(), xPoints.end());
//    long medIndex = floor(xPoints.size() / 2.0);
//    float medDist = xPoints.size() % 2 == 0 ? (xPoints.at(medIndex - 1) + xPoints.at(medIndex) )/ 2.0 : xPoints.at(medIndex); // compute median dist. ratio to remove outlier influence
//    return medDist;
//}
