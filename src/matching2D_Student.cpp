#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
int matchDescriptors( boost::circular_buffer<DataFrame>  *dataBuffer, int descType, int matcherType, int selectorType)

{
    //// configure matcher
    //// BINARY descriptors ->  BRISK, BRIEF, ORB, FREAK, and AKAZE. */
    //// HOG descriptors i->  SIFT

    bool crossCheck = false;
    bool convert = true;
    if (selectorType==0) crossCheck = true;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    try{
        if (matcherType == 0) {//MAT_BF    Brute Force
            if(descType==5)matcher = cv::BFMatcher::create(cv::NORM_L2, crossCheck); //SIFT
            else matcher = cv::BFMatcher::create(cv::NORM_HAMMING, crossCheck);
        }
        else //MAT_FLANN
        {
            convert = false;
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        }

        //  perform matching task
        double t = (double)cv::getTickCount();

        if (selectorType == 0) //SEL_NN
        { // nearest neighbor (best match)
            if(convert ||dataBuffer->at(0).descriptors.type()==CV_32F )
            matcher->match(dataBuffer->at(0).descriptors, dataBuffer->at(1).descriptors, (dataBuffer->end()-1)->kptMatches);
            else{
                cv::Mat  des1 ,des2 ;
                dataBuffer->at(0).descriptors.convertTo(des1, CV_32F);
                dataBuffer->at(1).descriptors.convertTo(des2, CV_32F);
                matcher->match(des1, des2, (dataBuffer->end()-1)->kptMatches);
              }
        }
        else //SEL_KNN
        { // k nearest neighbors k=2
            std::vector<std::vector<cv::DMatch>> knn_matches;

            if(convert ||dataBuffer->at(0).descriptors.type()==CV_32F )
            matcher->match(dataBuffer->at(0).descriptors, dataBuffer->at(1).descriptors, (dataBuffer->end()-1)->kptMatches);
            else{
                cv::Mat  des1 ,des2 ;
                dataBuffer->at(0).descriptors.convertTo(des1, CV_32F);
                dataBuffer->at(1).descriptors.convertTo(des2, CV_32F);
                matcher->match(des1, des2, (dataBuffer->end()-1)->kptMatches);
              }
            // Filter matches using descriptor distance ratio test
            double minDescDistRatio = 0.8;
            for (auto it : knn_matches) {
                if(it.size()==2)
                if (it[0].distance < minDescDistRatio * it[1].distance)  {
                    (dataBuffer->end()-1)->kptMatches.push_back(it[0]);
                }
            }
        }
        return (((double)cv::getTickCount() - t)*1000000.0) / cv::getTickFrequency();
    }
    catch (cv::Exception) {
        return -1;
    }
}


// Use one of several types of state-of-art descriptors to uniquely identify keypoints
//int descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors,  int descType)
int descKeypoints(boost::circular_buffer<DataFrame> *dataBuffer,  int descType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
    double t = (double)cv::getTickCount();

    try {
    switch (descType) {
    case 0: // BRISK_:
        descriptor = cv::BRISK::create(threshold, octaves, patternScale);break;
    case 1: //BRIEF_:
        descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();break;
    case 2: //ORB_:
        descriptor = cv::ORB::create();break;
    case 3: //FREAK_:
        descriptor = cv::xfeatures2d::FREAK::create();break;
    case 4: //AKAZE_:
        descriptor = cv::AKAZE::create();break;
    case 5: // SIFT_:
       //descriptor = cv::SIFT::create();break;
       descriptor = cv::xfeatures2d::SIFT::create();break;
    default:;
        return -1;
    }

    descriptor->compute((dataBuffer->end()-1)->cameraImg, (dataBuffer->end()-1)->keypoints, (dataBuffer->end()-1)->descriptors);
    } catch (cv::Exception) {
      return -1;
    }
     return (((double)cv::getTickCount() - t)*1000000.0) / cv::getTickFrequency();
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
int detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
     return (((double)cv::getTickCount() - t)*1000000.0) / cv::getTickFrequency();
}

int detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,  int detType, bool bVis)
{
    cv::Ptr<cv::Feature2D> detector ;

    switch (detType) {
    case 2: //FAST:
        detector = cv::FastFeatureDetector::create();break;
    case 3:// BRISK:
        detector = cv::BRISK::create();break;
    case 4: //ORB:
        detector = cv::ORB::create();break;
    case 5: //AKAZE:
        detector = cv::AKAZE::create();break;
    case 6: //SIFT:
       //detector = cv::SIFT::create();break;
       detector = cv::xfeatures2d::SIFT::create();break;
    default: return -1 ;
    }

     double t = (double)cv::getTickCount();
     detector->detect(img, keypoints);
     return (((double)cv::getTickCount() - t)*1000000.0) / cv::getTickFrequency();
}

int  detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters

    int blockSize = 2;     // A blockSize neighborhood is considered
    int apertureSize = 3;  // Aperture parameter for the Sobel operator Aperture size must be odd
    int minResponse = 100; // Minimum value for a corner in the scaled (0...255) response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Non-maximum suppression (NMS) settings
    double maxOverlap = 0.0;  // Maximum overlap between two features in %

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Apply non-maximum suppression (NMS)
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);

            // Apply the minimum threshold for Harris cornerness response
            if (response < minResponse) continue;

            // Otherwise create a tentative new keypoint
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f(i, j);
            newKeyPoint.size = 2 * apertureSize;
            newKeyPoint.response = response;

            // Perform non-maximum suppression (NMS) in local neighbourhood around the new keypoint
            bool bOverlap = false;
            // Loop over all existing keypoints
            for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                // Test if overlap exceeds the maximum percentage allowable
                if (kptOverlap > maxOverlap) {
                    bOverlap = true;
                    // If overlapping, test if new response is the local maximum
                    if (newKeyPoint.response > (*it).response) {
                        *it = newKeyPoint;  // Replace the old keypoint
                        break;  // Exit for loop
                    }
                }
            }

            // If above response threshold and not overlapping any other keypoint
            if (!bOverlap) {
                keypoints.push_back(newKeyPoint);  // Add to keypoints list
            }
        }
    }

     return (((double)cv::getTickCount() - t)*1000000.0) / cv::getTickFrequency();
}

double show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, std::string windowName)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));
    float ret;
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
      ret = xwmin;
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
    //string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName,800,600);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(100); // wait for key to be pressed

 return ret;
}




