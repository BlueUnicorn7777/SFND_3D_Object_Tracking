/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "helper.h"
#include "dataStructures.h"

using namespace std;
const string DetectorTypes[]=  {"SHITOMASI ","HARRIS   ", "FAST      ", "BRISK    ", "ORB       ", "AKAZE    ", "SIFT     "};
const string DescreptorTypes[]={"BRISK     ","BRIEF    ", "ORB       ", "FREAK    ", "AKAZE    ", "SIFT     "};
const string MatcherTypes[]={"MAT_BF     ","MAT_FLANN    "};
const string SelectorTypes[]={"SEL_NN     ","SEL_KNN    "};

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    if(argc<3){
        std::cout<<"Please specify the following arguments to execute \n"<<
                   "./2D_feature_tracking path bVis\n"
                   " path = path to image folder\n"
                   "bVis = true , false , avg \n" ;
        return 0;
    }

    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 77;   // last file index to load
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time


    initData(argv[1],imgEndIndex);

    cout<<"MacherTypes\tSelectorTypes\tDetectorType\tdescriptorType\t"
          "FrameNo\tProcessing Time ms\tTTC_Lidar s \tTTC_Camers s\t Differenc s \n";

    for(int matcherType=0;matcherType<2;matcherType++){ //MAT_BF , MAT_FLANN 2
        for(int selectorType=0 ;selectorType<2;selectorType++){ //SEL_NN , SEL_KNN 2
            for(int detector = 0 ; detector<6 ;detector++){ //Dtector Loop 7
                for(int Descriptor = 0 ; Descriptor<6;Descriptor++){ //descriptor Loop 6
                   boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize);
                    float TD=0,KM=0;
              //      for (size_t imgIndex = 50; imgIndex <= 60; imgIndex++)
                   for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
                    {
                        int total_time = 0;
                        int t0 = readImage(argv[1],imgStartIndex + imgIndex,&dataBuffer);
                        //std::cout << "#1 : LOAD IMAGE INTO BUFFER done time(ms) = "<< t0 << std::endl;
                        int t1 = detectKeypoints(&dataBuffer,detector);
                        //std::cout << "#2 : KEYPOINT DETECTION done time(ms) = "<< t1 << std::endl;
                        int t3 = descKeypoints_helper(&dataBuffer, Descriptor);

                        if (!(t3==-1)){
                        int t4 = 0 ,t5 = 0 ,t6 = 0;
                        if (dataBuffer.size() > 1) { // wait until at least two images have been processed
                        t4 = matchDescriptors_helper(&dataBuffer, Descriptor, matcherType, selectorType);
                        t5 = matchBoundingBoxes(&dataBuffer);
                        t6 = estimateTTC(&dataBuffer,!strcmp(argv[2],"true"));
                        }
                        total_time=t0+t1+t3+t4+t5+t6;
                        }
                        else{
                        total_time = -1000;}

                        //if (strcmp(argv[2],"avg"))
                        cout<<MatcherTypes[matcherType]<<"\t"<<SelectorTypes[selectorType]<<"\t"<<DetectorTypes[detector]<<"\t"<<DescreptorTypes[Descriptor]<<"\t"
                           <<imgIndex<<"\t"<< total_time/1000<<"\t"<<(dataBuffer.end()-1)->ttcLidar<<"\t"<<(dataBuffer.end()-1)->ttcCamera<<
                             "\t"<<(dataBuffer.end()-1)->ttcLidar-(dataBuffer.end()-1)->ttcCamera <<"\t"<<
                             abs((dataBuffer.end()-1)->ttcLidar-(dataBuffer.end()-1)->ttcCamera) <<"\n";

////                            TD+=t1+t3;
////                            KM+=(dataBuffer.end()-1)->kptMatches.size();

//                        if ((dataBuffer.size() > 1)&(!strcmp(argv[2],"true"))){


//                        string windowname =MatcherTypes[matcherType]+SelectorTypes[selectorType]+DetectorTypes[detector]+
//                                DescreptorTypes[Descriptor];
//                        cv::Mat visImg = ((dataBuffer.end() - 1)->cameraImg).clone();
//                                    cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
//                                                    (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
//                                                    (dataBuffer.end() - 1)->kptMatches, visImg,
//                                                    cv::Scalar::all(-1), cv::Scalar::all(-1),
//                                                    std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

//                       // cv::drawKeypoints((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

//                        cv::namedWindow(windowname, cv::WINDOW_NORMAL);
//                        cv::moveWindow(windowname, 20, 20);
//                        cv::resizeWindow(windowname, 800, 600);
//                        cv::imshow(windowname, visImg);
//                        //std::cout << "Press key to continue to next image" << std::endl;
//                        cv::waitKey(100); // wait for key to be pressed

//                    }
                   }
//                    if (!strcmp(argv[2],"avg"))
//                    cout<<MatcherTypes[matcherType]<<SelectorTypes[selectorType]<<DetectorTypes[detector]<<DescreptorTypes[Descriptor]<<"\t"
//                      <<TD/(imgEndIndex+1)<<"\t"<<KM/imgEndIndex<<"\n";
//                    //cv::waitKey(100); // wait for key to be pressed
//                    //cv::destroyAllWindows();
                }

            }
        }
    }

    return 0;
}
