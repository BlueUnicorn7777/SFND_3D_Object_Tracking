/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include "helper.h"



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

    helper HelperClass(argv[1],imgEndIndex);


    cout<<"MacherTypes\tSelectorTypes\tDetectorType\tdescriptorType\t"
          "FrameNo\tProcessing Time(ms)\tTTC_Lidar(s) \tTTC_Camera (s) \t Difference (s)\t Abs Difference (s) \n";

    for(int matcherType=0;matcherType<2;matcherType++){ //MAT_BF , MAT_FLANN 2
        for(int selectorType=0 ;selectorType<2;selectorType++){ //SEL_NN , SEL_KNN 2
            for(int detector = 0 ; detector<6 ;detector++){ //Dtector Loop 7
                for(int Descriptor = 0 ; Descriptor<6;Descriptor++){ //descriptor Loop 6
                    boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize);

                    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
                    {
                        int total_time = 0;
                        total_time+= HelperClass.readImage(imgStartIndex + imgIndex,&dataBuffer);
                        //std::cout << "#1 : LOAD IMAGE INTO BUFFER done time(ms) = "<< t0 << std::endl;
                        total_time+= HelperClass.detectKeypoints(&dataBuffer,detector);
                        //std::cout << "#2 : KEYPOINT DETECTION done time(ms) = "<< t1 << std::endl;
                        int t3 = HelperClass.descKeypoints_helper(&dataBuffer, Descriptor);
                        if (!(t3==-1)){ //if no error
                            total_time+=t3;
                            if (dataBuffer.size() > 1) { // wait until at least two images have been processed
                                total_time+= HelperClass.matchDescriptors_helper(&dataBuffer, Descriptor, matcherType, selectorType);
                                total_time+= HelperClass.matchBoundingBoxes_helper(&dataBuffer);
                                total_time+= HelperClass.estimateTTC(&dataBuffer,!strcmp(argv[2],"true"));
                            }

                        }
                        else{
                            total_time = -1000;}
                        cout<<MatcherTypes[matcherType]<<"\t"<<SelectorTypes[selectorType]<<"\t"<<DetectorTypes[detector]<<"\t"<<DescreptorTypes[Descriptor]<<"\t"
                           <<imgIndex<<"\t"<< total_time/1000<<"\t"<<(dataBuffer.end()-1)->ttcLidar<<"\t"<<(dataBuffer.end()-1)->ttcCamera<<
                             "\t"<<(dataBuffer.end()-1)->ttcLidar-(dataBuffer.end()-1)->ttcCamera <<"\t"<<
                             abs((dataBuffer.end()-1)->ttcLidar-(dataBuffer.end()-1)->ttcCamera) <<"\n";
                    }

                }

            }
        }
    }
    return 0;
}
