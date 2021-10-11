
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "objectDetection2D.hpp"


using namespace std;

// detects objects in an image using the YOLO library and a set of pre-trained objects from the COCO database;
// a set of 80 classes is listed in "coco.names" and pre-trained weights are stored in "yolov3.weights"

ObjectDetection2D::ObjectDetection2D(std::string dataPath){

    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    this->dataPath=dataPath;
}


int ObjectDetection2D::detectObjects(cv::Mat& img, boost::circular_buffer<DataFrame> *dataBuffer){
    double t = (double)cv::getTickCount();
    // load neural network
    string yoloModelConfiguration = dataPath + "dat/yolo/yolov3.cfg";
    string yoloModelWeights = dataPath + "dat/yolo/yolov3.weights";
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
    // generate 4D blob from input image
    cv::Mat blob;
    vector<cv::Mat> netOutput;
    double scalefactor = 1/255.0;
    cv::Size size = cv::Size(416, 416);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);

    // Get names of output layers
    vector<cv::String> names;
    float shrinkFactor = 0.10;
    vector<int> outLayers = net.getUnconnectedOutLayers(); // get  indices of  output layers, i.e.  layers with unconnected outputs
    vector<cv::String> layersNames = net.getLayerNames(); // get  names of all layers in the network

    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) // Get the names of the output layers in names
        names[i] = layersNames[outLayers[i] - 1];

    // invoke forward propagation through network
    net.setInput(blob);
    net.forward(netOutput, names);

    // Scan through all bounding boxes and keep only the ones with high confidence
    vector<int> classIds; vector<float> confidences; vector<cv::Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
        float* data = (float*)netOutput[i].data;
        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
        {
            cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
            cv::Point classId;
            double confidence;

            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold)
            {
                cv::Rect box; int cx, cy;
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
                box.width = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width/2; // left
                box.y = cy - box.height/2; // top

                boxes.push_back(box);
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);
            }
        }
    }

    // perform non-maxima suppression
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for(auto it=indices.begin(); it!=indices.end(); ++it) {

        BoundingBox bBox;
        cv::Rect smallerBox;
        bBox.roi = boxes[*it];
        smallerBox.x = bBox.roi.x + shrinkFactor * bBox.roi.width / 2.0;
        smallerBox.y = bBox.roi.y + shrinkFactor * bBox.roi.height / 2.0;
        smallerBox.width = bBox.roi.width * (1 - shrinkFactor);
        smallerBox.height = bBox.roi.height * (1 - shrinkFactor);
        bBox.shrinkroi=smallerBox;
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)(dataBuffer->end() - 1)->boundingBoxes.size(); // zero-based unique identifier for this bounding box
        (dataBuffer->end() - 1)->boundingBoxes.push_back(bBox);
    }

    // show results
    if(false) {
        cv::Mat visImg = img.clone();
        for(auto it=(dataBuffer->end() - 1)->boundingBoxes.begin(); it!=(dataBuffer->end() - 1)->boundingBoxes.end(); ++it) {

            // Draw rectangle displaying the bounding box
            int top, left, width, height;
            top = (*it).roi.y;
            left = (*it).roi.x;
            width = (*it).roi.width;
            height = (*it).roi.height;
            cv::rectangle(visImg, cv::Point(left, top), cv::Point(left+width, top+height),cv::Scalar(0, 255, 0), 2);
            top = (*it).shrinkroi.y;
            left = (*it).shrinkroi.x;
            width = (*it).shrinkroi.width;
            height = (*it).shrinkroi.height;
            cv::rectangle(visImg, cv::Point(left, top), cv::Point(left+width, top+height),cv::Scalar(0, 255, 0), 2);

            string label = cv::format("%.2f", (*it).confidence);
            label = classes[((*it).classID)] + ":" + label;

            // Display label at the top of the bounding box
            int baseLine;
            cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
            top = max(top, labelSize.height);
            rectangle(visImg, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);
        }

        string windowName = "Object classification";
        cv::namedWindow( windowName, 1 );
        cv::imshow( windowName, visImg );
        cv::waitKey(100); // wait for key to be pressed
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    return 1000 * t / 1.0;
}
