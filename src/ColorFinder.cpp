#include "ColorFinder.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

using namespace cv;

const int SATURATION_MAX = 255;
const int BRIGHTNESS_MAX = 255;

ColorFinder::ColorFinder(){
}

ColorFinder::~ColorFinder(){
}

void ColorFinder::runRed(){
    int hueL = 12;
    int hueH = 17;
    runFilter(hueL, hueH);
}

void ColorFinder::runBlue(){
    int hueL = 107;
    int hueH = 117;
    runFilter(hueL, hueH);
}

void ColorFinder::runFilter(int& hueL, int& hueU){

    VideoCapture cap(0);    

    if(!cap.isOpened()){  // check if we succeeded
        std::cout << "Couldn't open camera device" << std::endl;
        return;
    }

    namedWindow("cam",1);

    // Filter parameters
    int filterIterations = 3;
    int filterSize = 3;
    int b = 68; //BRIGHTNESS_MAX/2;
    int s = 223; //SATURATION_MAX/2;

    createTrackbar("HueL", "cam", &hueL, 180);
    createTrackbar("HueU", "cam", &hueU, 180);
    createTrackbar("Filter Iterations", "cam", &filterIterations, 25);
    createTrackbar("Filter Size", "cam", &filterSize, 20);
    createTrackbar("Brightness Cutoff", "cam", &b, BRIGHTNESS_MAX);
    createTrackbar("Saturation Cutoff", "cam", &s, SATURATION_MAX);

    while (true){

        // Ensures that the filter size and iterations never go to zero
        if (getTrackbarPos("Filter Iterations", "cam") == 0){
            filterIterations = 1;
        }
        if (getTrackbarPos("Filter Size", "cam") == 0){
            filterSize = 1;
        }

        // Obtain the frame
        Mat frame;
        cap >> frame; // get a new frame from camera

        // Change to HSB for computation
        cvtColor(frame, frame, CV_BGR2HSV);

        for (int i = 0; i < filterIterations; i++){

            // Blur the input
            blur(frame, frame, Size(filterSize, filterSize));

            // Filter out the colors
            for (int x = 0; x < frame.cols; x++){
                for (int y = 0; y < frame.rows; y++){

                    // Retrieve the color vector at each pixel position
                    Vec3b &colors = frame.at<Vec3b>(y, x);

                    // Filter out unwanted colors
                    if (colors.val[0] > hueU ||
                        colors.val[0] < hueL ||
                        colors.val[1] < s    ||
                        colors.val[2] < b)
                    {
                        colors.val[2] = 0;
                    }

                    // Augment the desired colors
                    else{

                        /*// Assign to whichever is closer? Upper, Lower, or Middle
                        int d = hueU - colors.val[0];

                        if (d > colors.val[0] - hueL){
                            d = colors.val[0] - hueL;
                            colors.val[0] = hueL;
                        }
                        else if (d >= abs(colors.val[0] - (hueU + hueL)/2)){
                            colors.val[0] = (hueU + hueL)/2;
                        }
                        else{
                            colors.val[0] = hueU;
                        }*/

                        // Assign to the average value
                        //colors.val[0] = (hueU + hueL)/2;

                        // Set the saturation and brightness to be the same
                        colors.val[1] = SATURATION_MAX;
                        colors.val[2] = BRIGHTNESS_MAX;
                    }
                }
            }

        }

        // Convert back to BGR
        cvtColor(frame, frame, CV_HSV2BGR);

        // Find the edges associated with the objects in the frame
        //Canny(frame, frame, 0, 30, 3);    

        imshow("cam", frame);
        if(waitKey(30) >= 0) break;

    }
}
