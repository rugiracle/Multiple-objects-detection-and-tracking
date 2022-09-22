//---------------------------------------------------------------------------- 
// Class for moving objects detection using Background modeling method.
// it takes input image and subtrack it from the background image
// the diffference image is used to extract blobs
//
// a post processing step is performed to refine the extracted blob regions
// and update the background accordingly 
//----------------------------------------------------------------------------

#pragma once
#include <opencv2/opencv.hpp>
constexpr int detectionThresh = 500;  // threshold for detectable objets size 100, 200, ...
class BackgroundLearning
{
public:
	BackgroundLearning();
	~BackgroundLearning();
	void process(const cv::Mat& img_input, cv::Mat& img_output, cv::Mat& img_bgmodel, std::vector<cv::Rect>* det_obj);

private:
	cv::Mat img_background;
	double alphaLearn;        // during background learning: bkgr_img = alphaLearn*input_img +(1-alphaLearn)* bkgr_img
	double alphaDetection;    //during background update: Adaptive and Selective update of the background model
	long learningFrames;     // the number of frame the background model is learned before starting detection
	int counter;
	double minVal;
	double maxVal;
	int threshold;           // threshold for foreground detection
	std::vector < std::vector < cv::Point > >* contours = new std::vector < std::vector < cv::Point > >; 
};

