//---------------------------------------------------------------------------- 
// Class for Tracked objects.
// ...
//----------------------------------------------------------------------------
#pragma once
#include<opencv2/opencv.hpp>
class ObjectTracking
{
public:
	ObjectTracking(cv::Rect detect, int whitePxlcnt, int idx);
	~ObjectTracking();
	void updateTracklet(cv::Rect detect, int whitePxlcnt);
	void predictTracklet();
	int getIdx();
	int getRadius();
	double getWhitePxlCnt();
	cv::Point getCenter();
	cv::Rect getBbox();
	cv::Point getPrevCenter();
	int getStatus();
	std::vector<cv::Point> path;

private:
	cv::Rect bbox;
	cv::Rect preBox;
	cv::Point center;
	cv::Point start;
	double whiteArea;
	int missing;        // how long a tracked object has been missing
	int alive;		    // how long a tracked object has been successfuly tracked
	int idx;            // unique tracked object identification 
	
};

