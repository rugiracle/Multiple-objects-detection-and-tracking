#include <iostream>
#include "BackgroundLearning.h"
#include"ObjectTracking.h"
#include"Helpers.h"


int main()
{
	const char* vid_location = "D:/2021/CODE_old/online_vid/PVTEA301a.mp4";
	cv::VideoCapture cap = cv::VideoCapture(vid_location);
	std::cout << "Object detection using background modeling     ... " << std::endl;
	
	cv::Mat frgrd, frgrd_, img_mask, img_bkgmodel;
	if (!cap.isOpened())
	{
		std::cerr << "Cannot initialize video!" << std::endl;
		return -1;
	}
	BackgroundLearning frgrd_detection;
	std::vector<cv::Rect> detect_blobs;
	std::vector<ObjectTracking> tracked;
	cv::Size resol(640, 480);
	//for displaying detection and tracking results
	cv::Mat Display(cv::Size(resol.width * 3, resol.height), CV_8UC3, 255);
	cv::Mat mask_show(cv::Size(resol.width, resol.height), CV_8UC3);
	cv::Mat bgDisplay(cv::Size(resol.width, resol.height), CV_8UC3);
	int key = 0;
	//to write the output video
	int frame_width = resol.width * 3;  //resolution of the output video
	int frame_height = resol.height;
	cv::VideoWriter video("out_demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frame_width, frame_height));
	int frame_count = 0;
	
	for (;;)
	{
		cap.read(frgrd_);
		if (frgrd_.empty())
		{
			std::cout << "ENd of the viedo file" << std::endl;
			return -1;
		}
		if (frame_count == 290)
			break;
		cv::resize(frgrd_, frgrd, resol);
		frgrd_detection.process(frgrd, img_mask, img_bkgmodel, &detect_blobs);		
		if (!img_mask.empty() && detect_blobs.size())		
		{
			for (int i = 0; i < detect_blobs.size(); i++)
			{
				draw_rectangle(frgrd, detect_blobs.at(i), cv::Scalar(0, 255, 200), 3);
			}
			if (tracked.size())
			{
				ObjectTrackingHandler(img_mask, frgrd, detect_blobs, tracked);
				drawTracklet(frgrd, tracked);
			}
			else
			{
				initTracking(detect_blobs, tracked, img_mask);
			}
			cv::cvtColor(img_mask, mask_show, CV_GRAY2BGR);
			cv::cvtColor(img_bkgmodel, bgDisplay, CV_GRAY2BGR);
			frgrd.copyTo(Display(cv::Rect(0, 0, resol.width, resol.height)));
			mask_show.copyTo(Display(cv::Rect(resol.width, 0, resol.width, resol.height)));
			bgDisplay.copyTo(Display(cv::Rect(resol.width * 2, 0, resol.width, resol.height)));
			video.write(Display);

			cv::imshow("Surveillance @ objects detection & tracking     ... ", Display);
			key = cv::waitKey(30);
		}
		detect_blobs.clear(); 
		frame_count++;
	}
	cap.release();
	video.release();
	cvDestroyAllWindows();
	return 0;
}