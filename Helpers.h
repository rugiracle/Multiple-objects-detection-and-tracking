#pragma once
#include <opencv2/opencv.hpp>
#include"ObjectTracking.h"
#include <algorithm>
#include"Parameters.h"

void draw_rectangle(cv::Mat& img, cv::Rect roi, cv::Scalar color, int thickness)
{
	cv::rectangle(img, roi, color, thickness);
}
void drawTracklet(cv::Mat& im_ptr, std::vector<ObjectTracking>& tracked)
{
	char text[50] = "_%d";
	char identify[100];	
	cv::Scalar color(0, 0, 200);	
	for(int i = 0 ; i < tracked.size(); i++)
	{
		cv::rectangle(im_ptr, tracked.at(i).getBbox(), color, 2);		
		sprintf(identify, text, tracked.at(i).getIdx());
		cv::putText(im_ptr, identify, tracked.at(i).getCenter(),
			cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(0, 0, 250), 1, CV_AA);
		if (tracked.at(i).path.size()>= 2)
		{
			for (int p = tracked.at(i).path.size() - (int)1; p > std::max((int)tracked.at(i).path.size() - pathLengt, 1); p--)
			{
				cv::line(im_ptr, tracked.at(i).path.at(p), tracked.at(i).path.at(p - 1), color, 2 , 8, 0);
			}
		}
	}
}

void initTracking(std::vector<cv::Rect>& detected, std::vector<ObjectTracking>& tracked, cv::Mat mask)
{
	for (int i = 0; i < detected.size(); i++)
	{

		ObjectTracking temp(detected.at(i), cv::countNonZero(mask(detected.at(i))), ++uniqueId);
		tracked.push_back(temp);
	}
}

void addTracklet(std::vector<ObjectTracking>& tracked, cv::Rect detect, double area)
{
	tracked.push_back(ObjectTracking(detect, area, ++uniqueId));
}

void deleteTracklet(std::vector<ObjectTracking>& tracked, int idx)
{
	for (int i = 0; i < tracked.size(); i++)
	{
		if (tracked.at(i).getIdx() == idx)
		{
			tracked.erase(tracked.begin() + i);
			break;
		}
	}
}

int liesWithinCircle(cv::Point pt_left, cv::Point center, int radius)
{
	int res = 0;
	if ((pow((pt_left.x - center.x), 2) + pow((pt_left.y - center.y), 2)) < pow(radius / (gate), 2))
		res = 1;
	return res;
}
double euclidian(cv::Point a, cv::Point b)
{
	double res = sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
	return res;
}
double formula(double in)
{
	return (1 / (1 + in));
}
cv::Mat DataAssoc(cv::Mat mask, std::vector<cv::Rect>& detected, std::vector<ObjectTracking>& tracked)
{
	cv::Mat DA(tracked.size(), detected.size(), CV_64F, -1);
	for (int i = 0; i < tracked.size(); i++)
	{
		int radius = tracked.at(i).getRadius();
		int idx = -1;
		for (int j = 0; j < detected.size(); j++)
		{
			cv::Point border_left(detected.at(j).x, detected.at(j).y + detected.at(j).height / 2);
			cv::Point center(detected.at(j).x+detected.at(j).width/2, detected.at(j).y + detected.at(j).height / 2);
			cv::Point border_right(detected.at(j).x + detected.at(j).width, detected.at(j).y + detected.at(j).height / 2);
			if (liesWithinCircle(border_left, tracked.at(i).getCenter(), radius) || liesWithinCircle(border_right, tracked.at(i).getCenter(), radius))
			{
				double distance = euclidian(tracked.at(i).getCenter(), center);
				double dsize = euclidian(cv::Point(tracked.at(i).getBbox().width, tracked.at(i).getBbox().height), cv::Point(detected.at(j).width, detected.at(j).height));//@size
				double vel = euclidian(tracked.at(i).getPrevCenter(), center);//@error in velocity estimation
				double fit = 2 * formula(distance) + formula(dsize) + formula(vel);
				DA.at<double>(i, j) = fit;			
			}
		}
	}
	return DA;
}
cv::Mat pairingObs2Det(cv::Mat A)
{
	double maxnum;
	cv::Mat res(A.rows, 2, CV_64F, -1);
	int idx = -1;
	for (int i = 0; i < A.rows; i++)
	{
		maxnum = -10;
		for (int j = 0; j < A.cols; j++)
		{
			if (A.at<double>(i, j) > maxnum)
			{
				maxnum = A.at<double>(i, j);
				idx = j;
			}
		}
		res.at<double>(i, 0) = idx;
		res.at<double>(i, 1) = A.at<double>(i, idx);
	}
	return res;
}

void updateTranklets(cv::Mat binImg, cv::Mat BGR, cv::Mat Paired, std::vector<ObjectTracking>& prev_detect_ptr, std::vector<cv::Rect>& detect_ptr)
{
	int i, j;
	std::vector<int> newborn;	//@newly detected objects(appearance)	
	int tr_size = prev_detect_ptr.size(); //@tracklets size
	int dt_size = detect_ptr.size();      //@detection size
	for (int i = 0; i < dt_size; i++)
	{
		int m = 0;//@ how many time is a single detection assigned to tracklet
		double maxfit = 0.0;
		int idx = -1;
		for (j = 0; j < tr_size; j++)
		{
			if (Paired.at<double>(j, 0) == i && Paired.at<double>(j, 1) != -1) //multiple assignement of the ith detection
			{
				m++;
				if (Paired.at<double>(j, 1) > maxfit)
				{
					maxfit = Paired.at<double>(j, 1);
					idx = j;
				}
			}
		}
		if (m == 0)
		{
			newborn.push_back(i); //ith detection not assigned						
		}
		else if (m == 1)
		{
			prev_detect_ptr.at(idx).updateTracklet(detect_ptr.at(i), cv::countNonZero(binImg(detect_ptr.at(i)) != 0));//@ remove
		}
		else //@ i.e. multiple assignement
		{
			double growing = 100 * abs(prev_detect_ptr.at(idx).getWhitePxlCnt() - cv::countNonZero(binImg(detect_ptr.at(i)) != 0)) / (prev_detect_ptr.at(idx).getWhitePxlCnt());
			if (growing < grow_thr)// normal growth 
			{
				prev_detect_ptr.at(idx).updateTracklet(detect_ptr.at(i), cv::countNonZero(binImg(detect_ptr.at(i)) != 0));
				for (j = 0; j < Paired.rows; j++)
				{
					if (Paired.at<double>(j, 0) == i && Paired.at<double>(j, 1) != maxfit && Paired.at<double>(j, 1) != -1)
					{
						prev_detect_ptr.at(idx).predictTracklet();
					}
				}
			}
			else if (growing >= grow_thr && prev_detect_ptr.at(idx).getStatus() == 0)
			{
				prev_detect_ptr.at(idx).predictTracklet();
			}
			else
			{
				prev_detect_ptr.at(idx).updateTracklet(detect_ptr.at(i), cv::countNonZero(binImg(detect_ptr.at(i)) != 0));
			}
		}
	}
	for (int i = 0; i < newborn.size(); i++)
	{	
			addTracklet(prev_detect_ptr, detect_ptr.at(newborn.at(i)), cv::countNonZero(binImg(detect_ptr.at(newborn.at(i))) != 0));
	}
}

void deleteMissingTracklets(cv::Mat binImg, std::vector < ObjectTracking>& tracked)
{
	for (int i = 0; i < tracked.size(); i++)
	{
		cv::Mat Points;
		cv::findNonZero(binImg(tracked.at(i).getBbox()), Points);
		if (!Points.empty())
		{
			cv::Rect bRect = cv::boundingRect(Points);
			if (bRect.width < 0.5 * tracked.at(i).getBbox().width || bRect.height < 0.5 * tracked.at(i).getBbox().height)
			{
				tracked.erase(tracked.begin() + i);				
			}
		}
		else if (Points.empty())
		{
			tracked.erase(tracked.begin() + i);
		}
		else if (tracked.at(i).getStatus() >= life_thr)
		{
			tracked.erase(tracked.begin() + i);
		}
	}
}

void ObjectTrackingHandler(cv::Mat binImg, cv::Mat BGR, std::vector<cv::Rect>& detect_ptr, std::vector<ObjectTracking>& tracked)
{
	if (detect_ptr.size() && !tracked.size())
		initTracking(detect_ptr, tracked, binImg);
	else if (detect_ptr.size() && tracked.size())
	{
		cv::Mat Matched = DataAssoc(binImg, detect_ptr, tracked);
		cv::Mat pairs = pairingObs2Det(Matched);
		updateTranklets(binImg, BGR, pairs, tracked, detect_ptr);				
		deleteMissingTracklets(binImg, tracked);
	}
}
