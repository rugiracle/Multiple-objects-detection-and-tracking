#include "BackgroundLearning.h"

BackgroundLearning::BackgroundLearning():alphaLearn(0.05), alphaDetection(0.05), learningFrames(-1), counter(0), minVal(0.0), maxVal(1.0),
threshold(15)
{
	std::cout << "BackgroundLearning()" << std::endl;
}

BackgroundLearning::~BackgroundLearning()
{
	delete[] contours;
	std::cout << "~BackgroundLearning()" << std::endl;
}

void BackgroundLearning::process(const cv::Mat& img_input_, cv::Mat& img_output, cv::Mat& img_bgmodel, std::vector<cv::Rect>* det_obj)
{
	if (img_input_.empty())
		return;

	cv::Mat img_input;                            
	if (img_input_.channels() == 3)
	{
		cv::cvtColor(img_input_, img_input, CV_BGR2GRAY);
		cv::equalizeHist(img_input, img_input);  // @equalize image
	}
	else
	{
		img_input_.copyTo(img_input);
		cv::equalizeHist(img_input, img_input);
	}

	if (img_background.empty())
		img_input.copyTo(img_background);

	cv::GaussianBlur(img_background, img_background, cv::Size(5, 5), 0.3, 0.3);
	
	cv::Mat img_input_f(img_input.size(), CV_32F);
	img_input.convertTo(img_input_f, CV_32F, 1. / 255.);

	cv::Mat img_background_f(img_background.size(), CV_32F);
	img_background.convertTo(img_background_f, CV_32F, 1. / 255.);

	cv::Mat img_diff_f(img_input.size(), CV_32F);
	cv::absdiff(img_input_f, img_background_f, img_diff_f);//@ compute imag difference, bkgrd subtraction

	cv::Mat img_foreground(img_input.size(), CV_8U);
	img_diff_f.convertTo(img_foreground, CV_8U, 255.0 / (maxVal - minVal), -minVal);
	cv::threshold(img_foreground, img_foreground, threshold, 255, cv::THRESH_BINARY);
	cv::medianBlur(img_foreground, img_foreground, 5);

	//@This will help in foreground update by removing shadow and ghosts
	cv::Mat diff_img = img_background - img_input;
	cv::threshold(diff_img, diff_img, 0, 255, CV_THRESH_OTSU);

	//@ when it is learning skip detection
	if (learningFrames > 0 && counter <= learningFrames)
	{
		img_background_f = alphaLearn * img_input_f + (1 - alphaLearn) * img_background_f;
		counter++;
	}
	else
	{
		cv::findContours(img_foreground, *contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		det_obj->clear();
		for (int c = 0; c < contours->size(); c++)
		{
			cv::Rect rec = boundingRect(contours->at(c));
			if (rec.area() > detectionThresh)  //@if the size of detected blob is accesptable
			{
				//@from here : blob post processing: remove shadow and or ghost
				double white = cv::countNonZero(diff_img(rec) != 0);
				if (white < double(rec.height) + double(rec.width)) //@update this ...it is probabaly noise 
				{
					img_foreground(rec).setTo(-1);
				}
				else
				{
					cv::Rect rec_;
					rec_ = boundingRect(diff_img(rec));
					cv::Mat temp;
					img_foreground(rec)(rec_).copyTo(temp);
					img_foreground(rec).setTo(-1);
					////update the rectangle
					rec.x += rec_.x; rec.y += rec_.y; rec.width = rec_.width; rec.height = rec_.height;
					temp.copyTo(img_foreground(rec));
					//to there
					det_obj->push_back(rec);
				}				
			}		
			else
			{
				img_foreground(rec).setTo(-1);;
			}
			contours->at(c).clear();
		}
		int rows = img_input.rows;
		int cols = img_input.cols;
		//@Update the background
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				// Adaptive and Selective update of the background model
				if (img_foreground.at<uchar>(i, j) == 0)
				{
					img_background_f.at<float>(i, j) = alphaDetection * img_input_f.at<float>(i, j) + (1 - alphaDetection) * img_background_f.at<float>(i, j);
				}
				else if (img_foreground.at<uchar>(i, j) == -1)
					img_background_f.at<float>(i, j) = (.5 - alphaDetection) * img_input_f.at<float>(i, j) + alphaDetection * img_background_f.at<float>(i, j);
			}
		}
	}
	cv::Mat img_new_background(img_input.size(), CV_8U);
	img_background_f.convertTo(img_new_background, CV_8U, 255.0 / (maxVal - minVal), -minVal);
	img_new_background.copyTo(img_background);
	img_foreground.copyTo(img_output);
	img_background.copyTo(img_bgmodel);
	
	if (learningFrames > 0 && counter <= learningFrames)
	{
		cv::Point loc_d(img_bgmodel.cols * .3, img_bgmodel.rows / 2);
		cv::putText(img_bgmodel, "Initialization *****  ", loc_d,
			cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, cvScalar(255, 255, 255), 2, CV_AA);
	}
}