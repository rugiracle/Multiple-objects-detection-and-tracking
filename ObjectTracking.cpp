#include "ObjectTracking.h"

ObjectTracking::ObjectTracking(cv::Rect detect, int whitePxlcnt, int idx):bbox(detect), whiteArea(whitePxlcnt), missing(0), alive(1), idx(idx)
{
	start.x = bbox.x + (bbox.width / 2);
	start.y = bbox.y + (bbox.height / 2);
	center.x = start.x;
	center.y = start.y;
	path.push_back(center);
	preBox = bbox;
}
ObjectTracking::~ObjectTracking()
{
	;
}
void ObjectTracking::updateTracklet(cv::Rect detect, int whitePxlcnt)
{
	preBox = bbox;
	bbox = detect;
	whiteArea = whitePxlcnt;
	center.x = bbox.x + (bbox.width / 2);
	center.y = bbox.y + (bbox.height / 2);
	alive += 1;
	path.push_back(center);
}
void ObjectTracking::predictTracklet()
{
	missing += 1;
}
int ObjectTracking::getIdx()
{
	return idx;
}
int ObjectTracking::getRadius()
{
	return (bbox.height + bbox.width);
}
cv::Point ObjectTracking::getCenter()
{
	return(center);
}
cv::Rect ObjectTracking::getBbox()
{
	return(bbox);
}
cv::Point ObjectTracking::getPrevCenter()
{
	return(cv::Point(preBox.x +preBox.width/2, preBox.y + preBox.height / 2));
}
double ObjectTracking::getWhitePxlCnt() 
{
	return whiteArea;
}
int ObjectTracking::getStatus()
{
	return missing;
}