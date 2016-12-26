#pragma once
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class Utilities {
public:
	static bool openImage(const std::string &filename, Mat &image);
	static void drawKeypoints(string windowName, Mat &image, std::vector<KeyPoint> &keypoints, std::vector<int> &words);
	static void drawKeypoints(string windowName, Mat &image, std::vector<KeyPoint> &keypoints);
	static void vectorToMat(const vector<float>& in, Mat& out);
};