#include "Utilities.h"

bool Utilities::openImage(const std::string & filename, Mat & image)
{
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cout << " --(!) Error reading image " << filename << std::endl;
		return false;
	}
	return true;
}

void Utilities::drawKeypoints(string windowName, Mat & image, std::vector<KeyPoint>& keypoints, std::vector<int>& words)
{
	if (keypoints.size() != words.size())
		return;

	Mat newImage;
	cvtColor(image, newImage, CV_GRAY2RGB);
	int max = 0;
	for (int i = 0; i<words.size(); i++)
		if (words[i]>max)
			max = words[i];

	int steps = (int)(255 / (log(max + 1) / log(3)));
	vector<Scalar> colors;
	for (int r = 1; r<256; r += steps)
		for (int g = 1; g<256; g += steps)
			for (int b = 1; b<256; b += steps)
				colors.push_back(cvScalar(b, g, r));

	for (int i = 0; i<keypoints.size(); i++)
	{
		circle(newImage, keypoints[i].pt, 4, colors[words[i]], 2);
	}

	namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	imshow(windowName, newImage);
}

void Utilities::drawKeypoints(string windowName, Mat & image, std::vector<KeyPoint>& keypoints){
	for (int i = 0; i < keypoints.size(); i++){
		circle(image, keypoints[i].pt, 4, Scalar(255, 0, 0));
	}

	namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	imshow(windowName, image);
}

void Utilities::vectorToMat(const vector<float>& in, Mat & out)
{
	int k = 0;
	for (int i = 0; i < out.rows; ++i)
	{
		for (int j = 0; j < out.cols; ++j)
		{
			out.at<float>(i, j) = in[k];
			++k;
		}
	}

	////obtem o tamanho do descritor
	//int len = in.size();

	////converte o descriptor numa imagem
	//Mat data(1, len, CV_32FC1);
	//for (int i = 0; i < len; i++) {
	//	data.at<float>(0, i) = in[i];
	//}
	//out = data.clone();
}
