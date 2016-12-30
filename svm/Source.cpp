#include <opencv2\opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include "Utilities.h"
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;


int main(){

	initModule_nonfree();

	string imagePath = "train/"; //"DatasetSample/Train/";
	string imageTestPath =  "test2/"; //"DatasetSample/Test2/";

	Mat example;
	vector<int> labels;
	CvSVM svm;

	vector <vector <string> > data;
	ifstream infile("Dataset/Train/trainLabels.csv");
	
	int nPictures = 3000;

	SiftDescriptorExtractor siftDescExtractor;
	Mat siftDescriptor;
	vector<KeyPoint> keypoints;
	Mat siftFeatures;


	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	
	bool train = false;

	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.50625000000000009;
	params.C = 200.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.000001);

	if (train){

		cout << "Reading Labels" << endl;
		while (infile)
		{
			string s;
			if (!getline(infile, s)) break;

			istringstream ss(s);
			vector <string> record;

			while (ss)
			{
				string s;
				if (!getline(ss, s, ',')) break;
				record.push_back(s);
			}

			data.push_back(record);
		}
		if (!infile.eof())
		{
			cerr << "Fooey!\n";
		}

		//for (int i = 1; i < data.size(); i++){
		for (int i = 1; i < nPictures+1; i++){
			string label = data[i][1];
			if (label == "airplane") labels.push_back(1);
			if (label == "automobile") labels.push_back(2);
			if (label == "bird") labels.push_back(3);
			if (label == "cat") labels.push_back(4);
			if (label == "deer") labels.push_back(5);
			if (label == "dog") labels.push_back(6);
			if (label == "frog") labels.push_back(7);
			if (label == "horse") labels.push_back(8);
			if (label == "ship") labels.push_back(9);
			if (label == "truck") labels.push_back(10);
		}

		cout << "Reading features for vocabulary" << endl;

		for (int i = 0; i < nPictures; i++){
			string imageName = imagePath + to_string(i+1) + ".png";
			//string imageName = imagePath + to_string(i + 1) + ".jpg";
			Utilities::openImage(imageName, example);
			//resize(example, example, Size(256, 256));
			siftDescExtractor.detect(example, keypoints);
			siftDescExtractor.compute(example, keypoints, siftDescriptor);
			siftFeatures.push_back(siftDescriptor);
		}

		Mat featuresUnclustered;
		vector<KeyPoint> keypoints;
		Mat img;

		int dictionarySize = 100;
		TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
		int retries = 1;
		int flags = KMEANS_PP_CENTERS;

		cout << "Constructing vocabulary" << endl;

		BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
		Mat dictionary = bowTrainer.cluster(siftFeatures);
		//Mat dictionary = bowTrainer.cluster(features);

		FileStorage fs("dictionary.yml", FileStorage::WRITE);
		fs << "vocabulary" << dictionary;
		fs.release();


		//Set the dictionary with the vocabulary we created in the first step
		bowDE.setVocabulary(dictionary);
		Mat allBowDescriptors;

		cout << "Getting Descriptors" << endl;

		for (int i = 0; i < nPictures; i++){
			string imageName = imagePath + to_string(i + 1) + ".png";
			//string imageName = imagePath + to_string(i + 1) + ".jpg";
			Utilities::openImage(imageName, img);
			//resize(img, img, Size(256, 256));
			vector<KeyPoint> keypoints;
			//Detect SIFT keypoints (or feature points)
			detector->detect(img, keypoints);
			//To store the BoW (or BoF) representation of the image
			Mat bowDescriptor;
			//extract BoW (or BoF) descriptor from given image
			bowDE.compute(img, keypoints, bowDescriptor);
			if (bowDescriptor.empty()) {
				labels.erase(labels.begin() + i);
			}
			allBowDescriptors.push_back(bowDescriptor);
			//Utilities::drawKeypoints("img keypoints", img, keypoints);
			//waitKey(0);

		}

		cout << "Adjusting Labels to Descriptors Found" << endl;

		Mat learningLabels(labels.size(), 1, CV_32SC1);

		for (int i = 0; i<labels.size(); i++){
			
			learningLabels.at<int>(i) = labels[i];
		}

		/*CvSVMParams params;
		params.kernel_type = CvSVM::RBF;
		params.svm_type = CvSVM::C_SVC;
		params.gamma = 0.51125000000000009;
		params.C = 200.50000000000000;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);*/

		cout << "Training SVM" << endl;

		svm.train(allBowDescriptors, learningLabels, Mat(), Mat(), params);

		//CvSVMParams params2;
		//params2.kernel_type = CvSVM::RBF;
		//params.svm_type = CvSVM::C_SVC;
		//params.gamma = 0.50625000000000009;
		//params.C = 312.50000000000000;
		//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);

		//bool something = svm.train_auto(allBowDescriptors, learningLabels, Mat(), Mat(),params2,3);
		//svm.train(allBowDescriptors, learningLabels, Mat(), Mat(), params2);

		string filename = "smallTest.xml";
		cout << "SVM saved under the name: " << filename << endl;
		svm.save(filename.c_str());

	}
	else{
		string filename = "smallTest.xml";
		cout << "Loading SVM" << endl;
		svm.load(filename.c_str());
		Mat img;
		Mat bowDescriptor;
		//prepare BOW descriptor extractor from the dictionary    
		Mat dictionary;
		FileStorage fs("dictionary.yml", FileStorage::READ);
		fs["vocabulary"] >> dictionary;
		fs.release();

		//create a nearest neighbor matcher
		Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
		//create Sift feature point extracter
		Ptr<FeatureDetector> detector(new SiftFeatureDetector());
		//create Sift descriptor extractor
		Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
		//create BoF (or BoW) descriptor extractor
		BOWImgDescriptorExtractor bowDE(extractor, matcher);
		//Set the dictionary with the vocabulary we created in the first step
		bowDE.setVocabulary(dictionary);

		for (int i = 0; i < 20; i++){
			string imageName = imageTestPath + to_string(i + 1) + ".png";
			//string imageName = imageTestPath + to_string(i + 1) + ".jpg";
			Utilities::openImage(imageName, img);
			//resize(img, img, Size(256, 256));
			detector->detect(img, keypoints);
			bowDE.compute(img, keypoints, bowDescriptor);
			float prediction = svm.predict(bowDescriptor);
			string whatItIs;
			if (prediction == 1) whatItIs = "Airplane";
			if (prediction == 2) whatItIs = "Automobile";
			if (prediction == 3) whatItIs = "Bird";
			if (prediction == 4) whatItIs = "Cat";
			if (prediction == 5) whatItIs = "Deer";
			if (prediction == 6) whatItIs = "Dog";
			if (prediction == 7) whatItIs = "Frog";
			if (prediction == 8) whatItIs = "Horse";
			if (prediction == 9) whatItIs = "Ship";
			if (prediction == 10) whatItIs = "Truck";

			cout << "Filename: " << imageName << " -> Prediction: " << whatItIs << endl;
		}
	}

	system("pause");

	return 0;
}