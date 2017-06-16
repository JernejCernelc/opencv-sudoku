#include <cstdint>
#include <typeinfo>

#include "TrainDetector.hpp"
#include "mnist/mnist_reader.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace mnist;
using namespace ml;


void trainDetector() {
	try {
		auto dataset = read_dataset<vector, vector, uint8_t, uint8_t>();
		int num_files = 0;
		vector<Mat> training_imagesMat;
		vector<int> labels;
		for (int i = 0; i < dataset.training_images.size(); i++) {
			if ((int)dataset.training_labels.at(i) != 0) {
				Mat img = Mat(28, 28, CV_8UC1, dataset.training_images.at(i).data());
				Mat post;
				normalize(img, post, -1, 1, NORM_MINMAX, CV_32FC1);
				training_imagesMat.push_back(post);
				labels.push_back((int)dataset.training_labels.at(i));
				num_files++;
			}
		}

		int img_area = 28 * 28;
		Mat training_mat(num_files, img_area, CV_32FC1);

		int file_num = 0;
		for (Mat training_image : training_imagesMat) {
			int ii = 0;

			for (int i = 0; i < training_image.rows; i++) {
				for (int j = 0; j < training_image.cols; j++) {
					training_mat.at<float>(file_num, ii++) = (float)(training_image.at<float>(i, j));
				}
			}
			file_num++;
		}

		Ptr<SVM> svm = SVM::create();
		svm->setType(SVM::C_SVC);
		svm->setKernel(SVM::RBF);
		svm->trainAuto(training_mat, ROW_SAMPLE, labels, 10);
		svm->save("digit_recognize_cascadeAutoRBF.xml");

	}
	catch (cv::Exception& e) {
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
}

