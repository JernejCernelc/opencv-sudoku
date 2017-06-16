#include "utilityMy.hpp"
#include "sudokuFunc.hpp"

#include <cstdint>
#include <typeinfo>
#include <iostream>
#include <stdio.h>
#include <math.h>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace ml;
using namespace Sud;

vector<Mat> preprocess(Mat image) {
	Size size(WINDOW_WIDTH, WINDOW_HEIGHT);

	Mat thr = cropSudoku(image);

	resize(thr, thr, size);

	vector<Mat> polja;
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			Rect roi(Point(j*BLOCK_WIDTH, i*BLOCK_HEIGHT), Point((j + 1)*BLOCK_WIDTH, (i + 1)*BLOCK_HEIGHT));
			Mat izrez = thr(roi);
			polja.push_back(izrez);
		}
	}

	return polja;
}

Mat cropSudoku(Mat image) {
	Mat thresh;
	vector<vector<Point> > contours;

	/*-----------------------------------------------------------*//*
	Use threshold on input image
	Invert working Mat so contaur doesnt detect frame edge
	Find contours
	*//*-----------------------------------------------------------*/

	threshold(image, thresh, 100, 255, THRESH_BINARY);
	thresh = 255 - thresh;

	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	findContours(thresh, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > largest_area) {
			largest_area = area;
			largest_contour_index = i;
			bounding_rect = boundingRect(contours[i]);
		}
	}

	//findContours(thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


	/*-----------------------------------------------------------*//*
	If working Mat is tilting to side fix it with rotation
	*//*-----------------------------------------------------------*/
	if (contours.size() > 0) {
		int maxX = 0, maxY = 0, minX = thresh.cols, minY = thresh.rows;
		int yDesne, yLeve, xZgornje, xSpodnje;
		for (int i = 0; i < contours.at(largest_contour_index).size(); i++) {
			Point coi = contours.at(largest_contour_index).at(i);
			if (maxX < coi.x) {
				maxX = coi.x;
				yDesne = coi.y;
			}
			if (minX > coi.x) {
				minX = coi.x;
				yLeve = coi.y;
			}
			if (maxY < coi.y) {
				maxY = coi.y;
				xSpodnje = coi.x;
			}
			if (minY > coi.y) {
				minY = coi.y;
				xZgornje = coi.x;
			}
		}

		double kot;
		if (yDesne > yLeve) {
			kot = -angle(Point(bounding_rect.tl().x, yLeve), Point(bounding_rect.tl().x, bounding_rect.tl().y), Point(xZgornje, bounding_rect.tl().y));
		}
		else {
			kot = angle(Point(bounding_rect.br().x, yDesne), Point(bounding_rect.br().x, bounding_rect.tl().y), Point(xZgornje, bounding_rect.tl().y));
		}

		Rect roi(bounding_rect.tl(), bounding_rect.br());
		thresh = thresh(roi);

		Mat rot = getRotationMatrix2D(Point2f(thresh.rows / 2, thresh.cols / 2), kot, 1);
		warpAffine(thresh, thresh, rot, size(thresh));

		contours.clear();
		largest_area = 0;
		largest_contour_index = 0;

		findContours(thresh, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

		for (size_t i = 0; i < contours.size(); i++) {
			double area = contourArea(contours[i]);
			if (area > largest_area) {
				largest_area = area;
				largest_contour_index = i;
				bounding_rect = boundingRect(contours[i]);
			}
		}

		Rect roi2(bounding_rect.tl(), bounding_rect.br());
		thresh = thresh(roi2);
	}

	double dilation_size = 1;
	double erosion_size = 1;
	Mat elementDil = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	Mat elementEro = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));

	dilate(thresh, thresh, elementDil);
	erode(thresh, thresh, elementEro);

	/*while (true) {
		int tmp = waitKey(30);
		if (tmp >= 0) {
			destroyAllWindows();
			break;
		}
		imshow("1", thresh);
	}*/

	return thresh;
}

double angle(Point pt1, Point pt2, Point pt0) {
	double p01 = sqrt(pow(pt1.x - pt0.x, 2) + pow(pt1.y - pt0.y, 2));
	double p02 = sqrt(pow(pt2.x - pt0.x, 2) + pow(pt2.y - pt0.y, 2));
	double p12 = sqrt(pow(pt2.x - pt1.x, 2) + pow(pt2.y - pt1.y, 2));
	double kot = acos((pow(p01, 2) + pow(p02, 2) - pow(p12, 2)) / (2 * p01 * p02)) * 180 / PI;
	/*-----------------------------------------------------------*//*
	Check if kot is NaN
	*//*-----------------------------------------------------------*/
	if (kot != kot) {
		return 0;
	}
	else {
		return kot;
	}
}

Mat cropDigit(Mat in) {
	Mat out;
	in.copyTo(out);
	vector<Point> nonBlackList;
	nonBlackList.reserve(out.rows*out.cols);
	for (int j = 0; j < out.rows; j++) {
		for (int i = 0; i < out.cols; i++) {
			if (in.at<uchar>(j, i) != 0) {
				nonBlackList.push_back(Point(i, j));
			}
		}
	}

	Rect bb = boundingRect(nonBlackList);
	out = out(bb);
	return out;
}

Sudoku::Sudoku() {
	//Init empty sudoku
	solveTrigger = false;
	sudoku = Mat(9, 9, CV_32S);

	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			sudoku.at<int>(i, j) = 0;
		}
	}

	solvedSudokuDraw = sudoku.clone();

	pre = Mat(WINDOW_WIDTH, WINDOW_HEIGHT, CV_8UC3, Scalar(255, 255, 255));
	post = Mat(WINDOW_WIDTH, WINDOW_HEIGHT, CV_8UC3, Scalar(255, 255, 255));
}

void Sudoku::setSolution() {
	solvedSudoku = sudoku.clone();
}

void Sudoku::ConstructSudoku(vector<Mat> polja, Ptr<SVM> svm, CascadeClassifier cascade) {
	vector<Rect> digits;
	for (int i = 8; i >= 0; i--) {
		for (int j = 8; j >= 0; j--) {
			Mat pre = polja.back();

			digits.clear();
			cascade.detectMultiScale(pre, digits, 1.05, 1, 0, Size(26, 26), Size(40, 40));

			float response = -1.0;
			if (digits.size() == 1) {
				Mat digit(28, 28, CV_8UC1, Scalar(0, 0, 0));
				pre = pre(digits.at(0));
				Size size(pre.cols / 1.38, pre.rows / 1.38);
				resize(pre, pre, size);
				Mat croped = cropDigit(pre);

				int xCopy = 14 - (croped.cols / 2);
				int yCopy = 14 - (croped.rows / 2);

				croped.copyTo(digit(Rect(xCopy, yCopy, croped.cols, croped.rows)));

				int img_area = 28 * 28;
				Mat tmp(1, img_area, CV_32FC1);
				int ii = 0;
				for (int i = 0; i < digit.rows; i++) {
					for (int j = 0; j < digit.cols; j++) {
						tmp.at<float>(0, ii++) = (float)(digit.at<uchar>(i, j));
					}
				}

				response = svm->predict(tmp);
				sudoku.at<int>(i, j) = (int)response;

				Mat neki;
				normalize(digit, neki, -1, 1, NORM_MINMAX, CV_32F);

				/*cout << "Digit recognized:" << response << endl;
				while (true) {
					int tmp = waitKey(30);
					if (tmp >= 0) {
						destroyAllWindows();
						break;
					}
					imshow("1", neki);
				}*/

				tmp.release();
				croped.release();
				digit.release();
			}
			else {
				sudoku.at<int>(i, j) = 0;
			}
			polja.pop_back();
		}
	}
}

void Sudoku::drawSudokuPre() {
	pre.setTo(Scalar(255, 255, 255));
	for (int i = 1; i < 9; i++) {
		int thickness = 2;
		Scalar color(50, 50, 50);
		if (i % 3 == 0) {
			thickness = 6;
			color = Scalar(0, 0, 0);
		}
		line(pre, Point(i * BLOCK_WIDTH, 0), Point(i * BLOCK_WIDTH, WINDOW_HEIGHT), color, thickness, 8, 0);
		line(pre, Point(0, i * BLOCK_HEIGHT), Point(WINDOW_WIDTH, i * BLOCK_HEIGHT), color, thickness, 8, 0);
	}

	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			if (sudoku.at<int>(i, j) != 0) {
				string s = to_string(sudoku.at<int>(i, j));
				putText(pre, s, Point(10 + j * 56, 46 + i * 56), 1, 3.2, Scalar(0, 0, 0), 2, 8, false);
			}
		}
	}
}

void Sudoku::drawSudokuPost() {
	post.setTo(Scalar(255, 255, 255));
	for (int i = 1; i < 9; i++) {
		int thickness = 2;
		Scalar color(50, 50, 50);
		if (i % 3 == 0) {
			thickness = 6;
			color = Scalar(0, 0, 0);
		}
		line(post, Point(i * BLOCK_WIDTH, 0), Point(i * BLOCK_WIDTH, WINDOW_HEIGHT), color, thickness, 8, 0);
		line(post, Point(0, i * BLOCK_HEIGHT), Point(WINDOW_WIDTH, i * BLOCK_HEIGHT), color, thickness, 8, 0);
	}

	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			string s;
			if (sudoku.at<int>(i, j) != 0) {
				s = to_string(sudoku.at<int>(i, j));
				putText(post, s, Point(10 + j * 56, 46 + i * 56), 1, 3.2, Scalar(0, 0, 0), 2, 8, false);
			}
			if (solvedSudokuDraw.at<int>(i, j) != 0) {
				s = to_string(solvedSudokuDraw.at<int>(i, j));
				putText(post, s, Point(10 + j * 56, 46 + i * 56), 1, 3.2, Scalar(0, 0, 255), 2, 8, false);

			}
		}
	}
}

void sudokuPreCall(int event, int x, int y, int flags, void* param) {
	Sudoku* sudoku = reinterpret_cast<Sudoku*>(param);
	if (event == EVENT_LBUTTONDOWN)
	{
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		Rect okvir;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (j * 56 <= x && (j + 1) * 56 >= x && i * 56 <= y && (i + 1) * 56 >= y) {
					okvir = Rect(Point(j * 56 + 5, i * 56 + 5), Point((((j + 1) * 56) - 5), ((i + 1) * 56) - 5));
				}
			}
		}

		int tmp;
		while (true) {
			sudoku->drawSudokuPre();
			imshow("sudokuPre", sudoku->pre);
			tmp = waitKey(200);

			if (tmp >= 49 && tmp <= 57) {
				break;
			}
			else if ((tmp > 0 && tmp < 49) || tmp > 57) {
				cout << "Enter number between 1 and 9." << endl;
			}

			
			rectangle(sudoku->pre, okvir, Scalar(255, 0, 0), 2, 8, 0);
			imshow("sudokuPre", sudoku->pre);
			tmp = waitKey(200);

			if (tmp >= 49 && tmp <= 57) {
				break;
			}
			else if ((tmp > 0 && tmp < 49) || tmp > 57) {
				cout << "Enter number between 1 and 9." << endl;
			}
		}
		
		int previous;
		
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (j * 56 <= x && (j + 1) * 56 >= x && i * 56 <= y && (i + 1) * 56 >= y) {
					previous = sudoku->sudoku.at<int>(i, j);
					sudoku->sudoku.at<int>(i, j) = tmp - 48;
					if (sudoku->solveTrigger) {
						if (solveSudoku(sudoku->sudoku, sudoku->solvedSudoku)) {
							cout << "solution found" << endl;
						}
						else {
							sudoku->sudoku.at<int>(i, j) = previous;
							cout << "no sollution" << endl;
						}
					}
				}
			}
		}
		ResetMat(sudoku->solvedSudokuDraw);
		sudoku->drawSudokuPre();
		sudoku->drawSudokuPost();
		imshow("sudokuPre", sudoku->pre);
		imshow("sudokuPost", sudoku->post);
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		//cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		int previous;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (j * 56 <= x && (j + 1) * 56 >= x && i * 56 <= y && (i + 1) * 56 >= y) {
					previous = sudoku->sudoku.at<int>(i, j);
					if (previous != 0) {
						sudoku->sudoku.at<int>(i, j) = 0;
						if (sudoku->solveTrigger) {
							if (solveSudoku(sudoku->sudoku, sudoku->solvedSudoku)) {
								cout << "solution found" << endl;
							}
							else {
								cout << "no sollution" << endl;
							}
						}
					}
				}
			}
		}
		if (previous != 0) {
			ResetMat(sudoku->solvedSudokuDraw);
		}
		sudoku->drawSudokuPre();
		sudoku->drawSudokuPost();
		imshow("sudokuPre", sudoku->pre);
		imshow("sudokuPost", sudoku->post);
	}
}

void sudokuPostCall(int event, int x, int y, int flags, void* param) {
	Sudoku* sudoku = reinterpret_cast<Sudoku*>(param);
	if (event == EVENT_LBUTTONDOWN)
	{
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (j * 56 <= x && (j + 1) * 56 >= x && i * 56 <= y && (i + 1) * 56 >= y) {
					if (sudoku->sudoku.at<int>(i, j) == 0) {
						sudoku->solvedSudokuDraw.at<int>(i, j) = sudoku->solvedSudoku.at<int>(i, j);
					}
				}
			}
		}
		sudoku->drawSudokuPost();
		imshow("sudokuPost", sudoku->post);
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		//cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (sudoku->sudoku.at<int>(i, j) == 0) {
					sudoku->solvedSudokuDraw.at<int>(i, j) = sudoku->solvedSudoku.at<int>(i, j);
				}
			}
		}
		sudoku->drawSudokuPost();
		imshow("sudokuPost", sudoku->post);
	}
}

void ResetMat(Mat m) {
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++) {
			m.at<int>(i, j) = 0;
		}
	}
}

