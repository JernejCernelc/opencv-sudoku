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

Mat preprocess(Mat image) {
	//Size size(WINDOW_WIDTH, WINDOW_HEIGHT);

	Mat thr = cropSudoku(image);

	while (true) {
		int tmp = waitKey(30);
		if (tmp >= 0) {
			break;
		}
		imshow("prewiev", thr);
	}
	//resize(thr, thr, size);
	return thr;
}

Mat cropSudoku(Mat image) {
	Mat thresh;
	vector<vector<Point> > contours;

	/*-----------------------------------------------------------*//*
	Use threshold on input image
	Invert working Mat so contaur doesnt detect frame edge
	Find contours
	*//*-----------------------------------------------------------*/

	//adaptiveThreshold(image, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
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

	/*while (true) {
		int tmp = waitKey(30);
		if (tmp >= 0) {
			destroyAllWindows();
			break;
		}
		imshow("1", thresh);
	}*/

	return thresh(Rect(Point(round(thresh.cols*0.01), round(thresh.rows*0.01)), Point(thresh.cols-round(thresh.cols*0.01), thresh.rows-round(thresh.rows*0.01))));
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

Mat removeEdges(Mat in) {
	Mat out;
	in.copyTo(out);
	vector<vector<Point> > contours;
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect = Rect(Point(0, 0),Point(out.cols-1, out.rows-1));
	
	findContours(out, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > largest_area) {
			largest_area = area;
			largest_contour_index = i;
			bounding_rect = boundingRect(contours[i]);
		}
	}

	/*Mat tmp = Mat(image.rows, image.cols, CV_8UC3, Scalar(0, 0, 0));
	rectangle(tmp, bounding_rect, Scalar(255, 0, 0), 1, 8, 0);
	imshow("neki", tmp);*/

	return out(bounding_rect);
}

Sudoku::Sudoku() {
	//Init empty sudoku
	sudoku = Mat(9, 9, CV_32S);
	uncompatible = Mat(9, 9, CV_32S);

	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			sudoku.at<int>(i, j) = 0;
			uncompatible.at<int>(i, j) = 0;
		}
	}
	solvable = false;
	solvedSudokuDraw = sudoku.clone();

	pre = Mat(WINDOW_WIDTH, WINDOW_HEIGHT, CV_8UC3, Scalar(255, 255, 255));
	post = Mat(WINDOW_WIDTH, WINDOW_HEIGHT, CV_8UC3, Scalar(255, 255, 255));
}

void Sudoku::setSolution() {
	solvedSudoku = sudoku.clone();
}

void Sudoku::clear() {
	ResetMat(sudoku);
	ResetMat(solvedSudoku);
	ResetMat(solvedSudokuDraw);
	ResetMat(uncompatible);
	solvable = false;
	drawSudokuPre();
	drawSudokuPost();
	imshow("sudokuPre", pre);
	imshow("sudokuPost", post);
}

void Sudoku::check() {
	bool vseOk = true;
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			if (sudoku.at<int>(i, j) != 0) {
				if (numberAppears(sudoku.at<int>(i, j), j, i)) {
					uncompatible.at<int>(i, j) = sudoku.at<int>(i, j);
					vseOk = false;
				}
			}
		}
	}
	if (vseOk) {
		solvable = true;
	}
	else {
		solvable = false;
	}
}

bool Sudoku::isInRow(int st, int x, int y) {
	bool returnValue = false;
	for (int i = 0; i < 9; i++) {
		if (i != x && sudoku.at<int>(y, i) == st) {
			uncompatible.at<int>(y, i) = st;
			returnValue = true;
		}
	}
	return returnValue;
}

bool Sudoku::isInCol(int st, int x, int y) {
	bool returnValue = false;
	for (int i = 0; i < 9; i++) {
		if (i != y && sudoku.at<int>(i, x) == st) {
			uncompatible.at<int>(i, x) = st;
			returnValue = true;
		}
	}
	return returnValue;
}

bool Sudoku::isInBox(int st, int x, int y, int xBox, int yBox) {
	bool returnValue = false;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (!eq(x, y, xBox + j, yBox + i) && sudoku.at<int>(yBox + i, xBox + j) == st) {
				uncompatible.at<int>(y, x) = st;
				returnValue = true;
			}
		}
	}
	return returnValue;
}

bool eq(int x, int y, int j, int i) {
	if (x == j && y == i) {
		return true;
	}
	return false;
}

bool Sudoku::numberAppears(int st, int x, int y) {
	if (isInRow(st, x, y) || isInCol(st, x, y) || isInBox(st, x, y, x - (x % 3), y - (y % 3))) {
		return true;
	}
	else {
		return false;
	}
}

void Sudoku::ConstructSudoku(Mat polje, Ptr<SVM> svm, CascadeClassifier cascade) {
	vector<Rect> digits;
	int poljeW = polje.cols;
	int poljeH = polje.rows;
	
	digits.clear();
	cascade.detectMultiScale(polje, digits, 1.05, 3, 0, Size(round(poljeW/24), round(poljeW / 26)), Size(round(poljeW / 12), round(poljeW / 14)));

	for (Rect stevilo : digits) {
		rectangle(polje, stevilo, Scalar(255, 255, 255), 1, 8, 0);
		//Mat detect = polje(Rect(Point(stevilo.tl().x-round(poljeW / 40), stevilo.tl().y- round(poljeH / 40)), Point(stevilo.br().x+round(poljeW / 40), stevilo.br().y+round(poljeH/40))));
		float response = -1.0;
		Mat digit(28, 28, CV_8UC1, Scalar(0, 0, 0));
		//detect = detect(Rect(Point(stevilo.tl().x, stevilo.tl().y), Point(stevilo.br().x, detect.rows-1)));

		while (true) {
			int tmp = waitKey(30);
			if (tmp >= 0) {
				destroyAllWindows();
				break;
			}
			imshow("1", polje);
		}

		/*double dilation_size = 0.7;
		double erosion_size = 0.7;
		Mat elementDil = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
		Mat elementEro = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
		erode(detect, detect, elementEro);
		dilate(detect, detect, elementDil);*/

		/*Mat croped = removeEdges(detect);
		Size size(croped.cols*20/croped.rows, 20);
		resize(croped, croped, size);

		

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

		Mat norm;
		normalize(tmp, norm, -1, 1, NORM_MINMAX, CV_32F);

		response = svm->predict(norm);
		cout << response << endl;
		//sudoku.at<int>(i, j) = (int)response;

		//cout << "Digit recognized:" << response << endl;

		tmp.release();
		croped.release();
		digit.release();
		//else	sudoku.at<int>(i, j) = 0;
		*/
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
			if (uncompatible.at<int>(i, j) != 0) {
				Rect okvir = Rect(Point(j * 56 + 5, i * 56 + 5), Point((((j + 1) * 56) - 5), ((i + 1) * 56) - 5));
				rectangle(pre, okvir, Scalar(0, 0, 255), 2, 8, 0);
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
				cout << "Please enter number between 1 and 9." << endl;
			}

			
			rectangle(sudoku->pre, okvir, Scalar(255, 0, 0), 2, 8, 0);
			imshow("sudokuPre", sudoku->pre);
			tmp = waitKey(200);

			if (tmp >= 49 && tmp <= 57) {
				break;
			}
			else if ((tmp > 0 && tmp < 49) || tmp > 57) {
				cout << "Please enter number between 1 and 9." << endl;
			}
		}
		
		int previous;
		
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (j * 56 <= x && (j + 1) * 56 >= x && i * 56 <= y && (i + 1) * 56 >= y) {
					previous = sudoku->sudoku.at<int>(i, j);
					sudoku->sudoku.at<int>(i, j) = tmp - 48;
					ResetMat(sudoku->uncompatible);
					sudoku->check();
					cout << "solvable: " << sudoku->solvable << endl;
					cout << sudoku->uncompatible << endl << endl;
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
						ResetMat(sudoku->uncompatible);
						sudoku->check();
						cout << "solvable: " << sudoku->solvable << endl;
						cout << sudoku->uncompatible << endl << endl;
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
		if (sudoku->solvable) {
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
}

void ResetMat(Mat m) {
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++) {
			m.at<int>(i, j) = 0;
		}
	}
}

