#include "utilityMy.hpp"
#include "TrainDetector.hpp"
#include "sudokuFunc.hpp"

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace cv;
using namespace ml;
using namespace Sud;

#define WINDOW_NAME "DigitDetection"

CascadeClassifier cascade;

int main(int argc, char** argv) {
	try {
		//UNCOMMENT TO TRAIN NEW 
		/*cout << "processing";
		trainDetector();
		cout << "finished";
		while (true) {

		}*/

		Mat image, gray;
		Ptr<SVM> svm;

		if (argc != 2) {
			cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
			return -1;
		}

		if (!cascade.load("digit_detect_cascade.xml")) {
			cerr << "Could not load classifier digit_detect_cascade.xml" << endl;
			return -1;
		}

		if (!(svm = Algorithm::load<SVM>("digit_recognize_cascadeAutoRBF.xml"))) {
			cerr << "Could not load classifier digit_recognize_cascade.xml" << endl;
			return -1;
		}

		image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	
		cvtColor(image, gray, CV_BGR2GRAY);
		Mat polje = preprocess(gray);

		Sudoku sudoku;
		sudoku.Sudoku::Sudoku();
		sudoku.ConstructSudoku(polje, svm, cascade);
		sudoku.setSolution();
		sudoku.check();

		namedWindow("sudokuPre", WINDOW_AUTOSIZE);
		namedWindow("sudokuPost", WINDOW_AUTOSIZE);
		
		setMouseCallback("sudokuPre", sudokuPreCall, &sudoku);
		setMouseCallback("sudokuPost", sudokuPostCall, &sudoku);

		sudoku.drawSudokuPre();
		sudoku.drawSudokuPost();

		imshow("sudokuPre", sudoku.pre);
		imshow("sudokuPost", sudoku.post);

		while (true) {
			int key = waitKey(30);
			if (key == 'x') {
				break;
			}
			else if (key == 's') {
				if (sudoku.solvable) {
					clock_t t;
					t = clock();
					if (solveSudoku(sudoku.sudoku, sudoku.solvedSudoku)) {
						cout << "Solution found in ";
					}
					else {
						cout << "No sollution for this input found in ";
					}
					t = clock() - t;
					cout << t*1.0 / CLOCKS_PER_SEC << " seconds." << endl;
				}
				else {
					cout << "This sudoku cannot be solved." << endl;
				}
			}
			else if (key == 'c') {
				sudoku.clear();
				cout << "Sudoku cleared." << endl;
			}
		}

		return 0;
	}
	catch (cv::Exception& e) {
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
}