#ifndef UTILITYMY_
#define UTILITYMY_

#include <cstdint>
#include <typeinfo>

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

const double PI = 3.141592653589793238463;

#define WINDOW_WIDTH 504
#define WINDOW_HEIGHT 504

#define BLOCK_WIDTH  (WINDOW_WIDTH / 9)
#define BLOCK_HEIGHT  (WINDOW_HEIGHT / 9)

namespace Sud {
	class Sudoku {
		public:
			Mat sudoku;
			Mat solvedSudoku;
			Mat solvedSudokuDraw;
			Mat uncompatible;
			
			Mat pre;
			Mat post;
			Mat clicked;
			
			bool solvable;
			
			Sudoku();
			void ConstructSudoku(vector<Mat> polja, Ptr<SVM> svm, CascadeClassifier cascade);
			void setSolution();
			
			void drawSudokuPre();
			void drawSudokuPost();
			
			void clear();
			void check();
			bool isInRow(int st, int x, int y);
			bool isInCol(int st, int x, int y);
			bool isInBox(int st, int x, int y, int xBox, int yBox);
			bool numberAppears(int st, int x, int y);
	};
};

void sudokuPreCall(int event, int x, int y, int flags, void* param);
void sudokuPostCall(int event, int x, int y, int flags, void* param);

Mat removeEdges(Mat in);
vector<Mat> preprocess(Mat image);
Mat cropSudoku(Mat image) ;
double angle(Point pt1, Point pt2, Point pt0);
void ResetMat(Mat m);
bool eq(int x, int y, int j, int i);

#endif