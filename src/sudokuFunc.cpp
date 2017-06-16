#include "sudokuFunc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/highgui.hpp>

using namespace cv;

bool solveSudoku(Mat sudoku, Mat resitev) {
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			if (sudoku.at<int>(i, j) != 0) {
				resitev.at<int>(i, j) = sudoku.at<int>(i, j);
			}
			else {
				resitev.at<int>(i, j) = 0;
			}
		}
	}
	return solveSudokuRek(resitev);
}

bool solveSudokuRek(Mat resitev) {
	int x, y;
	if (!findEmptySpace(resitev, &x, &y)) {
		return true;
	}

	for (int i = 1; i <= 9; i++) {
		if (validLoccation(resitev, i, x, y)) {
			resitev.at<int>(y, x) = i;
			if (solveSudokuRek(resitev)) {
				return true;
			}
			resitev.at<int>(y, x) = 0;
		}
	}
	return false;
}

bool findEmptySpace(Mat sudoku, int *x, int *y) {
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			if (sudoku.at<int>(i, j) == 0) {
				*x = j;
				*y = i;
				return true;
			}
		}
	}
	return false;
}

bool isInRow(Mat sudoku, int st, int y) {
	for (int i = 0; i < 9; i++) {
		if (sudoku.at<int>(y, i) == st) {
			return true;
		}
	}
	return false;
}

bool isInCol(Mat sudoku, int st, int x) {
	for (int i = 0; i < 9; i++) {
		if (sudoku.at<int>(i, x) == st) {
			return true;
		}
	}
	return false;
}

bool isInBox(Mat sudoku, int st, int xBox, int yBox) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (sudoku.at<int>(yBox + i, xBox + j) == st) {
				return true;
			}
		}
	}
	return false;
}

bool validLoccation(Mat sudoku, int st, int x, int y) {
	if (!isInRow(sudoku, st, y) && !isInCol(sudoku, st, x) && !isInBox(sudoku, st, x-(x%3), y-(y%3))) {
		return true;
	}
	else {
		return false;
	}
}
