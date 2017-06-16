#ifndef SUDOKURECOGNIZER_
#define	SUDOKURECOGNIZER_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/highgui.hpp>

using namespace cv;

bool solveSudoku(Mat sudoku, Mat resitev);
bool solveSudokuRek(Mat resitev);
bool findEmptySpace(Mat sudoku, int *x, int *y);
bool isInRow(Mat sudoku, int st, int y);
bool isInCol(Mat sudoku, int st, int x);
bool isInBox(Mat sudoku, int st, int xBox, int yBox);
bool validLoccation(Mat sudoku, int st, int x, int y);

#endif