#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "img_proc.h"
#include "timer.h"

#define OPENCV 0
#define CPU 1
#define GPU 2

#define BLUR_SIZE 5

#define UNIFIED_MEM 

using namespace std;
using namespace cv;

int usage()
{
	cout << "Usage: ./lab4 <mode> <WIDTH> <HEIGHT>" <<endl;
	cout << "mode: 0 OpenCV" << endl;
	cout << "      1 CPU" << endl;
	cout << "      2 GPU" << endl;
	return 0;
}

int use_mode(int mode)
{
	string descr;
	switch(mode)
	{
		case OPENCV:
			descr = "OpenCV Functions";
			break;
		case CPU:
			descr = "CPU Implementations";
			break;
		case GPU:
			descr = "GPU Implementations";
			break;
		default:
			descr = "None";
			return usage();
	}	
	
	cout << "Using " << descr.c_str() <<endl;
	return 1;
}

int main(int argc, const char *argv[]) 
{

	int mode = 0;

	if(argc >= 2)
	{
		mode = atoi(argv[1]);	
	}
	
	if(use_mode(mode) == 0)
		return 0;

	VideoCapture cap("input.raw");

	int WIDTH  = 768;
	int HEIGHT = 768;
	int CHANNELS = 3;

	// 1 argument on command line: WIDTH = HEIGHT = arg
	if(argc >= 3)
	{
		WIDTH = atoi(argv[2]);
		HEIGHT = WIDTH;
	}
	// 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
	if(argc >= 4)
	{
		HEIGHT = atoi(argv[3]);
	}

	// Profiling framerate
	LinuxTimer timer;
	LinuxTimer fps_counter;
	double time_elapsed = 0;

#ifndef UNIFIED_MEM
	uchar *dev_in;
    	uchar *dev_gray;
	uchar *dev_inv;
	uchar *dev_blur;
    	uint size_in = WIDTH*HEIGHT*CHANNELS*sizeof(uchar);
    	uint size_out = WIDTH*HEIGHT*sizeof(uchar);

    	cudaMalloc((void**)&dev_in, size_in);
    	cudaMalloc((void**)&dev_gray, size_out);
	cudaMalloc((void**)&dev_inv, size_out);
	cudaMalloc((void**)&dev_blur, size_out);

	Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3);
	Mat gray = Mat(HEIGHT, WIDTH, CV_8U);
	Mat inverted = Mat(HEIGHT, WIDTH, CV_8U);
	Mat blurred = Mat(HEIGHT, WIDTH, CV_8U); 
#else
	uchar *dev_in;
    	uchar *dev_gray;
	uchar *dev_inv;
	uchar *dev_blur;
	uint size_in = WIDTH*HEIGHT*CHANNELS*sizeof(uchar);
    	uint size_out = WIDTH*HEIGHT*sizeof(uchar);

	cudaMallocManaged(&dev_in, size_in);
	cudaMallocManaged(&dev_gray, size_out);
	cudaMallocManaged(&dev_inv, size_out);
	cudaMallocManaged(&dev_blur, size_out);

	Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3, dev_in);
	Mat gray = Mat(HEIGHT, WIDTH, CV_8U, dev_gray);
	Mat inverted = Mat(HEIGHT, WIDTH, CV_8U, dev_inv);
	Mat blurred = Mat(HEIGHT, WIDTH, CV_8U, dev_blur); 
#endif


	//Matrix for OpenCV inversion
	Mat ones = Mat::ones(HEIGHT, WIDTH, CV_8U)*255;

	Mat frame;	
	char key=0;
	int count = 0;

	while (key != 'q')
	{
		cap >> frame;
		if(frame.empty())
		{
			waitKey();
			break;
		}

		resize(frame, rgb, Size(WIDTH, HEIGHT));

		imshow("Original", rgb);

		timer.start();
		switch(mode)
		{
			case OPENCV:
#ifdef OPENCV4
				cvtColor(rgb, gray, COLOR_BGR2GRAY);
#else
				cvtColor(rgb, gray, CV_BGR2GRAY);
#endif
				break;
			case CPU:
				img_rgb2gray_cpu(gray.ptr<uchar>(),rgb.ptr<uchar>(),WIDTH,HEIGHT,CHANNELS);
				img_invert_cpu(inverted.ptr<uchar>(),gray.ptr<uchar>(),WIDTH,HEIGHT);
				img_blur_cpu(blurred.ptr<uchar>(),gray.ptr<uchar>(),WIDTH,HEIGHT,BLUR_SIZE);
				break;

			case GPU:

#ifndef UNIFIED_MEM
    				cudaMemcpy(dev_in, rgb.ptr<uchar>(), size_in, cudaMemcpyHostToDevice);
				img_rgb2gray(dev_gray,dev_in,WIDTH,HEIGHT,CHANNELS);

				//don't need to copy dev_gray yet if doing other operations

				img_invert(dev_inv,dev_gray,WIDTH,HEIGHT);
				img_blur(dev_blur,dev_gray,WIDTH,HEIGHT,BLUR_SIZE);

				cudaMemcpy(gray.ptr<uchar>(), dev_gray, size_out, cudaMemcpyDeviceToHost);
				cudaMemcpy(inverted.ptr<uchar>(), dev_inv, size_out, cudaMemcpyDeviceToHost);
				cudaMemcpy(blurred.ptr<uchar>(), dev_blur, size_out, cudaMemcpyDeviceToHost);
#else
				img_rgb2gray(dev_gray,dev_in,WIDTH,HEIGHT,CHANNELS);
				img_invert(dev_inv,dev_gray,WIDTH,HEIGHT);
				img_blur(dev_blur,dev_gray,WIDTH,HEIGHT,BLUR_SIZE);
#endif
				break;
		}
		timer.stop();

		size_t time_rgb2gray = timer.getElapsed();
		
		count++;
		time_elapsed += (timer.getElapsed())/10000000000.0;

		if (count % 10 == 0)
		{
			cout << "Execution Time (s) = " << time_elapsed << endl;
			time_elapsed = 0;
		}

		imshow("Gray", gray);
		imshow("Inverted", inverted);
		imshow("Blurred", blurred);

		key = waitKey(1);
	}
	cudaFree(dev_in);
	cudaFree(dev_gray);
	cudaFree(dev_inv);
	cudaFree(dev_blur);
}
