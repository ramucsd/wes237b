#include "main.h"
#include "timer.h"
#include "pmu_counters.h"

#define FRAME_NUMBER 5 //set to 0 or -1 to run while loop

using namespace std;
using namespace cv;

int main(int argc, const char * argv[])
{
    
	unsigned int c_start;
	unsigned int opencv_c, student_c;

	cout << "WES237B lab 2" << endl;

	VideoCapture cap("input.raw");

	Mat frame, gray, dct_org, dct_student, diff_img;
	char key=0;
	float mse;
	int fps_cnt = 0;

	int WIDTH  = 64;
	int HEIGHT = 64;
    
    int algo = 0;
    
    // 1st argument - DCT algorithm to use
    //0 = Naive
    //1 = Separable
    //2 = Matrix Multiplication
    //3 = Block Matrix Multiplication
    if(argc >= 2)
	{
		algo = atoi(argv[1]);
	}


	// 2 argument on command line: WIDTH = HEIGHT = arg
	if(argc >= 3)
	{
		WIDTH = atoi(argv[2]);
		HEIGHT = WIDTH;
	}
	// 3 arguments on command line: WIDTH = arg1, HEIGHT = arg2
	if(argc >= 4)
	{
		HEIGHT = atoi(argv[3]);
	}

	initDCT(WIDTH, HEIGHT);

	float avg_perf = 0.f;
	int count = 0;
    
    uint64_t totalCycles = 0;

#if FRAME_NUMBER <= 0
	while(key != 'q')
#else
    for(int f = 0; f < FRAME_NUMBER; f++)
#endif
	{
		cap >> frame;
		if(frame.empty()){ break; }

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		resize(gray, gray, Size(WIDTH, HEIGHT));
		gray.convertTo(gray, CV_32FC1);

		// OpenCV DCT
		dct(gray, dct_org);

		// Your DCT
        float myTimer;
        uint32_t startCount = 0;
        uint32_t endCount = 0;
        //initialize PMU counters

        endCount = get_cyclecount();
        switch (algo)
        {
            case 0: //Naive
                {
                    LinuxTimer t;
                    init_counters(1,0);
                    startCount = get_cyclecount();
                    dct_student = student_dct_naive(gray);
                    endCount = get_cyclecount();
                    t.stop();
                    myTimer = t.getElapsed();
                    break;
                }
            case 1: //separable
                {
                    LinuxTimer t;
                    init_counters(1,0);
                    startCount = get_cyclecount();
                    dct_student = student_dct_opt(gray);
                    endCount = get_cyclecount();
                    t.stop();
                    myTimer = t.getElapsed();
                    break;
                }
            case 2: //Matrix Multiply
                {
                    LinuxTimer t;
                    init_counters(1,0);
                    startCount = get_cyclecount();
                    dct_student = student_dct_MM(gray);
                    endCount = get_cyclecount();
                    t.stop();
                    myTimer = t.getElapsed();
                    break;
                }
            case 3:
                {
                    LinuxTimer t;
                    init_counters(1,0);
                    startCount = get_cyclecount();
                    dct_student = student_dct_BMM(gray);
                    endCount = get_cyclecount();
                    t.stop();
                    myTimer = t.getElapsed();
                    break;
                }
            default:
                {
                    LinuxTimer t;
                    init_counters(1,0);
                    startCount = get_cyclecount();
                    dct_student = student_dct_naive(gray);
                    endCount = get_cyclecount();
                    t.stop();
                    myTimer = t.getElapsed();
                }
        }
        
        uint32_t numCycles = endCount - startCount;
        totalCycles += numCycles;


		gray.convertTo(gray, CV_8UC1);

		absdiff(dct_org, dct_student, diff_img); 

		/* calculating RMSE */
		diff_img = diff_img.mul(diff_img);
		Scalar se = sum(diff_img);
		mse = se[0]/((float)HEIGHT*WIDTH);

		count++;

		cout <<  "Execute time: "
			<< (double)myTimer/1000000000.0 << endl;
        cout <<  "Cycle count: " << numCycles << endl;
		printf("RMSE: %.4f\n", sqrt(mse));

		Mat inverse;
		idct(dct_student, inverse);
		inverse.convertTo(inverse, CV_8U);

#ifndef __arm__
		imshow("Original", gray);
		imshow("IDCT Output", inverse);
		moveWindow("IDCT Output", 500, 0);
		key = waitKey(10);
#endif
	}
    
    uint32_t avgCycles = round((double)totalCycles/(double)FRAME_NUMBER);
    cout << "Average number of cycles: " << avgCycles << endl;

	return 0;
}




