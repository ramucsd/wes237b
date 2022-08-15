#include <iostream>
#include <stdlib.h>

#include <eigen3/Eigen/Dense>
// Or possibly just
//#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "MatrixMultiply.h"
#include "pmu_counters.h"


using namespace std;

using Eigen::MatrixXf;
using cv::Mat;

int main(int argc, const char* argv[]) {
    
    if (argc != 4)
    {
        cout << "Usage: ./matMult <size> <numTrials> <printFlag>\n";
        return 1;
    }
    
    
    int size = atoi(argv[1]); //size of each dimension of each matrix (assumed to be square)
    int numTrials = atoi(argv[2]);
    int printFlag = atoi(argv[3]);
    
    uint64_t totalCyclesMine = 0;
    uint64_t totalCyclesEigen = 0;
    uint64_t totalCyclesOpenCV = 0;
    
    for (int i = 0; i < numTrials; i++)
    {
        cout << "Trial " << i << endl;
        //initialize matrices
        //constant values for testing
        /*
        size = 3
        float m1 [size][size] = {
            {1., 2., 3.},
            {5., 6., 7.},
            {9., 10., 11.}
        };
        float m2 [size][size] = {
            {13., 14., 15.},
            {17., 18., 19.},
            {21., 22., 23.}
        };*/
        
        //using random values for greater variety in average calculation
        //Using Eigen random library
        MatrixXf em1 = MatrixXf::Random(size,size);
        MatrixXf em2 = MatrixXf::Random(size,size);
        
        //copy matrices to our 2D array
        float m1 [size][size];
        float m2 [size][size];
        
        for(int i = 0; i < size; i++)
        {
                for(int j = 0; j < size; j++)
                {
                        m1[i][j] = em1(i,j);
                        m2[i][j] = em2(i,j);
                }
        }
        
        float result [size][size] = {}; //initialize output matrix
    
        uint32_t startCount = 0;
        uint32_t endCount = 0;
        //initialize PMU counters
        init_counters(1,0);
        startCount = get_cyclecount();
        matMult(&m1[0][0],&m2[0][0],&result[0][0],size); //Multiply using my function
        endCount = get_cyclecount();

        uint32_t numCyclesMine = endCount - startCount;
        cout << "My implementation took " << numCyclesMine << " cycles\n";

        //testing compared to Eigen
        init_counters(1,0); //reset
        startCount = get_cyclecount();
        MatrixXf eigen_result = em1*em2; //Eigen Multiply
        endCount = get_cyclecount();

        uint32_t numCyclesEigen = endCount - startCount;
        cout << "Eigen took " << numCyclesEigen << " cycles\n";



        //testing compared to OpenCV
        Mat OCVMat1 = Mat(size, size, CV_32FC1);
        Mat OCVMat2 = Mat(size, size, CV_32FC1);
        float* ocv1_ptr = OCVMat1.ptr<float>();
        float* ocv2_ptr = OCVMat2.ptr<float>();

         //copy matrices to the OpenCV ones
        for(int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
            {
                ocv1_ptr[i * size + j] = m1[i][j];
                ocv2_ptr[i * size + j] = m2[i][j];
            }
        }

        init_counters(1,0); //reset
        startCount = get_cyclecount();
        Mat OCVResult = OCVMat1*OCVMat2; //OpenCV Multiply
        endCount = get_cyclecount();

        uint32_t numCyclesOpenCV = endCount - startCount;
        cout << "OpenCV took " << numCyclesOpenCV << " cycles\n";

        //print results
        if (printFlag != 0)
        {
            cout << "Result:" << endl;
            bool match = true;
            for(int i = 0; i < size; i++)
            {
                for(int j = 0; j < size; j++)
                {
                        if(result[i][j] != eigen_result(i,j))
                        {
                            match = false;
                        }
                        cout << result[i][j] << " ";
                }
                cout << endl;
            }
            cout << "Eigen Result: " << endl << eigen_result << endl;
            if (match)
            {
                cout << "Matrices match" << endl;
            }
            else
            {
                cout << "ERROR: Matrices do not match" << endl;
            }
            cout << "OpenCV Result: " << endl << OCVResult << endl;
        }
        

        totalCyclesMine += numCyclesMine;
        totalCyclesEigen += numCyclesEigen;
        totalCyclesOpenCV += numCyclesOpenCV;
    }
    
    uint32_t avgCyclesMine = round((double)totalCyclesMine/(double)numTrials);
    uint32_t avgCyclesEigen = round((double)totalCyclesEigen/(double)numTrials);
    uint32_t avgCyclesOpenCV = round((double)totalCyclesOpenCV/(double)numTrials);
    cout << "Averages:\n" << "Mine: " << avgCyclesMine << endl;
    cout << "Eigen: " << avgCyclesEigen << endl;
    cout << "OpenCV: " << avgCyclesOpenCV << endl;
    
    
    return 0;
	
}