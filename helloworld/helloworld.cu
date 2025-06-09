#include"stdio.h"
// defining the function that run on GPU
//__global__ specifier defines that this runs on GPU device 
__global__ void cuda_hello(){
    printf("Hello, This is running in GPU!\n");
}

//host code/ kernels
//can call the the code to run on GPU
int main(){
    // execution is called using <<<..>>>
    // this is called kernel Launch
    cuda_hello<<<1,1>>>();
    return 0;
}