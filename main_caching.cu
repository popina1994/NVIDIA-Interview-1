#include <stdio.h>
#include <limits.h>
#include <time.h>
// dimx is a number of columns
// dimy is a number of rowssdfsfd
bool checkResults (float *gold, float *d_data, int dimx, int dimy, float rel_tol) {
	for (int iy = 0; iy < 4; ++iy)
	{
		for (int ix = 0; ix < 4; ++ix)
		{
			int idx = iy * dimx + ix;
      printf("(%.4f %.4f)", d_data[idx], gold[idx]);
		}
    printf("\n");

	}
	for (int iy = 0; iy < dimy; ++iy) 
  {
		for (int ix = 0; ix < dimx; ++ix) 
    {
			int idx = iy * dimx + ix;

			float gdata = gold[idx];
			float ddata = d_data[idx];

			if (isnan(gdata) || isnan(ddata)) 
      {
				printf("Nan detected: gold %f, device %f\n", gdata, ddata);
				return false;
			}

			float rdiff;
			if (fabs(gdata) == 0.f)
				rdiff = fabs(ddata);
			else
				rdiff = fabs(gdata - ddata) / fabs(gdata);

			if (rdiff > rel_tol) 
      {
				printf("Error solutions don't match at iy=%d, ix=%d.\n", iy, ix);
				printf("gold: %f, device: %f\n", gdata, ddata);
				printf("rdiff: %f\n", rdiff);
				return false;
			}
		}
	}
	return true;
}

void computeCpuResults(float *g_data, int dimx, int dimy, int niterations, int nreps) 
{
	for (int r = 0; r < nreps; r++) 
  {
		printf("Rep: %d\n", r);
#pragma omp parallel for
		for (int iy = 0; iy < dimy; ++iy) 
    {
			for (int ix = 0; ix < dimx; ++ix)
      {
				int idx = iy * dimx + ix;

				float value = g_data[idx];

				for (int i = 0; i < niterations; i++) 
        {
					if (ix % 4 == 0) 
          {
						value += sqrtf(logf(value) + 1.f);
					} else if (ix % 4 == 1) 
          {
						value += sqrtf(cosf(value) + 1.f);
					} else if (ix % 4 == 2) 
          {
						value += sqrtf(sinf(value) + 1.f);
					} else if (ix % 4 == 3) 
          {
						value += sqrtf(tanf(value) + 1.f);
					}
				}
				g_data[idx] = value;
			}
		}
	}
}
#define DEBUG_P 0

#if DEBUG_P
#define PRINT_X() printf("PX: BIDY: %d TIDY: %d BIDX: %d TIDX: %d\n", blockIdx.y, threadIdx.y, blockIdx.x, threadIdx.x);
#else
#define PRINT_X()
#endif 
__global__ void kernel_A(float *g_data, const int dimx, const int dimy, const int niterations) 
{
  //printf("blockDim.y %d blockIdx.y %d threadIdx.y %d \n", blockDim.y, blockIdx.y,  threadIdx.y);
  //printf("blockDim.x %d blockIdx.y %d threadIdx.x %d \n", blockDim.x, blockIdx.x, threadIdx.x);
  int idx; 
  float value;
  int chunkY = (dimy + blockDim.y * gridDim.y - 1) / (blockDim.y * gridDim.y);
  int chunkX = (dimx + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);

  int gridYStart = chunkY * (blockIdx.y * blockDim.y + threadIdx.y);
  int gridYEnd = chunkY * (blockIdx.y * blockDim.y + threadIdx.y + 1);
  int gridXStart = chunkX * (blockIdx.x * blockDim.x + threadIdx.x);
  int gridXEnd = chunkX * (blockIdx.x * blockDim.x + threadIdx.x +  1);
 // printf("Chunk size %d %d %d \n", chunk, blockDim.y * gridDim.y, dimy); 
  //printf("S: %d E: %d Y: %d\n", gridYStart, gridYEnd, threadIdx.y);
  gridYEnd = min(dimy, gridYEnd);
  gridXEnd = min(dimx, gridXEnd);
  if (threadIdx.x  == 0)
  {
    //printf("%d %d %d BIDX: %d TIDX: %d BIDXY: %d TIDXY: %d\n", gridYStart, gridYEnd, chunk, blockIdx.x, threadIdx.x, blockIdx.y, threadIdx.y);
  }
  if (threadIdx.x  == 0)
  {
    for (int ix = gridXStart; ix < gridXEnd; ix+=4)
    {
      for (int iy = gridYStart; iy < gridYEnd ; iy ++)
      {
        idx = iy * dimx + ix;
        value = g_data[idx];

        for (int i = 0; i < niterations; i++)
        {
          value += sqrtf(logf(value) + 1.f);
        }
        g_data[idx] = value;
        if ((iy == 512) && (ix == 0))
        {
          PRINT_X();
          //printf("START: %d END: %d\n", gridYStart, gridYEnd);
        }
      }
    }
  }
  else if (threadIdx.x == 1)
  {
    for (int ix = gridXStart + 1; ix < gridXEnd; ix+=4)
    {
      for (int iy = gridYStart; iy < gridYEnd; iy++)
      {
        idx = iy * dimx + ix;
        value = g_data[idx];

        for (int i = 0; i < niterations; i++)
        {
          value += sqrtf(cosf(value) + 1.f);
        }
        g_data[idx] = value;
      }
    }
  }
  else if (threadIdx.x ==  2)
  {
    for (int ix = gridXStart + 2; ix < gridXEnd; ix+=4)
    {
      for (int iy = gridYStart; iy < gridYEnd; iy++)
      {
        idx = iy * dimx + ix;
        value = g_data[idx];

        for (int i = 0; i < niterations; i++)
        {
          value += sqrtf(sinf(value) + 1.f);
        }
        g_data[idx] = value;
      }
    }
  }
  else if (threadIdx.x == 3)
  {
    for (int ix = gridXStart+3; ix < gridXEnd; ix += 4)
    {
      for (int iy = gridYStart; iy < gridYEnd; iy++)
      {
        idx = iy * dimx + ix;
        value = g_data[idx];

        for (int i = 0; i < niterations; i++)
        {
          value += sqrtf(tanf(value) + 1.f);
        }
        g_data[idx] = value;
      }
    }
  }

    /*
    for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += blockDim.y * gridDim.y) 
    {
			int idx = iy * dimx + ix;
      //printf("%d\n", idx);
			float value = g_data[idx];

			for (int i = 0; i < niterations; i++) 
      {
				if (ix % 4 == 0) 
        {
					value += sqrtf(logf(value) + 1.f);
				} else if (ix % 4 == 1) 
        {
					value += sqrtf(cosf(value) + 1.f);
				} else if (ix % 4 == 2) 
        {
					value += sqrtf(sinf(value) + 1.f);
				} else if (ix % 4 == 3) 
        {
					value += sqrtf(tanf(value) + 1.f);
				}
			}
			g_data[idx] = value;
		}
    */
}


void launchKernel(float * d_data, int dimx, int dimy, int niterations) 
{
	// Only change the contents of this function and the kernel(s). You may
	// change the kernel's function signature as you see fit.

	//query number of SMs
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int num_sms = prop.multiProcessorCount;

	dim3 block(4, 8);
	dim3 grid(4, num_sms/4);
	kernel_A<<<grid, block>>>(d_data, dimx, dimy, niterations);
}

float timing_experiment(float *d_data, int dimx, int dimy, int niterations, int nreps) 
{
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++) 
  {
		launchKernel(d_data, dimx, dimy, niterations);
	}
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= nreps;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsed_time_ms;
}

int main() {
	//int dimx = 8 * 1024;
	//int dimy = 8 * 1024;
	int dimx = 8 * 1024;
	int dimy = 8 * 1024;

	int nreps = 10;
	int niterations = 5;

	int nbytes = dimx * dimy * sizeof(float);

	float *d_data = 0, *h_data = 0, *h_gold = 0;
	cudaMalloc((void **)&d_data, nbytes);
	if (0 == d_data)
  {
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
	printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));
	h_data = (float *)malloc(nbytes);
	h_gold = (float *)malloc(nbytes);
	if (0 == h_data || 0 == h_gold) 
  {
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", 2.0f * nbytes / (1024.f * 1024.f));
	for (int i = 0; i < dimx * dimy; i++)
    h_gold[i] = 1.0f + 0.01 * (float)rand() / (float)RAND_MAX;
	cudaMemcpy(d_data, h_gold, nbytes, cudaMemcpyHostToDevice);

	float runtime_cuda = timing_experiment(d_data, dimx, dimy, niterations, 1);
  printf("Cuda runtime %.3f\n", runtime_cuda);
	printf("Verifying solution\n");

	cudaMemcpy(h_data, d_data, nbytes, cudaMemcpyDeviceToHost);

	float rel_tol = .001;
	computeCpuResults(h_gold, dimx, dimy, niterations, 1);

	bool pass = checkResults(h_gold, h_data, dimx, dimy, rel_tol);

	if (pass) 
  {
		printf("Results are correct\n");
	} 
  else 
  {
		printf("FAIL: results are incorrect\n");
	}

	float elapsed_time_ms = 0.0f;

	elapsed_time_ms = timing_experiment(d_data, dimx, dimy, niterations,
			nreps);
	printf("A: %8.2f ms\n", elapsed_time_ms);

	printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

	if (d_data) cudaFree(d_data);
	if (h_data) free(h_data);

	cudaDeviceReset();

	return 0;
}
