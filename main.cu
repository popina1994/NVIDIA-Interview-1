#include <stdio.h>
#include <limits.h>
#include <time.h>


#ifndef IMPLEM
#define IMPLEM 2
#endif
const int IMPL = IMPLEM;

#ifndef X_BLOCK
#define X_BLOCK 1024
#endif
#ifndef Y_GRID
#define Y_GRID 68
#endif
      

// dimx is a number of columns
// dimy is a number of rows
bool checkResults (float *gold, float *d_data, int dimx, int dimy, float rel_tol) {
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


__global__ void kernel_A1(float *g_data, const int dimx, const int dimy, const int niterations, const int xInc, const int yInc)
{
	int idx;
	float value;
	for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += yInc)
	{
		for (int ix = threadIdx.x * 4; ix < dimx; ix += xInc)
		{
			idx = iy * dimx + ix;
			value = g_data[idx];

			for (int i = 0; i < niterations; i++)
			{
				value += sqrtf(logf(value) + 1.f);
			}
			g_data[idx] = value;
		}
	}
}




__global__ void kernel_A2(float *g_data, const int dimx, const int dimy, const int niterations, const int xInc, const int yInc)
{
	int idx;
	float value;
	for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += yInc)
	{
		for (int ix = threadIdx.x * 4 + 1; ix < dimx; ix += xInc)
		
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

__global__ void kernel_A3(float *g_data, const int dimx, const int dimy, const int niterations, const int xInc, const int yInc)
{
	int idx;
	float value;
	for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += yInc)
	{
		for (int ix = threadIdx.x * 4 + 2; ix < dimx; ix += xInc)
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
	

__global__ void kernel_A4(float *g_data, const int dimx, const int dimy, const int niterations, const int xInc, const int yInc)
{
	int idx;
	float value;
	
	for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += yInc)
	{
		for (int ix = threadIdx.x * 4 + 3; ix < dimx; ix += xInc)
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


__global__ void kernel_AJoin(float *g_data, const int dimx, const int dimy, const int niterations, const int xInc, const int yInc, 
		const int numSms, const int numXBlocks)
{
	dim3 block(numXBlocks, 1);
	dim3 grid(1, numSms);
	kernel_A1<<<grid, block>>>(g_data, dimx, dimy, niterations, xInc, yInc);
	kernel_A2<<<grid, block>>>(g_data, dimx, dimy, niterations, xInc, yInc);
	kernel_A3<<<grid, block>>>(g_data, dimx, dimy, niterations, xInc, yInc);
	kernel_A4<<<grid, block>>>(g_data, dimx, dimy, niterations, xInc, yInc);
}


__global__ void kernel_A(float *g_data, const int dimx, const int dimy, const int niterations)
{
	int idx;
	float value;
	for (int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < dimx; ix += blockDim.x * gridDim.x)
	{
		if (ix % 4 == 0)
		{
			for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += blockDim.y * gridDim.y)
			{
				idx = iy * dimx + ix;
				value = g_data[idx];

				for (int i = 0; i < niterations; i++)
				{
					value += sqrtf(logf(value) + 1.f);
				}
				g_data[idx] = value;
			}
		}
		else if (ix % 4 == 1)
		{
			for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += blockDim.y * gridDim.y)
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
		else if (ix % 4 == 2)
		{
			for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += blockDim.y * gridDim.y)
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
		else if (ix % 4 == 3)
		{
			for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += blockDim.y * gridDim.y)
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
}


void launchKernel(float * d_data, int dimx, int dimy, int niterations)
{
	// Only change the contents of this function and the kernel(s). You may
	// change the kernel's function signature as you see fit.
	//query number of SMs
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int num_sms = prop.multiProcessorCount;
	/* This implementation prevents thread divergences, thus the speed up is equivalent to the number of 
	    different possible thread outcomes evaluated. However, it does not exploit L1 cache. 
	   */
	if (IMPL == 0)
	{
		dim3 block(1, 128);
		dim3 grid(1, num_sms);
		kernel_A<<<grid, block>>>(d_data, dimx, dimy, niterations);
	}
	/* This implemenetaion tries to exploit L1 cache that loads blocks of continious memory locations 
	  in L1 cachde and that are accessed by the threads from the same block. I broke this in 4 kernel calls to  reduces thread divergence. 
	  The size of the block for x is the biggest possible to exploit the cache as much as posssible. I noticed the trend as
	  I increased block size thta the runtime decreased. The reasoning is that there are only 5 register used by threads, 
	  thus for each block everything can fit L1 of one SM
	  I also tried to exploit the property that the memory is row-major order by iterating first over xs and then over y.
	  with four consecutive kernes, this prevents memory being dirtied by adjacents blocks.
	  Later I improved this to chaning 4 grids that evaluate different code depending on the grid number
	  (all threads in the same block will evaluate the same ocde
	  I changed the signature of the function to exploit the constant cache since these variables can be computed only once.
	   */
	else if (IMPL == 1)
	{
		int xBlock = X_BLOCK;
		//int yGrid = Y_GRID;
		int yGrid = num_sms;
		dim3 block(xBlock, 1);
		dim3 grid(1, num_sms);

		kernel_A1<<<grid, block>>>(d_data, dimx, dimy, niterations, block.x * 4, block.y * grid.y);
		kernel_A2<<<grid, block>>>(d_data, dimx, dimy, niterations, block.x * 4, block.y * grid.y);
		kernel_A3<<<grid, block>>>(d_data, dimx, dimy, niterations, block.x * 4, block.y * grid.y);
		kernel_A4<<<grid, block>>>(d_data, dimx, dimy, niterations, block.x * 4, block.y * grid.y);
	
	}
	/* Thi sis a unified call of all 4 kernels from one parent kernel*/
	else if (IMPL == 2)
	{
		int xBlock = X_BLOCK;
		dim3 blockDummy(1, 1);
		dim3 gridDummy(1, 1);
		dim3 block(xBlock, 1);
		dim3 grid(1, num_sms);

		kernel_AJoin<<<gridDummy, blockDummy>>>(d_data, dimx, dimy, niterations, block.x * 4, block.y * grid.y, num_sms, xBlock);
	}

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
