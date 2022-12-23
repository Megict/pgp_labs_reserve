#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>

#define silent false 	//no stderr except actual errors
#define verbal true		//print every component of every pixel before and after applying classification
#define visual false	//print avg of each pixel components in grid with img sides
#define debug  false	//do printf in kernel

#define INDEX_ERROR 800

#define EPS 1e-11

#define CSC(call)  																											\
do {																														\
	cudaError_t err = call;																									\
	if (err != cudaSuccess) {																								\
		std::cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << ". Message: " << cudaGetErrorString(err) << "\n";		\
		exit(0);																											\
	}																														\
} while(0)

//вариант 6
//Нахождение ранга матрицы

__global__ void kernel_swap_rows(double* elements, int row_1, int row_2, int n, int m) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;

	while(idx < m) {
		double swp = elements[idx*n + row_1];
		elements[idx*n + row_1] = elements[idx*n + row_2];
		elements[idx*n + row_2] = swp;
		
		if (debug) printf("KERNEL: swaping %lf and %lf at idx %d\n",elements[idx*n + row_1], elements[idx*n + row_2], idx);

		idx += offset_x;
	}
}

__global__ void kernel_gaussian_step(double* elements, int n, int m, int start_row_index, int active_colomn, double* result) {
	//n - количество строк (элементов в столбце)
	//m - количество столбцов (элементов в строке)

	int cur_row = blockIdx.x;
	int cur_col = threadIdx.x; 
	int in_row_offset = blockDim.x; //если закончились нити блока, а строка не закончилась
	int other_row_offset = gridDim.x; //если закончились блоки, а строки еще есть

	while(cur_row < n) {

		__shared__ double coef[32];

		//заполение разделяемой памяти
		if(cur_col < 32) { //первый варп приносится в жертву во имя производительности остальных
			if (cur_row > start_row_index) { //этот if не делит варпы, так что все норм
				coef[threadIdx.x] = - elements[active_colomn*n + cur_row] / elements[active_colomn*n + start_row_index];
			}
			else {
				coef[threadIdx.x] = 0; //нужно, чтобы потоки вообще не запускались для таких случаев 
			}
		}
		__syncthreads();
		//по сути остальные варпы все равно ждут, когда первый варп закончит запись, но, с другой стороны, когда строка больше количества потоков в блоке новые вычисления уже не требуются


		while(cur_col < m) {

			result[cur_col*n + cur_row] = elements[cur_col*n + cur_row] + coef[threadIdx.x % 32]*elements[cur_col*n + start_row_index];
			if (debug) printf("KERNEL: cur_row = %d cur_col = %d element = %lf coef = %lf\n",cur_row,cur_col,elements[cur_col*n + cur_row],coef[threadIdx.x % 32]);
			cur_col += in_row_offset;
		}
		
		if (debug) printf("KERNEL: to proceed? cur_row is %d, offset is %d, limit is %d\n",cur_row,other_row_offset,n);
		cur_row += other_row_offset;
		cur_col = threadIdx.x;
	}
}

bool close_to_zero(double val) {
	if(val < EPS && val > -EPS) {
		return true;
	}
	return false;
}

struct Compare {
    __host__ __device__ bool operator()(double num1, double num2) {
        return fabs(num1) < fabs(num2);
    }
};

int find_max_elm(double* array,int array_size) {
	//находит максимальный элемент массива и возвращает его индекс

	Compare cmp;

	double* device_array;
	CSC(cudaMalloc (&device_array, sizeof(double)*array_size));
	CSC(cudaMemcpy (device_array, array, sizeof(double)*array_size, cudaMemcpyHostToDevice));

	thrust::device_ptr<double> device_ptr = thrust::device_pointer_cast(device_array);
	
    int max_elm_pos = thrust::max_element(device_ptr, device_ptr + array_size, cmp) - device_ptr;

	CSC(cudaFree(device_array));

	printf("max element pos is %d\n",max_elm_pos);
	
	return max_elm_pos;
}

class matrix{
	int n;
	int m;
	double* array;

public:
	matrix(int n_, int m_, double* array_) {
		n = n_;
		m = m_;
		array = array_;
	}

	matrix(int n_, int m_){ //считывание матрицы с stdin
		n = n_; m = m_;
		//n - количество строк (элементов в столбце)
		//m - количество столбцов (элементов в строке)
		double* arr_all = (double*)malloc(sizeof(double)*m*n);

		for (int i = 0; i < n; ++i){ //проход по строкам
			for (int j = 0; j < m; ++j){ //проход по столбцам
				double elm = 0;
				std::cin >> elm;
				arr_all[j*n + i] = elm;
			}
		}

		array = arr_all;
	}

	void print() {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				std::cout << array[i*n + j] << " ";
			}
			std::cout << "\n---\n";
		}
	}

	void printf() {
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				//std::cout.precision(10);
				if(array[i*n + j] >= 0) {
					std::printf(" ");
				}
				std::printf("%.2lf ",array[i*n + j]);
				//std::cout << array[i*n + j] << " ";
			}
			std::printf("\n");
		}
	}

	void swap_rows(int lhs,int rhs) {
		//меняет местами строки lhs и rhs
		//должен работать на мультипроцессоре
		if(lhs >= n || rhs >= n) {
			throw INDEX_ERROR;
		}

		if(lhs == rhs) {
			return;
		}

		double* device_matrix;
		CSC(cudaMalloc (&device_matrix, sizeof(double)*m*n));
		CSC(cudaMemcpy (device_matrix, array, sizeof(double)*m*n, cudaMemcpyHostToDevice));

		kernel_swap_rows<<<512,512>>>(device_matrix, lhs, rhs, n, m);

		CSC(cudaMemcpy (array, device_matrix, sizeof(double)*m*n, cudaMemcpyDeviceToHost));
		
		CSC(cudaFree(device_matrix));
	}

	//1 - определить ведущий элемент в столбце i (thrust)
	//2 - переставить строки местами (О(m)) (параллельно, одномерной сеткой)
	//3 - вычислить коэфециенты для каждой строки (O(n)) ДЕЛАЕТСЯ ВНУТРИ ЯДРА
	//4 - записать коэфициенты в разд. память. 
	//		Каждый блок работает с одной из строк, ему нужен только один коэфициент
	//		Первый варп каждого блока считает коэфициент для соответствующей строки, следующие вары извлекают его из разделяемой памяти
	//		У каждого блока должен быть поток-лидер, который помещает нужный элемент в разделяемую память, остальные потоки должны начать работу только после завершения перемещения
	//		варп потока-лидера будет работать неоптимально
	//5 - преобразовать строки (O(n*n)) (параллельно, двумерной сеткой)

	int rank() {
		int rank = -1;
		int active_element_idx = 0;

		for (int i = 0; i < n - 1; ++i) { //i - текущая строка.
			//надо определить активную колонку для текущего шага
			//если первая колонка вся в нулях, то надо работать со следующей

			bool zero_col = true;
			for (int probe_idx = active_element_idx; probe_idx < m; ++probe_idx) {
				for (int idx_in_col = i; idx_in_col < n; ++idx_in_col) { //проверка, есть ли ненулевые элементы в рассматриваемой колонке. Если есть, то она будет основной.
					if(verbal) std::cerr  << array[probe_idx*n + idx_in_col] << " ";
					if(!close_to_zero(array[probe_idx*n + idx_in_col])) {
						zero_col = false;
						break;
					}
				}
				if(verbal) std::cerr << "\n";

				if(zero_col) {
					continue;
				}
				else {
					if(verbal) std::cerr << "prev active colomn: " << active_element_idx << " new active colomn: " << probe_idx << "\n";
					active_element_idx = probe_idx;
					break;
				}
			}

			if (zero_col) { //все следующие столбцы имеют нулевую активную часть
				rank = i;
				break;
			}



			int max_elm_idx = find_max_elm(&array[active_element_idx*n + i],n - i) + i;

			if(verbal) std::cerr << "swp\n" << "\tcur index " << i << "\n\tindex with max elm " << max_elm_idx <<"\n";

			swap_rows(i,max_elm_idx);
						
			double* device_matrix;
			CSC(cudaMalloc (&device_matrix, sizeof(double)*m*n));
			CSC(cudaMemcpy (device_matrix, array, sizeof(double)*m*n, cudaMemcpyHostToDevice));
			
			double* device_result;
			CSC(cudaMalloc (&device_result, sizeof(double)*m*n));


			kernel_gaussian_step<<<512,512>>> (device_matrix, n, m, i, active_element_idx, device_result);


			CSC(cudaMemcpy (array, device_result, sizeof(double)*m*n, cudaMemcpyDeviceToHost));
			
			if(verbal) std::cerr << "after transformation\n";
			if(verbal) printf();

			cudaFree(device_matrix);
			cudaFree(device_result);

			
		}

		if(rank == -1) {
			rank = n;
		}
		
		return rank;

	}

};

int main() {
	
	try{ 
		int n,m;
		std::cin >> n >> m;

		matrix matr(n,m);

		if(!silent) {
			std::cerr << "--\n";
			matr.print();
			std::cerr << "\n";
			matr.printf();
			std::cerr << "\n";
		}

		int rank = matr.rank();

		if(!silent) {
			std::cerr << "-- RANK: " << rank << " --\n";
		}
		std::cout << rank << "\n";
	}
	catch(int err) {
		if (err == 101) {
			std::cerr << "error opening file\n";
		} else
		if (err == 105){
			std::cerr << "error new length\n";
		} else 
		if (err == 800) {
			std::cerr << "error index\n";
		} else{
			std::cerr << "unknown error detected\n";
		}
	}

	return 0;
}


