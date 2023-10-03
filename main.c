#include <stdio.h>
#include <mpi.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#define N_SIZE 9000
#define TAU 0.00016
#define EPS 1e-9
#define MAX_NUM_ITERATIONS 10000


#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif

typedef struct range_s Range;
struct range_s {
	int begin_idx;
	int end_idx;
};

void sub_matrix(double* res, const double* A, const double* B, const size_t N, const size_t M) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < M; ++j) {
			res[i * N + j] = A[i * N + j] - B[i * N + j];
		}
	}
}

void mult_matrix_scalar(double* res, const double* A, const double scalar, const size_t N, const size_t M) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < M; ++j) {
			res[i * N + j] = A[i * N + j] * scalar;
		}
	}
}

void mult_matrix_to_vector(double* res, const double* matrix, const double* vector, const size_t row, const size_t column) {
	for (size_t i = 0; i < row; ++i) {
		res[i] = 0.0;
		const double* t = matrix + i * column;
		for (size_t j = 0; j < column; ++j) {
			res[i] += t[j] * vector[j];
		}
	}
}

double calc_metric(const double* matrix, const size_t N) {
	double res = 0;
	for (size_t i = 0; i < N; ++i) {
		res += matrix[i] * matrix[i];
	}
	return sqrt(res);
}

double calc_metric_sqr(const double* matrix, const size_t N) {
	double res = 0;
	for (size_t i = 0; i < N; ++i) {
		res += matrix[i] * matrix[i];
	}
	return res;
}

double calc_error(const double* A, const double* b, const double* x, const size_t N) {
	double* Ax = (double*)malloc(N * sizeof(double));
	mult_matrix_to_vector(Ax, A, x, N, N);
	double* sub = (double*)malloc(N * sizeof(double));
	sub_matrix(sub, Ax, b, 1, N);
	double err = calc_metric(sub, N) / calc_metric(b, N);
	free(Ax);
	free(sub);
	return err;
}

int main(int argc, char** argv) {
	double begin, end;
	MPI_Init(&argc, &argv);
	begin = MPI_Wtime();
	int num_process;
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double* A = NULL; //Исходная матрица
	double* b = (double*)malloc(N_SIZE * sizeof(double)); //Вектор правой части
	double* x_t = (double*)malloc(N_SIZE * sizeof(double)); //Текущее решение
	double* x_0 = (double*)malloc(N_SIZE * sizeof(double)); //Начальное приближение
	double* A_p = NULL; //Кусок исходной матрицы
	double* res = NULL; //Итоговый вектор решения
	Range range;

	int* m_arr = NULL;
	Range* ranges = NULL;
	int* displs = NULL;

	if (rank == 0) {
		res = (double*)malloc(N_SIZE * sizeof(double));
		//Заполнение матриц
		A = (double*)malloc(N_SIZE * N_SIZE * sizeof(double));
		for (size_t i = 0; i < N_SIZE; ++i) {
			for (size_t j = 0; j < N_SIZE; ++j) {
				if (i == j) {
					A[i * N_SIZE + j] = 2.0;
				}
				else {
					A[i * N_SIZE + j] = 1.0;
				}
			}
		}
		for (size_t i = 0; i < N_SIZE; ++i) {
			b[i] = N_SIZE + 1;
		}
		for (size_t i = 0; i < N_SIZE; ++i) {
			x_0[i] = -1 * 1e10;
		}
		//Распределение строк матрицы по процессам
		m_arr = (int*)malloc(num_process * sizeof(int));
		for (int i = 0; i < num_process; ++i) {
			m_arr[i] = N_SIZE / num_process;
		}
		for (int i = 0; i < N_SIZE % num_process; ++i) {
			m_arr[i]++;
		}
		for (int i = 0; i < num_process; ++i) {
			m_arr[i] *= N_SIZE;
		}
		ranges = (Range*)malloc(num_process * sizeof(Range));
		ranges[0].begin_idx = 0;
		ranges[0].end_idx = m_arr[0];
		for (int i = 1; i < num_process; ++i) {
			ranges[i].begin_idx = ranges[i - 1].end_idx;
			ranges[i].end_idx = ranges[i].begin_idx + m_arr[i];
		}
		range = ranges[0];
		displs = (int*)malloc(num_process * sizeof(int));
		for (int i = 0; i < num_process; ++i) {
			displs[i] = ranges[i].begin_idx;
		}
	}
	//Отправка границ процессам
	MPI_Scatter(ranges, sizeof(Range), MPI_UNSIGNED_CHAR, &range, sizeof(Range), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	free(ranges);
	//Отправка кусков матрицы
	A_p = (double*)malloc((range.end_idx - range.begin_idx) * sizeof(double));
	MPI_Scatterv(A, m_arr, displs, MPI_DOUBLE, A_p, range.end_idx - range.begin_idx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//Отправка вектора
	MPI_Bcast(b, N_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//Отправка начального приближения
	MPI_Bcast(x_0, N_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	memcpy(x_t, x_0, N_SIZE * sizeof(double));

	//Итерационный решала
	size_t numIterations = 0;
	double* Ax_p = (double*)malloc(
			(range.end_idx - range.begin_idx) / N_SIZE * sizeof(double)); //Промежуточный результат
	double* sub1_p = (double*)malloc((range.end_idx - range.begin_idx) / N_SIZE * sizeof(double));
	double* tauSub1_p = (double*)malloc((range.end_idx - range.begin_idx) / N_SIZE * sizeof(double));
	double* res_p = (double*)malloc((range.end_idx - range.begin_idx) / N_SIZE * sizeof(double));
	double metric_x_p;
	double metric_x;
	double metric_b = calc_metric(b, N_SIZE);
	if (rank == 0) {
		for (int i = 0; i < num_process; ++i) {
			m_arr[i] /= N_SIZE;
			displs[i] /= N_SIZE;
		}
	}
	double err = 1e15;
	for (; (numIterations < MAX_NUM_ITERATIONS) && (err > EPS); ++numIterations) {
		metric_x_p = 0;
		metric_x = 0;
		mult_matrix_to_vector(Ax_p, A_p, x_t, (range.end_idx - range.begin_idx) / N_SIZE, N_SIZE);
		for (size_t i = 0; i < (range.end_idx - range.begin_idx) / N_SIZE; ++i) {
			sub1_p[i] = Ax_p[i] - b[i + (range.begin_idx / N_SIZE)];
			tauSub1_p[i] = sub1_p[i] * TAU;
			res_p[i] = x_t[i + (range.begin_idx / N_SIZE)] - tauSub1_p[i];
		}
		MPI_Gatherv(res_p, (range.end_idx - range.begin_idx) /
		                   N_SIZE, MPI_DOUBLE, res, m_arr, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (rank == 0) {
			memcpy(x_t, res, N_SIZE * sizeof(double));
		}
		MPI_Bcast(x_t, N_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		mult_matrix_to_vector(Ax_p, A_p, x_t, (range.end_idx - range.begin_idx) / N_SIZE, N_SIZE);
		for (size_t i = 0; i < (range.end_idx - range.begin_idx) / N_SIZE; ++i) {
			sub1_p[i] = Ax_p[i] - b[i + (range.begin_idx / N_SIZE)];
		}
		metric_x_p = calc_metric_sqr(sub1_p, (range.end_idx - range.begin_idx) / N_SIZE);
		MPI_Reduce(&metric_x_p, &metric_x, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0) {
			metric_x = sqrt(metric_x);
			err = metric_x / metric_b;
			printf("err = %.15f\n", err);
		}
		MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	end = MPI_Wtime();

	if (rank == 0) {
		printf("Iterations = %zu, Time = %f seconds\n", numIterations, end - begin);
		if (N_SIZE <= 20) {
			for (size_t i = 0; i < N_SIZE; ++i) {
				printf("x[%zu] = %f\n", i, res[i]);
			}
		}
		else {
			printf("x[0] = %f\n", res[0]);
		}
	}


	free(A);
	free(b);
	free(x_t);
	free(x_0);
	free(A_p);
	free(res);
	free(m_arr);
	free(displs);
	free(Ax_p);
	free(sub1_p);
	free(tauSub1_p);
	free(res_p);
	MPI_Finalize();
	return 0;
}
