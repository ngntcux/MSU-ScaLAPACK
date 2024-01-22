#include <cassert>
#include <cmath>
#include <complex>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include "mpi.h"

using namespace std;

extern "C" {
/* Cblacs declarations */
void Cblacs_pinfo(int*, int*);
void Cblacs_get(int, int, int*);
void Cblacs_gridinit(int*, const char*, int, int);
void Cblacs_pcoord(int, int, int*, int*);
void Cblacs_gridexit(int);
void Cblacs_barrier(int, const char*);
void Cdgerv2d(int, int, int, double*, int, int, int);
void Cdgesd2d(int, int, int, double*, int, int, int);
void Czgerv2d(int, int, int, complex<double>*, int, int, int);
void Czgesd2d(int, int, int, complex<double>*, int, int, int);

int numroc_(int*, int*, int*, int*, int*);
void pdgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA,
             double* A, int* IA, int* JA, int* DESCA, double* B, int* IB,
             int* JB, int* DESCB, double* BETA, double* C, int* IC, int* JC,
             int* DESCC);
void pzgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K,
             complex<double>* ALPHA, complex<double>* A, int* IA, int* JA,
             int* DESCA, complex<double>* B, int* IB, int* JB, int* DESCB,
             complex<double>* BETA, complex<double>* C, int* IC, int* JC,
             int* DESCC);
void descinit_(int* idescal, int* m, int* n, int* mb, int* nb, int* dummy1,
               int* dummy2, int* icon, int* procRows, int* info);

void pdelget_(char*, char*, double*, double*, int*, int*, int*);
void pzelget_(char*, char*, complex<double>*, complex<double>*, int*, int*,
              int*);
}

namespace Constants {
int ONE = 1;
int ZERO = 0;
int MAX_SQUARE = 100;
};  // namespace Constants

class Shape {
 public:
  Shape(int first = 0, int second = 0) : first_(first), second_(second) {}

  int operator[](bool second) const { return second ? second_ : first_; }
  int& operator[](bool second) { return second ? second_ : first_; }

  int square() const { return first_ * second_; }

 private:
  int first_;
  int second_;
};

template <typename T>
class Matrix {
 public:
  explicit Matrix() = default;
  Matrix(size_t n, size_t m) : shape(n, m), n_(n), m_(m), data_(n * m, T()) {}
  Matrix(Shape shape)
      : shape(shape),
        n_(shape[0]),
        m_(shape[1]),
        data_(shape[0] * shape[1], T()) {}

  void resize(size_t n, size_t m) {
    data_.resize(n * m);
    n_ = n;
    m_ = m;
    shape = Shape(n, m);
  }
  void resize(Shape new_shape) {
    data_.resize(new_shape[0] * new_shape[1]);
    n_ = new_shape[0];
    m_ = new_shape[1];
    shape = new_shape;
  }

  T& operator()(size_t i, size_t j) { return data_[j * n_ + i]; }
  const T operator()(size_t i, size_t j) const { return data_[j * n_ + i]; }

  T* data() { return data_.data(); }

  Shape shape;

 private:
  size_t n_;
  size_t m_;
  vector<T> data_;
};

template <typename T>
ostream& operator<<(ostream& out, const Matrix<T>& matrix) {
  for (int i = 0; i < matrix.shape[0]; ++i) {
    for (int j = 0; j < matrix.shape[1]; ++j) {
      out << matrix(i, j) << '\t';
    }
    out << '\n';
  }
  return out;
}

template <typename T>
void init_matrix(Matrix<T>& matrix, T lower_bound, T upper_bound) {
  random_device rd;
  mt19937 gen(rd());

  if constexpr (is_same_v<T, double>) {
    uniform_real_distribution<double> distrib(lower_bound, upper_bound);
    for (int i = 0; i < matrix.shape[0]; ++i) {
      for (int j = 0; j < matrix.shape[1]; ++j) {
        matrix(i, j) = distrib(gen);
      }
    }
  } else if constexpr (is_same_v<T, complex<double>>) {
    uniform_real_distribution<double> distrib_real(lower_bound.real(),
                                                   upper_bound.real());
    uniform_real_distribution<double> distrib_imag(lower_bound.imag(),
                                                   upper_bound.imag());
    for (int i = 0; i < matrix.shape[0]; ++i) {
      for (int j = 0; j < matrix.shape[1]; ++j) {
        matrix(i, j) = complex<double>(distrib_real(gen), distrib_imag(gen));
      }
    }
  } else
    throw runtime_error("Check list of possible types.");
}

template <typename T>
void read_matrix(Matrix<T>& matrix, ifstream& in) {
  for (int i = 0; i < matrix.shape[0]; ++i) {
    for (int j = 0; j < matrix.shape[1]; ++j) {
      in >> matrix(i, j);
    }
  }
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b) {
  if (a.shape[1] != b.shape[0]) throw runtime_error("Check matrixes shapes.");

  Matrix<T> c(a.shape[0], b.shape[1]);

  for (int i = 0; i < a.shape[0]; ++i) {
    for (int k = 0; k < a.shape[1]; ++k) {
      for (int j = 0; j < b.shape[1]; ++j) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return c;
}

void define_shapes(Shape& matrix_shape, Shape& requested_block_shape,
                   Shape& real_block_shape, int proc_rows, int proc_cols,
                   int row, int col) {
  requested_block_shape =
      Shape(2, 2);
  if (!requested_block_shape[0]) requested_block_shape[0] = 1;
  if (!requested_block_shape[1]) requested_block_shape[1] = 1;

  real_block_shape = Shape(numroc_(&matrix_shape[0], &requested_block_shape[0],
                                   &row, &Constants::ZERO, &proc_rows),
                           numroc_(&matrix_shape[1], &requested_block_shape[1],
                                   &col, &Constants::ZERO, &proc_cols));
}

template <typename T>
void scatter(Matrix<T>& matrix, const Shape& matrix_shape, Matrix<T>& block,
             const Shape& requested_block_shape, int rank, int context,
             int proc_rows, int proc_cols, int row, int col) {
  int send_row = -1, send_col = -1;
  int recv_row = 0, recv_col = 0;
  for (int block_row = 0; block_row < matrix_shape[0];
       block_row += requested_block_shape[0]) {
    send_row = (send_row + 1) % proc_rows;
    auto block_rows_shape =
        min(requested_block_shape[0], matrix_shape[0] - block_row);

    send_col = -1;
    for (int block_col = 0; block_col < matrix_shape[1];
         block_col += requested_block_shape[1]) {
      send_col = (send_col + 1) % proc_cols;
      auto block_cols_shape =
          min(requested_block_shape[1], matrix_shape[1] - block_col);

      if (!rank) {
        if constexpr (is_same_v<T, double>) {
          Cdgesd2d(context, block_rows_shape, block_cols_shape,
                   matrix.data() + matrix_shape[0] * block_col + block_row,
                   matrix_shape[0], send_row, send_col);
        } else if constexpr (is_same_v<T, complex<double>>) {
          Czgesd2d(context, block_rows_shape, block_cols_shape,
                   matrix.data() + matrix_shape[0] * block_col + block_row,
                   matrix_shape[0], send_row, send_col);
        }
      }

      if (row == send_row && col == send_col) {
        if constexpr (is_same_v<T, double>) {
          Cdgerv2d(context, block_rows_shape, block_cols_shape,
                   block.data() + block.shape[0] * recv_col + recv_row,
                   block.shape[0], 0, 0);
        } else if constexpr (is_same_v<T, complex<double>>) {
          Czgerv2d(context, block_rows_shape, block_cols_shape,
                   block.data() + block.shape[0] * recv_col + recv_row,
                   block.shape[0], 0, 0);
        }
        recv_col = (recv_col + block_cols_shape) % block.shape[1];
      }
    }
    if (row == send_row)
      recv_row = (recv_row + block_rows_shape) % block.shape[0];
  }
}

template <typename T>
void gather(Matrix<T>& matrix, const Shape& matrix_shape, Matrix<T>& block,
            const Shape& requested_block_shape, int rank, int context,
            int proc_rows, int proc_cols, int row, int col) {
  int send_row = -1, send_col = -1;
  int recv_row = 0, recv_col = 0;
  for (int block_row = 0; block_row < matrix_shape[0];
       block_row += requested_block_shape[0]) {
    send_row = (send_row + 1) % proc_rows;
    auto block_rows_shape =
        min(requested_block_shape[0], matrix_shape[0] - block_row);

    send_col = -1;
    for (int block_col = 0; block_col < matrix_shape[1];
         block_col += requested_block_shape[1]) {
      send_col = (send_col + 1) % proc_cols;
      auto block_cols_shape =
          min(requested_block_shape[1], matrix_shape[1] - block_col);

      if (row == send_row && col == send_col) {
        if constexpr (is_same_v<T, double>) {
          Cdgesd2d(context, block_rows_shape, block_cols_shape,
                   block.data() + block.shape[0] * recv_col + recv_row,
                   block.shape[0], 0, 0);
        } else if constexpr (is_same_v<T, complex<double>>) {
          Czgesd2d(context, block_rows_shape, block_cols_shape,
                   block.data() + block.shape[0] * recv_col + recv_row,
                   block.shape[0], 0, 0);
        }
        recv_col = (recv_col + block_cols_shape) % block.shape[1];
      }

      if (!rank) {
        if constexpr (is_same_v<T, double>) {
          Cdgerv2d(context, block_rows_shape, block_cols_shape,
                   matrix.data() + matrix_shape[0] * block_col + block_row,
                   matrix_shape[0], send_row, send_col);
        } else if constexpr (is_same_v<T, complex<double>>) {
          Czgerv2d(context, block_rows_shape, block_cols_shape,
                   matrix.data() + matrix_shape[0] * block_col + block_row,
                   matrix_shape[0], send_row, send_col);
        }
      }
    }
    if (row == send_row)
      recv_row = (recv_row + block_rows_shape) % block.shape[0];
  }
}

template <typename T>
void print(const Matrix<T>& matrix, int rank, int proc_num, int context) {
  for (int cur_rank = 0; cur_rank < proc_num; ++cur_rank) {
    if (cur_rank == rank) {
      cout << matrix << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, proc_num;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  Shape a_shape{stoi(argv[1]), stoi(argv[2])};
  Shape b_shape{stoi(argv[3]), stoi(argv[4])};
  bool show = a_shape.square() < Constants::MAX_SQUARE &&
              b_shape.square() < Constants::MAX_SQUARE;

  Matrix<double> a, b, c;

  if (!rank) {
    a.resize(a_shape);
    b.resize(b_shape);

    init_matrix(a, 1.0, 5.0);
    init_matrix(b, 1.0, 5.0);

    if (show) {
      cout << "Matrix A:\n" << a << '\n';
      cout << "Matrix B:\n" << b << '\n';
    }

    c = a * b;

    if (show) {
      cout << "Matrix C (sequential):\n" << c << '\n';
    }
  }

  int proc_rows = int(sqrt(proc_num));
  int proc_cols = proc_num / proc_rows;
  int myid, context;

  Cblacs_pinfo(&myid, &proc_num);
  Cblacs_get(0, 0, &context);
  Cblacs_gridinit(&context, "Row-major", proc_rows, proc_cols);

  int row, col;
  Cblacs_pcoord(context, myid, &row, &col);

  Shape a_req_shape, a_real_shape;
  Shape b_req_shape, b_real_shape;

  define_shapes(a_shape, a_req_shape, a_real_shape, proc_rows, proc_cols, row,
                col);
  define_shapes(b_shape, b_req_shape, b_real_shape, proc_rows, proc_cols, row,
                col);

  Matrix<double> a_block(a_real_shape), b_block(b_real_shape);
  Shape c_real_shape(
      numroc_(&a_shape[0], &a_req_shape[0], &row, &Constants::ZERO, &proc_rows),
      numroc_(&b_shape[1], &b_req_shape[1], &col, &Constants::ZERO,
              &proc_cols));

  for (int id = 0; id < proc_num; ++id) {
    Cblacs_barrier(context, "All");
  }
  Matrix<double> c_block(c_real_shape);

  scatter(a, a_shape, a_block, a_req_shape, rank, context, proc_rows, proc_cols,
          row, col);
  scatter(b, b_shape, b_block, b_req_shape, rank, context, proc_rows, proc_cols,
          row, col);

  int a_desc[9], b_desc[9], c_desc[9];
  int info;
  descinit_(a_desc, &a_shape[0], &a_shape[1], &a_req_shape[0], &a_req_shape[1],
            &Constants::ZERO, &Constants::ZERO, &context, &a_real_shape[0],
            &info);
  descinit_(b_desc, &b_shape[0], &b_shape[1], &b_req_shape[0], &b_req_shape[1],
            &Constants::ZERO, &Constants::ZERO, &context, &b_real_shape[0],
            &info);
  descinit_(c_desc, &a_shape[0], &b_shape[1], &a_req_shape[0], &b_req_shape[1],
            &Constants::ZERO, &Constants::ZERO, &context, &c_real_shape[0],
            &info);

  double alpha = 1.0;
  double betta = 0.0;
  char N = 'N';
  int one = 1;

  MPI_Barrier(MPI_COMM_WORLD);
  pdgemm_(&N, &N, &a_shape[0], &b_shape[1], &a_shape[1], &alpha, a_block.data(),
          &Constants::ONE, &Constants::ONE, a_desc, b_block.data(),
          &Constants::ONE, &Constants::ONE, b_desc, &betta, c_block.data(),
          &Constants::ONE, &Constants::ONE, c_desc);

  Shape result_shape(a_shape[0], b_shape[1]);
  Matrix<double> result;

  if (!rank) {
    result.resize(result_shape);
  }
  auto gg = c_block;
  gather(result, result_shape, c_block, c_real_shape, rank, context, proc_rows,
         proc_cols, row, col);

  print(c_block, rank, proc_num, context);
  MPI_Barrier(MPI_COMM_WORLD);
  flush(cout);

  char scope = 'A', top = 'I';
  vector<double> aval(min(result_shape[0], result_shape[1]));
  for (int i = 1; i < aval.size() + 1; ++i) {
    pdelget_(&scope, &top, &aval[i - 1], c_block.data(), &i, &i, c_desc);
  }

  if (!rank) {
    cout << "Diag:" << endl;
    for (auto el : aval) {
      cout << el << ' ';
    }
    cout << '\n';
  }

  Cblacs_gridexit(context);
  MPI_Finalize();
}
