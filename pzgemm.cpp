#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "mkl_blacs.h"
#include "mkl_pblas.h"
#include "mkl_scalapack.h"
#include "mpi.h"

using namespace std;

namespace {
const int MainProcessID = 0;
}

template <typename T>
class FortranMatrix {
 public:
  explicit FortranMatrix() = default;
  FortranMatrix(size_t rows, size_t columns)
      : rows_(rows), columns_(columns), matrix_(rows_ * columns_) {}
  size_t index(size_t i, size_t j) const { return j * rows_ + i; }
  T& operator()(size_t i, size_t j) { return matrix_[this->index(i, j)]; }
  const T operator()(size_t i, size_t j) const {
    return matrix_[this->index(i, j)];
  }
  T* data() { return matrix_.data(); }
  const T* data() const { return matrix_.data(); }
  void print() const {
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < columns_; ++j) {
        auto element = matrix_[index(i, j)];
        cout << fixed << setprecision(1) << "(" << element.real << ",\t"
             << element.imag << ")"
             << "\t";
      }
      cout << endl;
    }
    cout << endl;
  }
  size_t rows() const { return rows_; }
  size_t columns() const { return columns_; }

 private:
  size_t rows_;
  size_t columns_;
  vector<T> matrix_;
};

FortranMatrix<MKL_Complex16> get_complex_random_matrix(size_t n, size_t m,
                                                       double mean,
                                                       double std) {
  FortranMatrix<MKL_Complex16> A(n, m);

  random_device rd;
  mt19937 gen(rd());
  std::normal_distribution<double> distribution(mean, std);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      MKL_Complex16 element;
      element.real = distribution(gen);
      element.imag = distribution(gen);
      A(i, j) = element;
    }
  }

  return A;
}

MKL_Complex16 to_mkl_complex(complex<double> x) {
  MKL_Complex16 ans;
  ans.real = x.real();
  ans.imag = x.imag();
  return ans;
}

complex<double> to_complex(MKL_Complex16 x) {
  complex<double> ans;
  ans.real(x.real);
  ans.imag(x.imag);
  return ans;
}

FortranMatrix<MKL_Complex16> multiply_matrices_sequentially(
    const FortranMatrix<MKL_Complex16>& A,
    const FortranMatrix<MKL_Complex16>& B) {
  size_t A_rows = A.rows();
  size_t A_cols = A.columns();
  size_t B_cols = B.columns();

  FortranMatrix<MKL_Complex16> C(A_rows, B_cols);

  for (size_t i = 0; i < A_rows; i++) {
    for (size_t j = 0; j < B_cols; j++) {
      complex<double> result(0, 0);
      for (size_t k = 0; k < A_cols; k++) {
        result += to_complex(A(i, k)) * to_complex(B(k, j));
      }
      C(i, j) = to_mkl_complex(result);
    }
  }

  return C;
}

void send_complex_block(int nr, int nc, int r, int c,
                        const FortranMatrix<MKL_Complex16>& A, int target_id) {
  FortranMatrix<MKL_Complex16> localA(nr, nc);

  for (size_t i = r; i < r + nr; i++) {
    for (size_t j = c; j < c + nc; j++) {
      localA(i - r, j - c) = A(i, j);
    }
  }

  MPI_Send(localA.data(), nr * nc, MPI_DOUBLE_COMPLEX, target_id, 0,
           MPI_COMM_WORLD);
}

void receive_complex_block(int nr, int nc, int offset,
                           FortranMatrix<MKL_Complex16>& localA,
                           int source_id) {
  FortranMatrix<MKL_Complex16> tmp(nr, nc);

  int row_offset = offset % localA.rows();
  int col_offset = offset / localA.rows();

  MPI_Recv(tmp.data(), nr * nc, MPI_DOUBLE_COMPLEX, source_id, MPI_ANY_TAG,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (size_t i = 0; i < nr; i++) {
    for (size_t j = 0; j < nc; j++) {
      localA(row_offset + i, col_offset + j) = tmp(i, j);
    }
  }
}

FortranMatrix<MKL_Complex16> scatter_blacs_complex_matrix(
    const FortranMatrix<MKL_Complex16>& A, int& N, int& M, int& NB, int& MB,
    int& nrows, int& ncols, int ctxt, MPI_Comm comm, int NB_FORCE = 0,
    int MB_FORCE = 0) {
  int iZERO = 0;
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  int myrow, mycol;
  int proc_rows, proc_cols;
  blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);

  int root_row, root_col;
  int bcast_data[4];
  if (my_rank == MainProcessID) {
    bcast_data[0] = A.rows();
    bcast_data[1] = A.columns();
    bcast_data[2] = myrow;
    bcast_data[3] = mycol;
  }

  MPI_Bcast(&bcast_data, 4, MPI_INT, MainProcessID, MPI_COMM_WORLD);

  N = bcast_data[0];
  M = bcast_data[1];
  root_row = bcast_data[2];
  root_col = bcast_data[3];
  NB = N / proc_rows;
  MB = M / proc_cols;

  if (NB == 0) {
    NB = 1;
  }
  if (MB == 0) {
    MB = 1;
  }

  nrows = numroc_(&N, &NB, &myrow, &iZERO, &proc_rows);
  ncols = numroc_(&M, &MB, &mycol, &iZERO, &proc_cols);

  FortranMatrix<MKL_Complex16> localA(nrows, ncols);

  int sendr = 0, sendc = 0, recvr = 0, recvc = 0;
  for (int r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
    sendc = 0;

    int nr = NB;
    if (N - r < NB) {
      nr = N - r;
    }

    for (int c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
      int nc = MB;

      if (M - c < MB) {
        nc = M - c;
      }

      if (my_rank == MainProcessID) {
        int send_id = blacs_pnum(&ctxt, &sendr, &sendc);
        send_complex_block(nr, nc, r, c, A, send_id);
      }

      if (myrow == sendr && mycol == sendc) {
        receive_complex_block(nr, nc, nrows * recvc + recvr, localA,
                              MainProcessID);
        recvc = (recvc + nc) % ncols;
      }
    }

    if (myrow == sendr) {
      recvr = (recvr + nr) % nrows;
    }
  }

  return localA;
}

void initialize_blacs_grid(int& ctxt) {
  char order = 'R';
  int IZERO = 0;
  int IMINUS = -1;
  int communicator_size;
  MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);

  int proc_rows = sqrt(communicator_size),
      proc_cols = communicator_size / proc_rows;

  blacs_get(&IMINUS, &IZERO, &ctxt);
  blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
}

template <typename T>
void print_local_matrix(const FortranMatrix<T>& A, MPI_Comm comm) {
  int myid, numproc;
  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &numproc);

  for (int id = 0; id < numproc; ++id) {
    if (id == myid) {
      cout << "ID: " << myid << ". Local matrix:" << endl;

      A.print();
      flush(cout);
      cout << endl;
    }

    MPI_Barrier(comm);
  }
}

void gather_blacs_complex_matrix(const FortranMatrix<MKL_Complex16>& localC,
                                 FortranMatrix<MKL_Complex16>& C, int N, int M,
                                 int NB, int MB, int nrows, int ncols,
                                 int ctxt) {
  int iZERO = 0;
  int my_rank, communicator_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);

  int myrow, mycol;
  int proc_rows, proc_cols;
  blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);

  int root_row, root_col;
  int bcast_data[2];
  if (my_rank == MainProcessID) {
    bcast_data[0] = myrow;
    bcast_data[1] = mycol;
  }

  MPI_Bcast(&bcast_data, 2, MPI_INT, MainProcessID, MPI_COMM_WORLD);

  root_row = bcast_data[0];
  root_col = bcast_data[1];

  int localC_LD = localC.rows();

  int sendr = 0, sendc = 0, recvr = 0, recvc = 0;
  for (int r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
    sendc = 0;
    int nr = NB;
    if (N - r < NB) {
      nr = N - r;
    }

    for (int c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
      int nc = MB;
      if (M - c < MB) {
        nc = M - c;
      }

      if (myrow == sendr && mycol == sendc) {
        send_complex_block(nr, nc, recvr, recvc, localC, MainProcessID);
        recvc = (recvc + nc) % ncols;
      }

      if (my_rank == MainProcessID) {
        int source_id = blacs_pnum(&ctxt, &sendr, &sendc);
        receive_complex_block(nr, nc, N * c + r, C, source_id);
      }
    }

    if (myrow == sendr) {
      recvr = (recvr + nr) % nrows;
    }
  }
}

int main(int argc, char** argv) {
  int my_rank, communicator_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);

  if (argc != 5) {
    if (my_rank == 0) {
      cout << "Usage: " << argv[0] << " A_rows A_cols B_rows B_cols" << endl;
    }
    return 1;
  }

  int A_rows = atoi(argv[1]);
  int A_cols = atoi(argv[2]);
  int B_rows = atoi(argv[3]);
  int B_cols = atoi(argv[4]);

  FortranMatrix<MKL_Complex16> A, B, C, x;
  if (my_rank == 0) {
    A = get_complex_random_matrix(A_rows, A_cols, 0, 3);
    B = get_complex_random_matrix(B_rows, B_cols, 0, 3);
    cout << "Matrix A:" << endl;
    A.print();
    cout << "Matrix B:" << endl;
    B.print();
    C = multiply_matrices_sequentially(A, B);
    cout << "Matrix C (sequential):" << endl;
    C.print();
  }

  int iZERO = 0;
  int ctxt, myrow, mycol, proc_rows, proc_cols;
  initialize_blacs_grid(ctxt);
  blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);
  int NB_A, MB_A, NB_B, MB_B;
  int nrows_A, ncols_A, nrows_B, ncols_B;

  auto localA = scatter_blacs_complex_matrix(
      A, A_rows, A_cols, NB_A, MB_A, nrows_A, ncols_A, ctxt, MPI_COMM_WORLD);
  auto localB = scatter_blacs_complex_matrix(
      B, B_rows, B_cols, NB_B, MB_B, nrows_B, ncols_B, ctxt, MPI_COMM_WORLD);

  auto nrows_C = numroc_(&A_rows, &NB_A, &myrow, &iZERO, &proc_rows);
  auto ncols_C = numroc_(&B_cols, &MB_B, &mycol, &iZERO, &proc_cols);

  FortranMatrix<MKL_Complex16> localC(nrows_C, ncols_C);

  int LLD_A = nrows_A;
  int LLD_B = nrows_B;
  int LLD_C = nrows_C;

  int rsrc = 0, csrc = 0, info;
  int* desca = new int[9];
  int* descb = new int[9];
  int* descc = new int[9];

  descinit_(desca, &A_rows, &A_cols, &NB_A, &MB_A, &rsrc, &csrc, &ctxt, &LLD_A,
            &info);
  if (info != 0) {
    cout << "descinit error (A): " << my_rank << " " << info << endl;
  }

  descinit_(descb, &B_rows, &B_cols, &NB_B, &MB_B, &rsrc, &csrc, &ctxt, &LLD_B,
            &info);
  if (info != 0) {
    cout << "descinit error (B): " << my_rank << " " << info << endl;
  }

  descinit_(descc, &A_rows, &B_cols, &NB_A, &MB_B, &rsrc, &csrc, &ctxt, &LLD_C,
            &info);
  if (info != 0) {
    cout << "descinit error (C): " << my_rank << " " << info << endl;
  }

  char N = 'N';
  int iONE = 1;
  MKL_Complex16 alpha;
  MKL_Complex16 betta;
  alpha.real = 1.0;
  alpha.imag = 0.0;
  betta.real = 1.0;
  betta.imag = 0.0;

  pzgemm_(&N, &N, &A_rows, &B_cols, &A_cols, &alpha, localA.data(), &iONE,
          &iONE, desca, localB.data(), &iONE, &iONE, descb, &betta,
          localC.data(), &iONE, &iONE, descc);

  print_local_matrix(localC, MPI_COMM_WORLD);

  if (my_rank == 0) {
    C = FortranMatrix<MKL_Complex16>(A_rows, B_cols);
  }

  gather_blacs_complex_matrix(localC, C, A_rows, B_cols, NB_A, MB_B, nrows_C,
                              ncols_C, ctxt);
  delete[] desca;
  delete[] descb;
  delete[] descc;
  blacs_gridexit(&ctxt);

  if (my_rank == 0) {
    cout << "Matrix C (pdgemm):" << endl;
    C.print();
  }

  MPI_Finalize();
  return 0;
}
