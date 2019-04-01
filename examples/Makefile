all: test_dot test_dot_dyn test_cpy test_swap test_axpy test_asum test_max test_nrm2 test_gemv test_gemm

INC_DIR=../include
SRC_DIR =../src
LIB_DIR=../lib

LIBST=-lmnblas
LIBDYN=-lmnblasdyn

OPTIONS_COMPIL  =-Wall -fPIC -I$(INC_DIR) -O2

OPTIONS_LINK_STATIC  =   -L$(LIB_DIR) $(LIBST) -lm
OPTIONS_LINK_DYN  = -L$(LIB_DIR) $(LIBDYN) -lm


flop.o: flop.c flop.h
	gcc $(OPTIONS_COMPIL) -c flop.c

test_cpy.o: test_cpy.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_cpy.c

test_cpy: test_cpy.o $(SRC_DIR)/copy.o $(LIB_DIR)/libmnblas.a
	gcc -o test_cpy test_cpy.o $(SRC_DIR)/copy.o $(OPTIONS_LINK_DYN)

test_swap.o: test_swap.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_swap.c

test_swap: test_swap.o $(SRC_DIR)/swap.o $(LIB_DIR)/libmnblas.a
	gcc -o test_swap test_swap.o $(SRC_DIR)/swap.o $(OPTIONS_LINK_DYN)

test_max.o: test_max.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_max.c

test_max: test_max.o $(SRC_DIR)/amax.o $(LIB_DIR)/libmnblas.a
	gcc -o test_max test_max.o $(SRC_DIR)/amax.o $(OPTIONS_LINK_DYN)

test_dot: test_dot.o flop.o $(LIB_DIR)/libmnblas.a
	gcc -o test_dot test_dot.o flop.o $(OPTIONS_LINK_STATIC)

test_dot_dyn: test_dot.o flop.o
	gcc -o test_dot_dyn flop.o test_dot.o $(OPTIONS_LINK_DYN)

test_dot.o: test_dot.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_dot.c

test_axpy: test_axpy.o flop.o $(LIB_DIR)/libmnblas.a
	gcc -o test_axpy test_axpy.o flop.o $(OPTIONS_LINK_STATIC)

test_axpy_dyn: test_axpy.o flop.o
	gcc -o test_axpy_dyn flop.o test_axpy.o $(OPTIONS_LINK_DYN)

test_axpy.o: test_axpy.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_axpy.c

test_asum: test_asum.o flop.o $(LIB_DIR)/libmnblas.a
	gcc -o test_asum test_asum.o flop.o $(OPTIONS_LINK_STATIC)

test_asum_dyn: test_asum.o flop.o
	gcc -o test_asum_dyn flop.o test_asum.o $(OPTIONS_LINK_DYN)

test_asum.o: test_asum.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_asum.c

test_nrm2: test_nrm2.o flop.o $(LIB_DIR)/libmnblas.a
	gcc -o test_nrm2 test_nrm2.o flop.o $(OPTIONS_LINK_STATIC)

test_nrm2_dyn: test_nrm2.o flop.o
	gcc -o test_nrm2_dyn flop.o test_nrm2.o $(OPTIONS_LINK_DYN)

test_nrm2.o: test_nrm2.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_nrm2.c

test_gemv: test_gemv.o flop.o $(LIB_DIR)/libmnblas.a
	gcc -o test_gemv test_gemv.o flop.o $(OPTIONS_LINK_STATIC)

test_gemv_dyn: test_gemv.o flop.o
	gcc -o test_gemv_dyn flop.o test_gemv.o $(OPTIONS_LINK_DYN)

test_gemv.o: test_gemv.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_gemv.c

test_gemm: test_gemm.o flop.o $(LIB_DIR)/libmnblas.a
	gcc -o test_gemm test_gemm.o flop.o $(OPTIONS_LINK_STATIC)

test_gemm_dyn: test_gemm.o flop.o
	gcc -o test_gemm_dyn flop.o test_gemm.o $(OPTIONS_LINK_DYN)

test_gemm.o: test_gemm.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c test_gemm.c

clean:
	rm -f *.o test_cpy test_cpy_dyn *~
	rm -f *.o test_swap test_swap_dyn *~
	rm -f *.o test_dot test_dot_dyn *~
	rm -f *.o test_axpy test_axpy_dyn *~
	rm -f *.o test_max test_max_dyn *~
	rm -f *.o test_asum test_asum_dyn *~
	rm -f *.o test_nrm2 test_nrm2_dyn *~
	rm -f *.o test_gemv test_gemv_dyn *~
	rm -f *.o test_gemm test_gemm_dyn *~