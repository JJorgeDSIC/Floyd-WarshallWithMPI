/*
 * programa.c
 * @author Pedro Alonso
 * @version 1.0
 * @date 2017/12/12
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>

#define min(a,b) ((a)<(b)?(a):(b))

void rutina( const int n_nodos, const float *A, float *D );

/*
 * MAIN: Programa principal
 */

int main(int argc, char *argv[]) {

  int rank, size;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  //MPI_Status     status;

	/* COMPROBAR ARGUMENTOS */
	if ( argc < 2 ) {
		if( !rank ) fprintf( stderr, "\nUso: %s <numero_de_nodos> \n\n", argv[0] );
                MPI_Finalize();
		return 0;
	}
  int n_nodos;
  sscanf(argv[1],"%d",&n_nodos);

  float *A = NULL, *D = NULL;
  /* Si soy el proceso 0 genero los datos */
  if( ! rank ) {
    /* Generacion de la matriz de adyacencia */
    if ( ( A = (float *) malloc( n_nodos*n_nodos*sizeof(float) ) ) == NULL ) {
      fprintf( stderr, "Error: Imposible reservar memoria para %.0f GB\n", (int) n_nodos*n_nodos*sizeof(float)/1.0e9 );
      MPI_Finalize();
      return 0;
    }

    int i, j;
    for( i=0; i<n_nodos; i++ ) {
      for( j=0; j<n_nodos; j++ ) {
        if( i==j ) 
          A[i*n_nodos+j] = 0.0f;
        else 
          A[i*n_nodos+j] = (100.0f * ( (float) rand() / RAND_MAX )) + 1;
      }
    }

    /* Generacion de la matriz de distancias */
	  if( ( D = (float *) malloc( n_nodos*n_nodos*sizeof(float) ) ) == NULL ) {
      fprintf( stderr, "Error: Imposible reservar %.0f GB para la matriz D \n", (int) n_nodos*n_nodos*sizeof(float)/1.0e9 );
      return 0;
	  }
  }

  //MPI_Bcast( &A, n_nodos*n_nodos, MPI_FLOAT, 0, commcol); 
  double inicio = 0.0;
  if( ! rank ) {
	  inicio = MPI_Wtime();
  }
  /*
   * FUNCION: Algoritmo de Floyd para encontrar el coste del camino mínimo entre 
   * cualquier par de vértices. 
   * La version facilitada a continuación es secuencial, es decir, solo funciona para un proceso. 
   */
  int i, j, k;

  double sqrResult = sqrt(size);
  int value = (int)(sqrResult*10);
  int dim1, dim2;

  
    //printf("Es una raiz exacta");
  if(value % 10 != 0){
    if(size == 6){
      dim1 = 2;
      dim2 = 3;
    }else if(size == 36){
      dim1 = 1;
      dim2 = 32;
    }else{
	return -1;
    }
    //valor = dim1 + dim2
    if(DEBUG){
      if(!rank){
        printf("D1 %d D2 %d\n", dim1, dim2);
        printf("No Es una raiz exacta, value=%d, sqrt=%f\n", value, sqrResult);
      }
    }
  }else{
    dim1 = (int)sqrResult;
    dim2 = (int)sqrResult;
  }

  int chunk1 = n_nodos/dim1;
  int chunk2 = n_nodos/dim2;

  //if(!rank){
  //  printf("chunk1=%d\n", chunk1);
  //  printf("chunk2=%d\n", chunk2);
  //}

  MPI_Comm comm2D;
  int ndims, reorder, periods[2], dim_size[2];
  int id2D, coords2D[2];
  ndims = 2;

  dim_size[0] = dim1;
  dim_size[1] = dim2;

  periods[0] = 0;
  periods[1] = 0;
  reorder = 1;
  /********************************/
  /* Creación del grid cartesiano */
  /********************************/
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dim_size, periods, reorder, &comm2D);

  /*************************************************/
  /* División en comunicadores de filas y columnas */
  /*************************************************/
  MPI_Comm commrow, commcol;

  int belongs[2];
  /* Create 2D Cartesian topology for processes */
  MPI_Comm_rank(comm2D, &id2D);
  MPI_Cart_coords(comm2D, id2D, ndims, coords2D);
  /* Create 1D row subgrids */
  belongs[0] = 0;
  belongs[1] = 1;     // this dimension belongs to subgrid
  MPI_Cart_sub(comm2D, belongs, &commrow);
  /* Create 1D column subgrids */
  belongs[0] = 1;      /* this dimension belongs to subgrid */
  belongs[1] = 0;
  MPI_Cart_sub(comm2D, belongs, &commcol);


  float *Dloc = (float *) malloc( chunk1*chunk2*sizeof(float));
  float *DKloc = (float *) malloc( chunk1*chunk2*sizeof(float));
  int p;
  int coordsDist[2];

  if(!rank){

    //Relleno la parte de la matriz D local al procesador P_N, N \in [1,...,Num_procs-1]
    for(p = 1; p < size; p++){
      //(obtengo Cord. Cart => P_N => P_X_Y)
      MPI_Cart_coords(comm2D, p, 2, coordsDist);
      //Relleno su matriz
      for(i = 0; i < chunk1; i++){
        for(j = 0; j < chunk2; j++){
          DKloc[i*chunk2+j] = A[(i+chunk1*coordsDist[0])*n_nodos+(j+chunk2*coordsDist[1])];
        }
      }
      //Y se la envio
      MPI_Send(DKloc, chunk1*chunk2, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
    }

    //Cuando acabo con todos, relleno la del root (P_0)
    MPI_Cart_coords(comm2D, 0, 2, coordsDist);
    //printf("P(%d,%d)\n", coordsDist[0], coordsDist[1]);
    for(i = 0; i < chunk1; i++){
        for(j = 0; j < chunk2; j++){
            DKloc[i*chunk2+j] = A[(i+chunk1*coordsDist[0])*n_nodos+(j+chunk2*coordsDist[1])];
        }
    }

    if(DEBUG){
      printf("A=%d\n", rank);
      for( i=0; i<n_nodos; i++ ) {
        for( j=0; j<n_nodos; j++ ) {
          printf("%.f ", A[i*n_nodos+j]);
        }
        printf("\n");
      }
    }

  }else{
    //Recibo mi trozo
    MPI_Recv(DKloc, chunk1*chunk2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
    
  }

  if(DEBUG){
    //Imprime cada uno su trozo (sanity check)
    printf("Proc=%d\n", rank);
      for( i=0; i<chunk1; i++ ) {
        for( j=0; j<chunk2; j++ ) {
          printf("%.f ", DKloc[i*chunk2+j]);
        }
        printf("\n");
    }
  }

  float localDkRow[chunk2];
  float localDkCol[chunk1];
    
  //Itero sobre las k etapas, tantas como nodos
  for( k=0; k<n_nodos; k++ ) {

    //Si soy un proceso que tiene esta fila 
    //(la primera coordenada es igual a la division entera k/chunk)
    //i.e: P_1_3 tiene la fila k=5 si el chunk es de 3? 5/3 = 1 SI, 
    if(k/chunk1 == coords2D[0]){
      for( j=0; j<chunk2; j++ ) {
        localDkRow[j] = DKloc[(k%chunk1) * chunk2 + j];
      }
      //printf("I'm ROW root=%d\n", id2D);
    }

    //each process Pi,j that has a segment of the k_th row broadcast it to the P*,j proc
    MPI_Bcast( &localDkRow, chunk2, MPI_FLOAT, k/chunk1, commcol); 

    //Si soy un proceso que tiene esta columna 
    //(la segunda coordenada es igual a la division entera k/chunk)
    //i.e: P_1_3 tiene la columna k=5 si el chunk es de 3? 5/3 = 1 NO
    //i.e: Y P_2_1? SI!  
    if(k/chunk2 == coords2D[1]){
      for( i=0; i<chunk1; i++ ) {
        localDkCol[i]= DKloc[i * chunk2 + (k%chunk2)];
      }
      //printf("I'm COL root=%d\n", id2D);
    }
    //each process Pi,j that has a segment of the k_th row broadcast it to the P*,j proc
    MPI_Bcast( &localDkCol, chunk1, MPI_FLOAT, k/chunk2, commrow); 

    //Local computation on each block individually
    //INCLUDE OPENMP HERE
    int n_th;
    #pragma omp parallel for private(j)
    for( i=0; i<chunk1; i++ ) {
      for( j=0; j<chunk2; j++ ) {
        Dloc[i*chunk2+j] = min( DKloc[i*chunk2+j], localDkCol[i] + localDkRow[j]);
      }
    }

    if( k < n_nodos-1 ) {
      #pragma omp parallel for private(j)
      for( i=0; i<chunk1; i++ ) {
        for( j=0; j<chunk2; j++ ) {
          DKloc[i*chunk2+j] = Dloc[i*chunk2+j];
        }
      }
    }
  }

  if(DEBUG){
    printf("P(sol)=%d\n", rank);
      for( i=0; i<chunk1; i++ ) {
        for( j=0; j<chunk2; j++ ) {
          printf("%.f ", DKloc[i*chunk2+j]);
        }
        printf("\n");
    }
  }

  //Gathering the results...
 int Bi,Bj;
 int coordRank;
 int inner_i, inner_j;

 
 if(!rank){
    //Bloque a bloque voy recogiendo
    for (Bi = 0; Bi < dim1; ++Bi)
    {

      for (Bj = 0; Bj < dim2; ++Bj){

        if(Bi == 0 && Bj == 0){
          //Myself
          for( i=0; i<chunk1; i++ ) {
             for( j=0; j<chunk2; j++ ) {
               inner_i = i + Bi * chunk1;
               inner_j = j + Bj * chunk2;
               D[inner_i * n_nodos + inner_j] = DKloc[i*chunk2+j];
               //printf("%.f ", DKloc[i*chunk+j]);
             }
             //printf("\n");
          }
          

        }else{

          coordsDist[0] = Bi;
          coordsDist[1] = Bj;
          MPI_Cart_rank(comm2D, coordsDist, &coordRank);
          //MPI_Recv(&num, 1, MPI_FLOAT, coordRank, 0, comm2D,  MPI_STATUS_IGNORE);
          //printf("coordRank=%d, with coord=[%d,%d] sent=%.f\n", coordRank, coordsDist[0], coordsDist[1], num);
          
          MPI_Recv(DKloc, chunk1*chunk2, MPI_FLOAT, coordRank, 0, comm2D,  MPI_STATUS_IGNORE);

          for( i=0; i<chunk1; i++ ) {
             for( j=0; j<chunk2; j++ ) {
               inner_i = i + Bi * chunk1;
               inner_j = j + Bj * chunk2;
               D[inner_i * n_nodos + inner_j] = DKloc[i*chunk2+j];
             }
          }

        }
      }
    }

  }else{
    MPI_Send(DKloc, chunk1*chunk2, MPI_FLOAT, 0, 0, comm2D);
  }
 
  if(!rank){
    for( i=0; i<n_nodos; i++ ) {
      for( j=0; j<n_nodos; j++ ) {
        printf("%.f ", D[i*n_nodos+j]);
      }
      printf("\n");
    }
  }
  


  free(Dloc);
  free(DKloc);
  
  if( ! rank ) {
    double fin = MPI_Wtime();
    printf("process ended\n");
    printf("%d\t%.5f\n",n_nodos, (fin-inicio));
    if( A != NULL ) free(A);
    if( D != NULL ) free(D);
  }

  

  MPI_Finalize();
  return 0;
}
