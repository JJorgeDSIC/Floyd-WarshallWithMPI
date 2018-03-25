/*
 * programa.c
 * @author Pedro Alonso
 * @version 1.0
 * @date 2017/12/12
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

	double inicio = MPI_Wtime();
  /*
   * FUNCION: Algoritmo de Floyd para encontrar el coste del camino mínimo entre 
   * cualquier par de vértices. 
   * La version facilitada a continuación es secuencial, es decir, solo funciona para un proceso. 
   */
  int i, j, k;

  /* Generacion de una matriz auxiliar */
	float *Dk = (float *) malloc( n_nodos*n_nodos*sizeof(float) );
	if ( Dk == NULL ) {
		fprintf( stderr, "Error: Imposible reservar %.0f GB para la matriz Dk \n", (int) n_nodos*n_nodos*sizeof(float)/1.0e9 );
		exit( EXIT_FAILURE );
	}
  #pragma omp parallel for private(j)
  for( i=0; i<n_nodos; i++ ) {
    for( j=0; j<n_nodos; j++ ) {
      Dk[i*n_nodos+j] = A[i*n_nodos+j];
    }
  }

  
  for( k=0; k<n_nodos; k++ ) {
    #pragma omp parallel for private(j)
    for( i=0; i<n_nodos; i++ ) {
      for( j=0; j<n_nodos; j++ ) {
        D[i*n_nodos+j] = min( Dk[i*n_nodos+j], Dk[i*n_nodos+k] + Dk[k*n_nodos+j] );
      }
    }
    if( k < n_nodos-1 ) {
      #pragma omp parallel for private(j)
      for( i=0; i<n_nodos; i++ ) {
        for( j=0; j<n_nodos; j++ ) {
          Dk[i*n_nodos+j] = D[i*n_nodos+j];
        }
      }
    }
  }

  //printf("Solucion\n");
  for( i=0; i<n_nodos; i++ ) {
      for( j=0; j<n_nodos; j++ ) {
        printf("%.f ", Dk[i*n_nodos+j]);
      }
      printf("\n");
  }

  free(Dk);
	printf("process ended\n");
	double fin = MPI_Wtime();
  printf("%d\t%.5f\n",n_nodos, (fin-inicio));

  if( ! rank ) {
    if( A != NULL ) free(A);
    if( D != NULL ) free(D);
  }

  MPI_Finalize();
  return 0;
}

