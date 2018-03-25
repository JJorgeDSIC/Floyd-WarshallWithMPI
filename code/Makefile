
BASELINE = floyd
FLOYD = mpi_floyd

# Optimization & debug flags:
# DEBUG: mensajes de depuracion.
CFLAGS  =  -O3 -Wall -DDEBUG=0 -fopenmp
#CFLAGS  =  -Wall -DDEBUG=0 -fopenmp

CC = mpicc

all: $(BASELINE) $(FLOYD)

$(BASELINE): $(BASELINE).o Makefile 
	$(CC) -o $(BASELINE) $(BASELINE).o $(CFLAGS) 

$(BASELINE).o: $(BASELINE).c Makefile 
	$(CC) -c $(BASELINE).c $(CFLAGS) 


$(FLOYD): $(FLOYD).o Makefile 
	$(CC) -o $(FLOYD) $(FLOYD).o $(CFLAGS) 

$(FLOYD).o: $(FLOYD).c Makefile 
	$(CC) -c $(FLOYD).c $(CFLAGS) 


clean:
	- rm -f *.o core $(BASELINE) $(FLOYD) *~



