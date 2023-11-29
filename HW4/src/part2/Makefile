CC = mpicxx

FLAGS = -O3 -Wall

SRC = $(wildcard *.cc)
OBJ = $(patsubst %.cc, %.o, $(SRC))
TARGET = matmul

%.o: %.cc
	$(CC) $(FLAG) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $^ -o $@

all: $(TARGET)

clean:
	rm matmul *.o
