SRC_DIR = src
OBJ_DIR = obj

CC = gcc

CFLAG = -Wall -O3 -lpthread

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst %.c, $(OBJ_DIR)/%.o, $(notdir $(SRC)))
TARGET = pi.out

.PHONY: all

all: Create_OBJ_DIR $(TARGET)

Create_OBJ_DIR:
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) -c $< -o $@ $(CFLAG) 

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ 

run: all
	./$(TARGET) 8 100000000

time: all
	time ./$(TARGET) 8 100000000

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
