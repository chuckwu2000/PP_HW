SRC_DIR = src
OBJ_DIR = obj

CC = gcc

CFLAG = -Wall -I include -g -O3
LDFLAG =

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst %.c, $(OBJ_DIR)/%.o, $(notdir $(SRC)))
TARGET = pi.out
PROFILE = gmon.out

.PHONY: all

all: Create_OBJ_DIR $(TARGET)

Create_OBJ_DIR:
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAG) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $(LDFLAG) $(OBJ) -o $@ 

run: all
	./$(TARGET)

time: all
	time ./$(TARGET)

profile: run
	gprof ./$(TARGET) $(PROFILE) -b > $(PROFILE_TARGET)

perf: all
	sudo perf record -e cpu-cycles ./$(TARGET)
	sudo perf report

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
