CC = gcc

CFLAG = -Wall -O3 -lpthread

TARGET = pi.out

$(TARGET): clean
	$(CC) pi.c -o $@ $(CFLAG)

clean:
	rm -rf $(TARGET)
