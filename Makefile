EXEC = p1_exec

CC = g++
CFLAGS = -std=c++11 -I.

OBJS = main.o p1_process.o p1_threads.o p1_bubble.o

%.o: %.cpp
	$(CC) -c $< $(CFLAGS)

$(EXEC): $(OBJS)
	$(CC) -o $(EXEC) $(OBJS) -lpthread

.PHONY: test
test: $(EXEC)
	python3 autograder.py

.PHONY: clean
clean:
	rm -rf $(EXEC)
	rm -rf ./*.o
	rm -rf ./output/*

