build: main.cu
	nvcc main.cu -o main -rdc=true

run: build
	./main