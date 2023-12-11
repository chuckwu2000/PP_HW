NVCC = nvcc
CXX=g++
CXXFLAGS=-I./common -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm

APP_NAME1=mandelbrot1
APP_NAME2=mandelbrot2
APP_NAME3=mandelbrot3
APP_NAME4=mandelbrot4
OBJDIR=objs
COMMONDIR=./common

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

PPM_CXX=$(COMMONDIR)/ppm.cpp
PPM_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))


default: $(APP_NAME1) $(APP_NAME2) $(APP_NAME3) $(APP_NAME4)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME1) $(APP_NAME2) $(APP_NAME3) $(APP_NAME4) 

OBJS=$(OBJDIR)/main.o $(OBJDIR)/mandelbrotSerial.o $(OBJDIR)/mandelbrotThread.o $(PPM_OBJ)

$(APP_NAME1): dirs $(OBJS) $(OBJDIR)/kernel1.o
		$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS) $(OBJDIR)/kernel1.o mandelbrotThreadRef.a

$(APP_NAME2): dirs $(OBJS) $(OBJDIR)/kernel2.o
		$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS) $(OBJDIR)/kernel2.o mandelbrotThreadRef.a

$(APP_NAME3): dirs $(OBJS) $(OBJDIR)/kernel3.o
		$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS) $(OBJDIR)/kernel3.o mandelbrotThreadRef.a

$(APP_NAME4): dirs $(OBJS) $(OBJDIR)/kernel4.o
		$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS) $(OBJDIR)/kernel4.o mandelbrotThreadRef.a

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(COMMONDIR)/CycleTimer.h kernel.h

$(OBJDIR)/kernel1.o : kernel1.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel1.cu -o $@

$(OBJDIR)/kernel2.o : kernel2.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel2.cu -o $@

$(OBJDIR)/kernel3.o : kernel3.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel3.cu -o $@

$(OBJDIR)/kernel4.o : kernel4.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel4.cu -o $@
