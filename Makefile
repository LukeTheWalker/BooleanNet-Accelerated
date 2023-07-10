SHELL := /bin/zsh

IDIR = include
SDIR = src
BINDIR = bin

ifeq ($(shell which nvcc),)	
	CXX := g++
else
	CXX := nvcc
	CUFLAGS := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 --use_fast_math
endif

CXXFLAGS= -I$(IDIR) -std=c++17 -g $(CUFLAGS) $(CV_FLAGS) -O3
LDFLAGS = $(GMP) $(CV_LIBS)

ODIR=obj

DEPS = $(IDIR)/$(wildcard *.hpp *.cuh)

_CUFILES = $(wildcard $(SDIR)/*.cu)
_CXXFILES = $(wildcard $(SDIR)/*.cpp)

CUFILES = $(notdir $(_CUFILES))
CXXFILES = $(notdir $(_CXXFILES))

_OBJ = $(_CUFILES:.cu=.o) $(_CXXFILES:.cpp=.o)
OBJ = $(patsubst $(SDIR)/%,$(ODIR)/%,$(_OBJ))

TARGET = $(BINDIR)/app

UNAME_S := $(shell uname -s)

file_name = $(notdir $(input_file))

$(TARGET): $(OBJ) | $(BINDIR)
	$(CXX) -o $@ $^ $(LDFLAGS)

all: $(TARGET) run

run: $(TARGET)
	./$(TARGET)

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS) | $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(ODIR)/%.o: $(SDIR)/%.cu $(DEPS)  | $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean run

clean:
	rm -f $(ODIR)/*.o $(TARGET)

$(ODIR):
	mkdir -p $(ODIR)

$(BINDIR): 
	mkdir -p $(BINDIR)

$(DATADIR):
	mkdir -p $(DATADIR)