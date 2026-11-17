CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -fopenmp -Wa,--gsframe=no -flto=auto

TARGET = qsim
SRC = qsim.cc

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
