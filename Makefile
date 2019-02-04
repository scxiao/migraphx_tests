
MIGRAPHFOLDER=/home/scxiao/Workplace/projects/AMDMIGraphX
MIGRAPHINCLUDE= -I$(MIGRAPHFOLDER)/src/include \
                -I$(MIGRAPHFOLDER)/src/targets/gpu/include \
                -I$(MIGRAPHFOLDER)/src/targets/cpu/include \
                -I$(MIGRAPHFOLDER)/test/include \
                -I$(MIGRAPHFOLDER)/deps/include

DEBUG=_debug
MIGRAPHLIBDIR = -L $(MIGRAPHFOLDER)/build$(DEBUG)/src/onnx -L $(MIGRAPHFOLDER)/build$(DEBUG)/src -L$(MIGRAPHFOLDER)/build$(DEBUG)/src/targets/gpu -L$(MIGRAPHFOLDER)/build$(DEBUG)/src/targets/cpu
MIGRAPHLIBS = -lmigraphx -lmigraphx_cpu -lmigraphx_device -lmigraphx_gpu -lmigraphx_onnx
CXXFLAGS=-g -std=c++14 ${MIGRAPHINCLUDE}
MIGRAPHLIBPATH = /home/scxiao/Workplace/software/migraphlibs

EXE_FILES = myAdd \
            mySin \
            mySinGpu \
            load_onnx \
            myConv2d \
            testTypename \
            test_char_rnn \
            test_rnn \
            test_shape \
            test_gather \
            test_const_eval \
            test_mm \
            rnn_gpu

all: create copy $(EXE_FILES)

SOURCE_FILES=myAdd.cpp \
    mySin.cpp \
    mySinGpu.cpp \
    load_onnx.cpp \
    myConvolution.cpp \
    testTypename.cpp \
    test_char_rnn.cpp \
    test_rnn.cpp \
    test_shape.cpp \
    test_gather.cpp \
    test_const_eval.cpp \
    test_mm.cpp \
    rnn_gpu.cpp

OBJ_FILES=$(SOURCE_FILES:.cpp=.o)

OBJ=obj
$(OBJ)/%.o : %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

rnn_gpu: $(OBJ)/rnn_gpu.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

test_mm: $(OBJ)/test_mm.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

test_gather: $(OBJ)/test_gather.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

test_shape: $(OBJ)/test_shape.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

test_char_rnn: $(OBJ)/test_char_rnn.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

test_rnn: $(OBJ)/test_rnn.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

testTypename: $(OBJ)/testTypename.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

myConv2d: $(OBJ)/myConvolution.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

load_onnx: $(OBJ)/load_onnx.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

myAdd: $(OBJ)/myAdd.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

mySin: $(OBJ)/mySin.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

mySinGpu: $(OBJ)/mySinGpu.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

test_const_eval: $(OBJ)/test_const_eval.o
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

create:
	if [ ! -d "./obj" ]; then \
		mkdir obj;\
	fi

copy:
#cd $(MIGRAPHFOLDER)/build$(DEUBG); make -j32
	cp $(MIGRAPHFOLDER)/build$(DEBUG)/src/onnx/libmigraphx_onnx.so $(MIGRAPHLIBPATH)/.
	cp $(MIGRAPHFOLDER)/build$(DEBUG)/src/libmigraphx.so $(MIGRAPHLIBPATH)/.
	cp $(MIGRAPHFOLDER)/build$(DEBUG)/src/targets/cpu/libmigraphx_cpu.so $(MIGRAPHLIBPATH)/.
	cp $(MIGRAPHFOLDER)/build$(DEBUG)/src/targets/gpu/libmigraphx_gpu.so $(MIGRAPHLIBPATH)/.
	cp $(MIGRAPHFOLDER)/build$(DEBUG)/src/targets/gpu/libmigraphx_device.so $(MIGRAPHLIBPATH)/.

clean:
	rm -rf *.o $(EXE_FILES)
