
MIGRAPHFOLDER=/home/scxiao/Workplace/projects/AMDMIGraphX
#MIGRAPHFOLDER=/home/scxiao/Workplace/projects/MIGraph
MIGRAPHINCLUDE= -I$(MIGRAPHFOLDER)/src/include \
                -I$(MIGRAPHFOLDER)/src/targets/gpu/include \
                -I$(MIGRAPHFOLDER)/src/targets/cpu/include \
                -I$(MIGRAPHFOLDER)/test/include \
                -I$(MIGRAPHFOLDER)/deps_mo1.8_rocblas2.2/include \
                -I.

CXX=/opt/rocm/bin/hcc
DEBUG=_debug
#DEBUG=_f46c700
MIGRAPHLIBDIR = -L$(MIGRAPHFOLDER)/build$(DEBUG)/src/onnx \
                -L$(MIGRAPHFOLDER)/build$(DEBUG)/src \
                -L$(MIGRAPHFOLDER)/build$(DEBUG)/src/targets/gpu \
                -L$(MIGRAPHFOLDER)/build$(DEBUG)/src/targets/cpu \
                -L$(MIGRAPHFOLDER)/deps_mo1.8_rocblas2.2/lib

MIGRAPHLIBS = -lmigraphx -lmigraphx_cpu -lmigraphx_device -lmigraphx_gpu -lmigraphx_onnx -lhip_hcc -lMIOpen
CXXFLAGS=-g -std=c++14 ${MIGRAPHINCLUDE} 
MIGRAPHLIBPATH = /home/scxiao/Workplace/software/migraphlibs

#EXE_FILES = myAdd \
#            mySin \
#            load_onnx \
#            myConv2d \
#            testTypename \
#            test_char_rnn \
#            test_rnn_both \
#            test_rnn_single \
#            test_shape \
#            test_gather \
#            test_const_eval \
#            test_mm \
#            rnn_gpu \
#            mySinGpu \
#            gru_test_1direct \
#            gru_test_bidirect \
#            test_eleminate_contiguous
            


PROGRAM_FILES=test_dot.cpp
#PROGRAM_FILES=test_load_onnx.cpp
SOURCE_FILES=utilities.cpp $(PROGRAM_FILES)

OBJ_FILES=$(SOURCE_FILES:.cpp=.o)
EXE_FILES=$(PROGRAM_FILES:.cpp=)

all: create copy $(EXE_FILES)

OBJ=obj
$(OBJ)/%.o : %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(EXE_FILES): $(OBJ)/$(OBJ_FILES)
	$(CXX) -o $@ $^ ${MIGRAPHLIBDIR} ${MIGRAPHLIBS} 

create:
	echo $(EXE_FILES)
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
	cp $(MIGRAPHFOLDER)/deps_mo1.8_rocblas2.2/lib/libMIOpen.so.1 $(MIGRAPHLIBPATH)/.
	cp $(MIGRAPHFOLDER)/build$(DEBUG)/src/py/*.so $(MIGRAPHLIBPATH)/.

clean:
	rm -rf *.o $(EXE_FILES)
