#ifndef MIGRAPHX_GUARD_C_API_H
#define MIGRAPHX_GUARD_C_API_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// return code, more to be added later
typedef enum {
    MIGRAPHX_SUCCESS = 0,

    // shape related error code
    //...

    // program related error code
    MIGRAPHX_INVALID_PROGRAM = 0x1000,
    //...

    // argument related error code
    //...

    // instruction related error code
    //...

    // target related error code
    MIGRAPHX_UNKNOWN_TARGET_NAME = 0x2000,
    //.....

} migraphx_status;

// data types used in migraphx
typedef num {
    MIGRAPHX_SHAPE_TYPE_HALF,
    MIGRAPHX_SHAPE_TYPE_BFLOAT16,
    MIGRAPHX_SHAPE_TYPE_SINGLE,
    MIGRAPHX_SHAPE_TYPE_UINT8,
    MIGRAPHX_SHAPE_TYPE_INT8,
    MIGRAPHX_SHAPE_TYPE_UINT16,
    MIGRAPHX_SHAPE_TYPE_INT16,
    MIGRAPHX_SHAPE_TYPE_UINT32,
    MIGRAPHX_SHAPE_TYPE_INT32,
    MIGRAPHX_SHAPE_TYPE_UINT64,
    MIGRAPHX_SHAPE_TYPE_INT64
} migraphx_shape_datatype_t;

// struct to point to a shape instance
typedef struct migraphx_shape_t {
    uint64_t handle = 0;
} migraphx_shape_t;

// struct to point to a target instance
typedef struct migraphx_target_t {
    uint64_t handle;
} migraphx_target_t;

// struct to point to an argument instance
typedef struct migraphx_argument_t {
    uint64_t handle;
} migraphx_argument_t;

typedef struct migraphx_instruction_t {
    uint64_t handle;
} migraphx_instruction_t;

// struct to point to a program instance
typedef struct migraphx_program_t {
    uint64_t handle;
} migraphx_program_t;

//==================================
// APIs related to shape
// create a shape instance
migraphx_status migraphx_create_shape(migraphx_shape_datatype_t type,
                                      const std::size_t dim_num,
                                      const std::size_t *dims,
                                      const std::size_t *strides,
                                      migraphx_shape_t& shape);

// retrieve shape dims
migraphx_status migraphx_get_shape_dims(const migraphx_shape_t& shape,
                                        std::size_t* &dims,
                                        std::size_t& num_dims);

// retrieve shape strides
migraphx_status migraphx_get_shape_strides(const migraphx_shape_t& shape,
                                           std::size_t* &strides
                                           std::size_t& num_strides);

// retrieve shape type
migraphx_status migraphx_get_shape_type(const migraphx_shape_t& shape,
                                        migraphx_shape_datatype_t& type);

// check whether the shape is transposed
migraphx_status migraphx_is_shape_transposed(const migraphx_shape_t& shape,
                                             bool& is_transposed);

//==================================
// APIs related to target
// create a target based on target name
migraphx_status migraphx_create_target(const char *name, 
                                       migraphx_target_t& target);

// copy argument buffer from host to the target(e.g. GPU)
// if target is the host, return the input argument
migraphx_status migraphx_copy_to_target(const migraphx_target_t& target,
                                        const migraphx_argument_t& in_arg,
                                        migraphx_argument_t& out_arg);

// copy argument buffer from target(e.g. GPU) to the host
// if target is the host, return the input argument
migraphx_status migraphx_copy_from_target(const migraphx_target_t& target,
                                          const migraphx_argument_t& in_arg,
                                          migraphx_argument_t& out_arg);

// =========================
// APIs related to arguments
// create an argument
migraphx_status migraphx_create_argument(const migraphx_shape_t& shape, 
                                         const void *buffer_ptr, 
                                         migraphx_argument_t& argument);

// get argument buffer
migraphx_status migraphx_get_argument_buffer(const migraphx_argument_t& argument,
                                             const void* &buffer_ptr);

// get argument shape
migraphx_status migraphx_get_argument_shape(const migraphx_argument_t& argument,
                                            migraphx_shape_t& shape);

// ==========================
// APIs related to instruction
// get instruction name
migraphx_status migraphx_get_instruction_name(const migraphx_instruction_t& instruction,
                                              char* &name);

// get instruction shape
migraphx_status migraphx_get_instruction_shape(const migraphx_instruction_t& instruction,
                                               migraphx_shape_t& shape);

// get inputs
migraphx_status migraphx_get_instruction_inputs(const migraphx_instruction_t& instruction,
                                                migraphx_instruction_t* &inputs,
                                                std::size_t& num_inputs);

// get outputs
migraphx_status migraphx_get_instruction_outputs(const migraphx_instruction_t& instruction,
                                                 migraphx_instruction_t* &outputs,
                                                 std::size_t& num_outputs);

// is instruction equal
migraphx_status migraphx_is_instruction_equal(const migraphx_instruction_t& inst1,
                                              const migraphx_instruction_t& inst2);

// =========================
// APIs related to program

// create a program
// return an empty program instance
migraphx_status migraphx_create_program(migraphx_program_t& prog);

// add operations (we need a way to input the 
// attributes of various operators
migraphx_status migraphx_add_program_instruction(migraphx_program_t& prog,
                                                 char* name,
                                                 const migraphx_instruction_t* inputs,
                                                 std::size_t num_inputs,
                                                 migraphx_instruction_t* &outputs,
                                                 std::size_t& num_outputs);

// compile a program
migraphx_status migraphx_compile_program(const migraphx_program_t program, 
                                         migraphx_target_t& target);

// get input parameters
mgiraphx_status migraphx_get_program_input_name_shapes(const migraphx_program_t& program,
                                               char** &input_names,
                                               migraphx_shape_t *&input_shapes,
                                               std::size_t& num_inputs);

// get output parameters (may not be used for now)
mgiraphx_status migraphx_get_program_output_name_shapes(const migraphx_program_t& program,
                                               char** &output_names,
                                               migraphx_shape_t *&output_shapes,
                                               std::size_t& num_outputs);

// run a program (return as a pointer to support multiple outputs)
migraphx_status migraphx_run_program(const migraphx_program_t& program,
        const migraphx_argument_t *input_arguments, const std::size_t num_inputs,
        const migraphx_argument_t* results, const std::size_t num_results); 

// =========================
// APIs related to parse onnx
// parse a protobuf to return a program
migraphx_status migraphx_parse_protobuf(const char *buffer, 
                                        std::size_t size, 
                                        migraphx_program_t& program);

// parse a onnx input file
migraphx_status migraphx_parse_onnx_file(const char *file_name
                                         migraphx_program_t& program);

// ========================
// others
migraphx_status migraphx_get_error_info(migraphx_status code,
                                        char* &error_info);

// check whether an operator is suppored
migraphx_status migraphx_is_operator_supported(const char *name,
                                               bool& is_supported);

#ifdef __cplusplus
}
#endif

#endif
