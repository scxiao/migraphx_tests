import migraphx
import sys
import numpy as np

def run_model(model, target):
    print("Running {} ...".format(target))

    model.compile(migraphx.get_target(target))
    params = {}
    args = []
    for key, value in model.get_parameter_shapes().items():
        print("Parameter {} -> {}".format(key, value))
        hash_val = hash(key) % (2 ** 32 - 1)
        np.random.seed(hash_val)
        args.append(np.random.randn(value.elements()).astype(np.single).reshape(value.lens()))
        params[key] = migraphx.argument(args[-1])

        result = np.array(model.run(params))

    print("")
    print("Result = \n{}".format(result))
    print("")

    return result

def main():
    if len(sys.argv) != 3:
        print("Usage: python {} onnx cpu/gpu/both".format(sys.argv[0]))
        exit()

    model = migraphx.parse_onnx(sys.argv[1])
    target = sys.argv[2]

    if target == "both":
        model_clone = model.clone()
        gpu_res = run_model(model_clone, "cpu")
        cpu_res = run_model(model, "gpu")
    elif target == "gpu":
        run_model(model, "gpu")
    else:
        run_model(model, "cpu")

if __name__ == "__main__":
    main()

