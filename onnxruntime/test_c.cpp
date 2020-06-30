#include <iostream>
#include <map>

// onnx runtime include
#include <core/session/onnxruntime_cxx_api.h>
#include <core/framework/allocator.h>
//#include <core/providers/migraphx/migraphx_provider_factory.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " onnxfile" << std::endl;
        return 0;
    }

    OrtApiBase base;
	const OrtApi* g_ort = base.GetApi(ORT_API_VERSION);
	OrtEnv* env;
	g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
	OrtSessionOptions* session_option;
	g_ort->OrtCreateSessionOptions(&session_options);
//	g_ort->OrtSessionOptionsAppendExecutionProvider_MiGraphX(sessionOptions, 0);
	OrtSession* session;
	g_ort->CreateSession(env, argv[1], session_option, &session);

	return 0;
}
