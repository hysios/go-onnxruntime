#ifndef __ORT__
#define __ORT__

#include "../include/onnxruntime_c_api.h"
// #include "../include/cuda_provider_factory.h"
#include "../include/tensorrt_provider_factory.h"

const char *getVersionString(OrtApiBase *api)
{
    return api->GetVersionString();
}

const OrtApi *getApi(OrtApiBase *api)
{
    return api->GetApi(ORT_API_VERSION);
}

OrtErrorCode getStatus(OrtApi *api, OrtStatus *status)
{
    return api->GetErrorCode(status);
}

OrtStatus *createStatus(OrtApi *api, OrtErrorCode code, const char *msg)
{
    return api->CreateStatus(code, msg);
}

OrtStatus *ortCreateEnv(OrtApi *api, OrtLoggingLevel log_severity_level, const char *logid, OrtEnv **out)
{
    return api->CreateEnv(log_severity_level, logid, out);
}

OrtStatus *ortCreateSessionOptions(OrtApi *api, OrtSessionOptions **out)
{
    return api->CreateSessionOptions(out);
}

OrtStatus *ortCreateSession(OrtApi *api, OrtEnv *env, char *model_path, OrtSessionOptions *options, OrtSession **out)
{
    return api->CreateSession(env, model_path, options, out);
}

// OrtStatus *ortSessionOptionsAppendExecutionProvider_CUDAOLD(OrtApi *api, OrtSessionOptions *options, int device_id)
// {
//     return OrtSessionOptionsAppendExecutionProvider_CUDA(options, device_id);
// }

OrtStatus *ortSessionOptionsAppendExecutionProvider_ROCM(OrtApi *api, OrtSessionOptions *options, OrtROCMProviderOptions *rocm_options)
{
    return api->SessionOptionsAppendExecutionProvider_ROCM(options, rocm_options);
}

OrtStatus *ortSessionOptionsAppendExecutionProvider_OpenVINO(OrtApi *api, OrtSessionOptions *options, OrtOpenVINOProviderOptions *provider_options)
{
    return api->SessionOptionsAppendExecutionProvider_OpenVINO(options, provider_options);
}

OrtStatus *ortSessionOptionsAppendExecutionProvider_TensorRT(OrtApi *api, OrtSessionOptions *options, OrtTensorRTProviderOptions *tensorrt_options)
{
    return api->SessionOptionsAppendExecutionProvider_TensorRT(options, tensorrt_options);
}

OrtStatus *ortSessionOptionsAppendExecutionProvider_TensorRT_V2(OrtApi *api, OrtSessionOptions *options, OrtTensorRTProviderOptionsV2 *tensorrt_options)
{
    return api->SessionOptionsAppendExecutionProvider_TensorRT_V2(options, tensorrt_options);
}

OrtStatus *ortCreateTensorRTProviderOptions(OrtApi *api, OrtTensorRTProviderOptionsV2 **out)
{
    return api->CreateTensorRTProviderOptions(out);
}

OrtStatus *ortUpdateTensorRTProviderOptions(OrtApi *api, OrtTensorRTProviderOptionsV2 *options, const char **provider_options_keys, const char **provider_options_values, size_t num_provider_options)
{
    return api->UpdateTensorRTProviderOptions(options, provider_options_keys, provider_options_values, num_provider_options);
}

OrtStatus *ortGetTensorRTProviderOptionsAsString(OrtApi *api, OrtTensorRTProviderOptionsV2 *options, OrtAllocator *allocator, char **out)
{
    return api->GetTensorRTProviderOptionsAsString(options, allocator, out);
}

OrtStatus *ortCreateArenaCfg(OrtApi *api, size_t max_mem, int arean_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk, OrtArenaCfg **out)
{
    return api->CreateArenaCfg(max_mem, arean_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk, out);
}

void ortReleaseArenaCfg(OrtApi *api, OrtArenaCfg *arena_cfg)
{
    api->ReleaseArenaCfg(arena_cfg);
}

void ortReleaseTensorRTProviderOptions(OrtApi *api, OrtTensorRTProviderOptionsV2 *options)
{
    api->ReleaseTensorRTProviderOptions(options);
}

OrtStatus *ortSetCurrentGpuDeviceId(OrtApi *api, int device_id)
{
    return api->SetCurrentGpuDeviceId(device_id);
}

OrtStatus *ortGetCurrentGpuDeviceId(OrtApi *api, int *device_id)
{
    return api->GetCurrentGpuDeviceId(device_id);
}

OrtStatus *ortCreateValue(OrtApi *api, const OrtValue **in, size_t num, enum ONNXType value, OrtValue **out)
{
    return api->CreateValue(in, num, value, out);
}

OrtStatus *ortCreateMemoryInfo(OrtApi *api, const char *name, OrtAllocatorType type, int id, enum OrtMemType memtype, OrtMemoryInfo **out)
{
    return api->CreateMemoryInfo(name, type, id, memtype, out);
}

OrtStatus *ortCreateCpuMemoryInfo(OrtApi *api, OrtAllocatorType type, enum OrtMemType memtype, OrtMemoryInfo **out)
{
    return api->CreateCpuMemoryInfo(type, memtype, out);
}

OrtStatus *ortCreateTensorAsOrtValue(OrtApi *api, OrtAllocator *allocator, const int64_t *shape, size_t shape_len,
                                     ONNXTensorElementDataType type, OrtValue **out)
{
    return api->CreateTensorAsOrtValue(allocator, shape, shape_len, type, out);
}

OrtStatus *ortCreateAllocator(OrtApi *api, OrtSession *session, OrtMemoryInfo *mem_infom, OrtAllocator **allocator)
{
    return api->CreateAllocator(session, mem_infom, allocator);
}

OrtStatus *ortCreateTensorWithDataAsOrtValue(OrtApi *api, OrtMemoryInfo *info, void *p_data,
                                             size_t p_data_len, const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type,
                                             OrtValue **out)
{
    return api->CreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape, shape_len, type,
                                               out);
}

OrtStatus *ortCreateRunOptions(OrtApi *api, OrtRunOptions **out)
{
    return api->CreateRunOptions(out);
}

OrtStatus *ortRun(OrtApi *api, OrtSession *sess, const OrtRunOptions *run_options,
                  const char *const *input_names,
                  const OrtValue *const *input, size_t input_len,
                  const char *const *output_names1, size_t output_names_len, OrtValue **output)
{
    return api->Run(sess, run_options, input_names, input, input_len, output_names1, output_names_len, output);
}

OrtStatus *ortSessionGetInputCount(OrtApi *api, OrtSession *sess, size_t *input_len)
{
    return api->SessionGetInputCount(sess, input_len);
}

OrtStatus *ortSessionGetInputName(OrtApi *api, OrtSession *sess, size_t index,
                                  OrtAllocator *allocator,
                                  char **out)
{
    return api->SessionGetInputName(sess, index, allocator, out);
}

OrtStatus *ortSessionGetInputTypeInfo(OrtApi *api, OrtSession *sess, size_t index,
                                      OrtTypeInfo **type_info)
{
    return api->SessionGetInputTypeInfo(sess, index, type_info);
}

OrtStatus *ortGetAllocatorWithDefaultOptions(OrtApi *api, OrtAllocator **out)
{
    return api->GetAllocatorWithDefaultOptions(out);
}

OrtStatus *ortSessionGetOutputCount(OrtApi *api, OrtSession *sess, size_t *input_len)
{
    return api->SessionGetOutputCount(sess, input_len);
}

OrtStatus *ortSessionGetOutputName(OrtApi *api, OrtSession *sess, size_t index,
                                   OrtAllocator *allocator,
                                   char **out)
{
    return api->SessionGetOutputName(sess, index, allocator, out);
}

OrtStatus *ortSessionGetOutputTypeInfo(OrtApi *api, OrtSession *sess, size_t index,
                                       OrtTypeInfo **type_info)
{
    return api->SessionGetOutputTypeInfo(sess, index, type_info);
}

OrtStatus *ortGetOnnxTypeFromTypeInfo(OrtApi *api, OrtTypeInfo *typeinfo, enum ONNXType *out)
{
    return api->GetOnnxTypeFromTypeInfo(typeinfo, out);
}

OrtStatus *ortCastTypeInfoToTensorInfo(OrtApi *api, const OrtTypeInfo *type_info,
                                       const OrtTensorTypeAndShapeInfo **out)
{
    return api->CastTypeInfoToTensorInfo(type_info, out);
}

OrtStatus *ortCastTypeInfoToSequenceTypeInfo(OrtApi *api, const OrtTypeInfo *type_info,
                                             const OrtSequenceTypeInfo **out)
{
    return api->CastTypeInfoToSequenceTypeInfo(type_info, out);
}

OrtStatus *ortGetTensorElementType(OrtApi *api, const OrtTensorTypeAndShapeInfo *tensor,
                                   enum ONNXTensorElementDataType *out)
{
    return api->GetTensorElementType(tensor, out);
}

OrtStatus *ortGetTensorTypeAndShape(OrtApi *api, const OrtValue *value,
                                    OrtTensorTypeAndShapeInfo **out)
{
    return api->GetTensorTypeAndShape(value, out);
}

OrtStatus *ortGetTensorElementCount(OrtApi *api, const OrtTensorTypeAndShapeInfo *tensor, size_t *out)
{
    return api->GetTensorShapeElementCount(tensor, out);
}

OrtStatus *ortGetTensorDimensionsCount(OrtApi *api, const OrtTensorTypeAndShapeInfo *tensor, size_t *out)
{
    return api->GetDimensionsCount(tensor, out);
}

OrtStatus *ortGetTensorDimensions(OrtApi *api,
                                  const OrtTensorTypeAndShapeInfo *tensor,
                                  int64_t *dim_values,
                                  size_t dim_values_length)
{
    return api->GetDimensions(tensor, dim_values, dim_values_length);
}

OrtStatus *ortGetSymbolicDimensions(OrtApi *api, const OrtTensorTypeAndShapeInfo *tensor,
                                    const char *dim_params[], size_t dim_params_length)
{
    return api->GetSymbolicDimensions(tensor, dim_params, dim_params_length);
}

OrtStatus *ortGetValueCount(OrtApi *api, OrtValue *value, size_t *size)
{
    return api->GetValueCount(value, size);
}

OrtStatus *ortTensorMutableFloatData(OrtApi *api, OrtValue *value, void **out)
{
    return api->GetTensorMutableData(value, out);
}

void ortReleaseSession(OrtApi *api, OrtSession *sess)
{
    api->ReleaseSession(sess);
}

void ortReleaseEnv(OrtApi *api, OrtEnv *env)
{
    api->ReleaseEnv(env);
}

void ortReleaseMemoryInfo(OrtApi *api, OrtMemoryInfo *info)
{
    api->ReleaseMemoryInfo(info);
}

void ortReleaseSessionOptions(OrtApi *api, OrtSessionOptions *options)
{
    api->ReleaseSessionOptions(options);
}

void ortReleaseValue(OrtApi *api, OrtValue *value)
{
    api->ReleaseValue(value);
}

void ortReleaseRunOptions(OrtApi *api, OrtRunOptions *run_options)
{
    api->ReleaseRunOptions(run_options);
}

#endif