#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 3;
        NSError *error = nil;
        NSString *source = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void tile_probe(device float *output [[buffer(0)]], "
            "threadgroup uchar *scratch [[threadgroup(0)]], "
            "uint2 group [[threadgroup_position_in_grid]], "
            "uint2 local [[thread_position_in_threadgroup]]) {\n"
            "uint lane = local.y * 16 + local.x;\n"
            "threadgroup uint *ids = reinterpret_cast<threadgroup uint *>(scratch);\n"
            "threadgroup float2 *xy = reinterpret_cast<threadgroup float2 *>(scratch + 1024);\n"
            "threadgroup float4 *co = reinterpret_cast<threadgroup float4 *>(scratch + 3072);\n"
            "threadgroup float *rgb = reinterpret_cast<threadgroup float *>(scratch + 7168);\n"
            "ids[lane] = lane; xy[lane] = float2(2 * lane, 3 * lane);\n"
            "co[lane] = float4(4 * lane, 5 * lane, 6 * lane, 7 * lane);\n"
            "rgb[lane * 3] = 8 * lane; rgb[lane * 3 + 1] = 9 * lane; rgb[lane * 3 + 2] = 10 * lane;\n"
            "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "uint next = (lane + 17) % 256;\n"
            "output[(group.y * 2 + group.x) * 256 + lane] = "
            "float(ids[next]) + xy[next].x + co[next].w + rgb[next * 3 + 2];\n"
            "}\n";
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 4; }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tile_probe"] error:&error];
        if (!pipeline) return 5;
        if (pipeline.maxTotalThreadsPerThreadgroup < 256 || device.maxThreadgroupMemoryLength < 10240) return 6;
        id<MTLBuffer> output = [device newBufferWithLength:1024 * sizeof(float) options:MTLResourceStorageModeShared];
        memset(output.contents, 0xff, output.length);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:output offset:0 atIndex:0];
        [encoder setThreadgroupMemoryLength:10240 atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(2, 2, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return 7;
        unsigned mismatches = 0;
        const float *values = static_cast<const float *>(output.contents);
        for (unsigned i = 0; i < 1024; ++i)
            if (values[i] != float(20 * ((i % 256 + 17) % 256))) ++mismatches;
        NSDictionary *report = @{
            @"device": device.name,
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"max_threadgroup_memory_bytes": @(device.maxThreadgroupMemoryLength),
            @"probe_pipeline_max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
            @"probe_pipeline_simd_width": @(pipeline.threadExecutionWidth),
            @"probe_threadgroup_shape": @[@16, @16, @1],
            @"probe_dynamic_threadgroup_memory_bytes": @10240,
            @"probe_groups": @4,
            @"checked_values": @1024,
            @"mismatches": @(mismatches),
            @"passed": @(mismatches == 0),
            @"scope": @"Independent compute capability probe; not a Gaussian renderer or performance benchmark."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
        if (!json || ![json writeToFile:[NSString stringWithUTF8String:argv[1]] atomically:YES]) return 8;
        puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
        return mismatches ? 9 : 0;
    }
}
