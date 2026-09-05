#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>

// Run an exact-power-of-two singular projection against a chosen shader source.
int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        auto options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        NSError *error = nil;
        auto source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
        auto library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            std::cerr << error.localizedDescription.UTF8String;
            return 1;
        }
        auto pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"preprocess"]
                                                  error:&error];
        if (!pipeline)
            return 1;
        float input[13] = {0, 0, 2, 1073741824.0f, 1073741824.0f, 0, 1073741824.0f, 0, 1, 1, 0, 0, .5};
        float camera[45] = {};
        for (int i = 0; i < 4; i++)
            camera[5 * i] = 1;
        camera[16] = camera[21] = camera[26] = camera[27] = 1;
        camera[34] = camera[35] = .5;
        camera[36] = camera[37] = 1;
        auto in = [device newBufferWithBytes:input length:sizeof(input) options:MTLResourceStorageModeShared];
        auto cam = [device newBufferWithBytes:camera
                                       length:sizeof(camera)
                                      options:MTLResourceStorageModeShared];
        auto out = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        auto count = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        auto err = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        std::memset(err.contents, 0, 4);
        uint32_t dims[4] = {1, 1, 1, 1048576};
        auto queue = [device newCommandQueue];
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:in offset:0 atIndex:0];
        [encoder setBuffer:cam offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBuffer:count offset:0 atIndex:3];
        [encoder setBuffer:err offset:0 atIndex:4];
        [encoder setBytes:dims length:16 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return 1;
        std::cout << "{\"error\":" << *(uint32_t *)err.contents
                  << ",\"radius\":" << ((float *)out.contents)[3]
                  << ",\"tile_count\":" << *(uint32_t *)count.contents << "}\n";
    }
}
