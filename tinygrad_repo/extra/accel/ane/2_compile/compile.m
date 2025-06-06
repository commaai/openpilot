#import <Foundation/Foundation.h>
#include <os/log.h>
#include <stdio.h>

typedef unsigned int ANECStatus;

int ANECCompile(NSDictionary* param_1, NSDictionary* param_2,
    void (^param_3)(ANECStatus status,
        NSDictionary* statusDictionary));

int main(int argc, char* argv[])
{
    os_log(OS_LOG_DEFAULT, "start compiler");

    NSDictionary* iDictionary = @ {
        @"NetworkPlistName" : [NSString stringWithCString:argv[1]
                                                 encoding:NSUTF8StringEncoding],
        @"NetworkPlistPath" : @"./",
    };
    NSArray* plistArray = @[ iDictionary ];

    NSMutableDictionary* optionsDictionary =
        [NSMutableDictionary dictionaryWithCapacity:4];
    NSMutableDictionary* flagsDictionary =
        [NSMutableDictionary dictionaryWithCapacity:4];
    optionsDictionary[@"InputNetworks"] = plistArray;

    optionsDictionary[@"OutputFilePath"] = @"./";

    // h11 (or anything?) works here too, and creates different outputs that don't
    // run
    flagsDictionary[@"TargetArchitecture"] = @"h13";

    if (argc > 2) {
        optionsDictionary[@"OutputFileName"] = @"debug/model.hwx";

        flagsDictionary[@"CompileANEProgramForDebugging"] =
            [NSNumber numberWithBool:YES];
        int debug_mask = 0x7fffffff;
        flagsDictionary[@"DebugMask"] = [NSNumber numberWithInt:debug_mask];
    } else {
        optionsDictionary[@"OutputFileName"] = @"model.hwx";
    }

    void (^simpleBlock)(ANECStatus status, NSDictionary* statusDictionary) = ^(ANECStatus status, NSDictionary* statusDictionary) {
        NSLog(@"status = %d\n", status);
        // when status != 0 dump the dictionary
        if (status)
            NSLog(@"%@", statusDictionary);
    };

    printf("hello\n");
    int ret = ANECCompile(optionsDictionary, flagsDictionary, simpleBlock);
    printf("compile: %d\n", ret);

    return ret;
}
