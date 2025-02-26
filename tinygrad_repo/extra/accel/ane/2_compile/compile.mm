#include <os/log.h>
#include <stdio.h>
#import <CoreFoundation/CoreFoundation.h>
#include <string>
#include <iostream>

extern "C" {
  int ANECCompile(CFDictionaryRef param_1, CFDictionaryRef param_2, unsigned long param_3);
  std::string _ZN21ZinIrEnumToStringUtil14OpCodeToStringE22ZinIrOpLayerOpCodeType(int op);
  std::string _ZN21ZinIrEnumToStringUtil21NonLinearModeToStringE18ZinIrNonLinearMode(int op);
  std::string _ZN19ZinMirCacheHintUtil17CacheHintToStringE15ZinMirCacheHint(int op);
  std::string _ZN30ZinMirKernelSizeSplitterEngine16ConvKindToStringENS_8ConvKindE(int op);

  /*void _Z24ZinIrRegBitPrintOutDebugILj7EE11ZinIrStatusjRN11ZinHWTraitsIXT_EE6HwTypeEiRNSt3__113basic_ostreamIcNS5_11char_traitsIcEEEE(
    unsigned long param_1, void *param_2,int param_3, std::basic_ostream<char> *param_4);

void debugregs(int a1, void *dat, int a2) {
  _Z24ZinIrRegBitPrintOutDebugILj7EE11ZinIrStatusjRN11ZinHWTraitsIXT_EE6HwTypeEiRNSt3__113basic_ostreamIcNS5_11char_traitsIcEEEE(a1, dat, a2, &std::cout);
}*/

}

int main(int argc, char* argv[]) {
  os_log(OS_LOG_DEFAULT, "start compiler");

  /*for (int i = 0; i < 60; i++) {
    std::string tmp = _ZN21ZinIrEnumToStringUtil14OpCodeToStringE22ZinIrOpLayerOpCodeType(i);
    //std::string tmp = _ZN21ZinIrEnumToStringUtil21NonLinearModeToStringE18ZinIrNonLinearMode(i);
    printf("%2d: %s\n", i, tmp.c_str());
  }*/

  CFTypeRef ikeys[2];
  ikeys[0] = CFSTR("NetworkPlistName");
  ikeys[1] = CFSTR("NetworkPlistPath");

  CFTypeRef ivalues[2];
  ivalues[0] = CFStringCreateWithCString(kCFAllocatorDefault, argv[1], kCFStringEncodingUTF8);
  ivalues[1] = CFSTR("./");
  
  CFDictionaryRef iDictionary = CFDictionaryCreate(kCFAllocatorDefault, ikeys, ivalues, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFArrayRef array = CFArrayCreate(kCFAllocatorDefault, (const void**)&iDictionary, 1, &kCFTypeArrayCallBacks);

  CFMutableDictionaryRef optionsDictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFMutableDictionaryRef flagsDictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

  CFDictionaryAddValue(optionsDictionary, CFSTR("InputNetworks"), array);
  CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFilePath"), CFSTR("./"));
  //CFDictionaryAddValue(optionsDictionary, CFSTR("OptionsFilePath"), CFSTR("good.options"));

  // h11 (or anything?) works here too, and creates different outputs that don't run
  CFDictionaryAddValue(flagsDictionary, CFSTR("TargetArchitecture"), CFSTR("h13"));

  if (argc > 2) {
    CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFileName"), CFSTR("debug/model.hwx"));
    //CFDictionaryAddValue(flagsDictionary, CFSTR("DebugDetailPrint"), kCFBooleanTrue);
    CFDictionaryAddValue(flagsDictionary, CFSTR("CompileANEProgramForDebugging"), kCFBooleanTrue);
    int debug_mask = 0x7fffffff;
    CFDictionaryAddValue(flagsDictionary, CFSTR("DebugMask"), CFNumberCreate(kCFAllocatorDefault, 3, &debug_mask));
  } else {
    CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFileName"), CFSTR("model.hwx"));
  }
  //CFDictionaryAddValue(flagsDictionary, CFSTR("DisableMergeScaleBias"), kCFBooleanTrue);
  //CFDictionaryAddValue(flagsDictionary, CFSTR("Externs"), CFSTR("swag"));

  //CFShow(optionsDictionary);
  //CFShow(flagsDictionary);

  printf("hello\n");
  int ret = ANECCompile(optionsDictionary, flagsDictionary, 0);
  printf("compile: %d\n", ret);


  return ret;
}
