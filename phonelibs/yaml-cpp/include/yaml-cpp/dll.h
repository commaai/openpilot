#ifndef DLL_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define DLL_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

// The following ifdef block is the standard way of creating macros which make
// exporting from a DLL simpler. All files within this DLL are compiled with the
// yaml_cpp_EXPORTS symbol defined on the command line. This symbol should not
// be defined on any project that uses this DLL. This way any other project
// whose source files include this file see YAML_CPP_API functions as being
// imported from a DLL, whereas this DLL sees symbols defined with this macro as
// being exported.
#undef YAML_CPP_API

#ifdef YAML_CPP_DLL      // Using or Building YAML-CPP DLL (definition defined
                         // manually)
#ifdef yaml_cpp_EXPORTS  // Building YAML-CPP DLL (definition created by CMake
                         // or defined manually)
//	#pragma message( "Defining YAML_CPP_API for DLL export" )
#define YAML_CPP_API __declspec(dllexport)
#else  // yaml_cpp_EXPORTS
//	#pragma message( "Defining YAML_CPP_API for DLL import" )
#define YAML_CPP_API __declspec(dllimport)
#endif  // yaml_cpp_EXPORTS
#else   // YAML_CPP_DLL
#define YAML_CPP_API
#endif  // YAML_CPP_DLL

#endif  // DLL_H_62B23520_7C8E_11DE_8A39_0800200C9A66
