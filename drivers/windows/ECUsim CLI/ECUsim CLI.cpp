// ECUsim CLI.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ECUsim DLL\ECUsim.h"

std::unique_ptr<ECUsim> sim;

BOOL CtrlHandler(DWORD fdwCtrlType)
{
	if (fdwCtrlType != CTRL_C_EVENT) return FALSE;

	sim->stop();
	sim->join();

	return(TRUE);
}

int main(int argc,      // Number of strings in array argv
	     char *argv[],  // Array of command-line argument strings
	     char *envp[])  // Array of environment variable strings
{

	int count;

	// Display each command-line argument.
	std::cout << "\nCommand-line arguments:\n";
	for (count = 0; count < argc; count++)
		std::cout << "  argv[" << count << "]   " << argv[count] << "\n";

	SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE);

	sim.reset(new ECUsim("", 500000));
	sim->join();

    return 0;
}

