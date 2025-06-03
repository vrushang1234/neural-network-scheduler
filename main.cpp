#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unistd.h>

#include "p1_process.h"

using namespace std;

int main(int argc, char** argv) {
    printf("Main process is created. (pid: %d)\n", getpid());
    int num_processes = 0;
    int num_threads = 0;

    vector<string> class_name;
    class_name.push_back("input1"); // default test file

    if (argc == 4 && (atoi(argv[1]) > 0 && atoi(argv[2]) > 0)) {
        num_processes = atoi(argv[1]);
        num_threads = atoi(argv[2]);
        string sort_type = argv[3];

        create_processes_and_sort(class_name, num_processes, num_threads, sort_type);
    } else {
        printf("[ERROR] Expecting 3 arguments with integral values greater than zero.\n");
        printf("[USAGE] %s <number of processes> <number of threads> <sort_type: merge|bubble>\n", argv[0]);
    }

    printf("Main process is terminated. (pid: %d)\n", getpid());
    return 0;
}

