#pragma once
#include <string>
#include <vector>

using namespace std;

struct student {
    unsigned long id;
    double grade;

    student() : id(0), grade(0.0) {}
    student(unsigned long id_, double grade_) : id(id_), grade(grade_) {}
};

void create_processes_and_sort(vector<string> class_names, int num_processes, int num_threads, string sort_type);


class ParallelBubbleSorter {
public:
    vector<student> sorted_list;
    int num_threads;

    ParallelBubbleSorter(vector<student>& original_list, int num_threads);
    static void* thread_init(void* args);
    vector<student> run_sort();
};

