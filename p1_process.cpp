#include "p1_process.h"
#include "p1_threads.h"
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cmath>

using namespace std;

void process_classes(vector<string> classes, int num_threads, string sort_type) {
    printf("Child process is created. (pid: %d)\n", getpid());

    for (int i = 0; i < classes.size(); i++) {
        string class_name = classes[i];
        string input_file_name = "input/" + sort_type + ".csv";
        string output_sorted_file_name = "output/" + sort_type + "_sorted.csv";
        string output_stats_file_name = "output/" + sort_type + "_stats.csv";

        vector<student> students;
        ifstream infile(input_file_name);
        if (!infile.is_open()) {
            perror(("Error opening file: " + input_file_name).c_str());
            continue;
        }

        string line;
        getline(infile, line); // skip header
        while (getline(infile, line)) {
            stringstream ss(line);
            string id_str, grade_str;
            getline(ss, id_str, ',');
            getline(ss, grade_str);

            student s(strtoul(id_str.c_str(), NULL, 10), atof(grade_str.c_str()));
            students.push_back(s);
        }
        infile.close();

        vector<student> sorted;
        if (sort_type == "merge") {
            ParallelMergeSorter sorter(students, num_threads);
            sorted = sorter.run_sort();
        } else if (sort_type == "bubble") {
            ParallelBubbleSorter sorter(students, num_threads);
            sorted = sorter.run_sort();
        } else {
            cerr << "[ERROR] Unknown sort type: " << sort_type << endl;
            exit(1);
        }

        ofstream outfile(output_sorted_file_name);
        outfile << "Rank,Student ID,Grade\n" << fixed << setprecision(10);
        for (size_t j = 0; j < sorted.size(); ++j) {
            outfile << (j + 1) << "," << sorted[j].id << "," << sorted[j].grade << "\n";
        }
        outfile.close();

        double sum = 0.0;
        for (auto& s : sorted) sum += s.grade;
        double avg = sorted.size() ? sum / sorted.size() : 0.0;

        double med = 0.0;
        if (!sorted.empty()) {
            size_t n = sorted.size();
            med = n % 2 ? sorted[n/2].grade : (sorted[n/2-1].grade + sorted[n/2].grade)/2.0;
        }

        double sq_sum = 0.0;
        for (auto& s : sorted) {
            double d = s.grade - avg;
            sq_sum += d * d;
        }
        double stddev = sorted.empty() ? 0.0 : sqrt(sq_sum / sorted.size());

        ofstream statsfile(output_stats_file_name);
        statsfile << fixed << setprecision(3);
        statsfile << "Average,Median,Std. Dev\n";
        statsfile << avg << "," << med << "," << stddev << "\n";
        statsfile.close();
    }

    printf("Child process is terminated. (pid: %d)\n", getpid());
    exit(0);
}

void create_processes_and_sort(vector<string> class_names, int num_processes, int num_threads, string sort_type) {
    vector<pid_t> child_pids;
    int classes_per_process = max((int)(class_names.size() / num_processes), 1);

    int l = 0;
    for (int i = 0; i < num_processes && l < class_names.size(); i++) {
        int r = l + classes_per_process + (i < (class_names.size() % num_processes) ? 1 : 0);
        if (r > class_names.size()) r = class_names.size();

        vector<string> sub_classes(class_names.begin() + l, class_names.begin() + r);
        pid_t pid = fork();

        if (pid == 0) {
            process_classes(sub_classes, num_threads, sort_type);
        } else if (pid > 0) {
            child_pids.push_back(pid);
        } else {
            perror("fork error");
            exit(1);
        }

        l = r;
    }

    for (pid_t pid : child_pids) {
        int status;
        waitpid(pid, &status, 0);
    }
}

