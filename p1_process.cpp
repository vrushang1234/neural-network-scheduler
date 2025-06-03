#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <unistd.h>
#include <sys/wait.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "p1_process.h"
#include "p1_threads.h"

using namespace std;

void process_classes(vector<string> classes, int num_threads)
{
  printf("Child process is created. (pid: %d)\n", getpid());

  for (int i = 0; i < classes.size(); i++)
  {
    // get all the input/output file names here
    string class_name = classes[i];
    char buffer[40];
    sprintf(buffer, "input/%s.csv", class_name.c_str());
    string input_file_name(buffer);

    sprintf(buffer, "output/%s_sorted.csv", class_name.c_str());
    string output_sorted_file_name(buffer);

    sprintf(buffer, "output/%s_stats.csv", class_name.c_str());
    string output_stats_file_name(buffer);

    vector<student> students;

    //
    ifstream infile(input_file_name.c_str());
    if (!infile.is_open())
    {
      perror(("Error opening file: " + input_file_name).c_str());
      continue;
    }

    string line;
    getline(infile, line); // Skip header

    while (getline(infile, line))
    {
      stringstream ss(line);
      string id_str, grade_str;

      if (!getline(ss, id_str, ','))
        continue;
      if (!getline(ss, grade_str))
        continue;

      unsigned long id = strtoul(id_str.c_str(), NULL, 10);
      double grade = atof(grade_str.c_str());

      student s(id, grade);
      students.push_back(s);
    }

    infile.close();

    // Run multi threaded sort
    ParallelMergeSorter sorter(students, num_threads);
    vector<student> sorted = sorter.run_sort();

    // Write sorted output to output/%s_sorted.csv
    ofstream outfile(output_sorted_file_name.c_str());
    if (!outfile.is_open())
    {
      perror(("Error writing to file: " + output_sorted_file_name).c_str());
      continue;
    }

    outfile << "Rank,Student ID,Grade\n";
    outfile << std::fixed << std::setprecision(10);
    for (size_t j = 0; j < sorted.size(); ++j)
    {
      outfile << (j + 1) << "," << sorted[j].id << "," << sorted[j].grade << "\n";
    }

    outfile.close();

    // Compute statistics
    double sum = 0.0;
    for (size_t j = 0; j < sorted.size(); ++j)
    {
      sum += sorted[j].grade;
    }

    double average = sorted.size() > 0 ? sum / sorted.size() : 0.0;

    double median = 0.0;
    size_t n = sorted.size();
    if (n > 0)
    {
      if (n % 2 == 0)
        median = (sorted[n / 2 - 1].grade + sorted[n / 2].grade) / 2.0;
      else
        median = sorted[n / 2].grade;
    }

    double sq_sum = 0.0;
    for (size_t j = 0; j < n; ++j)
    {
      double diff = sorted[j].grade - average;
      sq_sum += diff * diff;
    }
    double std_dev = n > 0 ? sqrt(sq_sum / n) : 0.0;

    // Write statistics to stats file
    ofstream statsfile(output_stats_file_name.c_str());
    if (!statsfile.is_open())
    {
      perror(("Error writing to stats file: " + output_stats_file_name).c_str());
      continue;
    }

    statsfile << std::fixed << std::setprecision(3);
    statsfile << "Average,Median,Std. Dev\n";
    statsfile << std::fixed << std::setprecision(3)
              << average << "," << median << "," << std_dev << "\n";
    statsfile.close();
  }

  // child process done, exit the program
  printf("Child process is terminated. (pid: %d)\n", getpid());
  exit(0);
}

void create_processes_and_sort(vector<string> class_names, int num_processes, int num_threads)
{
  vector<pid_t> child_pids;
  int classes_per_process = max(class_names.size() / num_processes, 1ul);

  int l = 0;
  for (int i = 0; i < num_processes && l < class_names.size(); i++)
  {
    int r = l + classes_per_process + (i < (class_names.size() % num_processes) ? 1 : 0);
    if (r > class_names.size())
      r = class_names.size();
    vector<string> sub_classes(class_names.begin() + l, class_names.begin() + r);

    pid_t pid = fork();
    if (pid == 0)
    {
      process_classes(sub_classes, num_threads);
    }
    else if (pid > 0)
    {
      child_pids.push_back(pid);
    }
    else
    {
      perror("fork error");
      exit(1);
    }

    l = r;
  }

  for (size_t i = 0; i < child_pids.size(); ++i)
  {
    int status;
    waitpid(child_pids[i], &status, 0);
  }
}
