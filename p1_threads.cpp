#include <vector>
#include <pthread.h>
#include <cstring>
#include <string>
#include <cstdlib>
#include <iostream>

#include "p1_process.h"
#include "p1_threads.h"

using namespace std;

struct MergeSortArgs
{
  int thread_index;
  ParallelMergeSorter *ctx;

  MergeSortArgs(ParallelMergeSorter *ctx, int thread_index)
  {
    this->ctx = ctx;
    this->thread_index = thread_index;
  }
};

// Class constructor
ParallelMergeSorter::ParallelMergeSorter(vector<student> &original_list, int num_threads)
{
  this->threads = vector<pthread_t>();
  this->sorted_list = vector<student>(original_list);
  this->num_threads = num_threads;
}

// This function will be called by each child process to perform multithreaded sorting
vector<student> ParallelMergeSorter::run_sort()
{

  for (int i = 0; i < num_threads; i++)
  {

    MergeSortArgs *args = new MergeSortArgs(this, i);
    pthread_t thread;
    threads.push_back(thread);
    int rc = pthread_create(&threads[i], NULL, ParallelMergeSorter::thread_init, (void *)args);
    if (rc)
    {
      cerr << "[ERROR] pthread_create failed for thread " << i << " with code " << rc << "\n";
    }
  }
  for (int i = 0; i < num_threads; ++i)
  {
    pthread_join(threads[i], NULL);
  }
  this->merge_threads();

  return this->sorted_list;
}

// Standard merge sort implementation
void ParallelMergeSorter::merge_sort(int lower, int upper)
{

  if (upper - lower <= 1)
    return;

  int middle = (lower + upper) / 2;

  merge_sort(lower, middle);

  merge_sort(middle, upper);

  merge(lower, middle, upper);
}

// Standard merge implementation for merge sort
void ParallelMergeSorter::merge(int lower, int middle, int upper)
{

  vector<student> temp;
  int left = lower;
  int right = middle;

  while (left < middle && right < upper)
  {
    if (sorted_list[left].grade > sorted_list[right].grade ||
        (sorted_list[left].grade == sorted_list[right].grade &&
         sorted_list[left].id > sorted_list[right].id))
    {
      temp.push_back(sorted_list[left]);
      left++;
    }
    else
    {
      temp.push_back(sorted_list[right]);
      right++;
    }
  }

  while (left < middle)
  {
    temp.push_back(sorted_list[left]);
    left++;
  }

  while (right < upper)
  {
    temp.push_back(sorted_list[right]);
    right++;
  }

  for (int i = 0; i < temp.size(); ++i)
  {
    sorted_list[lower + i] = temp[i];
  }
}

// This function will be used to merge the resulting sorted sublists together
void ParallelMergeSorter::merge_threads()
{

  int total = sorted_list.size();
  int remainder = total % num_threads;
  int offset = 0;
  vector<student> temp(total);

  vector<int> bounds;
  for (int i = 0; i < num_threads; ++i)
  {
    int chunk_size = total / num_threads + (i < remainder ? 1 : 0);
    bounds.push_back(offset);
    offset += chunk_size;
  }
  bounds.push_back(total);

  vector<student> merged(sorted_list.begin() + bounds[0], sorted_list.begin() + bounds[1]);

  for (int i = 1; i < num_threads; ++i)
  {
    vector<student> next(sorted_list.begin() + bounds[i], sorted_list.begin() + bounds[i + 1]);
    vector<student> merged_temp;
    size_t a = 0, b = 0;

    while (a < merged.size() && b < next.size())
    {
      if (merged[a].grade > next[b].grade ||
          (merged[a].grade == next[b].grade &&
           merged[a].id > next[b].id))
        merged_temp.push_back(merged[a++]);
      else
        merged_temp.push_back(next[b++]);
    }

    while (a < merged.size())
      merged_temp.push_back(merged[a++]);
    while (b < next.size())
      merged_temp.push_back(next[b++]);

    merged = merged_temp;
  }

  sorted_list = merged;
}

// This function is the start routine for the created threads, it should perform merge sort on its assigned sublist
// Since this function is static (pthread_create must take a static function), we cannot access "this" and must use ctx instead
void *ParallelMergeSorter::thread_init(void *args)
{
  MergeSortArgs *sort_args = (MergeSortArgs *)args;
  int thread_index = sort_args->thread_index;
  ParallelMergeSorter *ctx = sort_args->ctx;

  int work_per_thread = ctx->sorted_list.size() / ctx->num_threads;

  int lower = thread_index * work_per_thread + (thread_index < (ctx->sorted_list.size() % ctx->num_threads) ? thread_index : (ctx->sorted_list.size() % ctx->num_threads));
  int upper = lower + work_per_thread + (thread_index < (ctx->sorted_list.size() % ctx->num_threads) ? 1 : 0);
  ctx->merge_sort(lower, upper);
  // Free the heap allocation
  delete sort_args;
  return NULL;
}
