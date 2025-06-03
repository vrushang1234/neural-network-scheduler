#include "p1_process.h"
#include <pthread.h>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

struct BubbleSortArgs {
    int thread_index;
    ParallelBubbleSorter* ctx;

    BubbleSortArgs(ParallelBubbleSorter* c, int i) : ctx(c), thread_index(i) {}
};

ParallelBubbleSorter::ParallelBubbleSorter(vector<student>& original_list, int num_threads)
{
    this->num_threads = num_threads;
    this->sorted_list = original_list;
}

void* ParallelBubbleSorter::thread_init(void* args) {
    BubbleSortArgs* bs_args = (BubbleSortArgs*)args;
    ParallelBubbleSorter* ctx = bs_args->ctx;
    int tid = bs_args->thread_index;

    int size = ctx->sorted_list.size();
    int chunk = size / ctx->num_threads;
    int rem = size % ctx->num_threads;

    int start = tid * chunk + min(tid, rem);
    int end = start + chunk + (tid < rem ? 1 : 0);

    for (int i = start; i < end - 1; ++i) {
        for (int j = start; j < end - i + start - 1; ++j) {
            if (ctx->sorted_list[j].grade < ctx->sorted_list[j + 1].grade ||
                (ctx->sorted_list[j].grade == ctx->sorted_list[j + 1].grade &&
                 ctx->sorted_list[j].id < ctx->sorted_list[j + 1].id)) {
                swap(ctx->sorted_list[j], ctx->sorted_list[j + 1]);
            }
        }
    }

    delete bs_args;
    return nullptr;
}

vector<student> ParallelBubbleSorter::run_sort() {
    vector<pthread_t> threads(num_threads);
    for (int i = 0; i < num_threads; i++) {
        BubbleSortArgs* args = new BubbleSortArgs(this, i);
        pthread_create(&threads[i], nullptr, ParallelBubbleSorter::thread_init, args);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
    return sorted_list;
}

