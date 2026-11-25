/*
 * Dining Philosophers Problem Simulation (CO3)
 *
 * Build:  gcc -O2 -pthread -o dining dining_philosophers.c
 * Run:    ./dining <mode> [seconds]
 *
 * Modes:
 *   naive    - each philosopher takes LEFT fork then RIGHT fork.
 *              Demonstrates DEADLOCK (circular wait). A watchdog detects it.
 *   ordered  - Resource hierarchy: always pick the lower-numbered fork first.
 *              Breaks circular wait => deadlock is impossible.
 *   fair     - Monitor (mutex + condition variable) with ticket ordering:
 *              a philosopher may eat only if neither neighbour is eating AND
 *              no hungry neighbour holds an older ticket. Prevents deadlock
 *              and reduces starvation (oldest request is served first).
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define N 5
#define MIN_MS 50
#define MAX_MS 200
#define DEADLOCK_TIMEOUT 2.0 /* seconds without any meal => deadlock */

typedef enum { NAIVE, ORDERED, FAIR } Mode;
enum { THINKING, HUNGRY, EATING };

static Mode mode;
static int run_secs = 10;

static pthread_mutex_t fork_m[N];
static pthread_mutex_t mon = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;

static int state[N];
static long ticket[N];
static long next_ticket = 0;

static atomic_int stop_flag;
static atomic_long total_meals;
static atomic_int holds_left[N];

/* per-philosopher stats (each written only by its own thread) */
static int meals[N];
static double wait_sum[N], wait_max[N];

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void msleep(int ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

static int rnd_ms(unsigned *seed) {
    return MIN_MS + rand_r(seed) % (MAX_MS - MIN_MS + 1);
}

/* ---------- fair-mode monitor helpers ---------- */
static int can_eat(int i) {
    int l = (i + N - 1) % N, r = (i + 1) % N;
    if (state[l] == EATING || state[r] == EATING) return 0;
    if (state[l] == HUNGRY && ticket[l] < ticket[i]) return 0;
    if (state[r] == HUNGRY && ticket[r] < ticket[i]) return 0;
    return 1;
}

/* returns 1 if forks acquired, 0 if simulation stopped meanwhile */
static int acquire(int i) {
    int left = i, right = (i + 1) % N;

    if (mode == NAIVE) {
        pthread_mutex_lock(&fork_m[left]);
        atomic_store(&holds_left[i], 1);
        msleep(50); /* widen the race window so deadlock is reproducible */
        pthread_mutex_lock(&fork_m[right]);
        return 1;
    }
    if (mode == ORDERED) {
        int first = left < right ? left : right;
        int second = left < right ? right : left;
        pthread_mutex_lock(&fork_m[first]);
        pthread_mutex_lock(&fork_m[second]);
        return 1;
    }
    /* FAIR */
    pthread_mutex_lock(&mon);
    state[i] = HUNGRY;
    ticket[i] = next_ticket++;
    while (!atomic_load(&stop_flag) && !can_eat(i))
        pthread_cond_wait(&cv, &mon);
    if (atomic_load(&stop_flag)) {
        state[i] = THINKING;
        pthread_mutex_unlock(&mon);
        return 0;
    }
    state[i] = EATING;
    pthread_mutex_unlock(&mon);
    return 1;
}

static void release(int i) {
    int left = i, right = (i + 1) % N;
    if (mode == FAIR) {
        pthread_mutex_lock(&mon);
        state[i] = THINKING;
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mon);
        return;
    }
    pthread_mutex_unlock(&fork_m[left]);
    pthread_mutex_unlock(&fork_m[right]);
    atomic_store(&holds_left[i], 0);
}

static void *philosopher(void *arg) {
    int i = (int)(long)arg;
    unsigned seed = (unsigned)time(NULL) ^ (i * 7919u);
    int first = 1;

    while (!atomic_load(&stop_flag)) {
        if (!first) msleep(rnd_ms(&seed)); /* THINK (skipped first round so all start hungry) */
        first = 0;

        double t0 = now();
        if (!acquire(i)) break;
        double w = now() - t0;
        wait_sum[i] += w;
        if (w > wait_max[i]) wait_max[i] = w;

        msleep(rnd_ms(&seed)); /* EAT */
        meals[i]++;
        atomic_fetch_add(&total_meals, 1);
        release(i);
    }
    return NULL;
}

static void print_results(void) {
    printf("\n%-6s %8s %14s %14s\n", "Phil", "Meals", "AvgWait(ms)", "MaxWait(ms)");
    printf("------------------------------------------------\n");
    long sum = 0;
    double sq = 0, gmax = 0;
    for (int i = 0; i < N; i++) {
        double avg = meals[i] ? wait_sum[i] / meals[i] * 1000 : 0;
        printf("P%-5d %8d %14.1f %14.1f\n", i, meals[i], avg, wait_max[i] * 1000);
        sum += meals[i];
        sq += (double)meals[i] * meals[i];
        if (wait_max[i] > gmax) gmax = wait_max[i];
    }
    double jain = sq > 0 ? ((double)sum * sum) / (N * sq) : 0;
    printf("------------------------------------------------\n");
    printf("Total meals: %ld | Worst wait: %.1f ms | Jain fairness index: %.3f (1.0 = perfect)\n",
           sum, gmax * 1000, jain);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <naive|ordered|fair> [seconds]\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "naive")) mode = NAIVE;
    else if (!strcmp(argv[1], "ordered")) mode = ORDERED;
    else if (!strcmp(argv[1], "fair")) mode = FAIR;
    else { fprintf(stderr, "Unknown mode '%s'\n", argv[1]); return 1; }
    if (argc > 2) run_secs = atoi(argv[2]);

    for (int i = 0; i < N; i++) pthread_mutex_init(&fork_m[i], NULL);

    printf("=== Dining Philosophers: mode=%s, duration=%ds, philosophers=%d ===\n",
           argv[1], run_secs, N);

    pthread_t th[N];
    for (long i = 0; i < N; i++) pthread_create(&th[i], NULL, philosopher, (void *)i);

    double start = now(), last_change = start;
    long last_meals = 0;
    while (now() - start < run_secs) {
        msleep(100);
        long m = atomic_load(&total_meals);
        if (m != last_meals) { last_meals = m; last_change = now(); }

        if (mode == NAIVE && now() - last_change > DEADLOCK_TIMEOUT) {
            printf("\n*** DEADLOCK DETECTED after %.1fs (no meals for %.1fs) ***\n",
                   now() - start, DEADLOCK_TIMEOUT);
            for (int i = 0; i < N; i++)
                printf("  P%d holds fork %d and waits for fork %d (held by P%d)\n",
                       i, i, (i + 1) % N, (i + 1) % N);
            printf("  => circular wait: all four Coffman conditions hold.\n");
            printf("Meals eaten before deadlock: %ld\n", m);
            fflush(stdout);
            _exit(0); /* threads are stuck; cannot join */
        }
    }

    atomic_store(&stop_flag, 1);
    pthread_mutex_lock(&mon);
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mon);
    for (int i = 0; i < N; i++) pthread_join(th[i], NULL);

    if (mode == NAIVE) printf("\nNo deadlock occurred in this run (try again; it is timing dependent).\n");
    else printf("\nNo deadlock occurred.\n");
    print_results();
    return 0;
}
