#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sys/types.h>      /* key_t, sem_t, pid_t      */
#include <sys/wait.h>
#include <sys/shm.h>        /* shmat(), IPC_RMID        */
#include <errno.h>          /* errno, ECHILD            */
#include <semaphore.h>      /* sem_open(), sem_destroy(), sem_wait().. */
#include <fcntl.h>          /* O_CREAT, O_EXEC          */

#include <sys/mman.h>    /* New for shared memory */
#include <unistd.h>   /* to have fork() */
#include <pthread.h>

sem_t *sem;                   /*      synch semaphore         *//*shared */
int *pcount;
//macro to measure time taken by a block of code
// usage: MAKE_TIME_MEASUREMENT( code );
// the time taken will be stored in the variable time_taken
#define START_TIMER(start_ts) \
    if (clock_gettime(CLOCK_MONOTONIC, &(start_ts)) == -1) { \
        perror("clock_gettime START"); \
        exit(EXIT_FAILURE); \
    }

#define STOP_TIMER_NS(start_ts, end_ts, result_ns) \
    if (clock_gettime(CLOCK_MONOTONIC, &(end_ts)) == -1) { \
        perror("clock_gettime END"); \
        exit(EXIT_FAILURE); \
    } \
    (result_ns) = ((end_ts.tv_sec - start_ts.tv_sec) * 1000000000LL) + \
                  (end_ts.tv_nsec - start_ts.tv_nsec);

long long time_taken;
struct timespec start, end;
#define MAKE_TIME_MEASUREMENT(code) \
    START_TIMER(start) \
    code \
    STOP_TIMER_NS(start,end,time_taken) \




typedef struct config { //struct to hold command line arguments
    int num_threads;
    int num_count;
    int range;
    int seed;
    int padding;
    int collect_data;
}config_t;

typedef struct args{
    int *array;
    int start;
    int size;
    int value;
    int result;
    int padding; // should be null for the most part.
} args_t;
pthread_mutex_t my_mutex; // Declaration
int Global_count = 0;

void print_config(config_t *config){
    printf("Number of threads/prosses: %d\n", config->num_threads);
    printf("Number of random numbers to generate: %d\n", config->num_count);
    printf("Range of random numbers:  X mod %d\n", config->range);
    printf("Seed for random number generator: %d\n", config->seed);
    printf("Padding to avoid false sharing: %d\n", config->padding);
}

//function to count the number of times a value appears in an array
//this is more general than the book's version which only counts 3s 
// in the whole array this is for a subarray defined by start and size
// doing it this way will make it easier to parallelize later
int count_int_in_range(int *array,int start,int size, int value){
    int count = 0;
    int i = start;
    int end = start + size;
    do{
        if(array[i++] == value){
            ++count;
        }
    }while(i < end);
    return count;
}

/*
    this function parses the command line arguments and returns a config struct
    if an invalid argument is passed, the program will print an error message and exit
*/
config_t *parse_args(int argc, char *argv[]){
    config_t *config = malloc(sizeof(config_t));
    //default values
    config->num_threads = 1;
    config->num_count = 1000000;
    config->range = 10;//this is inclusive so the random numbers will be between 0 and range
    config->seed = time(NULL);
    config->padding = 0;

    for(int i=0; i<argc; i++){ //iterate through all arguments
        if(argv[i][0] == '-'){
            switch(argv[i][1]){ //check the character after the '-'
                case 't':
                    //number of threads
                    config->num_threads = atoi(argv[i+1]);
                    i++;
                    break;
                case 'n':
                    //number of random numbers to generate
                    config->num_count = atoi(argv[i+1]);
                    i++;
                    break;
                case 'r':
                    //range of random numbers
                    config->range = atoi(argv[i+1]);
                    i++;
                    break;
                case 's':
                    //seed for random number generator
                    config->seed = atoi(argv[i+1]);
                    i++;
                    break;
                case 'p':
                    config->padding = atoi(argv[i+1]);
                    i++;
                    //padding to avoid false sharing
                    break;
                case 'c':
                    // we run the data collections
                    config->collect_data = 1;
                    break;
                default:
                    printf("Invalid argument: %s\n", argv[i]);
                    exit(1);
            }
        }
    }
    config->range += 1; //make range inclusive
    return config;
};

int Global_count;
void *Threaded_count_function(void* args){
    args_t *args_cast = (args_t *)args;
    int i = args_cast->start;
    int end = args_cast->start + args_cast->size;
    do{
        if(args_cast->array[i++] == args_cast->value){
            args_cast->result++;
        }
    }while(i < end);
    pthread_mutex_lock(&my_mutex);
        Global_count+=args_cast->result;
    pthread_mutex_unlock(&my_mutex);
    return NULL;
};

// defalt general threading function that we will switch with different thread functions
void threded_count_int_in_range(int *array, config_t *config, int value){
    Global_count = 0;//set this to 0 
    pthread_t threads[config->num_threads];

    args_t *args = malloc(sizeof(args_t) * config->num_threads);
    args_t *args_padded = malloc(sizeof(args_t) * config->num_threads * config->padding);
    for(int i=0; i<config->num_threads; i++){
        //create threads and pass them the args struct
        args[i].array = array;
        args[i].start = i * (config->num_count / config->num_threads);
        args[i].size = config->num_count / config->num_threads;
        args[i].value = value;
        args[i].result = 0;
        if(config->padding >1){
            int d = i*config->padding;
            args_padded[d].array = array;
            args_padded[d].start = i * (config->num_count / config->num_threads);
            args_padded[d].size = config->num_count / config->num_threads;
            args_padded[d].value = value;
            args_padded[d].result = 0;
        }
        //create thread
        if(config->padding > 1){
            pthread_create(&threads[i], NULL, Threaded_count_function, (void *)&args_padded[i*config->padding]);
        }else{
            pthread_create(&threads[i], NULL, Threaded_count_function, (void *)&args[i]);
        }
    }
    for(int i=0; i<config->num_threads; i++){
        //join threads
        pthread_join(threads[i], NULL);
    }
    free(args);
    free(args_padded);

}


void make_data(config_t *config){
 
    printf("making the data.... (data will be saved in .dat files)\n");

    FILE *Serial_data = fopen("Serial_data.csv","w");
    fprintf(Serial_data,"size of n,number of threads,time in ns\n");

    FILE *cuda_data = fopen("cuda_data.csv","w");
    fprintf(Serial_data,"size of n,number of threads,time in ns\n");

    FILE *thread_data_16 = fopen("thread_data16.csv","w");
    fprintf(thread_data_16,"size of n,number of threads,time in ns\n");
    
    int arrsize[4] = {100000 , 1000000, 10000000, 100000000};
    for(int thread = 4 ; thread <= 64 ; thread*=2 ){
        for(int i=0;i < 4; i++ ){//loops though the arr for the size of n
            for(int r=0; r<6;r++){
                  config->num_threads = thread;
                  config->num_count = arrsize[i];
                  int *random_numbers = malloc(sizeof(int) * config->num_count);
                  for (int i = 0; i < config->num_count; i++) {
                       random_numbers[i] = rand() % config->range;
                   }
                  MAKE_TIME_MEASUREMENT(count_int_in_range(random_numbers,0,arrsize[i], 3);)
                  fprintf(Serial_data,"%d,%d,%lld\n",arrsize[i],thread,time_taken);
                  config->padding = 16;
                  MAKE_TIME_MEASUREMENT(threded_count_int_in_range(random_numbers,config,3);)
                  fprintf(thread_data_16 ,"%d,%d,%lld\n",arrsize[i],thread,time_taken);
                  config->padding = 0;
                  fflush(stdout);
                  fflush(Serial_data);

                  //MAKE_TIME_MEASUREMENT(make_processes(config,random_numbers);)
                  //fprintf(fork_data ,"%d,%d,%lld\n",arrsize[i],thread,time_taken);
                    
                  fflush(stdout);
                  free(random_numbers);
            }
        }
    }
    fprintf(stdout,"testing baseline.....");
    int *random_numbers = malloc(sizeof(int) * config->num_count);
    for(int i=0; i<1000; i++){
        for (int i = 0; i < config->num_count; i++) {
            random_numbers[i] = rand() % config->range;
        }
        MAKE_TIME_MEASUREMENT(count_int_in_range(random_numbers,0,config->num_count, 3);)
        fprintf(Serial_data,"%d,%d,%lld\n",config->num_count,0,time_taken);
   }
   free(random_numbers);
   fclose(Serial_data);
   //fclose(fork_data);
}







//command line arguments
/*
-t the number of threads to use
-n the number of random numbers to generate
-r the range of the random numbers (0 to r) inclusive
-s the seed for the random number generator
-h print a help message
-p padding to avoid false sharing
-c collects data
*/
int main(int argc , char *argv[]) {

    // FIX: Set stdout to unbuffered mode to prevent *any* buffering issues 
    // that lead to duplicated output after fork(). This is the most robust fix.
    setvbuf(stdout, NULL, _IONBF, 0); 
    
    config_t *config = parse_args(argc, argv); //parse command line arguments


    if(config->collect_data > 0){
        make_data(config);
    }
    return 0;
}
