/*
The MIT License (MIT)

Copyright (c) 2015 Mike Taghavi (mitghi) <mitghi@me.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <assert.h>

#include "thpool.h"
#include "pqueue.h"

#define GAP 2

#define MPANIC(x) ; assert(x != NULL) //조건에 맞지않으면 중단

static void insert_job(Priqueue *priqueue,struct job *newjob);  //add_job?
static struct job pop_job(Priqueue *priqueue);
static struct void swap_job(Priqueue *priqueue,unsigned int a, unsigned int b);

/*
static void insert_node(Priqueue *heap, Node* node);
static Node* pop_node(Priqueue *heap);
static void swap_node(Priqueue *heap, unsigned int a, unsigned int b);
*/

Priqueue* priqueue_init(int init_length){
  unsigned int mutex_status;
  Priqueue *priqueue = (Priqueue *) malloc(sizeof(Priqueue)) MPANIC(priqueue);  
  const size_t qsize = initial_length * sizeof(*priqueue->array);
  priqueue->hasjobs = (bsem *)malloc(sizeof(struct bsem));
  if(priqueue->hasjobs==NULL){
    return NULL;
  }

  mutex_status = pthread_mutex_init(&(priqueue->lock), NULL);
  bsem_init(priqueue->has_jobs, 0);
  if (mutex_status != 0) goto error;
  
  priqueue->head = NULL;
  priqueue->heap_size = initial_length; //need?
  priqueue->occupied = 0;
  priqueue->current = 1;
  priqueue->array = malloc(qsize) MPANIC(priqueue->array);

  memset(priqueue->array, 0x00, qsize);

  return priqueue;
  
 error:
  free(priqueue);

  return NULL;
}

static MHEAP_API MHEAPSTATUS realloc_heap(Priqueue *priqueue){

  if (priqueue->occupied >= priqueue->heap_size){
    const size_t arrsize = sizeof(*priqueue->array);
    
    void **resized_queue;
    resized_queue = realloc(priqueue->array, (2 * priqueue->heap_size) * arrsize);
    if (resized_queue != NULL){
      priqueue->heap_size *= 2;
      priqueue->array = (job**) resized_queue;
      memset( (priqueue->array + priqueue->occupied + 1) , 0x00, (priqueue->heap_size / GAP) * arrsize );
      return MHEAP_OK;
    } else return MHEAP_REALLOCERROR;
  }

  return MHEAP_NOREALLOC;
}

//jobqueue_push 
MHEAP_API void priqueue_insert(Priqueue *priqueue, job *newjob){

  //Node *node = (Node *) malloc(sizeof(Node)) MPANIC(node);
  //node->priority = priority;
  //node->data = data;
  newjob->arg
  pthread_mutex_lock(&(heap->lock));
  insert_node(priqueue,newjob);
  pthread_mutex_unlock(&(heap->lock));
}

static void insert_node(Priqueue *priqueue, job* newjob){

  if (priqueue->current == 1 && priqueue->array[1] == NULL){
    priqueue->head = newjob;
    priqueue->array[1] = newjob;
    priqueue->array[1]->index = priqueue->current;
    priqueue->occupied++;
    priqueue->current++;

    return;
  }

  if(priqueue->occupied >= priqueue->heap_size) {
    unsigned int realloc_status = realloc_heap(heap);
    assert(realloc_status == MHEAP_OK);
  }
  
  if(priqueue->occupied <= priqueue->heap_size){
    newjob->index = priqueue->current;
    priqueue->array[priqueue->current] = newjob;

    int parent = (priqueue->current / GAP);

    if (priqueue->array[parent]->arg->net.priority < newjob->arg->net.priority){ //확인 필요
      priqueue->occupied++;
      priqueue->current++;
      int depth = priqueue->current / GAP;
      int traverse = newjob->index;
      
      while(depth >= 1){
	
	if (traverse == 1) break;
	unsigned int parent = (traverse / GAP);
	
        if(priqueue->array[parent]->arg->net.priority < priqueue->array[traverse]->arg->net.priority){
	  swap_node(priqueue, parent , traverse);
          traverse = priqueue->array[parent]->index;
        }
	depth --;
      }
      priqueue->head = priqueue->array[1];
    } else {
      priqueue->occupied++;
      priqueue->current++;
    }
  }
  bsem_post(priqueue->hasjobs);
}

void swap_node(Priqueue *priqueue, unsigned int parent, unsigned int child){
  job *tmp = priqueue->array[parent];

  priqueue->array[parent] = priqueue->array[child];
  priqueue->array[parent]->index = tmp->index;

  priqueue->array[child] = tmp;
  priqueue->array[child]->index = child;
  
}
//jobqueue_pull 맞춰줘야함 pop_node 에서
MHEAP_API job *priqueue_pop(Priqueue *priqueue){
  job *job_p = NULL;
  
  pthread_mutex_lock(&(priqueue->lock));
  job_p = pop_node(priqueue);
  pthread_mutex_unlock(&(priqueue->lock));

  return job_p;
}

static job *pop_node(Priqueue *priqueue){
  job *job_p = NULL;
  unsigned int i;
  unsigned int depth;

  if (priqueue->current == 1) return job_p;
  
  else if (priqueue->current >= 2 ){
    job_p = priqueue->array[1];
    priqueue->array[1] = priqueue->array[priqueue->current - 1];
    priqueue->current -= 1;
    priqueue->occupied -= 1;
    
    depth = (priqueue->current -1) / 2;

    for(i = 1; i<=depth; i++){
      
      if (priqueue->array[i]->arg->net.priority < priqueue->array[i * GAP]->arg->net.priority || priqueue->array[i]->arg->net.priority < priqueue->array[(i * GAP)+1]->arg->net.priority){
        unsigned int biggest = priqueue->array[i * GAP]->arg->net.priority > priqueue->array[(i * GAP)+1]->arg->net.priority ?
	      priqueue->array[(i * GAP)]->index  :
	      priqueue->array[(i * GAP)+1]->index;
        swap_node(priqueue,i,biggest);
      }
    }bsem_post(priqueue->hasjobs);
  }

  return job_p;
}

MHEAP_API void priqueue_free(Priqueue *priqueue){  
  bsem_reset(priqueue->hasjobs);
  if (priqueue->current >= 2 ) {
    unsigned int i;
    for (i = 1; i <= priqueue->current; i++) priqueue_job_free(priqueue,priqueue->array[i]);
  }

  free(priqueue->head);
  free(*priqueue->array);
  free(priqueue->array);
  free(priqueue);
}

MHEAP_API void priqueue_job_free(Priqueue *priqueue,job *job_p){
  //if (node != NULL) free(node->data->data);
  free(job_p);  
}