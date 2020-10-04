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

#include "pqueue.h"

#define GAP 2

#define MPANIC(x) ; assert(x != NULL)

static void insert_node(Priqueue *heap, Node* node);
static Node* pop_node(Priqueue *heap);
static void swap_node(Priqueue *heap, unsigned int a, unsigned int b);

MHEAP_API Priqueue* priqueue_initialize(int initial_length){
  unsigned int mutex_status;
  
  Priqueue *heap = (Priqueue *) malloc(sizeof(Priqueue)) MPANIC(heap);  
  const size_t hsize = initial_length * sizeof(*heap->array);

  mutex_status = pthread_mutex_init(&(heap->lock), NULL);
  if (mutex_status != 0) goto error;
  
  heap->head = NULL;
  heap->heap_size = initial_length;
  heap->occupied = 0;
  heap->current = 1;
  heap->array = malloc(hsize) MPANIC(heap->array);

  memset(heap->array, 0x00, hsize);

  return heap;
  
 error:
  free(heap);

  return NULL;
}

static MHEAP_API MHEAPSTATUS realloc_heap(Priqueue *heap){

  if (heap->occupied >= heap->heap_size){
    const size_t arrsize = sizeof(*heap->array);
    
    void **resized_heap;
    resized_heap = realloc(heap->array, (2 * heap->heap_size) * arrsize);
    if (resized_heap != NULL){
      heap->heap_size *= 2;
      heap->array = (Node**) resized_heap;
      memset( (heap->array + heap->occupied + 1) , 0x00, (heap->heap_size / GAP) * arrsize );
      return MHEAP_OK;
    } else return MHEAP_REALLOCERROR;
  }

  return MHEAP_NOREALLOC;
}


MHEAP_API void priqueue_insert(Priqueue *heap, Data *data, int priority){

  Node *node = (Node *) malloc(sizeof(Node)) MPANIC(node);
  node->priority = priority;
  node->data = data;

  pthread_mutex_lock(&(heap->lock));
  insert_node(heap,node);
  pthread_mutex_unlock(&(heap->lock));
}

static void insert_node(Priqueue *heap, Node* node){

  if (heap->current == 1 && heap->array[1] == NULL){
    heap->head = node;
    heap->array[1] = node;
    heap->array[1]->index = heap->current;
    heap->occupied++;
    heap->current++;

    return;
  }

  if(heap->occupied >= heap->heap_size) {
    unsigned int realloc_status = realloc_heap(heap);
    assert(realloc_status == MHEAP_OK);
  }
  
  if(heap->occupied <= heap->heap_size){
    node->index = heap->current;
    heap->array[heap->current] = node;

    int parent = (heap->current / GAP);

    if (heap->array[parent]->priority < node->priority){
      heap->occupied++;
      heap->current++;
      int depth = heap->current / GAP;
      int traverse = node->index;
      
      while(depth >= 1){
	
	if (traverse == 1) break;
	unsigned int parent = (traverse / GAP);
	
        if(heap->array[parent]->priority < heap->array[traverse]->priority){
	  swap_node(heap, parent , traverse);
          traverse = heap->array[parent]->index;
        }
	depth --;
      }
      heap->head = heap->array[1];
    } else {
      heap->occupied++;
      heap->current++;
    }
  }
}

void swap_node(Priqueue *heap, unsigned int parent, unsigned int child){
  Node *tmp = heap->array[parent];

  heap->array[parent] = heap->array[child];
  heap->array[parent]->index = tmp->index;

  heap->array[child] = tmp;
  heap->array[child]->index = child;
  
}

MHEAP_API Node *priqueue_pop(Priqueue *heap){
  Node *node = NULL;
  
  pthread_mutex_lock(&(heap->lock));
  node = pop_node(heap);
  pthread_mutex_unlock(&(heap->lock));

  return node;
}

static Node *pop_node(Priqueue *heap){
  Node *node = NULL;
  unsigned int i;
  unsigned int depth;

  if (heap->current == 1) return node;
  
  else if (heap->current >= 2 ){
    node = heap->array[1];
    heap->array[1] = heap->array[heap->current - 1];
    heap->current -= 1;
    heap->occupied -= 1;
    
    depth = (heap->current -1) / 2;

    for(i = 1; i<=depth; i++){
      
      if (heap->array[i]->priority < heap->array[i * GAP]->priority ||
	  heap->array[i]->priority < heap->array[(i * GAP)+1]->priority){
	
	unsigned int biggest = heap->array[i * GAP]->priority > heap->array[(i * GAP)+1]->priority ?
	  heap->array[(i * GAP)]->index  :
	  heap->array[(i * GAP)+1]->index;

	swap_node(heap,i,biggest);
      }
    }
  }

  return node;
}

MHEAP_API void priqueue_free(Priqueue *heap){  
  if (heap->current >= 2 ) {
    unsigned int i;
    for (i = 1; i <= heap->current; i++) priqueue_node_free(heap,heap->array[i]);
  }
  
  free(heap->head);
  free(*heap->array);
  free(heap->array);
  free(heap);
}

MHEAP_API void priqueue_node_free(Priqueue *heap,Node *node){
  if (node != NULL) free(node->data->data);
  free(node);  
}
