#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <stdbool.h>

#define PACKET_SIZE 1500      // Standard MTU size
#define POOL_SIZE 10000       // Size of pre-allocated pool
#define CHUNK_SIZE 64         // Size of expansion chunks

typedef struct Packet {
    unsigned char data[PACKET_SIZE];
    size_t length;
    unsigned int packet_id;
    int priority;
    struct timespec timestamp;
} Packet;

typedef struct PacketNode {
    Packet packet;
    struct PacketNode* next;
    bool in_use;
} PacketNode;

typedef struct MemoryPool {
    PacketNode* free_list;
    PacketNode* all_nodes;
    size_t total_nodes;
    size_t available_nodes;
    pthread_mutex_t lock;
    bool initialized;
} MemoryPool;

static MemoryPool global_pool = {0};



int pool_initialize() {
    if (global_pool.initialized) {
        return 0;
    }

    if (pthread_mutex_init(&global_pool.lock, NULL) != 0) {
        return -1;
    }

    global_pool.all_nodes = (PacketNode*)calloc(POOL_SIZE, sizeof(PacketNode));
    if (!global_pool.all_nodes) {
        pthread_mutex_destroy(&global_pool.lock);
        return -1;
    }

    // Initialize the free list
    for (size_t i = 0; i < POOL_SIZE - 1; i++) {
        global_pool.all_nodes[i].next = &global_pool.all_nodes[i + 1];
        global_pool.all_nodes[i].in_use = false;
    }
    global_pool.all_nodes[POOL_SIZE - 1].next = NULL;
    global_pool.all_nodes[POOL_SIZE - 1].in_use = false;

    global_pool.free_list = &global_pool.all_nodes[0];
    global_pool.total_nodes = POOL_SIZE;
    global_pool.available_nodes = POOL_SIZE;
    global_pool.initialized = true;

    return 0;
}

static int expand_pool() {
    PacketNode* new_nodes = (PacketNode*)calloc(CHUNK_SIZE, sizeof(PacketNode));
    if (!new_nodes) {
        return -1;
    }

    // Initialize new nodes
    for (size_t i = 0; i < CHUNK_SIZE - 1; i++) {
        new_nodes[i].next = &new_nodes[i + 1];
        new_nodes[i].in_use = false;
    }
    new_nodes[CHUNK_SIZE - 1].next = NULL;
    new_nodes[CHUNK_SIZE - 1].in_use = false;

    // Add to free list
    new_nodes[CHUNK_SIZE - 1].next = global_pool.free_list;
    global_pool.free_list = &new_nodes[0];

    global_pool.total_nodes += CHUNK_SIZE;
    global_pool.available_nodes += CHUNK_SIZE;

    return 0;
}

Packet* pool_allocate_packet() {
    if (!global_pool.initialized && pool_initialize() != 0) {
        return NULL;
    }

    pthread_mutex_lock(&global_pool.lock);

    if (!global_pool.free_list) {
        if (expand_pool() != 0) {
            pthread_mutex_unlock(&global_pool.lock);
            return NULL;
        }
    }

    PacketNode* node = global_pool.free_list;
    global_pool.free_list = node->next;
    node->in_use = true;
    global_pool.available_nodes--;

    node->packet.length = 0;
    node->packet.packet_id = global_pool.total_nodes - global_pool.available_nodes;
    node->packet.priority = 0;
    clock_gettime(CLOCK_REALTIME, &node->packet.timestamp);

    pthread_mutex_unlock(&global_pool.lock);
    return &node->packet;
}

void pool_free_packet(Packet* packet) {
    if (!packet) {
        return;
    }

    PacketNode* node = (PacketNode*)((char*)packet - offsetof(PacketNode, packet));

    pthread_mutex_lock(&global_pool.lock);

    if (node->in_use) {
        node->next = global_pool.free_list;
        global_pool.free_list = node;
        node->in_use = false;
        global_pool.available_nodes++;
    }

    pthread_mutex_unlock(&global_pool.lock);
}

void pool_cleanup() {
    if (!global_pool.initialized) {
        return;
    }

    pthread_mutex_lock(&global_pool.lock);
    free(global_pool.all_nodes);
    global_pool.free_list = NULL;
    global_pool.all_nodes = NULL;
    global_pool.total_nodes = 0;
    global_pool.available_nodes = 0;
    global_pool.initialized = false;
    pthread_mutex_unlock(&global_pool.lock);
    pthread_mutex_destroy(&global_pool.lock);
}
