#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#define PACKET_SIZE 1500  // Standard MTU size
#define MAX_PACKETS 10000 // Maximum number of packets to track

typedef struct Packet {
    unsigned char* data;
    size_t length;
    unsigned int packet_id;
    int priority;
    struct timespec timestamp;
} Packet;

typedef struct PacketStats {
    unsigned long allocations;
    unsigned long deallocations;
    unsigned long failed_allocations;
    size_t total_bytes_allocated;
} PacketStats;

static PacketStats stats = {0};

Packet* packet_create(size_t size) {
    if (size == 0 || size > PACKET_SIZE) {
        errno = EINVAL;
        stats.failed_allocations++;
        return NULL;
    }

    Packet* packet = (Packet*)malloc(sizeof(Packet));
    if (!packet) {
        stats.failed_allocations++;
        return NULL;
    }

    packet->data = (unsigned char*)malloc(size);
    if (!packet->data) {
        free(packet);
        stats.failed_allocations++;
        return NULL;
    }

    packet->length = size;
    packet->packet_id = stats.allocations + 1;
    packet->priority = 0;
    clock_gettime(CLOCK_REALTIME, &packet->timestamp);

    stats.allocations++;
    stats.total_bytes_allocated += size;

    return packet;
}

void packet_destroy(Packet* packet) {
    if (!packet) {
        return;
    }

    free(packet->data);
    free(packet);
    stats.deallocations++;
}

int packet_write(Packet* packet, const unsigned char* data, size_t length) {
    if (!packet || !data || length > packet->length) {
        return -1;
    }

    memcpy(packet->data, data, length);
    return 0;
}

int packet_read(const Packet* packet, unsigned char* buffer, size_t buffer_size, size_t* bytes_read) {
    if (!packet || !buffer || buffer_size < packet->length) {
        return -1;
    }

    memcpy(buffer, packet->data, packet->length);
    *bytes_read = packet->length;
    return 0;
}

void packet_set_priority(Packet* packet, int priority) {
    if (packet) {
        packet->priority = priority;
    }
}

void get_packet_stats(PacketStats* current_stats) {
    if (current_stats) {
        memcpy(current_stats, &stats, sizeof(PacketStats));
    }
}

void reset_packet_stats() {
    memset(&stats, 0, sizeof(PacketStats));
}
