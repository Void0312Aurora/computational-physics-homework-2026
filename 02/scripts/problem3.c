#include <stdint.h>
#include <stdio.h>

static const char *endianness_name(void) {
    uint32_t probe = 0x01020304u;
    const unsigned char *bytes = (const unsigned char *)&probe;

    if (bytes[0] == 0x04) {
        return "little-endian";
    }
    if (bytes[0] == 0x01) {
        return "big-endian";
    }
    return "mixed/unknown";
}

int main(void) {
    uint32_t probe = 0x01020304u;
    const unsigned char *bytes = (const unsigned char *)&probe;

    puts("Problem 3 | Endianness");
    printf("machine byte order             : %s\n", endianness_name());
    printf("probe bytes                    : %02x %02x %02x %02x\n", bytes[0],
           bytes[1], bytes[2], bytes[3]);

    return 0;
}
