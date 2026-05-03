#include "ascon128.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int all_pass = 1;

static void print_hex(const char *label, const uint8_t *data, size_t len) {
    printf("%s = ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

static void check(const char *name, int cond) {
    if (cond) {
        printf("[PASS] %s\n", name);
    } else {
        printf("[FAIL] %s\n", name);
        all_pass = 0;
    }
}

/* Vector 1: empty PT, empty AD
   Key   = Nonce = 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
   Expected tag (from ASCON-c KAT #1): e355a03750b7da3868e79c93a8aee5f2 */
static void test_vector1(void) {
    printf("\n--- Vector 1: empty PT, empty AD ---\n");
    uint8_t key[16], nonce[16];
    for (int i = 0; i < 16; i++) { key[i] = (uint8_t)i; nonce[i] = (uint8_t)i; }

    /* encrypt */
    uint8_t ct[16]; /* only tag, no ciphertext */
    ascon128_encrypt(key, nonce, NULL, 0, NULL, 0, ct);
    print_hex("Tag", ct, 16);

    uint8_t expected_tag[16] = {
        0xe3,0x55,0x15,0x9f,0x29,0x29,0x11,0xf7,
        0x94,0xcb,0x14,0x32,0xa0,0x10,0x3a,0x8a
    };
    check("V1 tag matches", memcmp(ct, expected_tag, 16) == 0);

    /* decrypt — empty ciphertext = just tag */
    int ret = ascon128_decrypt(key, nonce, NULL, 0, ct, 16, NULL);
    check("V1 decrypt returns 0", ret == 0);

    /* tamper check */
    ct[0] ^= 0xff;
    ret = ascon128_decrypt(key, nonce, NULL, 0, ct, 16, NULL);
    check("V1 tampered tag returns -1", ret == -1);
}

/* Vector 2: PT = {0x00}, empty AD
   Key = Nonce = 00..0f
   Expected CT+Tag derived from ascon-c KAT count=2 */
static void test_vector2(void) {
    printf("\n--- Vector 2: PT = {0x00}, empty AD ---\n");
    uint8_t key[16], nonce[16];
    for (int i = 0; i < 16; i++) { key[i] = (uint8_t)i; nonce[i] = (uint8_t)i; }

    uint8_t pt[1] = {0x00};
    uint8_t ct[1 + 16];
    ascon128_encrypt(key, nonce, NULL, 0, pt, 1, ct);
    print_hex("CT+Tag", ct, 17);

    uint8_t expected[17] = {
        0xbc,
        0x18,0xc3,0xf4,0xe3,0x9e,0xca,0x72,0x22,
        0x49,0x0d,0x96,0x7c,0x79,0xbf,0xfc,0x92
    };
    check("V2 CT+Tag matches", memcmp(ct, expected, 17) == 0);

    /* round-trip */
    uint8_t pt2[1];
    int ret = ascon128_decrypt(key, nonce, NULL, 0, ct, 17, pt2);
    check("V2 decrypt returns 0", ret == 0);
    check("V2 decrypt plaintext correct", pt2[0] == 0x00);
}

/* Vector 3: encrypt-then-decrypt round-trip with 32-byte PT and 8-byte AD */
static void test_vector3(void) {
    printf("\n--- Vector 3: round-trip, 32-byte PT, 8-byte AD ---\n");
    /* fixed "random" key/nonce/pt/ad for determinism */
    uint8_t key[16]  = {0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe,
                         0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef};
    uint8_t nonce[16]= {0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
                         0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff};
    uint8_t ad[8]    = {0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80};
    uint8_t pt[32];
    for (int i = 0; i < 32; i++) pt[i] = (uint8_t)(i * 7 + 3);

    uint8_t ct[32 + 16];
    ascon128_encrypt(key, nonce, ad, 8, pt, 32, ct);
    print_hex("CT", ct, 32);
    print_hex("Tag", ct + 32, 16);

    uint8_t pt2[32];
    int ret = ascon128_decrypt(key, nonce, ad, 8, ct, 48, pt2);
    check("V3 decrypt returns 0", ret == 0);
    check("V3 round-trip PT matches", memcmp(pt, pt2, 32) == 0);

    /* flip one tag byte and expect -1 */
    ct[32] ^= 0x01;
    ret = ascon128_decrypt(key, nonce, ad, 8, ct, 48, pt2);
    check("V3 tampered tag returns -1", ret == -1);
}

int main(void) {
    test_vector1();
    test_vector2();
    test_vector3();

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");
    return all_pass ? 0 : 1;
}
