#ifndef ASCON128_H
#define ASCON128_H

#include <stdint.h>
#include <stddef.h>

#define ASCON128_KEY_LEN   16
#define ASCON128_NONCE_LEN 16
#define ASCON128_TAG_LEN   16

typedef struct {
    uint64_t x[5];
} ascon_state_t;

/* Encrypt plaintext. Writes ciphertext (same length as plaintext) + 16-byte tag.
   ct must have capacity plaintext_len + ASCON128_TAG_LEN. */
void ascon128_encrypt(
    const uint8_t *key,        /* 16 bytes */
    const uint8_t *nonce,      /* 16 bytes */
    const uint8_t *ad,         /* associated data, may be NULL */
    size_t         ad_len,
    const uint8_t *plaintext,
    size_t         plaintext_len,
    uint8_t       *ciphertext  /* out: plaintext_len + 16 bytes */
);

/* Returns 0 on success (tag valid), -1 on authentication failure. */
int ascon128_decrypt(
    const uint8_t *key,
    const uint8_t *nonce,
    const uint8_t *ad,
    size_t         ad_len,
    const uint8_t *ciphertext,
    size_t         ciphertext_len, /* includes 16-byte tag */
    uint8_t       *plaintext       /* out: ciphertext_len - 16 bytes */
);

/* Expose permutation for trace generation hooks */
void ascon_permutation(ascon_state_t *s, int rounds);

#endif /* ASCON128_H */
