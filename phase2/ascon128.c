#include "ascon128.h"
#include <string.h>

static uint64_t load64(const uint8_t *b) {
    return ((uint64_t)b[0] << 56) | ((uint64_t)b[1] << 48) |
           ((uint64_t)b[2] << 40) | ((uint64_t)b[3] << 32) |
           ((uint64_t)b[4] << 24) | ((uint64_t)b[5] << 16) |
           ((uint64_t)b[6] <<  8) |  (uint64_t)b[7];
}

static void store64(uint8_t *b, uint64_t v) {
    b[0]=v>>56; b[1]=v>>48; b[2]=v>>40; b[3]=v>>32;
    b[4]=v>>24; b[5]=v>>16; b[6]=v>>8;  b[7]=(uint8_t)v;
}

#define ROTR(x,n) (((x)>>(n))|((x)<<(64-(n))))

void ascon_sbox(ascon_state_t *s) {
    uint64_t t0, t1, t2, t3, t4;
    s->x[0] ^= s->x[4]; s->x[4] ^= s->x[3]; s->x[2] ^= s->x[1];
    t0=s->x[0]; t1=s->x[1]; t2=s->x[2]; t3=s->x[3]; t4=s->x[4];
    s->x[0] = t0 ^ (~t1 & t2);
    s->x[1] = t1 ^ (~t2 & t3);
    s->x[2] = t2 ^ (~t3 & t4);
    s->x[3] = t3 ^ (~t4 & t0);
    s->x[4] = t4 ^ (~t0 & t1);
    s->x[1] ^= s->x[0]; s->x[0] ^= s->x[4];
    s->x[3] ^= s->x[2]; s->x[2] = ~s->x[2];
}

static void ascon_linear(ascon_state_t *s) {
    s->x[0] ^= ROTR(s->x[0],19) ^ ROTR(s->x[0],28);
    s->x[1] ^= ROTR(s->x[1],61) ^ ROTR(s->x[1],39);
    s->x[2] ^= ROTR(s->x[2], 1) ^ ROTR(s->x[2], 6);
    s->x[3] ^= ROTR(s->x[3],10) ^ ROTR(s->x[3],17);
    s->x[4] ^= ROTR(s->x[4], 7) ^ ROTR(s->x[4],41);
}

void ascon_permutation(ascon_state_t *s, int rounds) {
    static const uint8_t RC[12] = {
        0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b
    };
    int start = 12 - rounds;
    for (int i = start; i < 12; i++) {
        s->x[2] ^= RC[i];
        ascon_sbox(s);
        ascon_linear(s);
    }
}

void ascon128_encrypt(
    const uint8_t *key,
    const uint8_t *nonce,
    const uint8_t *ad,
    size_t         ad_len,
    const uint8_t *plaintext,
    size_t         plaintext_len,
    uint8_t       *ciphertext)
{
    ascon_state_t s;
    uint64_t key0 = load64(key);
    uint64_t key1 = load64(key + 8);

    /* Initialization */
    s.x[0] = UINT64_C(0x80400c0600000000);
    s.x[1] = key0;
    s.x[2] = key1;
    s.x[3] = load64(nonce);
    s.x[4] = load64(nonce + 8);
    ascon_permutation(&s, 12);
    s.x[3] ^= key0;
    s.x[4] ^= key1;

    /* Associated Data */
    if (ad_len > 0) {
        size_t i = 0;
        while (i + 8 <= ad_len) {
            s.x[0] ^= load64(ad + i);
            ascon_permutation(&s, 6);
            i += 8;
        }
        /* Last (partial) AD block */
        uint8_t pad[8] = {0};
        size_t rem = ad_len - i;
        memcpy(pad, ad + i, rem);
        pad[rem] = 0x80;
        s.x[0] ^= load64(pad);
        ascon_permutation(&s, 6);
    }
    /* Domain separation */
    s.x[4] ^= UINT64_C(1);

    /* Encryption */
    size_t i = 0;
    while (i + 8 <= plaintext_len) {
        s.x[0] ^= load64(plaintext + i);
        store64(ciphertext + i, s.x[0]);
        ascon_permutation(&s, 6);
        i += 8;
    }
    /* Last (partial) plaintext block — always process */
    {
        uint8_t pad[8] = {0};
        size_t rem = plaintext_len - i;
        if (rem > 0) memcpy(pad, plaintext + i, rem);
        pad[rem] = 0x80;
        s.x[0] ^= load64(pad);
        /* Output only the rem ciphertext bytes */
        uint8_t tmp[8];
        store64(tmp, s.x[0]);
        memcpy(ciphertext + i, tmp, rem);
    }

    /* Finalization */
    s.x[1] ^= key0;
    s.x[2] ^= key1;
    ascon_permutation(&s, 12);
    s.x[3] ^= key0;
    s.x[4] ^= key1;

    /* Tag */
    store64(ciphertext + plaintext_len,     s.x[3]);
    store64(ciphertext + plaintext_len + 8, s.x[4]);
}

int ascon128_decrypt(
    const uint8_t *key,
    const uint8_t *nonce,
    const uint8_t *ad,
    size_t         ad_len,
    const uint8_t *ciphertext,
    size_t         ciphertext_len,
    uint8_t       *plaintext)
{
    if (ciphertext_len < ASCON128_TAG_LEN) return -1;
    size_t plaintext_len = ciphertext_len - ASCON128_TAG_LEN;

    ascon_state_t s;
    uint64_t key0 = load64(key);
    uint64_t key1 = load64(key + 8);

    /* Initialization */
    s.x[0] = UINT64_C(0x80400c0600000000);
    s.x[1] = key0;
    s.x[2] = key1;
    s.x[3] = load64(nonce);
    s.x[4] = load64(nonce + 8);
    ascon_permutation(&s, 12);
    s.x[3] ^= key0;
    s.x[4] ^= key1;

    /* Associated Data */
    if (ad_len > 0) {
        size_t i = 0;
        while (i + 8 <= ad_len) {
            s.x[0] ^= load64(ad + i);
            ascon_permutation(&s, 6);
            i += 8;
        }
        uint8_t pad[8] = {0};
        size_t rem = ad_len - i;
        memcpy(pad, ad + i, rem);
        pad[rem] = 0x80;
        s.x[0] ^= load64(pad);
        ascon_permutation(&s, 6);
    }
    s.x[4] ^= UINT64_C(1);

    /* Decryption */
    size_t i = 0;
    while (i + 8 <= plaintext_len) {
        uint64_t ct_block = load64(ciphertext + i);
        uint64_t pt_block = s.x[0] ^ ct_block;
        store64(plaintext + i, pt_block);
        s.x[0] = ct_block;
        ascon_permutation(&s, 6);
        i += 8;
    }
    /* Last (partial) plaintext block */
    {
        size_t rem = plaintext_len - i;
        uint8_t ct_partial[8] = {0};
        if (rem > 0) memcpy(ct_partial, ciphertext + i, rem);

        /* Recover plaintext */
        uint8_t sx0[8];
        store64(sx0, s.x[0]);
        for (size_t j = 0; j < rem; j++)
            plaintext[i + j] = sx0[j] ^ ct_partial[j];

        /* Update state with padded plaintext (mirrors what encrypt did) */
        uint8_t pad[8] = {0};
        if (rem > 0) memcpy(pad, plaintext + i, rem);
        pad[rem] = 0x80;
        s.x[0] ^= load64(pad);
    }

    /* Finalization */
    s.x[1] ^= key0;
    s.x[2] ^= key1;
    ascon_permutation(&s, 12);
    s.x[3] ^= key0;
    s.x[4] ^= key1;

    /* Constant-time tag comparison */
    const uint8_t *tag = ciphertext + plaintext_len;
    uint8_t computed_tag[16];
    store64(computed_tag,     s.x[3]);
    store64(computed_tag + 8, s.x[4]);

    uint8_t diff = 0;
    for (int j = 0; j < 16; j++)
        diff |= tag[j] ^ computed_tag[j];

    return diff == 0 ? 0 : -1;
}
