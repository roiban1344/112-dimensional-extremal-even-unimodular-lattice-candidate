/* F1: exact integer enumeration of {x in Z^n : sum_i y_i^2 <= T}, y = R x (R upper triangular,
 * integer, R_ii > 0), i.e. y_i = sum_{j>=i} R_ij x_j.  All arithmetic in __int128 with overflow guards;
 * no floating point decides anything (the square root is corrected to the exact integer floor).
 * Driver and the error bound relating this form to the lattice: code/crosscheck/all_data/min_L.py.
 *
 * input (stdin): n T maxprint, then the n*n entries of R row by row (decimal integers).
 * output: one line per found x != 0 (only the first maxprint), then a summary line
 *   "SUMMARY count=<found x != 0, both signs> nodes=<nodes> hash2=<sum (a.x)^2 mod 2^64> hash4=<sum (a.x)^4 mod 2^64>"
 * with a_j = 2j+1 (odd weights).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef __int128 i128;
#define NMAX 64

static i128 R[NMAX][NMAX];
static int n;
static const i128 LIM = ((i128)1) << 120;

static i128 parse(const char *s) {
    int neg = 0; i128 v = 0;
    if (*s == '-') { neg = 1; s++; }
    for (; *s; s++) { if (*s < '0' || *s > '9') { fprintf(stderr, "bad number\n"); exit(2); } v = v * 10 + (*s - '0'); if (v > LIM) { fprintf(stderr, "too big\n"); exit(2); } }
    return neg ? -v : v;
}

static void chk(i128 v) { if (v > LIM || v < -LIM) { fprintf(stderr, "overflow guard\n"); exit(3); } }

static i128 isqrt128(i128 v) {           /* exact floor(sqrt(v)), v >= 0 */
    if (v < 0) { fprintf(stderr, "isqrt of negative\n"); exit(3); }
    i128 r = (i128)sqrtl((long double)v);
    while (r > 0 && r * r > v) r--;
    while ((r + 1) * (r + 1) <= v) r++;
    return r;
}

static i128 floordiv(i128 a, i128 b) {   /* b > 0 */
    i128 q = a / b; if ((a % b != 0) && (a < 0)) q--; return q;
}
static i128 ceildiv(i128 a, i128 b) { return -floordiv(-a, b); }

int main(void) {
    char buf[128];
    long long maxprint;
    if (scanf("%d", &n) != 1 || n < 1 || n > NMAX) return 2;
    if (scanf("%127s", buf) != 1) return 2;
    i128 T = parse(buf);
    if (scanf("%lld", &maxprint) != 1) return 2;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        if (scanf("%127s", buf) != 1) return 2;
        R[i][j] = parse(buf);
        if (j < i && R[i][j] != 0) { fprintf(stderr, "not upper triangular\n"); return 2; }
    }
    for (int i = 0; i < n; i++) if (R[i][i] <= 0) { fprintf(stderr, "diag <= 0\n"); return 2; }

    long long x[NMAX], hi[NMAX];
    i128 rem[NMAX + 1], s[NMAX];
    unsigned long long count = 0, nodes = 0, printed = 0;
    uint64_t hash2 = 0, hash4 = 0;
    memset(x, 0, sizeof x);
    rem[n] = T;
    int i = n - 1;
    /* set up the range at level i */
    #define SETUP(lv) do { \
        i128 ss = 0; for (int j = (lv) + 1; j < n; j++) { ss += R[lv][j] * (i128)x[j]; chk(ss); } \
        s[lv] = ss; i128 b = isqrt128(rem[(lv) + 1]); \
        i128 lo = ceildiv(-b - ss, R[lv][lv]), h = floordiv(b - ss, R[lv][lv]); \
        if (lo < -(1LL << 40) || h > (1LL << 40)) { fprintf(stderr, "range guard\n"); exit(3); } \
        x[lv] = (long long)lo; hi[lv] = (long long)h; nodes++; \
        if ((nodes & ((1ULL << 30) - 1)) == 0) { fprintf(stderr, "progress nodes=%llu count=%llu top=%lld,%lld\n", nodes, count, x[n - 1], x[n - 2]); fflush(stderr); } \
    } while (0)
    SETUP(i);
    for (;;) {
        if (x[i] > hi[i]) {               /* level exhausted: go up */
            i++;
            if (i >= n) break;
            x[i]++;
            continue;
        }
        i128 y = R[i][i] * (i128)x[i] + s[i];
        chk(y);
        i128 r = rem[i + 1] - y * y;
        if (r < 0) { x[i]++; continue; }  /* cannot happen inside the exact range; kept as a guard */
        if (i == 0) {
            int nz = 0; for (int j = 0; j < n; j++) if (x[j]) { nz = 1; break; }
            if (nz) {
                count++;
                if (printed < (unsigned long long)maxprint) {
                    for (int j = 0; j < n; j++) printf("%lld%c", x[j], j + 1 < n ? ' ' : '\n');
                    printed++;
                }
                uint64_t a = 0; for (int j = 0; j < n; j++) a += (uint64_t)(2 * j + 1) * (uint64_t)(int64_t)x[j];
                uint64_t a2 = a * a; hash2 += a2; hash4 += a2 * a2;
            }
            x[0]++;
            continue;
        }
        rem[i] = r;
        i--;
        SETUP(i);
    }
    printf("SUMMARY count=%llu nodes=%llu hash2=%llu hash4=%llu\n", count, nodes,
           (unsigned long long)hash2, (unsigned long long)hash4);
    return 0;
}
