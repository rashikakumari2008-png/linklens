/*
 * scoring.c  -  the LinkLens pairwise scoring engine (written in C for speed)
 *
 * For every pair of reports (i, j) with i < j it computes:
 *   id_score    how strong the shared identifiers (phone / UPI / URL) are
 *   time_score  how close in time the two reports are (1 = same day, falls towards 0)
 *   match_mask  which identifiers matched (bit 1 = phone, 2 = UPI, 4 = URL)
 *   flags       conflicting signals (bit 1 = different category, 2 = large time gap,
 *               4 = very common identifier, 8 = different city (information only))
 *   final       weighted score = w_text*text + w_id*id + w_time*time - penalties
 *
 * The text similarity is computed in Python (TF-IDF) and passed in as a matrix.
 * Pairs are written in row-major order: (0,1) (0,2) ... (0,n-1) (1,2) ... (n-2,n-1).
 * That is the same order as numpy.triu_indices(n, 1), so Python can line them up.
 *
 * Compile:  gcc -O2 -shared -fPIC -o libscoring.so scoring.c -lm
 */
#include <math.h>
#include <stdlib.h>

#define NID 3   /* number of identifier types: phone, upi, url */

/* p[] layout (kept in sync with PARAM_ORDER in engine.py):
 *  0 w_text   1 w_id   2 w_time   3 tau (days)
 *  4 type weight phone   5 type weight upi   6 type weight url
 *  7 penalty category   8 penalty time gap   9 penalty common identifier
 * 10 gap_days (what counts as a "large" gap)   11 common_freq (what counts as "very common")
 */
void score_pairs(int n,
                 const int *day, const int *cat, const int *city,
                 const int *ids,    /* n x NID, -1 means "no identifier" */
                 const int *freq,   /* n x NID, how many reports share this identifier value */
                 const double *text,/* n x n TF-IDF cosine similarity */
                 const double *p,
                 double *out_id, double *out_time,
                 int *out_match, int *out_flags, double *out_score)
{
    long k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++, k++) {

            /* ---- shared identifiers (noisy-OR, rarer identifiers count more) ---- */
            double keep = 1.0;
            int match = 0, common = 0;
            for (int t = 0; t < NID; t++) {
                int a = ids[i * NID + t];
                int b = ids[j * NID + t];
                if (a >= 0 && a == b) {
                    int f = freq[i * NID + t];                 /* f >= 2 here */
                    double specificity = 1.0 / (1.0 + log((double)f - 1.0));
                    keep *= 1.0 - p[4 + t] * specificity;
                    match |= (1 << t);
                    if (f >= (int)p[11]) common = 1;
                }
            }
            double id_s = 1.0 - keep;

            /* ---- time closeness ---- */
            int gap = abs(day[i] - day[j]);
            double time_s = exp(-(double)gap / p[3]);

            /* ---- conflicting signals ---- */
            int flags = 0;
            if (cat[i] != cat[j])   flags |= 1;
            if (gap > (int)p[10])   flags |= 2;
            if (common)             flags |= 4;
            if (city[i] != city[j]) flags |= 8;

            /* ---- final weighted score ---- */
            double s = p[0] * text[(long)i * n + j] + p[1] * id_s + p[2] * time_s;
            if (flags & 1) s -= p[7];
            if (flags & 2) s -= p[8];
            if (flags & 4) s -= p[9];
            if (s < 0.0) s = 0.0;

            out_id[k] = id_s;
            out_time[k] = time_s;
            out_match[k] = match;
            out_flags[k] = flags;
            out_score[k] = s;
        }
    }
}
