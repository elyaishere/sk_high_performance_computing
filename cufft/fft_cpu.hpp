#include <bits/stdc++.h>

using namespace std;

using fcomplex = complex<float>;
using  f2complex = float2;

class FFT {
public:
    void fft(vector<fcomplex> &a, bool invert) {

        int n = (int)a.size();

        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j >= bit; bit >>= 1)
                j -= bit;
            j += bit;
            if (i < j)
                swap(a[i], a[j]);
        }

        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * M_PI / len * (invert ? 1 : -1);
            fcomplex wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len) {
                fcomplex w(1);
                for (int j = 0; j < len / 2; ++j) {
                    fcomplex u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (invert)
            for (int i = 0; i < n; ++i)
                a[i] /= n;
        return;
    }

    /// Function to multiply two polynomial
    vector<int> mult(const vector<int>& a, const vector<int>& b) {
        vector<fcomplex> fa(a.begin(), a.end()), fb(b.begin(), b.end());

        // padding with zero to make their size equal to power of 2
        size_t n = 1;
        while (n < max(a.size(), b.size()))
            n <<= 1;
        n <<= 1;

        fa.resize(n), fb.resize(n);

        fft(fa, false), fft(fb, false);

        for (size_t i = 0; i < n; ++i)
            fa[i] *= fb[i];

        fft(fa, true);

        vector<int> res;
        res.resize(n);
        for (size_t i = 0; i < n; ++i)
            res[i] = int(fa[i].real() + 0.5);

        return res;
    }
};
