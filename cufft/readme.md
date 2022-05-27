# CUfft final project
### usage
1. Replace body of `main` in `fft.cu` with your example.
e.g.
```c++
int main()
{	
    vector<int> a = {1,1};
    vector<int> b = {1,2,3};
    auto multiplier = FFT();
    cout << "A = " << a;
    cout << "B = " << b;
    cout << "A * B = " << multiplier.mult(a, b) << endl;
    return 0;
}
```
2. See inference results in `eval.ipynb`
