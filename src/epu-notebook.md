<!-- toc -->

# Dawn of ACCEPTEDs

## Default Source Code

```cpp
#include <bits/stdc++.h>
#define FOR(i,a,b) for (int i=(a),_b_=(b);i<_b_;i++)
#define ROF(i,a,b) for (int i=(a),_b_=(b);i>_b_;i--)
#define IT(i,v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
#define ALL(v) v.begin(), v.end()
#define MS(v) memset(v,0,sizeof(v))
using namespace std;
typedef long long LL;
typedef unsigned long long ULL;
template<typename T> vector<T> &operator += (vector<T> &v, T x) {v.push_back(x);return v;}

void solve() {
}

int main(){
  #ifdef NGOBACH
  freopen("input.txt","r",stdin);
//  freopen("output.txt","w",stdout);
  #endif
  ios_base::sync_with_stdio(0); cin.tie(0); solve();
}
```

## Một số phương pháp giải bài

### Nhân ma trận

Một số bài toán với bộ input lớn 1e18 cần xử lý trong **lgN** và có tính lặp lại, **đầu ra của một step là đầu vào của step tiếp theo**. Khi đó có thể đưa về bài toán nhân mà trận. Xem ví dụ về bài n-th Fibonacci.

```
[First Matrix] = [Second matrix] * [Third Matrix]
| F(n)   |     =   Matrix 'C'    *  | F(n-1) |
| F(n-1) |                          | F(n-2) |
| F(n-2) |                          | F(n-3) |
```
[[Tham khảo]](http://www.geeksforgeeks.org/matrix-exponentiation/).

### Sqrt Decomposition

Đưa bài toán về **O(sqrt(N))** để giải nếu có thể. Usually we use them in the problems with modify and ask queries.

## Game theory - Lý thuyết trò chơi

Có những vị trí được coi là vị chí **CHẾT**. Nếu người đi trước ở vị trí này thì chắc chắn thua, và ngược lại. Vị trí chết là vị trí dù đi kiểu gì cũng trở thành vị trí **SỐNG** của người chơi kia.

## Difference table

Giải bài toán xác định đa thức khi biết 1 số phần tử.

## Offline solution

Đôi khi có thể đọc toàn bộ query sau đó giải quyết một thể thay vì đọc từng query và duyệt trả lời.

# Giải thuật & thuật toán

## KMP String search

```cpp
void prepair(char *pat, int *lps) {
  int len = 0, i = 1, N = strlen(pat);
  lps[0] = 0;
  while (i < N) {
    if (pat[i] == pat[len]) 
      lps[i++] = ++len;
    else {
      // Not matching, fallback
      if (len) {
        len = lps[len-1];
      } else {
        lps[i++] = 0;
      }
    }
  }
  cout << "lps:" << endl;
  FOR(i,0,N) cout << lps[i] << ' ';
  cout << endl;
}

int search(char s[], char pat[]) {
  int lps[1000];
  prepair(pat, lps);
  int i = 0, j = 0, N = strlen(s), M = strlen(pat);
  while (i < N) {
    if (s[i] == pat[j]) {
      i++;
      j++;
      if (j == M) return i-j;
    } else {
      if (j) {
        j = lps[j-1];
      } else {
        i++;
      }
    }
  }
  return -1;
}

int main(){
  char s[] = "AABAACAADAABAB324CABABABABABCABABCADEFAAABABCDEABABCD";
  char pat[] = "ABCDEABABCD";
  cout << "Search result: " << search(s, pat);
}
```

## Bài toán 2SAT

```cpp
#define f1(i,n) for (int i=1; i<=n; i++)
#define SetLength2(a, n, t) a=((t*) calloc(n, sizeof(t))) + (n)/2

int n, m, cnt;
vector<int> * a;
int *Color, *Num, *Low; // Color 1=False, 2=True
stack<int> st;
bool Invalid=false;

bool minimize(int &a, int b){
    if (a>b) a=b; else return false; return true;
}

void init(int m, int n){
    SetLength2(a, n*2+5, vector<int>);
    SetLength2(Color, n*2+5, int);
    SetLength2(Num, n*2+5, int);
    SetLength2(Low, n*2+5, int);
}

void set_color(int u, int x){
    if (Color[u]==(x^3)) Invalid=true; else Color[u]=x;
    if (Color[-u]==x) Invalid=true; else Color[-u]=(x^3);
}

void tarzan(int u){
  Num[u]=Low[u]=++cnt; st.push(u);
  
  for (int i=0,v; v=a[u][i]; i++){
    if (Num[v]) minimize(Low[u], Num[v]);
    else tarzan(v), minimize(Low[u], Low[v]);
    if (Color[v]==1) set_color(u, 1); // False
  }
  if (Low[u]==Num[u]){
    int v=0;
    if (Color[u]==0) set_color(u, 2); // True
    do {
      v=st.top(); st.pop();
      set_color(v, Color[u]);
      Num[v]=Low[v]=0x33334444;
    } while (u!=v);
  }
}

void enter(){
  //srand((m+n)*1000);
  f1(i,m) {
    int p, q;
    cin >> p >> q;
    //p=(rand()%n+1) * (rand()&1 ? 1 : -1);
    //q=(rand()%n+1) * (rand()&1 ? 1 : -1);
    a[-p].push_back(q);
    a[-q].push_back(p);
  }
  f1(i,n) a[i].push_back(0);
  f1(i,n) a[-i].push_back(0);
}

int main(){
  ios :: sync_with_stdio(false);
  cin >> m >> n;
  init(m, n);
  enter();
  f1(i,n) if (!Num[i]) tarzan(i);
  f1(i,n) if (!Num[-i]) tarzan(-i);
  
  if (Invalid) cout << "NO" << endl;
  else {
    cout << "YES" << endl;
    int Answer=0;
    f1(i,n) if (Color[i]==2) Answer++;
    cout << Answer << endl;
    f1(i,n) if (Color[i]==2) cout << i << " "; cout << endl;
  }
  cin.ignore(2);
}
```

## Bài toán LCA

Tư tưởng dựa trên sparse table.

```cpp
const int N = 100005;
int n, Root, l[N], P[20][N];

int level(int u) {
  if (u==Root) return l[u]=1;
  if (l[u]==0) l[u]=level(P[0][u])+1;
    return l[u];
}

int lca(int x, int y) {
  for (int k=19; k>=0; k--) 
    if (l[P[k][x]]>=l[y]) 
    x=P[k][x];
  for (int k=19; k>=0; k--)
    if (l[P[k][y]]>=l[x]) 
    y=P[k][y];
  for (int k=19; k>=0; k--)
    if (P[k][x]!=P[k][y]) 
    { x=P[k][x]; y=P[k][y]; }
  while (x!=y)
    { x=P[0][x]; y=P[0][y]; }
  return x; 
}

void solve() {
  scanf("%d", &n);
  for (int i=1; i<=n; i++) {
    int p; scanf("%d", &p);
    while (p-->0) {
      int q; scanf("%d", &q);
      P[0][q] = i;
    }
  }
  for (int i=1; i<=n; i++)
    if (P[0][i]==0) Root=i;
  for (int i=1; i<=n; i++)
    level(i); // done l
  
  for (int k=1; k<=19; k++)
  for (int i=1; i<=n; i++)
  P[k][i] = P[k-1][P[k-1][i]];
  
  int m; scanf("%d", &m);
  while (m-->0) {
    int x, y;
    scanf("%d%d", &x, &y);
    printf("%d\n", lca(x, y));
  }
}

main() {
  int t; scanf("%d", &t);
  for (int i=1; i<=t; i++) {
    printf("Case %d:\n", i);
    solve();
    for (int j=1; j<=n; j++) 
    { l[j]=0; P[0][j]=0; }
  }
}
```

## Chặt nhị phân

TODO:

## Suffix Array

Độ phức tạp **O(N*lg^2N)**.

Cách 1: Nhân đôi tiền tố.

```cpp
/*
Suffix array O(n lg^2 n)
LCP table O(n)
*/
#include <cstdio>
#include <algorithm>
#include <cstring>

using namespace std;

#define REP(i, n) for (int i = 0; i < (int)(n); ++i)

namespace SuffixArray
{
  const int MAXN = 1 << 21;
  char * S;
  int N, gap;
  int sa[MAXN], pos[MAXN], tmp[MAXN], lcp[MAXN];

  bool sufCmp(int i, int j)
  {
    if (pos[i] != pos[j])
      return pos[i] < pos[j];
    i += gap;
    j += gap;
    return (i < N && j < N) ? pos[i] < pos[j] : i > j;
  }

  void buildSA()
  {
    N = strlen(S);
    REP(i, N) sa[i] = i, pos[i] = S[i];
    for (gap = 1;; gap *= 2)
    {
      sort(sa, sa + N, sufCmp);
      REP(i, N - 1) tmp[i + 1] = tmp[i] + sufCmp(sa[i], sa[i + 1]);
      REP(i, N) pos[sa[i]] = tmp[i];
      if (tmp[N - 1] == N - 1) break;
    }
  }

  void buildLCP()
  {
    for (int i = 0, k = 0; i < N; ++i) if (pos[i] != N - 1)
    {
      for (int j = sa[pos[i] + 1]; S[i + k] == S[j + k];)
      ++k;
      lcp[pos[i]] = k;
      if (k)--k;
    }
  }
} // end namespace SuffixArray
```

Cách 2: Sử dụng hash.

```cpp
namespace HashSuffixArray
{
  const int
    MAXN = 1 << 21;

  typedef unsigned long long hash;

  const hash BASE = 137;

  int N;
  char * S;
  int sa[MAXN];
  hash h[MAXN], hPow[MAXN];

  #define getHash(lo, size) (h[lo] - h[(lo) + (size)] * hPow[size])

  inline bool sufCmp(int i, int j)
  {
    int lo = 1, hi = min(N - i, N - j);
    while (lo <= hi)
    {
      int mid = (lo + hi) >> 1;
      if (getHash(i, mid) == getHash(j, mid))
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    return S[i + hi] < S[j + hi];
  }

  void buildSA()
  {
    N = strlen(S);
    hPow[0] = 1;
    for (int i = 1; i <= N; ++i)
      hPow[i] = hPow[i - 1] * BASE;
    h[N] = 0;
    for (int i = N - 1; i >= 0; --i)
      h[i] = h[i + 1] * BASE + S[i], sa[i] = i;

    stable_sort(sa, sa + N, sufCmp);
  }

} // end namespace HashSuffixArray
```

## String hashing

So sánh 2 substring trong O(1). Xây dựng trong O(N). TODO!

# Các cấu trúc dữ liệu

## Segment tree (IT) - Query trên đoạn

```cpp
const int N = 1e5 + 10;
int node[4*N];
void modify(int seg, int l, int r, int p, int val){
  if(l == r){
    node[seg] += val;
    return;
  }
  int mid = (l + r)/2;
  if(p <= mid){
    modify(2*seg + 1, l, mid, p, val);
  }else{
    modify(2*seg + 2, mid + 1, r, p, val);
  }
  node[seg] = node[2*seg + 1] + node[2*seg + 2];
}
int sum(int seg, int l, int r, int a, int b){
  if(l > b || r < a) return 0;
  if(l >= a && r <= b) return node[seg];
  int mid = (l + r)/2;
  return sum(2*seg + 1, l, mid, a, b) + sum(2*seg + 2, mid + 1, r, a, b);
}
```

Ngoài ra còn có [Segment tree with lazy propagation](https://www.hackerearth.com/practice/notes/segment-tree-and-lazy-propagation/).
Hỗ trợ cho việc update trên một đoạn.

Đặt biệt còn có segment tree với mỗi node là một *vector*, *set*, *fenwick tree* phục vụ các bài toán đặt biệt, hay **Segment tree 2D**. [Tham khảo thêm](http://codeforces.com/blog/entry/15890).

## Binary Indexed tree (Fenwick Tree)

Cây BIT ban đầu các phần tử là *0*. Update vị trí `A[t]` trong **lgN**. Và lấy tổng từ `A[1]` đến `A[t]` trong **lgN**.   
Implementation:

```cpp
int n, m, k;
long long arr[1000005];
long long tree[1000005];
void update(int idx, int val) {
  while (idx <= n) {
    tree[idx] += val;
    idx += (idx & -idx);
  }
}

long long read(int idx) {
  long long ret = 0;
  while (idx > 0) {
    ret += tree[idx];
    idx -= (idx & -idx);
  }
  return ret;
}
```

## Sparse table

Dùng với bài toán chỉ query trên đoạn, không có update. O(lgN) khi xây dựng và O(1) khi query.

```cpp
const int k = 16;
const int N = 1e5;
const int ZERO = 0; // ZERO + x = x + ZERO = x (for any x)
long long table[N][k + 1]; // k + 1 because we need to access table[r][k]
int Arr[N];
int main()
{
  int n, L, R, q;
  cin >> n; // array size
  for(int i = 0; i < n; i++)
    cin >> Arr[i];

  // build Sparse Table
  for(int i = 0; i < n; i++)
    table[i][0] = Arr[i];
  for(int j = 1; j <= k; j++) {
    for(int i = 0; i <= n - (1 << j); i++)
      table[i][j] = table[i][j - 1] + table[i + (1 << (j - 1))][j - 1];
  }

  cin >> q; // number of queries
  for(int i = 0; i < q; i++) {
    cin >> L >> R; // boundaries of next query, 0-indexed
    long long answer = ZERO;
    for(int j = k; j >= 0; j--) {
      if(L + (1 << j) - 1 <= R) {
        answer = answer + table[L][j];
        L += 1 << j; // instead of having L', we increment L directly
      }
    }
    cout << answer << endl;
  }
  return 0;
}
```

## Trie

Sử dụng khi làm việc trên xâu có số lượng lớn, số loại ký tự nhỏ và độ dài mỗi xâu nhỏ.

```cpp
const int ALPHABET_SIZE = 26;

struct TrieNode
{
  struct TrieNode *children[ALPHABET_SIZE];
  bool isEndOfWord;
};

struct TrieNode *getNode(void)
{
  struct TrieNode *pNode =  new TrieNode;
  pNode->isEndOfWord = false;
  for (int i = 0; i < ALPHABET_SIZE; i++)
    pNode->children[i] = NULL;
  return pNode;
}

void insert(struct TrieNode *root, string key)
{
  struct TrieNode *pCrawl = root;
  for (int i = 0; i < key.length(); i++)
  {
    int index = key[i] - 'a';
    if (!pCrawl->children[index])
      pCrawl->children[index] = getNode();
    pCrawl = pCrawl->children[index];
  }
  // mark last node as leaf
  pCrawl->isEndOfWord = true;
}
 
// Returns true if key presents in trie, else
// false
bool search(struct TrieNode *root, string key)
{
  struct TrieNode *pCrawl = root;
  for (int i = 0; i < key.length(); i++)
  {
    int index = key[i] - 'a';
    if (!pCrawl->children[index])
      return false;
    pCrawl = pCrawl->children[index];
  }
  return (pCrawl != NULL && pCrawl->isEndOfWord);
}
```

# Toán học

## Tìm ước chung lớn nhất (GCD)

```cpp
LL gcd(LL a, LL b) { return b?gcd(b,a%b):a; }
// or Builtin
__gcd(a,b);
```

## Power Modulo & Multiply Modulo

```cpp
// a^e MOD m, consider use with mulmod when mod > 1e9, O
LL powmod(LL a, LL e, LL m) {
  if (m==1) return 0;
  if (!e) return 1;
  LL t = 1;
  while (e > 1) {
    if (e&1) t = t*a%m;
    a = a*a%m;
    e >>= 1;
  }
  return t*a%m;
}

// a*b MOD m
LL mulmod(LL a, LL b, LL m) {
  LL res = 0;
  if (!a || !b) {
    return 0;
  }
  if (a<b) swap(a,b);
  
  while (b>1) {
    if (b&1) res = (res + a)%m;
    a = (a+a)%m;
    b >>= 1;
  }
  return (res+a)%m;
}
```

## Modulo Inverse

Tìm `b` để có *a<sup>-1</sup> mod m == b*.

```cpp
LL moduloInverse(LL a, LL m) {
  LL q,r,y0=0,y1=1,y,m0=m;
  while (a>0) {
    q = m/a; r = m%a;
    if (!r) return (y%m0 + m0) % m0;
    y = y0-y1*q; y0=y1; y1=y; m=a; a=r;
  }
}
```

## Euclidean extended

Tìm cặp số **x**, **y** sao cho **ax+by=gdc(a,b)**.

```cpp
LL xGCD(LL a, LL b, LL &x, LL &y) {
  if (!b) { x = 1; y = 0; return a; }
  LL xx, yy, t = xGCD(b,a%b,xx,yy);
  x = yy; y = xx - a/b*yy;
  return t;
}
```

## Giải hệ bậc nhất 3 ẩn Crammer

```cpp
#define long long long
long det(int a1, int b1, int c1, int a2, int b2, int c2, int a3, int b3, int c3){
   return a1*(b2*c3-b3*c2) - b1*(a2*c3-a3*c2) + c1*(a2*b3-a3*b2);
}
main(){
   int a1, b1, c1, d1;
   int a2, b2, c2, d2;
   int a3, b3, c3, d3;

   cin >> a1 >> b1 >> c1 >> d1;
   cin >> a2 >> b2 >> c2 >> d2;
   cin >> a3 >> b3 >> c3 >> d3;

   long   D  = det(a1,b1,c1, a2,b2,c2, a3,b3,c3);
   double DX = det(d1,b1,c1, d2,b2,c2, d3,b3,c3);
   double DY = det(a1,d1,c1, a2,d2,c2, a3,d3,c3);
   double DZ = det(a1,b1,d1, a2,b2,d2, a3,b3,d3);

   if (D==0) cout << "Math error" << endl;
   else cout << DX/D << " " << DY/D << " " << DZ/D << endl;
}
```

## Chinese Remainder Theorem

**Problem:** Find a **x** such that:

```
x % a[1] = b[1]
x % a[2] = b[2]
....
```

Với b[1]...b[n] đôi một nguyên tố cùng nhau, và `inv()` tìm *inverse modulo*. Ta có:

```cpp
int findMinX(int num[], int rem[], int k)
{
  int prod = 1;
  for (int i = 0; i < k; i++)
    prod *= num[i];
  int result = 0;
  for (int i = 0; i < k; i++)
  {
    int pp = prod / num[i];
    result += rem[i] * inv(pp, num[i]) * pp;
  }
  return result % prod;
}
```

## Lucas's theorem

Problem: Tìm <sup>n</sup>C<sub>k</sub> trong O(p<sup>2</sup> * log<sub>p</sub>n).

```cpp
// Returns nCr % p.  In this Lucas Theorem based program,
// this function is only called for n < p and r < p.
int nCrModpDP(int n, int r, int p)
{
  // The array C is going to store last row of
  // pascal triangle at the end. And last entry
  // of last row is nCr
  int C[r+1];
  memset(C, 0, sizeof(C));

  C[0] = 1; // Top row of Pascal Triangle

  // One="" by constructs remaining rows of Pascal
  // Triangle from top to bottom
  for (int i = 1; i <= n; i++) {
    // Fill entries of current row using previous
    // row values
    for (int j = min(i, r); j > 0; j--)
      // nCj = (n-1)Cj + (n-1)C(j-1);
      C[j] = (C[j] + C[j-1])%p;
  }
  return C[r];
}
 
// Lucas Theorem based function that returns nCr % p
// This function works like decimal to binary conversion
// recursive function.  First we compute last digits of
// n and r in base p, then recur for remaining digits
int nCrModpLucas(int n, int r, int p)
{
  // Base case
  if (r==0)
    return 1;

  // Compute last digits of n and r in base p
  int ni = n%p, ri = r%p;

  // Compute result for last digits computed above, and
  // for remaining digits.  Multiply the two results and
  // compute the result of multiplication in modulo p.
  return (nCrModpLucas(n/p, r/p, p) * // Last digits of n and r
    nCrModpDP(ni, ri, p)) % p;  // Remaining digits
}
```

## Dãy số catalan

Dãy số này xuất hiện trong nhiều bài toán đếm. Các số đầu tiên của dãy (từ số thứ 0) là : *1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, …*.
Tìm số catalan thứ n trong O(n) sử dụng:

```cpp
// Returns value of Binomial Coefficient C(n, k)
unsigned long int binomialCoeff(unsigned int n, unsigned int k)
{
  unsigned long int res = 1;
  if (k > n - k)
    k = n - k;
  for (int i = 0; i < k; ++i)
  {
    res *= (n - i);
    res /= (i + 1);
  }
  return res;
}

unsigned long int catalan(unsigned int n)
{
  unsigned long int c = binomialCoeff(2*n, n);
  return c/(n+1);
}
```

## Số nguyên tố

### Sàng nguyên tố

```cpp
const int N = 1e7+7;
bool m[N];

void sieveEra() {
  memset(m, 0, sizeof(m));
  int i, lim = sqrt(N);
  for (i=2; i <= lim; i++) {
    if (!m[i]) {
      for (LL j=i*i; j < N; j+=i)
        if (!m[j]) m[j] = true;
    }
  }
}
```

Ngoài ra tham khảo thêm về [sàng đoạn (segment sieve)](http://www.geeksforgeeks.org/segmented-sieve/).

### Kiểm tra xác suất (Rabin - Miller)

Kiểm tra một số có phải là số nguyên tố hay không. Áp dụng cho `n < 2^31`, nếu lớn hơn kết hợp dùng **Modulo Multiplication**.

```cpp
const int RAB[] = {3,5,7,11,13,17}, R = sizeof(RAB)/sizeof(RAB[0]);
LL pm(LL a, LL e, LL m) {
  if (m==1) return 0;
  if (!e) return 1;
  LL t = 1;
  while (e > 1) {
    if (e&1) t = t*a%m;
    a = a*a%m;
    e >>= 1;
  }
  return t*a%m;
}
bool primeTest(LL n) {
  if (n==2) return true;
  if (n<2 || (n&1)==0) return false;
  
  LL m = n-1, s = 0;
  while ((m&1)==0) {
    m >>= 1;
    s++;
  }
  
  FOR(i,0,R) {
    LL k = RAB[i], b = pm(k,m,n);
    if (n == k) return true;
    if (n%k == 0) return false;
    
    if (b == 1) continue;
    bool pass = false;
    FOR(j,0,s) {
      if (b == n-1) {
        pass = true;
        break;
      }
      b = b*b%n;
    }
    if (!pass) return false;
  return true;
  }
}
```

### Lehmer - Đếm số lượng số nguyên tố nhỏ hơn n
Thuật toán này dùng chạy tốt với n=10^10.

```cpp
#define long long long
const int N = 100005;
const int M = 1000000007;
bool np[N];
int p[N], pp=0;

void eratos() {
  np[0]=np[1]=true;
  for (int i=2; i*i<N; i++) if (!np[i])
    for (int j=i*i; j<N; j+=i) np[j]=true;
  for (int i=2; i<N; i++)
  if (!np[i]) p[++pp]=i;
}

long power(long a, long k) {
  long P = 1;
  while (k) {
    if (k&1) P=P*a;
    k/=2; a=a*a;
  }
  return P;
}

long power(long a, long k, long M) {
  long P=1;
  for (a=a%M; k; k/=2)
  { if (k&1) P=P*a%M; a=a*a%M; }
  return P;
}

long root(long n, long k) {
  long x = pow(n, 1.0/k);
  while (power(x, k)%M==power(x, k, M) && power(x, k)<n) x++;
  while (power(x, k)%M!=power(x, k, M) || power(x, k)>n) x--;
  return x;
}

map<long, long> Phi[N];

long phi(long x, int a) {
  if (Phi[a].count(x)) return Phi[a][x];
  if (a==1) return (x+1)/2;
  long Result = phi(x, a-1) - phi(x/p[a], a-1);
  return Phi[a][x] = Result;
}

long pi(long x) {
  if (x<N)
    return upper_bound(p+1, p+pp+1, x) - (p+1);
  long a = pi(root(x, 4));
  long b = pi(root(x, 2));
  long c = pi(root(x, 3));
  long Sum = phi(x, a) + (b+a-2)*(b-a+1)/2;
  for (int i=a+1; i<=b; i++)
    Sum -= pi(x/p[i]);
  for (int i=a+1; i<=c; i++) {
    long bi = pi(root(x/p[i], 2));
    for (int j=i; j<=bi; j++)
    Sum -= pi(x/p[i]/p[j]) - (j-1);
  }
  return Sum;
}

main(){
  eratos();
  long n;
  while (cin >> n)
  cout << pi(n) << endl;
}
```

### Phi hàm Euler

phi(n) = số lượng số nhỏ hơn hoặc bằng n mà là nguyên tố cùng nhau với n.

```cpp
long Power[230997][15];
long power(int a, int k){
  if (k==0) return 1;
  if (Power[a][k] > 0) return Power[a][k];
  long p=power(a, k/2);
  if (k&1) return Power[a][k] = p*p*a;
  else return Power[a][k] = p*p;
}

long phi(int p, int k){
  // phi of p^k with p is a prime
  if (k==0) return 1;
  return (p-1)*power(p, k-1);
}

long Phi[230997]; // positive

long phi(int m){
  int i, k, om=m;
  long r=1;

  if (Phi[om] > 0) return Phi[om];
  for (i=2; i*i<=m; i++){
    k=0;
    while (m%i==0) { k++; m/=i; }
    r *= phi(i, k);
  }
  if (m>1) r *= phi(m, 1);
  return Phi[om] = r;
}

int n;

main(){
  int i;
  long r=0;
  for (;;){
    scanf("%d", &n);
    if (n==0) return 0;
    r = phi(n);
    printf("%lld\n", r);
  }
}
```

## Big Integer

Cung cấp khả năng làm việc với *Số nguyên lớn* ([**Link**](https://gist.github.com/eightnoteight/0870682c36ac2ff55477)).

## n-th Fibonacci - số fibonacci thứ n

**Cách 1: Nhân ma trận:**

```
mat = [[1, 1], [1, 0]] ^ n   
```

Khi đó `F(n) = mat[0][1]`

**Cách 2: Sử dụng công thức sau, đệ quy có nhớ:**

```cpp
f[i] = f[i/2] ^2 + f[i/2-1]^2; // khi i chẵn
f[i] = f[i/2] * f[i/2-1] + f[i/2] * f[i/2 + 1]; // khi i lẻ
```

## Kamenetsky: tìm số chữ số của n!

```cpp
long long findDigits(int n)
{
  if (n < 0) return 0;
  if (n <= 1) return 1;
  return floor(((n*log10(n/M_E)+log10(2*M_PI*n)/2.0)))+1;
}
```

# Hình học

## Diện tích một đa giác

```
A = 1/2*(x1*y2 - x2*y1 + x2*y3-x3*y2 + ... + xn*y1-x1*yn)
```

## Pick's theorem: Tìm số điểm nguyên trong 1 polygon

* **A** Area of the polygon.
* **i** number of lattice points in the interior located in the polygon.
* **b** number of lattice points on the boundary placed on the polygon's perimeter.

```
A = i + b/2 - 1
```

## Đường tròn nhỏ nhất - Emo Welzl

Tìm đường tròn nhỏ nhất chứa tất cả các điểm cho trước.

```cpp
typedef pair<double, double> point;
typedef pair<point, double> circle;
#define X first
#define Y second

point operator + (point a, point b) { return point(a.X+b.X, a.Y+b.Y); }
point operator - (point a, point b) { return point(a.X-b.X, a.Y-b.Y); }
point operator / (point a, double x) { return point(a.X/x, a.Y/x); }
double abs(point a) { return sqrt(a.X*a.X+a.Y*a.Y); }

point center_from(double bx, double by, double cx, double cy) {
  double B=bx*bx+by*by, C=cx*cx+cy*cy, D=bx*cy-by*cx;
  return point((cy*B-by*C)/(2*D), (bx*C-cx*B)/(2*D));
}

circle circle_from(point A, point B, point C) {
  point I = center_from(B.X-A.X, B.Y-A.Y, C.X-A.X, C.Y-A.Y);
  return circle(I+A, abs(I));
}

const int N = 100005;
int n, x[N], y[N];
point a[N];

circle f(int n, vector<point> T) {
  if (T.size()==3 || n==0) {
    if (T.size()==0) return circle(point(0, 0), -1);
    if (T.size()==1) return circle(T[0], 0);
    if (T.size()==2) return circle((T[0]+T[1])/2, abs(T[0]-T[1])/2);
    return circle_from(T[0], T[1], T[2]);
  }
  random_shuffle(a+1, a+n+1);
  circle Result = f(0, T);
  for (int i=1; i<=n; i++)
  if (abs(Result.X - a[i]) > Result.Y+1e-9) {
    T.push_back(a[i]);
    Result = f(i-1, T);
    T.pop_back();
  }
  return Result;
}

main() {
  scanf("%d", &n);
  for (int i=1; i<=n; i++) {
    scanf("%d%d", &x[i], &y[i]);
    a[i] = point(x[i], y[i]);
  }
  
  circle C = f(n, vector<point>());
  (cout << fixed).precision(2);
  cout << 2*C.Y << endl;
}
```

# Đồ thị

## Prim algorithm (cây khung nhỏ nhất)

```cpp
#include<bits/stdc++.h>
#define pb push_back
#define mp make_pair
using namespace std;
const int MAX = 5005;
typedef pair<int, int> PII;
vector<PII> adj[MAX];
int visited[MAX];
long long prim(int s){
  long long minimumCost = 0;
  priority_queue<PII, vector<PII>, greater<PII> > q;
  q.push(mp(0, s)); //insert (0, s) to retrieve first node as the starting one
  while(!q.empty()){
    PII p = q.top();
    q.pop();
    int length = p.first;
    s = p.second;
    if(visited[s])  continue; //if the current node is visited earlier check the next value
    visited[s] = 1; //mark the node as visited
    minimumCost+=length;
    for(int i=0;i<adj[s].size();i++){
      if(!visited[adj[s][i].second]) q.push((adj[s][i])); //push all the neighbours of current node in the priority queue
    }
  }
  return minimumCost;
}
int main(){
  int nodes, edges, x, y, weight, s;
  cin >> nodes >> edges;  //Number of Nodes and Edges
  memset(visited, 0, sizeof(visited));  //Initialise values to 0 as none of the nodes are visited
  for(int i = 0;i<edges;i++){
    cin>> x >> y >> weight;
    adj[x].pb(mp(weight, y)); 
    adj[y].pb(mp(weight, x));
  }
  cin >> s;
  long long minimumCost = prim(s);
  cout<<minimumCost;
  return 0;
}
```

Ngoài ra nên tham khảo sử dụng **Kruskal** kết hợp với **Disjoint Set**.

## Dijkstra (Đường đi ngắn nhất từ 1 đỉnh đến các đỉnh còn lại)

```cpp
#include<queue>
#include<climits>
#include<vector>
using namespace std;
const int MAX = 300;
int n; 
int first, last;
int m_adj[MAX][MAX];
struct pairComparator
{
  bool operator() (pair<int, int> a, pair<int, int> b)
  {
    return a.second > b.second;
  }
};

void rebuildPath(int previous[], int i, vector<int>& v)
{
  if(previous[i] >= 0) 
    rebuildPath(previous, previous[i], v);
  v.push_back(i);
}

int dijkstra()
{
  int d[MAX];
  bool done[MAX] = {0};
  priority_queue<pair<int, int>, vector<pair<int, int> >, pairComparator> q;
    int previous[MAX];
  
  for(int i = 0; i < n; i++)
    d[i] = INT_MAX;
  d[first] = 0;

  previous[first] = -1;

  for(int i = 0; i < n; i++)
    q.push(make_pair(i, d[i]));

  while(!p.empty())
  {
    int u = q.top().first;
    q.pop();

    if (done[u]) continue;

    for(int i = 0; i < n; i++)
      if(m_adj[u][i] && d[i] > d[u] + m_adj[u][i])
      {
        d[i] = d[u] + m_adj[u][i];
        q.push(make_pair(i, d[i]));
                previous[i] = u;
      }
  done[u] = true;
  }
  vector<int> path;
  rebuildPath(previous, last, path);
  return path;
}
```

## Floyd-Warshall

Đường đi ngắn nhất từ 1 đỉnh đến 1 đỉnh khác

```cpp
const int N = 110;
int dist[N][N];
void floyd(int n){
  for(int k = 1; k <= n; k++){
    for(int i = 1; i <= n; i++){
      for(int j = 1; j <= n; j++){
        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
      }
    }
  }
}
```

## Tarjan (tìm Strong Connected Components)

```cpp
/* Complexity: O(E + V)
 Tarjan's algorithm for finding strongly connected
components.
 *d[i] = Discovery time of node i. (Initialize to -1)
 *low[i] = Lowest discovery time reachable from node
 i. (Doesn't need to be initialized)
 *scc[i] = Strongly connected component of node i. (Doesn't
 need to be initialized)
 *s = Stack used by the algorithm (Initialize to an empty
 stack)
 *stacked[i] = True if i was pushed into s. (Initialize to
 false)
 *ticks = Clock used for discovery times (Initialize to 0)
 *current_scc = ID of the current_scc being discovered
 (Initialize to 0)
*/
vector<int> g[MAXN];
int d[MAXN], low[MAXN], scc[MAXN];
bool stacked[MAXN];
stack<int> s;
int ticks, current_scc;
void tarjan(int u){
  d[u] = low[u] = ticks++;
  s.push(u);
  stacked[u] = true;
  const vector<int> &out = g[u];
  for (int k=0, m=out.size(); k<m; ++k){
    const int &v = out[k];
    if (d[v] == -1){
      tarjan(v);
      low[u] = min(low[u], low[v]);
    }else if (stacked[v]){
      low[u] = min(low[u], low[v]);
    }
  }
  if (d[u] == low[u]){
    int v;
    do{
      v = s.top();
      s.pop();
      stacked[v] = false;
      scc[v] = current_scc;
    }while (u != v);
    current_scc++;
  }
}
```

Solution của KC97BLE

```cpp
#include <stdio.h>
#include <vector>
#include <stack>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 100005;
const int oo = 0x3c3c3c3c;

int n, m, Num[N], Low[N], cnt=0;
vector<int> a[N];
stack<int> st;
int Count=0;

void visit(int u) {
  Low[u]=Num[u]=++cnt;
  st.push(u);
  
  for (int i=0; int v=a[u][i]; i++)
  if (Num[v]) 
    Low[u]=min(Low[u], Num[v]);
  else {
    visit(v);
    Low[u]=min(Low[u], Low[v]);
  }
  
  if (Num[u]==Low[u]) { // found one
    Count++;
    int v;
    do {
      v=st.top(); st.pop();
      Num[v]=Low[v]=oo; // remove v from graph
    } while (v!=u);
  }
}

main(){
  scanf("%d%d", &n, &m);
  for (int i=1; i<=m; i++) {
    int x, y;
    scanf("%d%d", &x, &y);
    a[x].push_back(y);
  }
  for (int i=1; i<=n; i++)
  a[i].push_back(0);
  for (int i=1; i<=n; i++)
  if (!Num[i]) visit(i);
  cout << Count << endl;
}
```
## Khớp và cầu - Articulation Points (or Cut Vertices)

```cpp
#include<iostream>
#include <list>
#define NIL -1
using namespace std;
 
// A class that represents an undirected graph
class Graph
{
  int V;    // No. of vertices
  list<int> *adj;    // A dynamic array of adjacency lists
  void APUtil(int v, bool visited[], int disc[], int low[], 
    int parent[], bool ap[]);
public:
  Graph(int V);   // Constructor
  void addEdge(int v, int w);   // function to add an edge to graph
  void AP();    // prints articulation points
};
 
Graph::Graph(int V)
{
  this->V = V;
  adj = new list<int>[V];
}
 
void Graph::addEdge(int v, int w)
{
  adj[v].push_back(w);
  adj[w].push_back(v);  // Note: the graph is undirected
}
 
// A recursive function that find articulation points using DFS traversal
// u --> The vertex to be visited next
// visited[] --> keeps tract of visited vertices
// disc[] --> Stores discovery times of visited vertices
// parent[] --> Stores parent vertices in DFS tree
// ap[] --> Store articulation points
void Graph::APUtil(int u, bool visited[], int disc[], 
                                      int low[], int parent[], bool ap[])
{
  // A static variable is used for simplicity, we can avoid use of static
  // variable by passing a pointer.
  static int time = 0;

  // Count of children in DFS Tree
  int children = 0;

  // Mark the current node as visited
  visited[u] = true;

  // Initialize discovery time and low value
  disc[u] = low[u] = ++time;

  // Go through all vertices aadjacent to this
  list<int>::iterator i;
  for (i = adj[u].begin(); i != adj[u].end(); ++i)
  {
    int v = *i;  // v is current adjacent of u
    // If v is not visited yet, then make it a child of u
    // in DFS tree and recur for it
    if (!visited[v])
    {
      children++;
      parent[v] = u;
      APUtil(v, visited, disc, low, parent, ap);
      // Check if the subtree rooted with v has a connection to
      // one of the ancestors of u
      low[u]  = min(low[u], low[v]);
      // u is an articulation point in following cases
      // (1) u is root of DFS tree and has two or more chilren.
      if (parent[u] == NIL && children > 1)
        ap[u] = true;
      // (2) If u is not root and low value of one of its child is more
      // than discovery value of u.
      if (parent[u] != NIL && low[v] >= disc[u])
        ap[u] = true;
    }
    // Update low value of u for parent function calls.
    else if (v != parent[u])
      low[u]  = min(low[u], disc[v]);
  }
}
 
// The function to do DFS traversal. It uses recursive function APUtil()
void Graph::AP()
{
  // Mark all the vertices as not visited
  bool *visited = new bool[V];
  int *disc = new int[V];
  int *low = new int[V];
  int *parent = new int[V];
  bool *ap = new bool[V]; // To store articulation points

  // Initialize parent and visited, and ap(articulation point) arrays
  for (int i = 0; i < V; i++)
  {
    parent[i] = NIL;
    visited[i] = false;
    ap[i] = false;
  }

  // Call the recursive helper function to find articulation points
  // in DFS tree rooted with vertex 'i'
  for (int i = 0; i < V; i++)
    if (visited[i] == false)
      APUtil(i, visited, disc, low, parent, ap);

  // Now ap[] contains articulation points, print them
  for (int i = 0; i < V; i++)
    if (ap[i] == true)
      cout << i << " ";
}
 
// Driver program to test above function
int main()
{
  // Create graphs given in above diagrams
  cout << "\nArticulation points in first graph \n";
  Graph g1(5);
  g1.addEdge(1, 0);
  g1.addEdge(0, 2);
  g1.addEdge(2, 1);
  g1.addEdge(0, 3);
  g1.addEdge(3, 4);
  g1.AP();

  cout << "\nArticulation points in second graph \n";
  Graph g2(4);
  g2.addEdge(0, 1);
  g2.addEdge(1, 2);
  g2.addEdge(2, 3);
  g2.AP();

  cout << "\nArticulation points in third graph \n";
  Graph g3(7);
  g3.addEdge(0, 1);
  g3.addEdge(1, 2);
  g3.addEdge(2, 0);
  g3.addEdge(1, 3);
  g3.addEdge(1, 4);
  g3.addEdge(1, 6);
  g3.addEdge(3, 5);
  g3.addEdge(4, 5);
  g3.AP();

  return 0;
}
```

KC97BLE Solution

```cpp
#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 100005;
int n, m;
vector<int> a[N];
int CriticalEdge=0;
bool CriticalNode[N];
int Num[N], Low[N], Time=0;

void visit(int u, int p) {
  int NumChild = 0;
  Low[u] = Num[u] = ++Time;
  for (int i=0; int v=a[u][i]; i++)
  if (v!=p) {
    if (Num[v]!=0)
      Low[u] = min(Low[u], Num[v]);
    else {
      visit(v, u);
      NumChild++;
      Low[u] = min(Low[u], Low[v]);
      if (Low[v] >= Num[v]) 
        CriticalEdge++;
      if (u==p) {
        if (NumChild >= 2) 
        CriticalNode[u] = true;
      } else {
        if (Low[v] >= Num[u]) 
        CriticalNode[u] = true;
      }
    }
  }
}

main() {
  scanf("%d%d", &n, &m);
  for (int i=1; i<=m; i++) {
    int x, y;
    scanf("%d%d", &x, &y);
    a[x].push_back(y);
    a[y].push_back(x);
  }
  for (int i=1; i<=n; i++)
    a[i].push_back(0);
  for (int i=1; i<=n; i++)
    if (!Num[i]) visit(i, i);
  
  int Count = 0;
  for (int i=1; i<=n; i++)
    if (CriticalNode[i]) Count++;
  printf("%d %d\n", Count, CriticalEdge);
}
```

## Disjoint Set

```cpp
#include <iostream>

using namespace std;

const int N = 5000 + 1;

int n, m, q, pre[N], rank[N] = { 0 };

int get_father(int x) {
  if (pre[x] == x)
    return x;
  return pre[x] = get_father(pre[x]);
}

void merge(int x, int y) {
  x = get_father(x);
  y = get_father(y);
  if (rank[x] < rank[y])
    swap(x, y);
  if (rank[x] == rank[y])
    ++ rank[x];
  pre[y] = x;
}

inline bool is_relative(int x, int y) {
  return get_father(x) == get_father(y);
}

void init_disjoint_set() {
  for (int i = 1; i <= n; ++i)
    pre[i] = i;
}

void print_pre() {
  for (int i = 1; i <= n; ++i)
    cout << "pre[" << i << "] = " << pre[i] << endl;
}

int main() {
  cin >> n >> m >> q;
  init_disjoint_set();
  int x, y;
  for (int i = 0; i < m; ++i) {
    cin >> x >> y;
    merge(x, y);
    print_pre();
  }
  for (int i = 0; i < q; ++i) {
    cin >> x >> y;
    cout << (is_relative(x, y) ? "Yes" : "No") << endl;
  }
  return 0;
}
```

## Cặp ghép cực đại

Code của bc97kle. Độ phức tạp **O(n<sup>3</sup>)**.

```cpp
const int N = 102;
int n, m, Assigned[N];
int Visited[N], t=0;
vector<int> a[N];

bool visit(int u){
  if (Visited[u]==t) return false;
  Visited[u] = t;
  for (int i=0;int v=a[u][i];i++) {
    if (!Assigned[v] || visit(Assigned[v])) {
      Assigned[v] = u;
      return true;
    }
  }
  return false;
}

main() {
  freopen("input.txt", "r", stdin);
  scanf("%d%d", &m, &n);
  int x, y;
  while (scanf("%d%d", &x, &y) > 0)
    a[x].push_back(y);
  for (int i=1; i<=m; i++)
    a[i].push_back(0);
  
  int Count = 0;
  for (int i=1; i<=m; i++) {
    t++;
    Count += visit(i);
  }
  printf("%d\n", Count);
  for (int i=1; i<=n; i++)
  if (int j=Assigned[i])
  printf("%d %d\n", j, i);
}
```

# Link tham khảo

- KC97BLE's - https://sites.google.com/site/kc97ble/
- Geeks4Geeks - http://www.geeksforgeeks.org/top-algorithms-and-data-structures-for-competitive-programming/
- [Algorithm Gym :: Graph Algorithms](http://codeforces.com/blog/entry/16221)
- [Algorithm Gym :: Data structures](http://codeforces.com/blog/entry/15729)
- [Algorithm Gym :: Everything About Segment Trees](http://codeforces.com/blog/entry/15890)
- [Many algorithms and datastructs implementations](https://sites.google.com/site/indy256/)
