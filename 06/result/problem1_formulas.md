# Problem 1 Formula Derivation

For stencil offsets $s_j$ and coefficients $c_j$, Taylor expansion gives

$$
\sum_j c_j f(x+s_j h) = \sum_{k=0}^{\infty} \frac{h^k}{k!}\left(\sum_j c_j s_j^k\right) f^{(k)}(x).
$$

Choose the coefficients so that

$$
\sum_j c_j s_j^k = 0 \quad (0 \le k < m), \qquad \sum_j c_j s_j^m = m!,
$$

then

$$
f^{(m)}(x) = \frac{1}{h^m}\sum_j c_j f(x+s_j h) + O(h^p),
$$

where $p=1$ for the minimum forward/backward stencil and $p=2$ for the symmetric central stencil used here.

The coefficients below are obtained by solving these moment equations directly rather than calling a black-box finite-difference template.

## Forward difference for the 2nd derivative

Offsets: `(0, 1, 2)`. Assume

$$
D(x) = \frac{c_{0}\,f(x) + c_{1}\,f(x+h) + c_{2}\,f(x+2h)}{h^{2}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} = 0\\
c_{1} + 2\,c_{2} = 0\\
c_{1} + 4\,c_{2} = 2
\end{cases}
$$

Solving gives

$$
c_{0} = 1,\quad c_{1} = -2,\quad c_{2} = 1
$$

Hence

$$
f^{(2)}(x) = \frac{f(x) - 2\,f(x+h) + f(x+2h)}{h^{2}} + O(h^{1})
$$

The first retained truncation term is

$$
1\,h^{1} f^{(3)}(x)
$$

## Backward difference for the 2nd derivative

Offsets: `(-2, -1, 0)`. Assume

$$
D(x) = \frac{c_{0}\,f(x-2h) + c_{1}\,f(x-h) + c_{2}\,f(x)}{h^{2}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} = 0\\
-2\,c_{0} - c_{1} = 0\\
4\,c_{0} + c_{1} = 2
\end{cases}
$$

Solving gives

$$
c_{0} = 1,\quad c_{1} = -2,\quad c_{2} = 1
$$

Hence

$$
f^{(2)}(x) = \frac{f(x-2h) - 2\,f(x-h) + f(x)}{h^{2}} + O(h^{1})
$$

The first retained truncation term is

$$
-1\,h^{1} f^{(3)}(x)
$$

## Central difference for the 2nd derivative

Offsets: `(-1, 0, 1)`. Assume

$$
D(x) = \frac{c_{0}\,f(x-h) + c_{1}\,f(x) + c_{2}\,f(x+h)}{h^{2}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} = 0\\
-c_{0} + c_{2} = 0\\
c_{0} + c_{2} = 2
\end{cases}
$$

Solving gives

$$
c_{0} = 1,\quad c_{1} = -2,\quad c_{2} = 1
$$

Hence

$$
f^{(2)}(x) = \frac{f(x-h) - 2\,f(x) + f(x+h)}{h^{2}} + O(h^{2})
$$

The first retained truncation term is

$$
\frac{1}{12}\,h^{2} f^{(4)}(x)
$$

## Forward difference for the 3rd derivative

Offsets: `(0, 1, 2, 3)`. Assume

$$
D(x) = \frac{c_{0}\,f(x) + c_{1}\,f(x+h) + c_{2}\,f(x+2h) + c_{3}\,f(x+3h)}{h^{3}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} = 0\\
c_{1} + 2\,c_{2} + 3\,c_{3} = 0\\
c_{1} + 4\,c_{2} + 9\,c_{3} = 0\\
c_{1} + 8\,c_{2} + 27\,c_{3} = 6
\end{cases}
$$

Solving gives

$$
c_{0} = -1,\quad c_{1} = 3,\quad c_{2} = -3,\quad c_{3} = 1
$$

Hence

$$
f^{(3)}(x) = \frac{-f(x) + 3\,f(x+h) - 3\,f(x+2h) + f(x+3h)}{h^{3}} + O(h^{1})
$$

The first retained truncation term is

$$
\frac{3}{2}\,h^{1} f^{(4)}(x)
$$

## Backward difference for the 3rd derivative

Offsets: `(-3, -2, -1, 0)`. Assume

$$
D(x) = \frac{c_{0}\,f(x-3h) + c_{1}\,f(x-2h) + c_{2}\,f(x-h) + c_{3}\,f(x)}{h^{3}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} = 0\\
-3\,c_{0} - 2\,c_{1} - c_{2} = 0\\
9\,c_{0} + 4\,c_{1} + c_{2} = 0\\
-27\,c_{0} - 8\,c_{1} - c_{2} = 6
\end{cases}
$$

Solving gives

$$
c_{0} = -1,\quad c_{1} = 3,\quad c_{2} = -3,\quad c_{3} = 1
$$

Hence

$$
f^{(3)}(x) = \frac{-f(x-3h) + 3\,f(x-2h) - 3\,f(x-h) + f(x)}{h^{3}} + O(h^{1})
$$

The first retained truncation term is

$$
- \frac{3}{2}\,h^{1} f^{(4)}(x)
$$

## Central difference for the 3rd derivative

Offsets: `(-2, -1, 0, 1, 2)`. Assume

$$
D(x) = \frac{c_{0}\,f(x-2h) + c_{1}\,f(x-h) + c_{2}\,f(x) + c_{3}\,f(x+h) + c_{4}\,f(x+2h)}{h^{3}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} + c_{4} = 0\\
-2\,c_{0} - c_{1} + c_{3} + 2\,c_{4} = 0\\
4\,c_{0} + c_{1} + c_{3} + 4\,c_{4} = 0\\
-8\,c_{0} - c_{1} + c_{3} + 8\,c_{4} = 6\\
16\,c_{0} + c_{1} + c_{3} + 16\,c_{4} = 0
\end{cases}
$$

Solving gives

$$
c_{0} = - \frac{1}{2},\quad c_{1} = 1,\quad c_{2} = 0,\quad c_{3} = -1,\quad c_{4} = \frac{1}{2}
$$

Hence

$$
f^{(3)}(x) = \frac{-\frac{1}{2}\,f(x-2h) + f(x-h) - f(x+h) + \frac{1}{2}\,f(x+2h)}{h^{3}} + O(h^{2})
$$

The first retained truncation term is

$$
\frac{1}{4}\,h^{2} f^{(5)}(x)
$$

## Forward difference for the 4th derivative

Offsets: `(0, 1, 2, 3, 4)`. Assume

$$
D(x) = \frac{c_{0}\,f(x) + c_{1}\,f(x+h) + c_{2}\,f(x+2h) + c_{3}\,f(x+3h) + c_{4}\,f(x+4h)}{h^{4}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} + c_{4} = 0\\
c_{1} + 2\,c_{2} + 3\,c_{3} + 4\,c_{4} = 0\\
c_{1} + 4\,c_{2} + 9\,c_{3} + 16\,c_{4} = 0\\
c_{1} + 8\,c_{2} + 27\,c_{3} + 64\,c_{4} = 0\\
c_{1} + 16\,c_{2} + 81\,c_{3} + 256\,c_{4} = 24
\end{cases}
$$

Solving gives

$$
c_{0} = 1,\quad c_{1} = -4,\quad c_{2} = 6,\quad c_{3} = -4,\quad c_{4} = 1
$$

Hence

$$
f^{(4)}(x) = \frac{f(x) - 4\,f(x+h) + 6\,f(x+2h) - 4\,f(x+3h) + f(x+4h)}{h^{4}} + O(h^{1})
$$

The first retained truncation term is

$$
2\,h^{1} f^{(5)}(x)
$$

## Backward difference for the 4th derivative

Offsets: `(-4, -3, -2, -1, 0)`. Assume

$$
D(x) = \frac{c_{0}\,f(x-4h) + c_{1}\,f(x-3h) + c_{2}\,f(x-2h) + c_{3}\,f(x-h) + c_{4}\,f(x)}{h^{4}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} + c_{4} = 0\\
-4\,c_{0} - 3\,c_{1} - 2\,c_{2} - c_{3} = 0\\
16\,c_{0} + 9\,c_{1} + 4\,c_{2} + c_{3} = 0\\
-64\,c_{0} - 27\,c_{1} - 8\,c_{2} - c_{3} = 0\\
256\,c_{0} + 81\,c_{1} + 16\,c_{2} + c_{3} = 24
\end{cases}
$$

Solving gives

$$
c_{0} = 1,\quad c_{1} = -4,\quad c_{2} = 6,\quad c_{3} = -4,\quad c_{4} = 1
$$

Hence

$$
f^{(4)}(x) = \frac{f(x-4h) - 4\,f(x-3h) + 6\,f(x-2h) - 4\,f(x-h) + f(x)}{h^{4}} + O(h^{1})
$$

The first retained truncation term is

$$
-2\,h^{1} f^{(5)}(x)
$$

## Central difference for the 4th derivative

Offsets: `(-2, -1, 0, 1, 2)`. Assume

$$
D(x) = \frac{c_{0}\,f(x-2h) + c_{1}\,f(x-h) + c_{2}\,f(x) + c_{3}\,f(x+h) + c_{4}\,f(x+2h)}{h^{4}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} + c_{4} = 0\\
-2\,c_{0} - c_{1} + c_{3} + 2\,c_{4} = 0\\
4\,c_{0} + c_{1} + c_{3} + 4\,c_{4} = 0\\
-8\,c_{0} - c_{1} + c_{3} + 8\,c_{4} = 0\\
16\,c_{0} + c_{1} + c_{3} + 16\,c_{4} = 24
\end{cases}
$$

Solving gives

$$
c_{0} = 1,\quad c_{1} = -4,\quad c_{2} = 6,\quad c_{3} = -4,\quad c_{4} = 1
$$

Hence

$$
f^{(4)}(x) = \frac{f(x-2h) - 4\,f(x-h) + 6\,f(x) - 4\,f(x+h) + f(x+2h)}{h^{4}} + O(h^{2})
$$

The first retained truncation term is

$$
\frac{1}{6}\,h^{2} f^{(6)}(x)
$$

## Forward difference for the 5th derivative

Offsets: `(0, 1, 2, 3, 4, 5)`. Assume

$$
D(x) = \frac{c_{0}\,f(x) + c_{1}\,f(x+h) + c_{2}\,f(x+2h) + c_{3}\,f(x+3h) + c_{4}\,f(x+4h) + c_{5}\,f(x+5h)}{h^{5}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} + c_{4} + c_{5} = 0\\
c_{1} + 2\,c_{2} + 3\,c_{3} + 4\,c_{4} + 5\,c_{5} = 0\\
c_{1} + 4\,c_{2} + 9\,c_{3} + 16\,c_{4} + 25\,c_{5} = 0\\
c_{1} + 8\,c_{2} + 27\,c_{3} + 64\,c_{4} + 125\,c_{5} = 0\\
c_{1} + 16\,c_{2} + 81\,c_{3} + 256\,c_{4} + 625\,c_{5} = 0\\
c_{1} + 32\,c_{2} + 243\,c_{3} + 1024\,c_{4} + 3125\,c_{5} = 120
\end{cases}
$$

Solving gives

$$
c_{0} = -1,\quad c_{1} = 5,\quad c_{2} = -10,\quad c_{3} = 10,\quad c_{4} = -5,\quad c_{5} = 1
$$

Hence

$$
f^{(5)}(x) = \frac{-f(x) + 5\,f(x+h) - 10\,f(x+2h) + 10\,f(x+3h) - 5\,f(x+4h) + f(x+5h)}{h^{5}} + O(h^{1})
$$

The first retained truncation term is

$$
\frac{5}{2}\,h^{1} f^{(6)}(x)
$$

## Backward difference for the 5th derivative

Offsets: `(-5, -4, -3, -2, -1, 0)`. Assume

$$
D(x) = \frac{c_{0}\,f(x-5h) + c_{1}\,f(x-4h) + c_{2}\,f(x-3h) + c_{3}\,f(x-2h) + c_{4}\,f(x-h) + c_{5}\,f(x)}{h^{5}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} + c_{4} + c_{5} = 0\\
-5\,c_{0} - 4\,c_{1} - 3\,c_{2} - 2\,c_{3} - c_{4} = 0\\
25\,c_{0} + 16\,c_{1} + 9\,c_{2} + 4\,c_{3} + c_{4} = 0\\
-125\,c_{0} - 64\,c_{1} - 27\,c_{2} - 8\,c_{3} - c_{4} = 0\\
625\,c_{0} + 256\,c_{1} + 81\,c_{2} + 16\,c_{3} + c_{4} = 0\\
-3125\,c_{0} - 1024\,c_{1} - 243\,c_{2} - 32\,c_{3} - c_{4} = 120
\end{cases}
$$

Solving gives

$$
c_{0} = -1,\quad c_{1} = 5,\quad c_{2} = -10,\quad c_{3} = 10,\quad c_{4} = -5,\quad c_{5} = 1
$$

Hence

$$
f^{(5)}(x) = \frac{-f(x-5h) + 5\,f(x-4h) - 10\,f(x-3h) + 10\,f(x-2h) - 5\,f(x-h) + f(x)}{h^{5}} + O(h^{1})
$$

The first retained truncation term is

$$
- \frac{5}{2}\,h^{1} f^{(6)}(x)
$$

## Central difference for the 5th derivative

Offsets: `(-3, -2, -1, 0, 1, 2, 3)`. Assume

$$
D(x) = \frac{c_{0}\,f(x-3h) + c_{1}\,f(x-2h) + c_{2}\,f(x-h) + c_{3}\,f(x) + c_{4}\,f(x+h) + c_{5}\,f(x+2h) + c_{6}\,f(x+3h)}{h^{5}}
$$

The Taylor-moment system is

$$
\begin{cases}
c_{0} + c_{1} + c_{2} + c_{3} + c_{4} + c_{5} + c_{6} = 0\\
-3\,c_{0} - 2\,c_{1} - c_{2} + c_{4} + 2\,c_{5} + 3\,c_{6} = 0\\
9\,c_{0} + 4\,c_{1} + c_{2} + c_{4} + 4\,c_{5} + 9\,c_{6} = 0\\
-27\,c_{0} - 8\,c_{1} - c_{2} + c_{4} + 8\,c_{5} + 27\,c_{6} = 0\\
81\,c_{0} + 16\,c_{1} + c_{2} + c_{4} + 16\,c_{5} + 81\,c_{6} = 0\\
-243\,c_{0} - 32\,c_{1} - c_{2} + c_{4} + 32\,c_{5} + 243\,c_{6} = 120\\
729\,c_{0} + 64\,c_{1} + c_{2} + c_{4} + 64\,c_{5} + 729\,c_{6} = 0
\end{cases}
$$

Solving gives

$$
c_{0} = - \frac{1}{2},\quad c_{1} = 2,\quad c_{2} = - \frac{5}{2},\quad c_{3} = 0,\quad c_{4} = \frac{5}{2},\quad c_{5} = -2,\quad c_{6} = \frac{1}{2}
$$

Hence

$$
f^{(5)}(x) = \frac{-\frac{1}{2}\,f(x-3h) + 2\,f(x-2h) - \frac{5}{2}\,f(x-h) + \frac{5}{2}\,f(x+h) - 2\,f(x+2h) + \frac{1}{2}\,f(x+3h)}{h^{5}} + O(h^{2})
$$

The first retained truncation term is

$$
\frac{1}{3}\,h^{2} f^{(7)}(x)
$$
