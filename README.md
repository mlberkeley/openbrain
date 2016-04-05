---
author:
- |
    William Guss\
    26793499\
    wguss@berkeley.edu
title: '<span>EE 16A</span>: Homework <span>9</span>'
...

**Friends** My friends are: Kunal Gosar (26074334), Amy (26137562),

**Mechanical: Correlation**

See iPython notebook.

See iPython botebook.

**Inner Products**

Let $x,y$ be vectors in a Banach space $X$ with norem $\|\cdot\|$. Then
$$\begin{aligned}
                \|x + y\|^2 \leq \|x\| + \|y\|
            \end{aligned}$$

Observe the following manipulation $$\begin{aligned}
         \|x + y\|^2 &= \langle x + y, x+ y \rangle \\
            &=  \langle x, x \rangle + \langle x, y \rangle  + \langle y, x \rangle  + \langle y, y \rangle  \\
            &= \|x\|^2 + 2\langle x, y \rangle + \|y\|^2.
        \end{aligned}$$ Then by the Cauchy-Schwartz inequality we have
that
$$\|x + y\|^2 \leq \|x\|^2 + 2\|x\|\|y\| + \|y\|^2 = (\|x\| + \|y\|)^2$$
By the semipositive definiteness of $\cdot^2$, we have
$$\|x + y\| \leq \|x\| + \|y\|.$$ This completes the proof.

**A Different View of Matrix Multiplication**

We calculate the inner and outer products using iPython, and get the
following $$\begin{aligned}
                x \cdot y = 6 \;\;\;\;\;\;& x \otimes y = \begin{bmatrix}
                    1 & 2 & 3 \\
                    1 & 2 & 3 \\
                    1 & 2 & 3 \\
                \end{bmatrix}
            \end{aligned}$$ In the second case, we have
$$\begin{aligned}
                x \cdot y = 36 \;\;\;\;\;\;& x \otimes y = \begin{bmatrix}
                    1 & 2 & 3 \\
                    4 & 8 & 12 \\
                    9 & 18 & 27\\
                \end{bmatrix}
            \end{aligned}$$

(0,-0.25) rectangle (2.2,0.25); (0,-.5) circle (0.5); (0,0.5)
circle(0.5);

In the case of inner product, order does not matter by definition.
*Note: we define inner product as a discriminatory, symmetric, bilinear,
etc. operator from $X \times X \to \mathbb{R}$*.

For outer products we know that $$x \otimes y = (y \otimes x)^T.$$
Therefore order does matter!

In this case $$(AB)_{ij} = \langle A_i, b_j \rangle$$ for every $i,j.$
This defines a matrix.

Observe that the definition of outer product
$$a_i\otimes B_i = \begin{bmatrix}
                a_{i1}B_{i1} &\cdots & a_{i1}B_{in} \\
                \vdots & & \vdots \\
                a_{in}B_{i1} & \cdots & a_{in}B_{in} 
            \end{bmatrix}$$ Observe that
$(AB)_{11} = \sum_i a_{i1}B_{i1}.$ And in fact we have that
$(AB)_{ij} = \sum_k a_{ki}B_{kj}.$ Observing our derivation for
$a_i \otimes B_i$, we can then say
$$AB = \sum_{i=1}^n a_i \otimes B_i.$$

We have observed something remarkable in this question. Outer products
and inner products are operators dual to one another for matrix
multiplication. Even more fire to this claim: the row space of a matrix
is dual to its column space under transposition.

If $A$ is a symetric positive definite matrix and $\lambda_i, v_i$
denotes the $i^{th}$ eigenpair. Then
$$A = \sum_i  \lambda_i v_i \otimes v_i$$

If you grant Theorem 2, ie. the spectral theorem, then $AB$ positive
definite implies
$$AB = \sum a_i \otimes B_i = \sum \lambda_i v_i \otimes v_i.$$ It would
be an interesting theoretical exploration, to entertain the difference
between the eigenvector auto-outerproduct and $a_i \otimes B_i.$ Idk bro

**Audio file matching**

The dot product of $X_1$ and $X_2$ is $n.$ However in the case that
$$X_2 = \sum_{i=1}^n (-1)^{n+1}e_i$$ we have that
$\langle X_1, X_2 \rangle = 0.$

We propose to take the dot product of $Y$ with each block
$[x_i\ x_{i+1}\ x_{i+2}]$, then take the argmax over $i$. In the first
case we get that $i = 2$. In the second case we get that every $i$ is a
maximum. So we can take $i = 1.$

In this case scale does matter! So we need to take the weighted
Euclidean norm as our metric of similarity. That is, if we have a set of
possible signals $\mathcal{X}$ which contains $X,$ then
$$sim(X_i, Y) = \frac{\langle X_i, Y \rangle}{\|X_i\|\|Y\|} e^{-\|X_i - Y\|^2
            }$$

Then for the case of our signal similarity question, we consider all
contiguous dimension $dim(Y)$ subvectors in $X$ and take the argmax with
respect to $i$ of $sim(X_i, Y)$

**The most complicated proof ever... not really!**\
In this question we are going to prove the following theorem!

Let $f,g$ be integrable functions whose integrals over $\mathbb{R}$ are
finite, then
$$\left|\int_\mathbb{R} f(x)g(x)\ dx\right| \leq \sqrt{\int_\mathbb{R} f(x)\ dx} \sqrt{\int_\mathbb{R} g(x)\ dx}.$$

At first this looks super complicated until you observe (proove!) the
following facts.

Show that $R(\mathbb{R}),$ the set of Riemann integrable (finite)
functions $f: \mathbb{R} \to \mathbb{R}$, is a vector space.

Show that if $f,g \in R(\mathbb{R})$ then
$$\langle \cdot, \cdot \rangle:f,g \mapsto \int_\mathbb{R} f(x)g(x)\ dx$$
is an inner product, ie that it is positive definite, bilinear, and
symmetric.

Show that $$\|\cdot\|: f \mapsto \sqrt{\int_\mathbb{R} f^2(x)\ dx}$$ is
an inner product.

Finally show Theorem 3.

*Solution.*

Clearly $f,g$ are vectors over $\mathbb{R}$ indexed by
$x \in \mathbb{R}.$ Then consider the following for $a \in \mathbb{R}$
$$f+ag = x \mapsto f(x) + ag(x)$$ is integrable by the linearity of the
integral and by the finiteness of $\int_\mathbb{R} f\ dx$ and
$\int_\mathbb{R} g\ dx$. Therefore $R(\mathbb{R})$ is closed under
addition and scalar multiplication and $R(\mathbb{R})$ is a vector
space.

Consider the function
$\langle \cdot, \cdot \rangle: R(\mathbb{R}) \times R(\mathbb{R}) \to \mathbb{R}.$
Then take $f,g,h \in R(\mathbb{R})$ and $a \in \mathbb{R}.$
$$\begin{aligned}
            \langle af, g+ h\rangle &= \int_\mathbb{R}af(x)(g(x) + h(x))\ dx \\
            &= a\left(\int_\mathbb{R} f(x) g(x)\ dx + \int_\mathbb{R} f(x)h(x)\ dx \right) \\
            &= a\left(\langle f, g\rangle + \langle f, h\rangle \right),
            \end{aligned}$$ by linearity of integration. Furthermore
$$\langle f, g \rangle =  \int_\mathbb{R} f(x)g(x)\ dx = \int_\mathbb{R} g(x)f(x)\ dx = \langle g, f \rangle.$$
Therefore the functional is symmetric, and it follows immediately by the
previous statements that it is also bilinear.

Consider $\langle f, f \rangle.$ This value is clearly positive since
the integral of a montone increasing function, ie $f^2$, is always
positive. If $f$ is $0$ almost everywhere then
$\langle f, f \rangle = 0$ and the functional is positive definite.

It is clear that $\|\cdot \|$ is norm since
$\langle f, f \rangle = \|f\|^2$; that is the inner product induces
$\|\cdot\|$ as a norm.

Cauchy-Schwartz gives us $$\begin{aligned}
         |\langle f,g \rangle| &\leq \|f\|\|g\| \\
         \left|\int_\mathbb{R} f(x)g(x)\ dx\right| &\leq \sqrt{\int_\mathbb{R} f(x)\ dx} \sqrt{\int_\mathbb{R} g(x)\ dx}.
        \end{aligned}$$ This completes a proof of Theorem 3.
