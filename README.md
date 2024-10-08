# SOC-2024
This module comprises of 30 functions related to Number Theory and Algebra.

Owner: Varun Luhadia

Mentor: Nilabha Saha

References used: "A Computational Introduction to Number Theory and Algebra" by Victor Shoup

1. pair_gcd(a: int, b: int) -> int : 
Returns the GCD of a and b [Hint: Euclidean Algorithm]

2. pair_egcd(a: int, b: int) -> tuple[int, int, int] : 
Returns (x,y,d) where d is the GCD of a and b, and x and y are integers such that ax+by=d

3. gcd(*args: int) -> int : 
Returns the GCD of all integers provided as arguments [Assume 2 or more arguments are always passed]

4. pair_lcm(a: int, b: int) -> int : 
Returns the LCM of a and b

5. lcm(*args: int) -> int : 
Returns the LCM of all integers provided as arguments [Assume 2 or more arguments are always passed]

6. are_relatively_prime(a: int, b: int) -> bool : 
Returns True if a and b are relatively prime, False otherwise

7. mod_inv(a: int, n: int) -> int : 
Return the modular inverse of a modulo n. For this function, raise an Exception if a and n are not coprime

8. crt(a: list[int], n: list[int]) -> int : 
This function applies the Chinese Remainder Theorem to find the unique value of 
a modulo product of all n[i] such that a = a[i] (mod n[i]) 
[Assume all the n[i] are pairwise coprime and that the length of the two lists are the same and is nonzero]

9. pow(a: int, m: int, n: int) -> int : 
Returns a^m (mod n) very fast using a method called fast exponentiation

10. is_quadratic_residue_prime(a: int, p: int) -> int : 
Return 1 if a is a quadratic residue modulo p, 
return -1 if a is a quadratic non-residue modulo p, 
return 0 if a is not coprime to p 
[Assume p is prime]

11. is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int : 
Return 1 if a is a quadratic residue modulo p^e, 
return -1 if a is a quadratic non-residue modulo p^e, 
return 0 if a is not coprime to p^e 
[Assume p is prime and e >= 1]

12. floor_sqrt(x: int) -> int : 
Returns the floor of the square root of x; assume x > 0

13. is_perfect_power(x: int) -> bool : 
Returns if x is a perfect power; assume x > 1

14. is_prime(n: int) -> bool : 
Use the Miller-Rabin test to return true if n is (probably) prime or false if it is composite; assume n > 1. 
Choose a good set of bases.

15. gen_prime(m : int) -> int : 
Generate a random prime number p such that 2 <= p <= m; assume m > 2

16. gen_k_bit_prime(k: int) -> int : 
Generate a random k-bit prime number, that is, a prime number p such that 2^(k-1) <= p < 2^k; assume k >= 1

17. factor(n: int) -> list[tuple[int, int]] : 
Returns the prime factorisation of n; assume n >= 1. 
This should return a list of tuple, where the first component of the tuples are the prime factors, 
and the second component of the tuple is the respective power 
to which the corresponding factor is raised in the prime factorisation.

18. euler_phi(n: int) -> int : 
Returns the Euler phi function of n.

19. QuotientPolynomialRing : 
This class represents elements in a univariate polynomial ring over the integers modulo some 
specified monic polynomial in the same ring. 
Polynomials would be represented using a list of ints, where the i^th index represents the coefficient of X^i. 
The length of the list would be the degree d of the quotienting polynomial. 
(a) An instance variable called pi_generator which would be the the "quotienting polynomial", 
and an instance variable called element to represent the element of the ring.
(b) init(self, poly: list[int], pi_gen: list[int]) -> None : 
This initialises the object as required. Return an exception if pi_gen is empty or not monic.
(c) A static method Add(poly1: QuotientPolynomialRing, poly2: QuotientPolynomialRing) -> 
QuotientPolynomialRing which adds two polynomials modulo pi_generator. 
Raise an exception if the two arguments have different pi_generators.
(d) A static method Sub(poly1: QuotientPolynomialRing, poly2: QuotientPolynomialRing) -> 
QuotientPolynomialRing which subtracts two polynomials modulo pi_generator. 
Raise an exception if the two arguments have different pi_generators.
(e) A static method Mul(poly1: QuotientPolynomialRing, poly2: QuotientPolynomialRing) -> 
QuotientPolynomialRing which multiplies two polynomials modulo pi_generator. 
Raise an exception if the two arguments have different pi_generators.
(f) A static method GCD(poly1: QuotientPolynomialRing, poly2: QuotientPolynomialRing) -> 
QuotientPolynomialRing which returns the GCD of two polynomials modulo pi_generator. 
Raise an exception if the two arguments have different pi_generators.
(g) A static method Inv(poly: QuotientPolynomialRing) -> 
QuotientPolynomialRing which returns the modular inverse of a polynomial modulo pi_generator. 
Raise an exception if the polynomial is not invertible in the ring.

20. aks_test(n: int) -> bool : 
Use the AKS deterministic primality testing to return true if n is prime or false if it is composite; assume n > 1.

21. get_generator(p : int) -> int : 
Returns a generator of (Z_p)^*; assume p is prime.

22. discrete_log(x: int, g: int, p: int) -> int : 
Returns the discrete logarithm of x to the base g in (Z_p)^*; assume p is prime. 
Raise an exception if the discrete logarithm does not exist.

23. legendre_symbol(a: int, p: int) -> int : 
Returns the value of the Legendre Symbol (a | p); assume p is prime.

24. jacobi_symbol(a: int, n: int) -> int : 
Returns the value of the Jacobi Symbol (a | n); assume n is positive.

25. modular_sqrt_prime(x: int, p: int) -> int : 
Returns the modular square root of x modulo p (where p is prime). Raise an exception if the square root does not exist.

26. modular_sqrt_prime_power(x: int, p: int, e: int) -> int : 
Returns the modular square root of x modulo p^e (where p is prime, e >= 1). 
Raise an exception if the square root does not exist.

27. modular_sqrt(x: int, n: int) -> int : 
Returns the modular square root of x modulo n (n >= 1). Raise an exception if the square root does not exist.

28. is_smooth(m: int, y: int) -> bool : 
Return True if m is y-smooth, False otherwise.

29. probabilistic_dlog(x: int, g: int, p: int) -> int : 
Returns the discrete log of x to the base g in (Z_p)^* using a subexponential probabilistic algorithm; 
assume p is prime, and g is a generator of (Z_p)^*. Raise and exception if the square root does not exist.

30. probabilistic_factor(n: int) -> list[tuple[int, int]] : 
Returns the prime factorisation of n using a subexponential probabilistic algorithm; assume n >= 1. 
This should return a list of tuple, where the first component of the tuples are the prime factors, 
and the second component of the tuple is the respective power 
to which the corresponding factor is raised in the prime factorisation. 
The first components must appear in ascending order in the list.
