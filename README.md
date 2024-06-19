# SOC-2024
Project ID 98: Algorithmic Number Theory and Algebra

Mentor: Nilabha Saha (Github ID: @Nilabha13)

This repository comprises of soc24mathlib.py, a python file comprising of various functions related to Number Theory and Algebra.
1. pair_gcd(a: int, b: int) -> int : Returns the GCD of a and b [Hint: Euclidean Algorithm]
2. pair_egcd(a: int, b: int) -> tuple[int, int, int] : Returns (x,y,d) where d is the GCD of a and b, and x and y are integers such that ax+by=d
3. gcd(*args: int) -> int: Returns the GCD of all integers provided as arguments [Assume 2 or more arguments are always passed]
4. pair_lcm(a: int, b: int) _> int : Returns the LCM of a and b
5. lcm(*args: int) -> int: Returns the LCM of all integers provided as arguments [Assume 2 or more arguments are always passed]
6. are_relatively_prime(a: int, b: int) -> bool: Returns True if a and b are relatively prime, False otherwise
7. mod_inv(a: int, n: int) -> int: Return the modular inverse of a modulo n. For this function, raise an Exception if a and n are not coprime
8. crt(a: list[int], n: list[int]) -> int: This function applies the Chinese Remainder Theorem to find the unique value of a modulo product of all n[i] such that a = a[i] (mod n[i]) [Assume all the n[i] are pairwise coprime and that the length of the two lists are the same and is nonzero)
9. The Python function pow(a,m,n) compute a^m (mod n) very fast using a method called fast exponentiation. Read about how it is done. This method is often also referred to as the double-and-add algorithm (you don't have to implement anything wrt to this now, but you might have to in a different assignment). Feel free to make use of the pow(a,m,n) function in your scripts.
10. is_quadratic_residue_prime(a: int, p: int) -> int: Return 1 if a is a quadratic residue modulo p, return -1 if a is a quadratic non-residue modulo p, return 0 if a is not coprime to p [Assume p is prime]
11. is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int: Return 1 if a is a quadratic residue modulo p^e, return -1 if a is a quadratic non-residue modulo p^e, return 0 if a is not coprime to p^e [Assume p is prime and e >= 1]

References used:
A Computational Introduction to Number Theory and Algebra by Victor Shoup
