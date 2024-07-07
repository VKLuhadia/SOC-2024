import random
# Assignment 1

# Part 1
def pair_gcd(a: int, b: int) -> int:
    if a == 0:
        return b
    else:
        return pair_gcd(b % a, a)


# Part 2
def pair_egcd(a: int, b: int) -> tuple[int, int, int]:
    if a == 0:
        return (0, 1, b)
    else:
        x1, y1, d = pair_egcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (x, y, d)


# Part 3
def gcd(*args: int) -> int:
    current_gcd = args[0]
    for num in args[1:]:
        current_gcd = pair_gcd(current_gcd, num)
    return current_gcd


# Part 4
def pair_lcm(a: int, b: int) -> int:
    return a * b // pair_gcd(a, b)


# Part 5
def lcm(*args: int) -> int:
    current_lcm = args[0]
    for num in args[1:]:
        current_lcm = pair_lcm(current_lcm, num)
    return current_lcm


# Part 6
def are_relatively_prime(a: int, b: int) -> bool:
    if pair_gcd(a, b) == 1:
        return True
    else:
        return False


# Part 7
def mod_inv(a: int, n: int) -> int:
    x, y, gcd = pair_egcd(a, n)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} modulo {n}")
    else:
        return x % n


# Part 9
def pow(a, b, m):
    result = 1
    a = a % m
    while b > 0:
        if b % 2 == 1:
            result = (result * a) % m
        b //= 2
        a = (a * a) % m
    return result


# Part 8
def crt(a: list[int], b: list[int]) -> int:
    if len(a) != len(b):
        raise ValueError("Both lists must have the same length")
    sum = 0
    M = 1
    for bi in b:
        M *= bi
    for ai, bi in zip(a, b):
        Mi = M // bi
        inv = mod_inv(Mi, bi)
        sum += ai * Mi * inv
    return sum % M


# Part 10
def is_quadratic_residue_prime(a: int, p: int) -> int:
    if not are_relatively_prime(a, p):
        return 0
    if pow(a, (p - 1) // 2, p) == 1:
        return 1
    else:
        return -1


# Part 11
def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:
    if not are_relatively_prime(a, p):
        return 0
    elif is_quadratic_residue_prime(a%p, p) == 1:
        return 1
    else:
        return -1


# Assignment 2

# Part 1
def floor_sqrt(x: int) -> int:
    if x == 0 or x == 1:
        return x
    start, end = 0, x
    result = 0
    while start <= end:
        mid = (start + end) // 2
        mid_squared = mid * mid
        if mid_squared == x:
            return mid
        elif mid_squared < x:
            start = mid + 1
            result = mid
        else:
            end = mid - 1
    return result


# Part 2
def is_perfect_power(x: int) -> bool:
    def integer_log2(x: int) -> int:
        result = 0
        while x > 1:
            x //= 2
            result += 1
        return result

    def integer_root(x: int, b: int) -> int:
        low, high = 1, x
        while low < high:
            mid = (low + high + 1) // 2
            if mid ** b > x:
                high = mid - 1
            else:
                low = mid
        return low

    if x <= 1:
        return False

    max_b = integer_log2(x)

    for b in range(2, max_b + 1):
        a = integer_root(x, b)
        if a ** b == x:
            return True

    return False


# Part 3
def is_prime(n: int) -> bool:
    def miller_rabin_test(d: int, n: int, a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        while d != n - 1:
            x = (x * x) % n
            d *= 2
            if x == 1:
                return False
            if x == n - 1:
                return True
        return False

    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    d = n - 1
    while d % 2 == 0:
        d //= 2

    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in bases:
        if a >= n:
            break
        if not miller_rabin_test(d, n, a):
            return False

    return True


# Part 5
def gen_k_bit_prime(k: int) -> int:
    if k < 1:
        raise ValueError("k must be >= 1")

    while True:
        candidate = random.randint(2**(k-1), 2**k - 1)
        if is_prime(candidate):
            return candidate


# Part 4
def gen_prime(m: int) -> int:
    if m <= 2:
        raise ValueError("m must be greater than 2")

    while True:
        candidate = random.randint(2, m)
        if is_prime(candidate):
            return candidate


# Part 6
def factor(n: int) -> list[tuple[int, int]]:
    if n == 1:
        return []

    factors = []
    i = 2
    while i * i <= n:
        count = 0
        while n % i == 0:
            n //= i
            count += 1
        if count > 0:
            factors.append((i, count))
        i += 1
    if n > 1:
        factors.append((n, 1))
    return factors


# Part 7
def euler_phi(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be greater than 0")

    if n == 1:
        return 1

    factors = factor(n)
    result = n

    for (p, _) in factors:
        result *= (p - 1)
        result //= p

    return result


class QuotientPolynomialRing:
    def __init__(self, poly: list[int], pi_gen: list[int]) -> None:
        if not pi_gen or pi_gen[-1] != 1:
            raise ValueError("Quotienting polynomial must be monic (leading coefficient must be 1)")

        self.element = poly
        self.pi_generator = pi_gen
        self.degree = len(pi_gen) - 1

    @staticmethod
    def add_mod(poly1, poly2, mod_poly):
        max_len = max(len(poly1), len(poly2))
        result = [0] * max_len
        for i in range(max_len):
            if i < len(poly1):
                result[i] += poly1[i]
            if i < len(poly2):
                result[i] += poly2[i]
        return QuotientPolynomialRing.reduce(result, mod_poly)

    @staticmethod
    def sub_mod(poly1, poly2, mod_poly):
        max_len = max(len(poly1), len(poly2))
        result = [0] * max_len
        for i in range(max_len):
            if i < len(poly1):
                result[i] += poly1[i]
            if i < len(poly2):
                result[i] -= poly2[i]
        return QuotientPolynomialRing.reduce(result, mod_poly)

    @staticmethod
    def _polymul(a, b):
        result = [0] * (len(a) + len(b) - 1)
        for i in range(len(a)):
            for j in range(len(b)):
                result[i + j] += a[i] * b[j]
        return result

    @staticmethod
    def mod(poly1, poly2):
        _, r = QuotientPolynomialRing.divmod(poly1, poly2)
        return r

    @staticmethod
    def gcd(poly1, poly2):
        while poly2 != [0]:
            poly1, poly2 = poly2, QuotientPolynomialRing.mod(poly1, poly2)
        return poly1

    @staticmethod
    def inv_mod(poly, mod_poly):
        g, x, _ = QuotientPolynomialRing.extended_gcd(poly, mod_poly)
        if g != [1]:
            raise ValueError("Polynomial is not invertible in this ring")
        return QuotientPolynomialRing.reduce(x, mod_poly)

    @staticmethod
    def extended_gcd(a, b):
        x0, x1, y0, y1 = [1], [0], [0], [1]
        while b != [0]:
            q, r = QuotientPolynomialRing.divmod(a, b)
            a, b = b, r
            x0, x1 = x1, QuotientPolynomialRing.sub_mod(x0, QuotientPolynomialRing.mul_mod(q, x1, [1]), [1])
            y0, y1 = y1, QuotientPolynomialRing.sub_mod(y0, QuotientPolynomialRing.mul_mod(q, y1, [1]), [1])
        return a, x0, y0

    @staticmethod
    def reduce(poly, mod_poly):
        while len(poly) >= len(mod_poly):
            if poly[-1] != 0:
                for i in range(len(mod_poly)):
                    poly[-1 - i] -= mod_poly[-1 - i] * poly[-1]
            poly.pop()
        while len(poly) < len(mod_poly) - 1:
            poly.append(0)
        return poly

    @staticmethod
    def _modulus(poly, mod):
        while len(poly) >= len(mod):
            if poly[-1] != 0:
                for i in range(len(mod)):
                    poly[len(poly) - len(mod) + i] -= poly[-1] * mod[i]
            poly.pop()
        return poly

    @staticmethod
    def divmod(poly1, poly2):
        if not poly2:
            raise ValueError("Division by zero polynomial")

        q = [0] * (len(poly1) - len(poly2) + 1)
        r = poly1[:]
        while len(r) >= len(poly2) and any(r):
            if r[-1] != 0:
                q[len(r) - len(poly2)] = r[-1]
                for i in range(len(poly2)):
                    if len(r) - 1 - i < 0:
                        break
                    r[len(r) - 1 - i] -= poly2[-1 - i] * q[len(r) - len(poly2)]
            if any(r):
                r.pop()
        return q, r

    @staticmethod
    def _empty(poly):
        return all(coef == 0 for coef in poly)

    @staticmethod
    def _polygcd(a, b):
        d = len(a)
        while (not QuotientPolynomialRing._empty(b)):
            r = QuotientPolynomialRing._modulus(a, b)
            a = b
            b = r
            if (b == [0]):
                break
            QuotientPolynomialRing._modulus(a, b)
        for _ in range(len(a) - 1, d - 1):
            a.append(0)
        return a

    @staticmethod
    def Add(poly1, poly2):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial")
        return QuotientPolynomialRing(
            QuotientPolynomialRing.add_mod(poly1.element, poly2.element, poly1.pi_generator),
            poly1.pi_generator
        )

    @staticmethod
    def Sub(poly1, poly2):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial")
        return QuotientPolynomialRing(
            QuotientPolynomialRing.sub_mod(poly1.element, poly2.element, poly1.pi_generator),
            poly1.pi_generator
        )

    @staticmethod
    def Mul(poly1, poly2):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial")
        result = QuotientPolynomialRing._polymul(poly1.element, poly2.element)
        result = QuotientPolynomialRing._modulus(result, poly1.pi_generator)
        return QuotientPolynomialRing(result, poly1.pi_generator)

    @staticmethod
    def GCD(poly1, poly2):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial")
        result = QuotientPolynomialRing._polygcd(poly1.element, poly2.element)
        result = QuotientPolynomialRing._modulus(result, poly1.pi_generator)
        return QuotientPolynomialRing(result, poly1.pi_generator)

    @staticmethod
    def Inv(poly):
        return QuotientPolynomialRing(
            QuotientPolynomialRing.inv_mod(poly.element, poly.pi_generator),
            poly.pi_generator
        )


# Part 8
def aks_test(n: int) -> bool:
    if n <= 1:
        return False

    if n <= 3:
        return True

    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5

    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True
