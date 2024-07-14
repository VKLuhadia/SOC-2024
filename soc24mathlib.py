import random


def pair_gcd(a: int, b: int) -> int:
    """ 
    Returns:
        The Greatest Common Divisor (GCD) using Euclidean Algorithm. 
    
    Args:
        a (int): A positive integer.
        b (int): A positive integer.
    """
    if a == 0:
        return b
    else:
        return pair_gcd(b % a, a)


def pair_egcd(a: int, b: int) -> tuple[int, int, int]:
    """ 
    Returns:
        (x, y, d) where d is the Greatest Common Divisor (GCD) and x and y are integers such that ax+by=d
        using Extended Euclidean Algorithm.
    
    Args:
        a (int): A positive integer.
        b (int): A positive integer.
    """
    if a == 0:
        return (0, 1, b)
    else:
        x1, y1, d = pair_egcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (x, y, d)


def gcd(*args: int) -> int:
    """ 
    Returns:
        The Greatest Common Divisor (GCD) of all positive integers provided using Euclidean Algorithm. 

    Args:
        Positive Integers
    """
    current_gcd = args[0]
    for num in args[1:]:
        current_gcd = pair_gcd(current_gcd, num)
    return current_gcd


def pair_lcm(a: int, b: int) -> int:
    """ 
    Returns:
        The Least Common Multiple (LCM) by evaluating GCD using Euclidean Algorithm and then dividing a*b by their GCD.
        
    Args:
        a (int): A positive integer.
        b (int): A positive integer.
    """
    return a * b // pair_gcd(a, b)


def lcm(*args: int) -> int:
    """ 
    Returns:
        The Least Common Multiple (LCM) of all positive integers provided using Euclidean Algorithm. 
    
    Args:
        Positive Integers
    """
    current_lcm = args[0]
    for num in args[1:]:
        current_lcm = pair_lcm(current_lcm, num)
    return current_lcm


def are_relatively_prime(a: int, b: int) -> bool:
    """ 
    Returns:
        True if the GCD of two positive integers is iff 1, else returns False.
        
    Args:
        a (int): A positive integer.
        b (int): A positive integer.
    """
    if pair_gcd(a, b) == 1:
        return True
    else:
        return False


def mod_inv(a: int, n: int) -> int:
    """
    Returns:
        The inverse of a % n.
    
    Raises:
        Exception if a and n are NOT coprime.
        
    Args:
        a (int): A positive integer.
        n (int): A positive integer.
    """
    x, y, gcd = pair_egcd(a, n)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} modulo {n}")
    else:
        return x % n


def pow(a: int, b: int, m: int) -> int:
    """
    Returns:
        (a ^ b) % m using Fast Exponentiation Method or Double-and-Add Algorithm.

    Args:
        a (int): A positive integer.
        b (int): A positive integer.
        m (int): A positive integer.
    """
    result = 1
    a = a % m
    while b > 0:
        if b % 2 == 1:
            result = (result * a) % m
        b //= 2
        a = (a * a) % m
    return result


def crt(a: list[int], b: list[int]) -> int:
    """
    Returns:
        Unique value of a % (product of all n[i]) such that a = a[i] % n[i].

    Args:
        a (list[int]): A list of positive integers.
        n (list[int]): A list of positive integers.
        Assume all the n[i] are pairwise coprime and that the length of the two lists are the same and is nonzero.
    """
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


def is_quadratic_residue_prime(a: int, p: int) -> int:
    """
    Returns:
        1 if a is a quadratic residue modulo p^e,
        -1 if a is a quadratic non-residue modulo p^e,
        0 if a is not coprime to p^e.

    Args:
        a (int): A positive integer.
        p (int): A prime positive integer.
    """
    if not are_relatively_prime(a, p):
        return 0
    if pow(a, (p - 1) // 2, p) == 1:
        return 1
    else:
        return -1


def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:
    """
    Returns:
        1 if a is a quadratic residue modulo p^e,
        -1 if a is a quadratic non-residue modulo p^e,
        0 if a is not coprime to p^e.

    Args:
        a (int): A positive integer.
        p (int): A prime positive integer.
        e (int): A positive integer.
    """
    if not are_relatively_prime(a, p):
        return 0
    elif is_quadratic_residue_prime(a%p, p) == 1:
        return 1
    else:
        return -1


def floor_sqrt(x: int) -> int:
    """
    Returns:
        Floor of the square root of x.

    Args:
        x (int): A positive integer.
    """
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


def is_perfect_power(x: int) -> bool:
    """
    Returns:
        True if n is a perfect power, else False.

    Args:
        m (int): A positive integer > 1.
    """

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


def is_prime(n: int) -> bool:
    """
    Returns:
        True if n is a prime number, else False.

    Args:
        m (int): A positive integer.
    """

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


def gen_k_bit_prime(k: int) -> int:
    """
    Returns:
        A random k-bit prime number p, that is, a prime number p such that 2^(k-1) <= p < 2^k.

    Args:
        k (int): A positive integer.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    while True:
        candidate = random.randint(2**(k-1), 2**k - 1)
        if is_prime(candidate):
            return candidate


def gen_prime(m: int) -> int:
    """
    Returns:
        A random prime number p, that is, a prime number p such that 2 <= p < m.

    Args:
        m (int): A positive integer >= 2.
    """
    if m <= 2:
        raise ValueError("m must be greater than 2")

    while True:
        candidate = random.randint(2, m)
        if is_prime(candidate):
            return candidate


def factor(n: int) -> list[tuple[int, int]]:
    """
    Returns:
        The prime factorisation of n.
        This should return a list of tuple,
        where the first component of the tuples are the prime factors,
        and the second component of the tuple is
        the respective power to which the corresponding factor is raised in the prime factorisation.

    Args:
        n (int): A positive integer.
    """
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


def euler_phi(n: int) -> int:
    """
    Returns:
        The Euler phi Function or Totient Function of n.
        Euler's Totient function represents the number of integers less than n and coprime with n.

    Args:
        n (int): A positive integer.
    """
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
    """
    This class represents elements in a univariate polynomial ring over the integers
    modulo some specified monic polynomial in the same ring. Polynomials are represented
    using a list of ints, where the i^th index represents the coefficient of X^i.
    The length of the list is the degree d of the quotienting polynomial.
    """

    def __init__(self, poly: list[int], pi_gen: list[int]) -> None:
        """
        Initializes the object as required.

        Args:
            poly (List[int]): The polynomial element of the ring.
            pi_gen (List[int]): The quotienting polynomial.

        Raises:
            ValueError: If pi_gen is empty or not monic.
        """
        if not pi_gen or pi_gen[-1] != 1:
            raise ValueError("Quotienting polynomial must be monic (leading coefficient must be 1)")

        self.element = poly
        self.pi_generator = pi_gen
        self.degree = len(pi_gen) - 1

    @staticmethod
    def add_mod(poly1: list[int], poly2: list[int], mod_poly: list[int]) -> list[int]:
        """
        Adds two polynomials modulo pi_generator.

        Args:
            poly1 (List[int]): The first polynomial.
            poly2 (List[int]): The second polynomial.
            mod_poly (List[int]): The modulus polynomial.

        Returns:
            List[int]: The resulting polynomial after addition and modulus reduction.
        """
        max_len = max(len(poly1), len(poly2))
        result = [0] * max_len
        for i in range(max_len):
            if i < len(poly1):
                result[i] += poly1[i]
            if i < len(poly2):
                result[i] += poly2[i]
        return QuotientPolynomialRing.reduce(result, mod_poly)

    @staticmethod
    def sub_mod(poly1: list[int], poly2: list[int], mod_poly: list[int]) -> list[int]:
        """
        Subtracts two polynomials modulo pi_generator.

        Args:
            poly1 (List[int]): The first polynomial.
            poly2 (List[int]): The second polynomial.
            mod_poly (List[int]): The modulus polynomial.

        Returns:
            List[int]: The resulting polynomial after subtraction and modulus reduction.
        """
        max_len = max(len(poly1), len(poly2))
        result = [0] * max_len
        for i in range(max_len):
            if i < len(poly1):
                result[i] += poly1[i]
            if i < len(poly2):
                result[i] -= poly2[i]
        return QuotientPolynomialRing.reduce(result, mod_poly)

    @staticmethod
    def _polymul(a: list[int], b: list[int]) -> list[int]:
        """
        Multiplies two polynomials.

        Args:
            a (List[int]): The first polynomial.
            b (List[int]): The second polynomial.

        Returns:
            List[int]: The resulting polynomial after multiplication.
        """
        result = [0] * (len(a) + len(b) - 1)
        for i in range(len(a)):
            for j in range(len(b)):
                result[i + j] += a[i] * b[j]
        return result

    @staticmethod
    def mod(poly1: list[int], poly2: list[int]) -> list[int]:
        """
        Computes the modulus of two polynomials.

        Args:
            poly1 (List[int]): The polynomial to be divided.
            poly2 (List[int]): The modulus polynomial.

        Returns:
            List[int]: The remainder polynomial after division.
        """
        _, r = QuotientPolynomialRing.divmod(poly1, poly2)
        return r

    @staticmethod
    def gcd(poly1: list[int], poly2: list[int]) -> list[int]:
        """
        Computes the greatest common divisor (GCD) of two polynomials.

        Args:
            poly1 (List[int]): The first polynomial.
            poly2 (List[int]): The second polynomial.

        Returns:
            List[int]: The GCD polynomial.
        """
        while poly2 != [0]:
            poly1, poly2 = poly2, QuotientPolynomialRing.mod(poly1, poly2)
        return poly1

    @staticmethod
    def inv_mod(poly: list[int], mod_poly: list[int]) -> list[int]:
        """
        Computes the modular inverse of a polynomial.

        Args:
            poly (List[int]): The polynomial to be inverted.
            mod_poly (List[int]): The modulus polynomial.

        Returns:
            List[int]: The inverse polynomial.

        Raises:
            ValueError: If the polynomial is not invertible in the ring.
        """
        g, x, _ = QuotientPolynomialRing.extended_gcd(poly, mod_poly)
        if g != [1]:
            raise ValueError("Polynomial is not invertible in this ring")
        return QuotientPolynomialRing.reduce(x, mod_poly)

    @staticmethod
    def extended_gcd(a: list[int], b: list[int]) -> tuple[list[int], list[int], list[int]]:
        """
        Extended Euclidean algorithm for polynomials.

        Args:
            a (List[int]): The first polynomial.
            b (List[int]): The second polynomial.

        Returns:
            Tuple[List[int], List[int], List[int]]: A tuple containing the GCD and the Bézout coefficients.
        """
        x0, x1, y0, y1 = [1], [0], [0], [1]
        while b != [0]:
            q, r = QuotientPolynomialRing.divmod(a, b)
            a, b = b, r
            x0, x1 = x1, QuotientPolynomialRing.sub_mod(x0, QuotientPolynomialRing.mul_mod(q, x1, [1]), [1])
            y0, y1 = y1, QuotientPolynomialRing.sub_mod(y0, QuotientPolynomialRing.mul_mod(q, y1, [1]), [1])
        return a, x0, y0

    @staticmethod
    def reduce(poly: list[int], mod_poly: list[int]) -> list[int]:
        """
        Reduces a polynomial modulo another polynomial.

        Args:
            poly (List[int]): The polynomial to be reduced.
            mod_poly (List[int]): The modulus polynomial.

        Returns:
            List[int]: The reduced polynomial.
        """
        while len(poly) >= len(mod_poly):
            if poly[-1] != 0:
                for i in range(len(mod_poly)):
                    poly[-1 - i] -= mod_poly[-1 - i] * poly[-1]
            poly.pop()
        while len(poly) < len(mod_poly) - 1:
            poly.append(0)
        return poly

    @staticmethod
    def _modulus(poly: list[int], mod: list[int]) -> list[int]:
        """
        Reduces a polynomial modulo another polynomial.

        Args:
            poly (List[int]): The polynomial to be reduced.
            mod (List[int]): The modulus polynomial.

        Returns:
            List[int]: The reduced polynomial.
        """
        while len(poly) >= len(mod):
            if poly[-1] != 0:
                for i in range(len(mod)):
                    poly[len(poly) - len(mod) + i] -= poly[-1] * mod[i]
            poly.pop()
        return poly

    @staticmethod
    def divmod(poly1: list[int], poly2: list[int]) -> tuple[list[int], list[int]]:
        """
        Computes the quotient and remainder of polynomial division.

        Args:
            poly1 (List[int]): The polynomial to be divided.
            poly2 (List[int]): The divisor polynomial.

        Returns:
            Tuple[List[int], List[int]]: The quotient and remainder polynomials.

        Raises:
            ValueError: If poly2 is empty (division by zero polynomial).
        """
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
    def _empty(poly: list[int]) -> bool:
        """
        Checks if a polynomial is empty (all coefficients are zero).

        Args:
            poly (List[int]): The polynomial to check.

        Returns:
            bool: True if the polynomial is empty, False otherwise.
        """
        return all(coef == 0 for coef in poly)

    @staticmethod
    def _polygcd(a: list[int], b: list[int]) -> list[int]:
        """
        Computes the greatest common divisor (GCD) of two polynomials.

        Args:
            a (List[int]): The first polynomial.
            b (List[int]): The second polynomial.

        Returns:
            List[int]: The GCD polynomial.
        """
        d = len(a)
        while not QuotientPolynomialRing._empty(b):
            r = QuotientPolynomialRing._modulus(a, b)
            a = b
            b = r
            if b == [0]:
                break
            QuotientPolynomialRing._modulus(a, b)
        for _ in range(len(a) - 1, d - 1):
            a.append(0)
        return a

    @staticmethod
    def Add(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Adds two polynomials modulo pi_generator.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The resulting polynomial after addition.

        Raises:
            ValueError: If the two arguments have different pi_generators.
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial")
        return QuotientPolynomialRing(
            QuotientPolynomialRing.add_mod(poly1.element, poly2.element, poly1.pi_generator),
            poly1.pi_generator
        )

    @staticmethod
    def Sub(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Subtracts two polynomials modulo pi_generator.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The resulting polynomial after subtraction.

        Raises:
            ValueError: If the two arguments have different pi_generators.
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial")
        return QuotientPolynomialRing(
            QuotientPolynomialRing.sub_mod(poly1.element, poly2.element, poly1.pi_generator),
            poly1.pi_generator
        )

    @staticmethod
    def Mul(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Multiplies two polynomials modulo pi_generator.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The resulting polynomial after multiplication.

        Raises:
            ValueError: If the two arguments have different pi_generators.
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial")
        result = QuotientPolynomialRing._polymul(poly1.element, poly2.element)
        result = QuotientPolynomialRing._modulus(result, poly1.pi_generator)
        return QuotientPolynomialRing(result, poly1.pi_generator)

    @staticmethod
    def GCD(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Returns GCD of two polynomials modulo pi_generator.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The resulting GCD polynomial.

        Raises:
            ValueError: If the two arguments have different pi_generators.
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial")
        result = QuotientPolynomialRing._polygcd(poly1.element, poly2.element)
        result = QuotientPolynomialRing._modulus(result, poly1.pi_generator)
        return QuotientPolynomialRing(result, poly1.pi_generator)

    @staticmethod
    def Inv(poly: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Returns the modular inverse of a polynomial modulo pi_generator.

        Args:
            poly (QuotientPolynomialRing): The polynomial to be inverted.

        Returns:
            QuotientPolynomialRing: The resulting inverse polynomial.

        Raises:
            ValueError: If the polynomial is not invertible in the ring.
        """
        return QuotientPolynomialRing(
            QuotientPolynomialRing.inv_mod(poly.element, poly.pi_generator),
            poly.pi_generator
        )


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


def get_generator(p: int) -> int:
    """ Returns a generator of (Z_p)^*, assuming p is prime. """

    if not is_prime(p):
        raise ValueError("p must be a prime number")

    if p == 2:
        return 1

    for g in range(2, p):
        if pow(g, (p - 1) // 2, p) != 1 and pow(g, p - 1, p) == 1:
            return g

    raise ValueError("Failed to find a generator")


def discrete_log(x: int, g: int, p: int) -> int:
    """ Returns the discrete logarithm of x to the base g in (Z_p)^*; assume p is prime. """

    if not is_prime(p):
        raise ValueError("p must be a prime number")

    if x <= 0 or x >= p or g <= 0 or g >= p:
        raise ValueError("x and g must be in the range [1, p-1]")

    m = int(p**0.5) + 1

    baby_steps = {}
    g_m = pow(g, m, p)
    for j in range(m):
        baby_steps[pow(g, j, p)] = j

    g_inv_m = mod_inv(pow(g, m, p), p)

    for i in range(m):
        if x in baby_steps:
            return i * m + baby_steps[x]
        x = (x * g_inv_m) % p

    raise ValueError("Discrete logarithm does not exist")


def legendre_symbol(a: int, p: int) -> int:
    """ Returns the Legendre symbol (a/p); assume p is prime. """

    if not is_prime(p):
        raise ValueError("p must be a prime number")

    a = a % p

    if a == 0:
        return 0
    elif a == 1:
        return 1
    elif a == p - 1:
        return -1 if p % 4 == 1 else 1
    else:
        return is_quadratic_residue_prime(a, p)


def jacobi_symbol(a: int, n: int) -> int:
    """ Returns the Jacobi symbol (a/n); assume n is positive. """

    if n <= 0:
        raise ValueError("n must be a positive integer")

    result = 1
    a = a % n

    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result

        a, n = n, a
        if a % 4 == n % 4 == 3:
            result = -result
        a = a % n

    if n == 1:
        return result
    else:
        return 0


def modular_sqrt_prime(x: int, p: int) -> int:
    """ Returns the modular square root of x modulo p (where p is prime).
        Raises an exception if the square root does not exist. """

    if p <= 1 or not is_prime(p):
        raise ValueError("p must be a prime number greater than 1")

    x = x % p

    if x == 0:
        return 0

    if p == 2:
        return x

    if legendre_symbol(x, p) != 1:
        raise ValueError(f"x={x} is not a quadratic residue modulo p={p}")

    if p % 4 == 3:
        sqrt_x = pow(x, (p + 1) // 4, p)
        return sqrt_x

    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1

    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(x, q, p)
    r = pow(x, (q + 1) // 2, p)

    while t != 0 and t != 1:
        t2i = t
        i = 0
        while t2i != 1:
            t2i = (t2i * t2i) % p
            i += 1

        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p

    return r


def is_smooth(m: int, y: int) -> bool:
    """ Return True if m is y-smooth, False otherwise. """

    if m == 1:
        return True

    factor = 2
    while factor * factor <= m:
        while (m % factor) == 0:
            if factor > y:
                return False
            m //= factor
        factor += 1
    if m > 1:
        return m <= y
    return True


def probabilistic_discrete_log(x: int, g: int, p: int) -> int:
    """ Returns the discrete logarithm of x to the base g in (Z_p)^* using
        a subexponential probabilistic algorithm (Baby-step Giant-step).
        Assumes p is prime and g is a generator of (Z_p)^*.
        Raises an exception if the discrete logarithm does not exist. """

    if x <= 0 or x >= p or g <= 0 or g >= p or p <= 1 or not is_prime(p):
        raise ValueError("Invalid arguments. x, g should be in (1, p-1), and p should be a prime number")

    m = floor_sqrt(p) + 1

    baby_steps = {}
    giant_step_multiplier = pow(g, m, p)
    current = x

    for j in range(m):
        if current not in baby_steps:
            baby_steps[current] = j
        current = (current * g) % p

    giant_step = 1
    for i in range(m):
        if giant_step in baby_steps:
            j = baby_steps[giant_step]
            return i * m - j
        giant_step = (giant_step * giant_step_multiplier) % p

    raise ValueError("Discrete logarithm does not exist")


def pollard_rho_factor(n):
    if n <= 1:
        return []

    def rho_factor(n):
        if n % 2 == 0:
            return 2
        x = random.randint(1, n-1)
        y = x
        c = random.randint(1, n-1)
        d = 1
        f = lambda x: (x * x + c) % n
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = gcd(abs(x - y), n)
        return d

    factors = []
    queue = [n]

    while queue:
        m = queue.pop(0)
        if is_prime(m):
            factors.append(m)
        else:
            d = rho_factor(m)
            while d == m:
                d = rho_factor(m)
            queue.extend([d, m // d])

    return factors

def probabilistic_factor(n):
    factors = pollard_rho_factor(n)
    prime_factors = sorted(set(factors))
    factorization = []

    for prime in prime_factors:
        power = 0
        m = n
        while m % prime == 0:
            m //= prime
            power += 1
        if power > 0:
            factorization.append((prime, power))

    return factorization
