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
    if pow(a, (p - 1) // 2, p) == 1 and a < p:
        return 1
    else:
        return -1


# Part 11
def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:
    if not are_relatively_prime(a, p):
        return 0
    elif is_quadratic_residue_prime(a%p, p) == 1  and a < p ** e:
        return 1
    else:
        return -1
