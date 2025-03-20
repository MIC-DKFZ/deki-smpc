import secrets
import string


class SecurityUtils:
    @staticmethod
    def generate_preshared_secret(length: int = 32) -> str:
        """Generate a cryptographically secure preshared secret.

        - Must be at least `length` characters long (default: 32)
        - Contains at least one uppercase letter, one digit, and one special character.
        - Uses `secrets` for true randomness.
        """
        if length < 16:
            raise ValueError("Secret length must be at least 16 characters")

        # Securely select one character from each required category
        uppercase = secrets.choice(string.ascii_uppercase)
        digit = secrets.choice(string.digits)
        special = secrets.choice(string.punctuation)

        # Generate the remaining characters securely
        all_characters = string.ascii_letters + string.digits + string.punctuation
        remaining_chars = "".join(
            secrets.choice(all_characters) for _ in range(length - 3)
        )

        # Combine and shuffle securely
        secret = list(uppercase + digit + special + remaining_chars)
        secrets.SystemRandom().shuffle(secret)

        return "".join(secret)
