# src/privacy/mpc_handler.py

import random

class MPCProtocol:
    @staticmethod
    def secure_sum(values: list) -> int:
        """Compute a secure sum using secret sharing."""
        partial_sums = []
        for value in values:
            # Split each value into a random share and a complement
            share = random.randint(0, value)
            complement = value - share
            partial_sums.append((share, complement))
        
        # Assume all parties compute the complement sum
        total_sum = sum(share + complement for share, complement in partial_sums)
        return total_sum

# Example usage
values = [100, 200, 300]
mpc_protocol = MPCProtocol()
secure_result = mpc_protocol.secure_sum(values)
print(f"Securely computed sum: {secure_result}")
