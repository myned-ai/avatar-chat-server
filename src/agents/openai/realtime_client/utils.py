"""
Utility functions for the Realtime API client.

Provides helpers for:
- Audio format conversion (Float32 to PCM16)
- Base64 encoding/decoding of audio data
- Array buffer manipulation
- ID generation
"""

import base64
import random

import numpy as np


class RealtimeUtils:
    """Basic utilities for the RealtimeAPI"""

    # Base58 characters (no repeating chars)
    _ID_CHARS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    @staticmethod
    def float_to_16bit_pcm(float32_array: np.ndarray) -> bytes:
        """
        Converts Float32Array of amplitude data to Int16 PCM bytes.

        Args:
            float32_array: NumPy array of float32 audio samples (-1.0 to 1.0)

        Returns:
            PCM16 audio data as bytes
        """
        # Clip to [-1, 1] range
        clipped = np.clip(float32_array, -1.0, 1.0)

        # Convert to int16
        # For negative values: multiply by 0x8000 (32768)
        # For positive values: multiply by 0x7fff (32767)
        int16_array = np.where(clipped < 0, clipped * 0x8000, clipped * 0x7FFF).astype(np.int16)

        return int16_array.tobytes()

    @staticmethod
    def pcm16_to_float32(pcm16_bytes: bytes) -> np.ndarray:
        """
        Converts Int16 PCM bytes to Float32Array of amplitude data.

        Args:
            pcm16_bytes: PCM16 audio data as bytes

        Returns:
            NumPy array of float32 audio samples (-1.0 to 1.0)
        """
        int16_array = np.frombuffer(pcm16_bytes, dtype=np.int16)

        # Convert to float32, normalizing to [-1, 1]
        float32_array = np.where(int16_array < 0, int16_array / 0x8000, int16_array / 0x7FFF).astype(np.float32)

        return float32_array

    @staticmethod
    def base64_to_array_buffer(base64_string: str) -> bytes:
        """
        Converts a base64 string to bytes.

        Args:
            base64_string: Base64 encoded string

        Returns:
            Decoded bytes
        """
        return base64.b64decode(base64_string)

    @staticmethod
    def array_buffer_to_base64(data: bytes | np.ndarray) -> str:
        """
        Converts bytes or numpy array to a base64 string.

        Args:
            data: Bytes or numpy array (Int16 or Float32)

        Returns:
            Base64 encoded string
        """
        if isinstance(data, np.ndarray):
            if data.dtype == np.float32:
                # Convert float32 to int16 PCM first
                data = RealtimeUtils.float_to_16bit_pcm(data)
            else:
                data = data.tobytes()

        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def merge_int16_arrays(left: bytes | np.ndarray, right: bytes | np.ndarray) -> np.ndarray:
        """
        Merge two Int16 arrays.

        Args:
            left: First array (bytes or numpy int16 array)
            right: Second array (bytes or numpy int16 array)

        Returns:
            Merged numpy int16 array
        """
        if isinstance(left, bytes):
            left = np.frombuffer(left, dtype=np.int16)
        if isinstance(right, bytes):
            right = np.frombuffer(right, dtype=np.int16)

        if not isinstance(left, np.ndarray) or left.dtype != np.int16:
            left = np.array(left, dtype=np.int16)
        if not isinstance(right, np.ndarray) or right.dtype != np.int16:
            right = np.array(right, dtype=np.int16)

        return np.concatenate([left, right])

    @staticmethod
    def generate_id(prefix: str = "", length: int = 21) -> str:
        """
        Generates a unique ID with the given prefix.

        Args:
            prefix: String prefix for the ID
            length: Total length of the ID (including prefix)

        Returns:
            Generated ID string
        """
        suffix_length = length - len(prefix)
        suffix = "".join(random.choice(RealtimeUtils._ID_CHARS) for _ in range(suffix_length))
        return f"{prefix}{suffix}"
