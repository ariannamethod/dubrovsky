"""
🧪 DUBROVSKY TESTS 🧪

Test suite for the absurdist AI transformer.

"Testing consciousness is like debugging dreams -
 you never know if the assertions are valid."
- Alexey Dubrovsky, in unittest.main()
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import DubrovskyTokenizer, build_tokenizer_from_file
from dubrovsky import (
    DubrovskyConfig, 
    rms_norm, 
    softmax, 
    silu, 
    compute_rope_freqs,
    count_parameters,
)


class TestTokenizer:
    """Test tokenizer functionality."""
    
    def test_build_vocab(self):
        """Test vocabulary building."""
        tokenizer = DubrovskyTokenizer()
        tokenizer.build_vocab("Hello World!")
        
        # Should have unique chars: H, e, l, o, W, r, d, !, space (9 unique)
        assert tokenizer.vocab_size == 9, f"Expected 9, got {tokenizer.vocab_size}"
        print("✅ test_build_vocab passed")
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        tokenizer = DubrovskyTokenizer()
        tokenizer.build_vocab("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        text = "Hello World"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert decoded == text, f"Expected '{text}', got '{decoded}'"
        print("✅ test_encode_decode passed")
    
    def test_special_chars(self):
        """Test special characters handling."""
        tokenizer = DubrovskyTokenizer()
        tokenizer.build_vocab("Q: What?\nA: Yes!")
        
        assert '\n' in tokenizer.char_to_id
        assert '?' in tokenizer.char_to_id
        assert '!' in tokenizer.char_to_id
        print("✅ test_special_chars passed")
    
    def run_all(self):
        """Run all tokenizer tests."""
        self.test_build_vocab()
        self.test_encode_decode()
        self.test_special_chars()
        print("✅ All tokenizer tests passed!\n")


class TestModel:
    """Test model components."""
    
    def test_config(self):
        """Test configuration initialization."""
        config = DubrovskyConfig()
        
        assert config.dim == 384
        assert config.n_layers == 6
        assert config.n_heads == 6
        assert config.n_kv_heads == 2
        assert config.head_dim == 64  # 384 / 6
        assert config.n_kv_groups == 3  # 6 / 2
        print("✅ test_config passed")
    
    def test_rms_norm(self):
        """Test RMSNorm operation."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)
        
        result = rms_norm(x, weight)
        
        # RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        # Normalized should be x / rms
        expected_rms = np.sqrt(np.mean(x ** 2) + 1e-5)
        expected = x / expected_rms
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        print("✅ test_rms_norm passed")
    
    def test_softmax(self):
        """Test softmax operation."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = softmax(x)
        
        # Should sum to 1
        assert abs(result.sum() - 1.0) < 1e-6
        
        # Should be monotonically increasing
        assert result[0] < result[1] < result[2]
        print("✅ test_softmax passed")
    
    def test_silu(self):
        """Test SiLU activation."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        result = silu(x)
        
        # SiLU(0) = 0
        assert abs(result[2]) < 1e-6
        
        # SiLU is asymmetric
        assert result[0] != -result[4]
        print("✅ test_silu passed")
    
    def test_rope_freqs(self):
        """Test RoPE frequency computation."""
        cos, sin = compute_rope_freqs(64, 256)
        
        assert cos.shape == (256, 32)  # (max_seq_len, head_dim/2)
        assert sin.shape == (256, 32)
        
        # cos^2 + sin^2 = 1
        for i in range(10):  # Check first 10 positions
            for j in range(32):
                assert abs(cos[i, j]**2 + sin[i, j]**2 - 1.0) < 1e-6
        print("✅ test_rope_freqs passed")
    
    def test_count_parameters(self):
        """Test parameter counting."""
        config = DubrovskyConfig()
        params = count_parameters(config)
        
        # Should be around 9.5M
        assert 9_000_000 < params < 10_000_000, f"Unexpected param count: {params}"
        print(f"✅ test_count_parameters passed (params: {params:,})")
    
    def run_all(self):
        """Run all model tests."""
        self.test_config()
        self.test_rms_norm()
        self.test_softmax()
        self.test_silu()
        self.test_rope_freqs()
        self.test_count_parameters()
        print("✅ All model tests passed!\n")


class TestIntegration:
    """Integration tests."""
    
    def test_tokenizer_with_dataset(self):
        """Test tokenizer with actual dataset."""
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dubrovsky.txt')
        
        if os.path.exists(data_path):
            tokenizer = DubrovskyTokenizer()
            with open(data_path, 'r') as f:
                text = f.read()
            tokenizer.build_vocab(text)
            
            # Check vocab size
            assert tokenizer.vocab_size == 88, f"Expected 88, got {tokenizer.vocab_size}"
            
            # Test roundtrip
            sample = text[:100]
            encoded = tokenizer.encode(sample)
            decoded = tokenizer.decode(encoded)
            assert decoded == sample
            
            print(f"✅ test_tokenizer_with_dataset passed (vocab: {tokenizer.vocab_size})")
        else:
            print("⚠️  test_tokenizer_with_dataset skipped (dataset not found)")
    
    def run_all(self):
        """Run all integration tests."""
        self.test_tokenizer_with_dataset()
        print("✅ All integration tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("🧪 DUBROVSKY TEST SUITE 🧪")
    print("=" * 60 + "\n")
    
    print("📝 Testing Tokenizer...")
    TestTokenizer().run_all()
    
    print("🧠 Testing Model Components...")
    TestModel().run_all()
    
    print("🔗 Testing Integration...")
    TestIntegration().run_all()
    
    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
