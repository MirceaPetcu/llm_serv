import pytest
import sys
import os

# Add the parent directory to the Python path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from llm_serv.structured_response.utils import extract_int, extract_float, extract_bool


class TestExtractInt:
    """Comprehensive tests for extract_int function."""
    
    def test_basic_integers(self):
        """Test basic integer extraction."""
        assert extract_int("42") == 42
        assert extract_int("-123") == -123
        assert extract_int("0") == 0
        assert extract_int("-0") == 0
    
    def test_embedded_integers(self):
        """Test integers embedded in text."""
        assert extract_int("The answer is 42.") == 42
        assert extract_int("Price: $25 today") == 25
        assert extract_int("Temperature: -10°C") == -10
        assert extract_int("Chapter 7 begins here") == 7
    
    def test_html_xml_tags(self):
        """Test integers with HTML/XML tags."""
        assert extract_int('42<ref id="23"/>') == 42
        assert extract_int('<span>99</span>') == 99
        assert extract_int('<div class="price">-15</div>') == -15
        assert extract_int('Value: <b>123</b> units') == 123
    
    def test_quoted_integers(self):
        """Test quoted integers."""
        assert extract_int('"42"') == 42
        assert extract_int("'42'") == 42
        assert extract_int('"-123"') == -123
        assert extract_int("'-456'") == -456
        assert extract_int('""789""') == 789
    
    def test_whitespace_handling(self):
        """Test integers with various whitespace."""
        assert extract_int("  42  ") == 42
        assert extract_int("\t-123\n") == -123
        assert extract_int("   0   ") == 0
    
    def test_multiple_numbers(self):
        """Test strings with multiple numbers (should return first)."""
        assert extract_int("First: 42, Second: 99") == 42
        assert extract_int("Numbers: -10, 20, 30") == -10
        assert extract_int("Mix: 1.5 and 7") == 7  # Should skip decimal and get integer
    
    def test_leading_zeros(self):
        """Test integers with leading zeros."""
        assert extract_int("007") == 7
        assert extract_int("-00123") == -123
        assert extract_int("0000") == 0
    
    def test_large_numbers(self):
        """Test very large integers."""
        assert extract_int("9223372036854775807") == 9223372036854775807  # max int64
        assert extract_int("-9223372036854775808") == -9223372036854775808  # min int64
    
    def test_numbers_with_separators(self):
        """Test numbers with comma separators (should extract digits only)."""
        assert extract_int("1,000") == 1000
        assert extract_int("1,234,567") == 1234567
        assert extract_int("-2,500") == -2500
    
    def test_edge_position_numbers(self):
        """Test numbers at beginning/end of strings."""
        assert extract_int("42 is the answer") == 42
        assert extract_int("The answer is 42") == 42
        assert extract_int("42") == 42
    
    def test_special_characters_around_numbers(self):
        """Test numbers surrounded by special characters."""
        assert extract_int("(42)") == 42
        assert extract_int("[123]") == 123
        assert extract_int("{-456}") == -456
        assert extract_int("$99") == 99
        assert extract_int("15%") == 15
    
    def test_unicode_and_mixed_content(self):
        """Test with Unicode and mixed content."""
        assert extract_int("价格: 42元") == 42
        assert extract_int("Température: -5°") == -5
        assert extract_int("α = 123 β") == 123
    
    def test_multiple_minus_signs(self):
        """Test handling of multiple minus signs."""
        assert extract_int("--42") == -42  # Should handle double minus
        assert extract_int("---123") == -123
    
    def test_error_cases(self):
        """Test cases that should raise ValueError."""
        with pytest.raises(ValueError):
            extract_int("")
        with pytest.raises(ValueError):
            extract_int("   ")
        with pytest.raises(ValueError):
            extract_int("no numbers here")
        with pytest.raises(ValueError):
            extract_int("abc def ghi")
        with pytest.raises(ValueError):
            extract_int("only-minus-")
        with pytest.raises(ValueError):
            extract_int("!!!")


class TestExtractFloat:
    """Comprehensive tests for extract_float function."""
    
    def test_basic_floats(self):
        """Test basic float extraction."""
        assert extract_float("3.14") == 3.14
        assert extract_float("-2.5") == -2.5
        assert extract_float("0.0") == 0.0
        assert extract_float("42") == 42.0
    
    def test_scientific_notation(self):
        """Test scientific notation."""
        assert extract_float("1.5e-3") == 0.0015
        assert extract_float("2E+4") == 20000.0
        assert extract_float("-1.23e2") == -123.0
        assert extract_float("6.022e23") == 6.022e23
        assert extract_float("1e0") == 1.0
    
    def test_embedded_floats(self):
        """Test floats embedded in text."""
        assert extract_float("The value is 3.14.") == 3.14
        assert extract_float("Temperature: -10.5°C") == -10.5
        assert extract_float("Price: $19.99") == 19.99
        assert extract_float("Progress: 85.7% complete") == 85.7
    
    def test_html_xml_tags(self):
        """Test floats with HTML/XML tags."""
        assert extract_float('2.5<ref id="23"/>') == 2.5
        assert extract_float('<span>-15.7</span>') == -15.7
        assert extract_float('Value: <b>123.456</b>') == 123.456
    
    def test_quoted_floats(self):
        """Test quoted floats."""
        assert extract_float('"3.14"') == 3.14
        assert extract_float("'-1.23'") == -1.23
        assert extract_float('""42.0""') == 42.0
        assert extract_float("' -99.9 '") == -99.9
    
    def test_integers_as_floats(self):
        """Test that integers are properly converted to floats."""
        assert extract_float("42") == 42.0
        assert extract_float("-123") == -123.0
        assert extract_float("0") == 0.0
    
    def test_leading_trailing_zeros(self):
        """Test floats with leading/trailing zeros."""
        assert extract_float("007.500") == 7.5
        assert extract_float("0.123") == 0.123
        assert extract_float("123.000") == 123.0
        assert extract_float(".5") == 0.5
        assert extract_float("5.") == 5.0
    
    def test_very_large_small_numbers(self):
        """Test very large and very small numbers."""
        assert extract_float("1.7976931348623157e+308") == 1.7976931348623157e+308
        assert extract_float("2.2250738585072014e-308") == 2.2250738585072014e-308
        assert extract_float("-1e100") == -1e100
    
    def test_multiple_decimal_points(self):
        """Test handling of multiple decimal points (should get first valid)."""
        assert extract_float("3.14.159") == 3.14  # Should stop at first valid number
        assert extract_float("1.2.3.4") == 1.2
    
    def test_numbers_with_separators(self):
        """Test numbers with comma separators."""
        assert extract_float("1,234.56") == 1234.56
        assert extract_float("-2,500.75") == -2500.75
    
    def test_special_float_values(self):
        """Test special float values like inf and nan."""
        # Note: These might not be handled by current implementation
        # We'll see what happens in the test run
        pass
    
    def test_programming_notation(self):
        """Test programming-style float notation."""
        assert extract_float("3.14f") == 3.14  # C-style float suffix
        assert extract_float("2.5d") == 2.5    # Java-style double suffix
    
    def test_error_cases(self):
        """Test cases that should raise ValueError."""
        with pytest.raises(ValueError):
            extract_float("")
        with pytest.raises(ValueError):
            extract_float("   ")
        with pytest.raises(ValueError):
            extract_float("no numbers here")
        with pytest.raises(ValueError):
            extract_float("abc def ghi")
        with pytest.raises(ValueError):
            extract_float("only dots ...")


class TestExtractBool:
    """Comprehensive tests for extract_bool function."""
    
    def test_basic_true_values(self):
        """Test basic true values."""
        assert extract_bool("true") == True
        assert extract_bool("True") == True
        assert extract_bool("TRUE") == True
        assert extract_bool("1") == True
    
    def test_basic_false_values(self):
        """Test basic false values."""
        assert extract_bool("false") == False
        assert extract_bool("False") == False
        assert extract_bool("FALSE") == False
        assert extract_bool("0") == False
    
    def test_case_variations(self):
        """Test various case combinations."""
        assert extract_bool("TrUe") == True
        assert extract_bool("tRuE") == True
        assert extract_bool("FaLsE") == False
        assert extract_bool("fAlSe") == False
    
    def test_whitespace_handling(self):
        """Test booleans with whitespace (current implementation may not handle this)."""
        # These tests will show if the current implementation handles whitespace
        pass
    
    def test_numeric_strings(self):
        """Test numeric representations of booleans."""
        assert extract_bool("1") == True
        assert extract_bool("0") == False
        # Let's see how other numbers are handled
    
    def test_alternative_boolean_words(self):
        """Test alternative boolean representations (may not be implemented)."""
        # These would be enhancements to test:
        # yes/no, on/off, enable/disable, etc.
        pass
    
    def test_empty_and_edge_cases(self):
        """Test empty strings and edge cases."""
        # Let's see how empty strings are handled
        result_empty = extract_bool("")
        result_space = extract_bool("   ")
        # These will show current behavior
    
    def test_non_boolean_strings(self):
        """Test how non-boolean strings are handled."""
        # The current implementation returns bool(text) for non-matches
        # Let's see what this produces
        result1 = extract_bool("hello")
        result2 = extract_bool("world")
        result3 = extract_bool("123")


def run_specific_test_group(test_class_name):
    """Helper function to run a specific test group."""
    import subprocess
    cmd = f"python -m pytest {__file__}::{test_class_name} -v"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    print(f"Return code: {result.returncode}")
    return result.returncode == 0


if __name__ == "__main__":
    # Run all tests when script is executed directly
    import subprocess
    cmd = f"python -m pytest {__file__} -v"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Test Results:")
    print("=" * 50)
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    print(f"\nOverall result: {'PASSED' if result.returncode == 0 else 'FAILED'}")
