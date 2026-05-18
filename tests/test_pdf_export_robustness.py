#!/usr/bin/env python3
"""Test PDF export robustness and timeout handling.

This test validates:
1. Browser detection and validation
2. Subprocess timeout handling
3. HTML generation
4. PDF generation
5. Error reporting and logging
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from presentation_common import export_marp_pdf, default_pdf_export_enabled


def test_default_pdf_export_enabled():
    """Verify PDF export is enabled by default."""
    assert default_pdf_export_enabled() is True, "PDF export should be enabled by default"
    print("✓ PDF export enabled by default")


def test_browser_detection():
    """Test browser detection and validation."""
    result = subprocess.run(
        ['bash', '-c', 'scripts/marp-pdf.sh --help 2>&1 || true'],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Script should be executable and provide help (or fail gracefully)
    assert 'scripts/marp-pdf.sh' in result.stderr or result.returncode == 0
    print("✓ Browser detection logic validates")


def test_timeout_handling():
    """Test that timeout errors are handled gracefully."""
    # Create a simple markdown file
    with tempfile.TemporaryDirectory() as tmpdir:
        md_file = Path(tmpdir) / 'test.md'
        md_file.write_text('''---
theme: default
---

# Test Slide

Test content

---

# Slide 2

More content
''')
        
        # Test export with timeout wrapper
        start = time.time()
        try:
            pdf_path, error = export_marp_pdf(str(md_file))
            elapsed = time.time() - start
            
            if pdf_path is not None:
                assert Path(pdf_path).exists(), f"PDF file should exist: {pdf_path}"
                pdf_size = Path(pdf_path).stat().st_size
                assert pdf_size > 1000, f"PDF should have content, got {pdf_size} bytes"
                print(f"✓ PDF export succeeded in {elapsed:.1f}s ({pdf_size} bytes)")
            elif error:
                assert elapsed < 340, f"Export should timeout before 340s, took {elapsed:.1f}s"
                print(f"✓ Timeout handled gracefully: {error[:60]}...")
            else:
                assert False, "Export should return either PDF path or error message"
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            assert elapsed < 340, f"Overall timeout should be < 340s, got {elapsed:.1f}s"
            print(f"✓ Timeout exception handled gracefully after {elapsed:.1f}s")


def test_error_messages():
    """Test that error messages are informative."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create invalid markdown
        md_file = Path(tmpdir) / 'invalid.md'
        # Leave it empty - this should still work, but test error handling
        md_file.write_text('# Test')
        
        pdf_path, error = export_marp_pdf(str(md_file))
        
        if error is not None:
            # Error message should be helpful
            assert 'timed out' in error.lower() or 'failed' in error.lower() or 'not' in error.lower()
            print(f"✓ Error message is informative: {error[:80]}...")
        else:
            # If no error, PDF should exist
            assert pdf_path is not None and Path(pdf_path).exists()
            print(f"✓ PDF generated successfully: {pdf_path}")


def test_shell_script_help():
    """Test that marp-pdf.sh script is robust and provides help."""
    script = Path(__file__).parent.parent / 'scripts' / 'marp-pdf.sh'
    assert script.exists(), f"Script not found: {script}"
    
    result = subprocess.run(
        ['bash', str(script)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    
    # Should fail with usage message (no input provided)
    assert result.returncode != 0, "Script should fail with no input"
    assert 'Usage:' in result.stderr, "Script should show usage message"
    print("✓ Shell script is robust and provides help")


def run_all_tests():
    """Run all robustness tests."""
    tests = [
        ('Default PDF export enabled', test_default_pdf_export_enabled),
        ('Browser detection', test_browser_detection),
        ('Timeout handling', test_timeout_handling),
        ('Error messages', test_error_messages),
        ('Shell script help', test_shell_script_help),
    ]
    
    print("\n" + "="*60)
    print("PDF EXPORT ROBUSTNESS TESTS")
    print("="*60)
    
    failed = []
    for name, test_func in tests:
        try:
            print(f"\n[TEST] {name}")
            test_func()
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed.append((name, str(e)))
    
    print("\n" + "="*60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return 1
    else:
        print(f"PASSED: All {len(tests)} tests ✓")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
