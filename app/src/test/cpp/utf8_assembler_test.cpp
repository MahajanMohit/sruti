// Host-side test for Utf8Assembler.
//
// Runs on the build machine, not a device: the logic is pure byte manipulation
// with no JNI or llama.cpp dependency, and it is exactly the kind of code that
// fails silently in production. Build and run with tools/run_native_tests.sh.

#include "../../main/cpp/utf8_assembler.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using sruti::Utf8Assembler;

namespace {

int failures = 0;

void expect_eq(const std::string & actual, const std::string & expected, const char * what) {
    if (actual != expected) {
        std::printf("FAIL %s\n  expected: \"%s\"\n  actual:   \"%s\"\n",
                    what, expected.c_str(), actual.c_str());
        ++failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

// Feeds `input` one byte at a time — the worst case for boundary handling —
// and returns the concatenation of everything emitted, including the flush.
std::string feed_byte_by_byte(const std::string & input) {
    Utf8Assembler a;
    std::string out;
    for (char c : input) {
        out += a.push(&c, 1);
    }
    out += a.flush();
    return out;
}

// Feeds `input` in chunks of `size`, which is how real token pieces arrive.
std::string feed_chunked(const std::string & input, size_t size) {
    Utf8Assembler a;
    std::string out;
    for (size_t i = 0; i < input.size(); i += size) {
        out += a.push(input.data() + i, std::min(size, input.size() - i));
    }
    out += a.flush();
    return out;
}

void test_ascii_passes_through() {
    Utf8Assembler a;
    expect_eq(a.push("hello", 5), "hello", "ascii emits immediately");
    expect_eq(a.flush(), "", "nothing held back after ascii");
}

void test_split_two_byte() {
    // U+00E9 (e-acute) = C3 A9
    Utf8Assembler a;
    expect_eq(a.push("\xC3", 1), "", "lead byte of 2-byte seq is held");
    expect_eq(a.push("\xA9", 1), "\xC3\xA9", "2-byte seq emitted once complete");
}

void test_split_three_byte() {
    // U+4E2D (CJK) = E4 B8 AD
    Utf8Assembler a;
    expect_eq(a.push("\xE4", 1), "", "3-byte lead held");
    expect_eq(a.push("\xB8", 1), "", "3-byte partial still held");
    expect_eq(a.push("\xAD", 1), "\xE4\xB8\xAD", "3-byte seq emitted once complete");
}

void test_split_four_byte_emoji() {
    // U+1F600 (grinning face) = F0 9F 98 80
    Utf8Assembler a;
    expect_eq(a.push("\xF0", 1), "", "4-byte lead held");
    expect_eq(a.push("\x9F", 1), "", "4-byte partial held (1)");
    expect_eq(a.push("\x98", 1), "", "4-byte partial held (2)");
    expect_eq(a.push("\x80", 1), "\xF0\x9F\x98\x80", "emoji emitted once complete");
}

void test_prefix_emitted_while_tail_held() {
    Utf8Assembler a;
    // "ab" plus the lead byte of a 2-byte sequence.
    expect_eq(a.push("ab\xC3", 3), "ab", "complete prefix emits, partial tail held");
    expect_eq(a.push("\xA9", 1), "\xC3\xA9", "held tail completes on next push");
}

void test_reassembly_is_lossless() {
    const std::string mixed =
        "Hello \xE4\xB8\xAD\xE6\x96\x87 world \xF0\x9F\x8E\xB8 caf\xC3\xA9 "
        "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82!";

    expect_eq(feed_byte_by_byte(mixed), mixed, "byte-by-byte round-trips losslessly");

    for (size_t chunk : {2u, 3u, 5u, 7u}) {
        char label[64];
        std::snprintf(label, sizeof(label), "chunk size %zu round-trips losslessly", chunk);
        expect_eq(feed_chunked(mixed, chunk), mixed, label);
    }
}

void test_truncated_tail_is_flushed_not_dropped() {
    Utf8Assembler a;
    expect_eq(a.push("ok\xF0\x9F", 4), "ok", "truncated emoji held back");
    // A stream that ends mid-character is malformed, but dropping bytes silently
    // is worse than passing them on — flush must surrender whatever it holds.
    expect_eq(a.flush(), "\xF0\x9F", "flush releases the incomplete tail");
    expect_eq(a.flush(), "", "flush is idempotent");
}

void test_invalid_lead_byte_does_not_stall() {
    // 0xFF is not a legal UTF-8 lead byte. It must pass straight through rather
    // than be buffered forever waiting for continuations that never come.
    Utf8Assembler a;
    expect_eq(a.push("\xFF", 1), "\xFF", "invalid lead byte passes through");
    assert(a.empty());
}

void test_stray_continuation_bytes_do_not_stall() {
    Utf8Assembler a;
    // Continuation bytes with no lead. Walking back finds no valid lead within
    // the 3-byte window, so they should be released rather than accumulate.
    const std::string out = a.push("\x80\x80\x80\x80", 4);
    expect_eq(out + a.flush(), "\x80\x80\x80\x80", "stray continuations are released");
}

}  // namespace

int main() {
    test_ascii_passes_through();
    test_split_two_byte();
    test_split_three_byte();
    test_split_four_byte_emoji();
    test_prefix_emitted_while_tail_held();
    test_reassembly_is_lossless();
    test_truncated_tail_is_flushed_not_dropped();
    test_invalid_lead_byte_does_not_stall();
    test_stray_continuation_bytes_do_not_stall();

    if (failures == 0) {
        std::printf("\nAll UTF-8 assembler tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
