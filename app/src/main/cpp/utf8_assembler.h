#pragma once

// Reassembles UTF-8 across token boundaries.
//
// A llama.cpp token piece is a byte string, not a character string. Multi-byte
// characters routinely straddle two tokens — emoji and CJK almost always do —
// so handing raw pieces to NewStringUTF produces corrupted text. This holds back
// any trailing bytes that form an unfinished sequence until the rest arrives.
//
// Header-only so it can be compiled into a host-side test without dragging in
// JNI or llama.cpp.

#include <cstddef>
#include <string>

namespace sruti {

class Utf8Assembler {
public:
    // Appends raw bytes and returns whatever now forms complete UTF-8, retaining
    // any incomplete trailing sequence for a later call.
    std::string push(const char * data, size_t len) {
        buffer_.append(data, len);

        const size_t keep = incomplete_tail_len(buffer_);
        const size_t emit = buffer_.size() - keep;
        if (emit == 0) {
            return {};
        }

        std::string out = buffer_.substr(0, emit);
        buffer_.erase(0, emit);
        return out;
    }

    std::string push(const std::string & s) { return push(s.data(), s.size()); }

    // Releases any held-back bytes. Call once at end of stream, otherwise a
    // truncated final character would be silently dropped.
    std::string flush() {
        std::string out = buffer_;
        buffer_.clear();
        return out;
    }

    bool empty() const { return buffer_.empty(); }

private:
    // Number of trailing bytes forming a started-but-unfinished sequence.
    static size_t incomplete_tail_len(const std::string & s) {
        const size_t n = s.size();
        // A UTF-8 sequence is at most 4 bytes, so only the last 3 can be partial.
        const size_t max_look = n < 3 ? n : 3;

        for (size_t back = 1; back <= max_look; ++back) {
            const auto c = static_cast<unsigned char>(s[n - back]);
            if ((c & 0xC0) == 0x80) {
                continue;  // continuation byte; keep walking back to the lead
            }

            size_t needed = 0;
            if      ((c & 0x80) == 0x00) needed = 1;
            else if ((c & 0xE0) == 0xC0) needed = 2;
            else if ((c & 0xF0) == 0xE0) needed = 3;
            else if ((c & 0xF8) == 0xF0) needed = 4;
            else return 0;  // invalid lead byte: pass through rather than stall

            return needed > back ? back : 0;
        }
        return 0;
    }

    std::string buffer_;
};

}  // namespace sruti
