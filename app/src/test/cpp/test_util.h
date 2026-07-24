#pragma once

// Minimal assertion and fixture helpers for the host-side native tests.
//
// Deliberately not a test framework: these tests run under ASan/UBSan in a shell
// script, and a dependency-free header keeps that setup honest.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>

namespace sruti_test {

inline int & failure_count() {
    static int failures = 0;
    return failures;
}

inline void record(bool ok, const std::string & what, const std::string & detail = {}) {
    if (ok) {
        std::printf("ok   %s\n", what.c_str());
    } else {
        ++failure_count();
        std::printf("FAIL %s%s%s\n", what.c_str(),
                    detail.empty() ? "" : " -- ", detail.c_str());
    }
}

template <typename A, typename B>
inline void expect_eq(const A & actual, const B & expected, const std::string & what) {
    if (actual == static_cast<A>(expected)) {
        record(true, what);
    } else {
        record(false, what, "values differ");
    }
}

/// A directory under /tmp that is removed when the object goes out of scope.
class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/sruti_test_XXXXXX";
        const char * made = ::mkdtemp(tmpl);
        if (made == nullptr) {
            std::perror("mkdtemp");
            std::abort();
        }
        path_ = made;
    }

    ~TempDir() {
        // Shallow removal is enough: tests only create flat files.
        DIR * d = ::opendir(path_.c_str());
        if (d != nullptr) {
            while (dirent * entry = ::readdir(d)) {
                const std::string name = entry->d_name;
                if (name != "." && name != "..") {
                    ::unlink((path_ + "/" + name).c_str());
                }
            }
            ::closedir(d);
        }
        ::rmdir(path_.c_str());
    }

    TempDir(const TempDir &) = delete;
    TempDir & operator=(const TempDir &) = delete;

    const std::string & path() const { return path_; }
    std::string file(const std::string & name) const { return path_ + "/" + name; }

private:
    std::string path_;
};

inline void write_text_file(const std::string & path, const std::string & text) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::perror(path.c_str());
        std::abort();
    }
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
}

inline int summary(const char * suite) {
    if (failure_count() == 0) {
        std::printf("\nAll %s tests passed.\n", suite);
        return 0;
    }
    std::printf("\n%d %s test(s) failed.\n", failure_count(), suite);
    return 1;
}

}  // namespace sruti_test

#define ASSERT_TRUE(cond, what)  ::sruti_test::record((cond), (what))
#define ASSERT_EQ(a, b, what)    ::sruti_test::expect_eq((a), (b), (what))

using ::sruti_test::TempDir;
using ::sruti_test::write_text_file;

inline int test_summary(const char * suite) { return ::sruti_test::summary(suite); }
