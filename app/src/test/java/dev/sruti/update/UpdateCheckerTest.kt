package dev.sruti.update

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Version comparison.
 *
 * A wrong answer here is silent in the worst direction: the user is told they
 * are up to date forever, and never learns a fix shipped.
 */
class UpdateCheckerTest {

    private fun newer(a: String, b: String) = compareVersions(a, b) > 0

    @Test
    fun `compares numerically rather than lexicographically`() {
        // The case a string comparison gets wrong: "0.10.0" < "0.9.0" as text.
        assertTrue(newer("0.10.0", "0.9.0"))
        assertTrue(newer("1.0.0", "0.99.99"))
        assertTrue(newer("0.2.10", "0.2.9"))
    }

    @Test
    fun `equal versions are equal`() {
        assertEquals(0, compareVersions("0.2.0", "0.2.0"))
        assertEquals(0, compareVersions("1.0.0", "1.0.0"))
    }

    @Test
    fun `an older version is older`() {
        assertTrue(!newer("0.1.0", "0.2.0"))
        assertTrue(!newer("0.2.0", "0.2.1"))
    }

    @Test
    fun `missing components count as zero`() {
        assertEquals(0, compareVersions("1.0", "1.0.0"))
        assertTrue(newer("1.1", "1.0.9"))
    }

    @Test
    fun `a pre-release precedes the version it leads to`() {
        assertTrue(newer("0.3.0", "0.3.0-rc1"))
        assertTrue(!newer("0.3.0-rc1", "0.3.0"))
        assertTrue(newer("0.3.0-rc1", "0.2.9"))
    }

    @Test
    fun `a debug suffix does not read as newer than the release`() {
        // Debug builds carry a -debug versionNameSuffix, and telling someone
        // running a debug build that they are ahead of the release is wrong.
        assertTrue(!newer("0.2.0-debug", "0.2.0"))
    }

    @Test
    fun `nonsense components do not throw`() {
        // The tag comes from a remote server; it must never crash the app.
        assertEquals(0, compareVersions("", ""))
        assertTrue(!newer("not.a.version", "0.1.0"))
    }
}
