package dev.sruti.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * Tests for the parts of the agent that are pure logic: retrieval, argument
 * validation, and the workspace boundary.
 *
 * These are where a mistake is silent. A grammar guarantees the shape of a call;
 * nothing guarantees the call is sensible, so validation is the last check before
 * something runs.
 */
class AgentTest {

    @get:Rule val temp = TemporaryFolder()

    private fun tool(
        name: String,
        description: String = "",
        keywords: List<String> = emptyList(),
        parameters: List<ToolParameter> = emptyList(),
    ) = Tool(name, description, parameters, keywords = keywords) { ToolResult.Success("") }

    // --- retrieval -----------------------------------------------------------

    @Test
    fun `retrieval ranks a name match above a description match`() {
        val registry = ToolRegistry(
            listOf(
                tool("send_email", "Deliver a message to someone"),
                tool("read_file", "Read a file from storage"),
            ),
        )
        val results = registry.retrieve("read the file please", setOf(ToolTier.InApp))
        assertEquals("read_file", results.first().name)
    }

    @Test
    fun `curated keywords retrieve a tool whose name does not match`() {
        // Users say "open"; the tool is called read_file. Without keywords the
        // right tool never reaches the model's candidate list at all.
        val registry = ToolRegistry(
            listOf(
                tool("read_file", "Read a text file", keywords = listOf("open", "cat")),
                tool("set_mode", "Change the mode"),
            ),
        )
        val results = registry.retrieve("open my notes", setOf(ToolTier.InApp))
        assertEquals("read_file", results.first().name)
    }

    @Test
    fun `retrieval caps the candidate list`() {
        // Accuracy collapses as candidates grow at this model size, so the cap is
        // a correctness measure rather than a performance one.
        val registry = ToolRegistry((1..20).map { tool("tool_$it", "does thing $it") })
        assertEquals(5, registry.retrieve("thing", setOf(ToolTier.InApp), limit = 5).size)
    }

    @Test
    fun `retrieval excludes tools whose tier is not enabled`() {
        val registry = ToolRegistry(
            listOf(
                tool("read_file", "Read a file"),
                Tool(
                    name = "run_shell",
                    description = "Run a shell command",
                    tier = ToolTier.Shell,
                ) { ToolResult.Success("") },
            ),
        )
        val results = registry.retrieve("run a shell command", setOf(ToolTier.InApp))
        assertTrue(results.none { it.name == "run_shell" })

        val withShell = registry.retrieve(
            "run a shell command",
            setOf(ToolTier.InApp, ToolTier.Shell),
        )
        assertEquals("run_shell", withShell.first().name)
    }

    @Test
    fun `retrieval is stable for an unmatched request`() {
        val registry = ToolRegistry(listOf(tool("read_file"), tool("set_mode")))
        val results = registry.retrieve("xyzzy", setOf(ToolTier.InApp))
        assertEquals(2, results.size)
    }

    // --- argument parsing ----------------------------------------------------

    @Test
    fun `parses a constrained JSON object into plain values`() {
        val parsed = parseToolArguments("""{"path":"/tmp/a.txt","max_lines":10}""")
        assertNotNull(parsed)
        assertEquals("/tmp/a.txt", parsed!!["path"])
        // Unquoted, so a tool receives "10" rather than "\"10\"".
        assertEquals("10", parsed["max_lines"])
    }

    @Test
    fun `parses booleans and negative numbers`() {
        val parsed = parseToolArguments("""{"flag":true,"offset":-5}""")!!
        assertEquals("true", parsed["flag"])
        assertEquals("-5", parsed["offset"])
    }

    @Test
    fun `rejects malformed or empty output`() {
        assertNull(parseToolArguments(""))
        assertNull(parseToolArguments("not json"))
        assertNull(parseToolArguments("[1,2]"))
    }

    // --- validation ----------------------------------------------------------

    private val readFile = Tool(
        name = "read_file",
        description = "Read a file",
        parameters = listOf(
            ToolParameter("path", ToolParameter.Type.String, "Path"),
            ToolParameter("max_lines", ToolParameter.Type.Integer, "Limit", required = false),
        ),
    ) { ToolResult.Success("") }

    @Test
    fun `accepts a well-formed call`() {
        assertNull(validateToolCall(readFile, mapOf("path" to "notes.txt")))
        assertNull(validateToolCall(readFile, mapOf("path" to "notes.txt", "max_lines" to "20")))
    }

    @Test
    fun `rejects a missing or empty required argument`() {
        // The grammar guarantees the key can appear; it cannot guarantee the model
        // put anything useful in it.
        assertNotNull(validateToolCall(readFile, emptyMap()))
        assertNotNull(validateToolCall(readFile, mapOf("path" to "")))
        assertNotNull(validateToolCall(readFile, mapOf("path" to "   ")))
    }

    @Test
    fun `rejects a value of the wrong type`() {
        val message = validateToolCall(readFile, mapOf("path" to "a", "max_lines" to "many"))
        assertNotNull(message)
        assertTrue(message!!.contains("integer"))
    }

    @Test
    fun `rejects an enum value outside the declared set`() {
        val setMode = Tool(
            name = "set_mode",
            description = "Set mode",
            parameters = listOf(
                ToolParameter(
                    "mode", ToolParameter.Type.Enum, "Mode",
                    values = listOf("fast", "quality"),
                ),
            ),
        ) { ToolResult.Success("") }

        assertNull(validateToolCall(setMode, mapOf("mode" to "fast")))
        assertNotNull(validateToolCall(setMode, mapOf("mode" to "turbo")))
    }

    @Test
    fun `rejects arguments the schema does not declare`() {
        // The grammar should make these unreachable, so seeing one means something
        // upstream is wrong and executing anyway would hide it.
        val message = validateToolCall(readFile, mapOf("path" to "a", "sudo" to "true"))
        assertNotNull(message)
        assertTrue(message!!.contains("sudo"))
    }

    // --- workspace boundary --------------------------------------------------

    @Test
    fun `resolves ordinary paths inside the workspace`() {
        val root = temp.newFolder("workspace")
        assertNotNull(resolveInWorkspace(root, "notes.txt"))
        assertNotNull(resolveInWorkspace(root, "sub/dir/notes.txt"))
        assertNotNull(resolveInWorkspace(root, "."))
    }

    @Test
    fun `rejects paths that escape the workspace`() {
        val root = temp.newFolder("workspace")
        // Judged on what they resolve to, not how they are spelled.
        assertNull(resolveInWorkspace(root, "../outside.txt"))
        assertNull(resolveInWorkspace(root, "sub/../../outside.txt"))
        assertNull(resolveInWorkspace(root, "a/b/../../../c"))
        assertNull(resolveInWorkspace(root, "../../../../../../etc/passwd"))
    }

    @Test
    fun `an absolute path is confined rather than honoured`() {
        // File(parent, child) treats the child as relative even with a leading
        // slash, so "/etc/passwd" lands at <workspace>/etc/passwd. That is the
        // safe outcome, and worth pinning: a future rewrite that resolves the
        // child on its own would silently start honouring absolute paths.
        val root = temp.newFolder("workspace")
        val resolved = resolveInWorkspace(root, "/etc/passwd")
        assertNotNull(resolved)
        assertTrue(resolved!!.path.startsWith(root.canonicalFile.path + "/"))
        assertTrue(resolved.path.endsWith("/workspace/etc/passwd"))
    }

    @Test
    fun `rejects a sibling directory sharing a name prefix`() {
        // "workspace-evil" starts with "workspace" as a string but is not inside
        // it; a naive startsWith check on the raw path would let this through.
        val root = temp.newFolder("workspace")
        temp.newFolder("workspace-evil")
        assertNull(resolveInWorkspace(root, "../workspace-evil/secret.txt"))
    }

    @Test
    fun `the workspace root itself resolves`() {
        val root = temp.newFolder("workspace")
        val resolved = resolveInWorkspace(root, "")
        assertNotNull(resolved)
        assertEquals(root.canonicalFile, resolved)
    }

    @Test
    fun `escape hatch name is not a real tool`() {
        // If a tool were ever called "none" the model could not decline.
        val registry = ToolRegistry(listOf(tool("read_file")))
        assertFalse(registry.all.any { it.name == GbnfGrammar.NO_TOOL })
    }
}
